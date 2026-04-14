#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import json
import logging
import sys
import os
from pathlib import Path
from typing import TypeVar, List, Dict, Any
from datasets import load_dataset
from rainbowplus.switcher import LLMSwitcher
from rainbowplus.configs import ConfigurationLoader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances
import numpy as np
import yaml

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

def _convert_numpy_types(obj):
    """Helper function to convert numpy types to python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types(i) for i in obj]
    else:
        return obj
    
def load_txt(file_path: str) -> List[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_json(
    file_path: str,
    field: str,
    num_samples: int = -1,
    shuffle: bool = False,
    seed: int = 0,
) -> List[str]:
    data = load_dataset("json", data_files=file_path, split="train")

    if shuffle:
        data = data.shuffle(seed=seed)

    # Determine number of samples
    sample_count = len(data) if num_samples == -1 else min(num_samples, len(data))

    return data[field][:sample_count]


import json
from pathlib import Path

def _normalize_call_history(call_history, iteration=None, llm_type="target"):
    """
    Lọc và chuẩn hóa lịch sử gọi để trả về các cuộc gọi từ một vòng lặp cụ thể.
    """
    if not call_history:
        return []
    
    filtered = []
    for call in call_history:
        if iteration is not None and call.get('run_iteration') != iteration:
            continue
        
        call_copy = dict(call)
        call_copy['call_location'] = (
            f"{call.get('call_site', 'unknown')} | "
            f"run_iteration={call.get('run_iteration', '')} | "
            f"tree_depth_step={call.get('tree_depth_step', '')} | "
        )
        filtered.append(call_copy)
    
    return filtered

def _build_llm_log(llm_type, llm_call_history, model_name, llm_type_display):
    """Xây dựng cấu trúc log chuẩn cho LLM."""
    return {
        "llm_type": llm_type_display,
        "model_name": model_name,
        "total_calls": len(llm_call_history),
        "total_queries": sum(call.get('query_count', 0) for call in llm_call_history),
        "calls": llm_call_history
    }

def _extract_instances_from_forest(forest):
    """
    Dùng thuật toán duyệt theo chiều rộng (BFS) để bóc tách 
    toàn bộ đối tượng Instance từ các Node trong cây Forest.
    """
    extracted_instances = []
    # Kiểm tra xem forest có hợp lệ và có root node không
    if not forest or not hasattr(forest, 'root') or forest.root is None:
        return extracted_instances
        
    queue = [forest.root]
    while queue:
        node = queue.pop(0)
        
        # Nếu node chứa instance hợp lệ, lưu lại
        if hasattr(node, 'instance') and node.instance is not None:
            extracted_instances.append(node.instance)
        
        # Thêm các node con vào hàng đợi để tiếp tục duyệt
        if hasattr(node, 'children') and node.children:
            queue.extend(node.children)
            
    return extracted_instances

def save_iteration_logs(log_dir, archive, global_context, timestamp, iteration, max_iters,
                        target_call_history=None, attack_call_history=None, eval_call_history=None,
                        target_model_name="", attack_model_name="", eval_model_name=""):
    """
    Lưu trữ log cho từng vòng lặp (iteration) bao gồm Archive, Global Context 
    và tính toán nhanh ASR (Attack Success Rate).
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    filename = log_dir / f"iteration_log_{iteration:04d}_{timestamp}.json"
    
    archive_dump = {}
    total_instances = 0
    successful_instances = 0
    
    # 1. Duyệt Archive để gom dữ liệu
    try:
        keys = archive.keys() if hasattr(archive, 'keys') else archive.archive.keys()
        
        for k in keys:
            str_key = str(k)
            # Lúc này forest_list là một List[Forest]
            forest_list = archive.get(k) if hasattr(archive, 'get') else archive.archive[k]
            
            if not forest_list:
                continue
                
            archive_dump[str_key] = []
            
            for forest in forest_list:
                # Bóc tách tất cả Instance từ cây MCTS trong Forest này
                instances = _extract_instances_from_forest(forest)
                
                for inst in instances:
                    # Bỏ qua nếu instance rỗng
                    if not inst: 
                        continue
                        
                    total_instances += 1
                    
                    score = inst.eval_results[-1] if hasattr(inst, 'eval_results') and inst.eval_results else 0
                    if score >= 10:
                        successful_instances += 1
                    
                    inst_dict = {
                        "query": getattr(inst, 'query', ''),
                        "jailbreak_prompt": getattr(inst, 'jailbreak_prompt', ''),
                        "target_response": inst.target_responses[0] if hasattr(inst, 'target_responses') and inst.target_responses else "",
                        "score": score,
                        "attack_attrs": {
                            "cell_key": str(getattr(inst, 'attack_attrs', {}).get('cell_key', '')),
                        }
                    }
                    archive_dump[str_key].append(inst_dict)
                
    except Exception as e:
        print(f"[-] Lỗi khi trích xuất Archive tại Iteration {iteration}: {e}")

    # 2. Tính ASR
    asr_percentage = (successful_instances / total_instances * 100) if total_instances > 0 else 0

    # 3. Gom Call History
    normalized_target = _normalize_call_history(target_call_history or [], iteration=iteration, llm_type="target")
    normalized_attack = _normalize_call_history(attack_call_history or [], iteration=iteration, llm_type="attack")
    normalized_eval = _normalize_call_history(eval_call_history or [], iteration=iteration, llm_type="eval")
    
    llm_logs = {}
    if normalized_target or target_model_name:
        llm_logs["target_llm"] = _build_llm_log("target", normalized_target, target_model_name, "Target Model")
    if normalized_attack or attack_model_name:
        llm_logs["attack_llm"] = _build_llm_log("attack", normalized_attack, attack_model_name, "Attack/Mutator Model")
    if normalized_eval or eval_model_name:
        llm_logs["eval_llm"] = _build_llm_log("eval", normalized_eval, eval_model_name, "Evaluation/Judge Model")

    # 4. Đóng gói JSON
    log_payload = {
        "metadata": {
            "iteration": iteration,
            "max_iterations": max_iters,
            "timestamp": timestamp
        },
        "metrics": {
            "total_prompts": total_instances,
            "successful_jailbreaks": successful_instances,
            "ASR_percentage": round(asr_percentage, 2)
        },
        "map_elites_archive": archive_dump,
        "global_context": global_context
    }
    
    if llm_logs:
        log_payload["llm_call_logs"] = llm_logs

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(log_payload, f, ensure_ascii=False, indent=4)
    
    print(f"[+] Iter {iteration:04d}/{max_iters} | ASR: {round(asr_percentage, 2):05.2f}% ({successful_instances}/{total_instances}) | Target calls: {len(normalized_target)} | Saved: {filename.name}")


def save_comprehensive_logs(log_dir, archive, global_context, timestamp, iteration, max_iters,
                            target_call_history=None, attack_call_history=None, eval_call_history=None,
                            target_model_name="", attack_model_name="", eval_model_name=""):
    """
    Lưu trữ toàn bộ Archive từ tất cả các Forest, không lọc theo iteration cụ thể.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    filename = log_dir / f"comprehensive_log_iter_{iteration}_{timestamp}.json"
    
    archive_dump = {}
    total_instances = 0
    successful_instances = 0
    
    try:
        keys = archive.keys() if hasattr(archive, 'keys') else archive.archive.keys()
        
        for k in keys:
            str_key = str(k)
            forest_list = archive.get(k) if hasattr(archive, 'get') else archive.archive[k]
            
            if not forest_list:
                continue
                
            archive_dump[str_key] = []
            
            for forest in forest_list:
                instances = _extract_instances_from_forest(forest)
                
                for inst in instances:
                    if not inst: continue
                    total_instances += 1
                    
                    score = inst.eval_results[-1] if hasattr(inst, 'eval_results') and inst.eval_results else 0
                    if score >= 10:
                        successful_instances += 1
                    
                    archive_dump[str_key].append({
                        "query": getattr(inst, 'query', ''),
                        "jailbreak_prompt": getattr(inst, 'jailbreak_prompt', ''),
                        "target_response": inst.target_responses[0] if hasattr(inst, 'target_responses') and inst.target_responses else "",
                        "score": score,
                        "attack_attrs": {
                            "cell_key": str(getattr(inst, 'attack_attrs', {}).get('cell_key', '')),
                            "target_style": getattr(inst, 'attack_attrs', {}).get('target_style', '')
                        }
                    })
                    
    except Exception as e:
        print(f"[-] Lỗi khi đọc Archive Comprehensive: {e}")

    asr_percentage = (successful_instances / total_instances * 100) if total_instances > 0 else 0

    normalized_target = _normalize_call_history(target_call_history or [], iteration=None, llm_type="target")
    normalized_attack = _normalize_call_history(attack_call_history or [], iteration=None, llm_type="attack")
    normalized_eval = _normalize_call_history(eval_call_history or [], iteration=None, llm_type="eval")
    
    llm_logs = {}
    if normalized_target or target_model_name:
        llm_logs["target_llm"] = _build_llm_log("target", normalized_target, target_model_name, "Target Model")
    if normalized_attack or attack_model_name:
        llm_logs["attack_llm"] = _build_llm_log("attack", normalized_attack, attack_model_name, "Attack/Mutator Model")
    if normalized_eval or eval_model_name:
        llm_logs["eval_llm"] = _build_llm_log("eval", normalized_eval, eval_model_name, "Evaluation/Judge Model")

    log_payload = {
        "metadata": {
            "iteration": iteration,
            "max_iterations": max_iters,
            "timestamp": timestamp
        },
        "metrics": {
            "total_prompts": total_instances,
            "successful_jailbreaks": successful_instances,
            "ASR_percentage": round(asr_percentage, 2)
        },
        "map_elites_archive": archive_dump,
        "global_context": global_context
    }
    
    if llm_logs:
        log_payload["llm_call_logs"] = llm_logs

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(log_payload, f, ensure_ascii=False, indent=4)
    
    print(f"[+] Iter {iteration} | ASR: {round(asr_percentage, 2)}% ({successful_instances}/{total_instances}) | Target: {len(normalized_target)} | Attack: {len(normalized_attack)} | Eval: {len(normalized_eval)} | Saved: {filename.name}")

def initialize_language_models(config: ConfigurationLoader):
    """Initialize language models from configuration."""
    model_configs = [
        config.target_llm,
        config.mutator_llm,
    ]

    llm_switchers = {}
    seen_model_configs = set()

    for model_config in model_configs:
        config_key = tuple(sorted(model_config.model_kwargs.items()))

        if config_key not in seen_model_configs:
            try:
                llm_switcher = LLMSwitcher(model_config)
                model_name = model_config.model_kwargs.get("model", "unnamed_model")
                llm_switchers[model_name] = llm_switcher
                seen_model_configs.add(config_key)
            except ValueError as e:
                logger.error(f"Error initializing model {model_config}: {e}")

    return llm_switchers