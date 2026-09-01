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

def build_run_metadata(algorithm: str, args, config=None) -> Dict[str, Any]:
    """
    Snapshot of "what produced this log": the algorithm name plus every CLI/
    config hyperparameter (num_samples, max_iters, tree_width, total, ...),
    saved into every saved log file's "run_metadata" key so a log can be
    identified/traced back to its exact run later without external
    bookkeeping (which was previously not recorded anywhere in the logs).

    api_key is deliberately never included in model_backend -- only the
    model name/type/base_url, which is enough to tell a local-vLLM run apart
    from an apply_api_override() run without leaking the secret into a log
    file that might get shared or committed.
    """
    hyperparameters = dict(vars(args))

    model_backend = None
    if config is not None:
        def _role_info(llm_cfg):
            if llm_cfg is None:
                return None
            return {
                "model": llm_cfg.model_kwargs.get("model"),
                "type": getattr(llm_cfg, "type_", "vllm"),
                "base_url": llm_cfg.base_url,  # None for local vllm
            }

        model_backend = {
            "target": _role_info(config.target_llm),
            "mutator": _role_info(config.mutator_llm),
            "fitness": _role_info(config.fitness_llm),
        }

    return {
        "algorithm": algorithm,
        "hyperparameters": hyperparameters,
        "model_backend": model_backend,
    }


def _extract_instances_from_forest(forest):
    """
    Dùng thuật toán duyệt theo chiều rộng (BFS) để bóc tách
    toàn bộ đối tượng Instance từ các Node trong cây Forest.

    Trước đây hàm này chỉ trả về list phẳng các Instance, bỏ hết metadata của
    Node (depth, visits, max_score, is_terminal, quan hệ cha-con) -- giờ trả
    về list[dict] để đủ dữ liệu tái dựng lại hình dạng cây MCTS từ log đã lưu.
    Mỗi dict: {"instance", "depth", "visits", "max_score", "is_terminal",
    "node_id", "parent_id"}. depth/node_id/parent_id chỉ dựa vào thứ tự BFS
    -- không cần thêm field mới trên MCTSNode.
    """
    extracted = []
    # Kiểm tra xem forest có hợp lệ và có root node không
    if not forest or not hasattr(forest, 'root') or forest.root is None:
        return extracted

    # (node, depth, parent_id) -- root gets node_id=0/parent_id=None even
    # though it's never itself appended (its instance is always None), so its
    # depth-1 children can still reference it as their parent_id.
    queue = [(forest.root, 0, None)]
    next_id = 0
    while queue:
        node, depth, parent_id = queue.pop(0)
        node_id = next_id
        next_id += 1

        # Nếu node chứa instance hợp lệ, lưu lại cùng metadata của Node
        if hasattr(node, 'instance') and node.instance is not None:
            extracted.append({
                "instance": node.instance,
                "depth": depth,
                "visits": getattr(node, 'visits', 0),
                "max_score": getattr(node, 'max_score', 0.0),
                "is_terminal": getattr(node, 'is_terminal', False),
                "node_id": node_id,
                "parent_id": parent_id,
            })

        # Thêm các node con vào hàng đợi để tiếp tục duyệt
        if hasattr(node, 'children') and node.children:
            queue.extend((child, depth + 1, node_id) for child in node.children)

    return extracted

def save_iteration_logs(log_dir, archive, global_context, timestamp, iteration, max_iters,
                        target_call_history=None, attack_call_history=None, eval_call_history=None,
                        target_model_name="", attack_model_name="", eval_model_name="", run_metadata=None,
                        cell_first_success=None):
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
                # Bóc tách tất cả Instance + metadata Node (depth/visits/
                # max_score/is_terminal/node_id/parent_id) từ cây MCTS trong Forest này
                extracted = _extract_instances_from_forest(forest)

                for item in extracted:
                    inst = item["instance"]
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
                        },
                        # MCTS tree-shape/search-progress metadata for this node
                        # (see _extract_instances_from_forest).
                        "tree": {
                            "node_id": item["node_id"],
                            "parent_id": item["parent_id"],
                            "depth": item["depth"],
                            "visits": item["visits"],
                            "max_score": item["max_score"],
                            "is_terminal": item["is_terminal"],
                        },
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

    # run_metadata identifies WHICH algorithm/hyperparameters produced this
    # log (see build_run_metadata()) -- put it first so it's the first thing
    # anyone opening the file sees, e.g. when digging back through old runs.
    if run_metadata:
        log_payload = {"run_metadata": run_metadata, **log_payload}

    if llm_logs:
        log_payload["llm_call_logs"] = llm_logs

    # Per-cell "cost to first success": iteration + cumulative target-query-
    # count at which each MAP-Elites cell first produced an instance scoring
    # raw >= 10 (same threshold as ASR above). Computed inline in the driver
    # loop (rainbowplus.py's cell_first_success) -- keyed by str(cell_key) to
    # match archive_dump's str_key.
    if cell_first_success:
        log_payload["cell_first_success"] = cell_first_success

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(log_payload, f, ensure_ascii=False, indent=4)

    print(f"[+] Iter {iteration:04d}/{max_iters} | ASR: {round(asr_percentage, 2):05.2f}% ({successful_instances}/{total_instances}) | Target calls: {len(normalized_target)} | Saved: {filename.name}")


def save_comprehensive_logs(log_dir, archive, global_context, timestamp, iteration, max_iters,
                            target_call_history=None, attack_call_history=None, eval_call_history=None,
                            target_model_name="", attack_model_name="", eval_model_name="", run_metadata=None,
                            cell_first_success=None):
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
                extracted = _extract_instances_from_forest(forest)

                for item in extracted:
                    inst = item["instance"]
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
                        },
                        # MCTS tree-shape/search-progress metadata for this node
                        # (see _extract_instances_from_forest).
                        "tree": {
                            "node_id": item["node_id"],
                            "parent_id": item["parent_id"],
                            "depth": item["depth"],
                            "visits": item["visits"],
                            "max_score": item["max_score"],
                            "is_terminal": item["is_terminal"],
                        },
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

    # run_metadata identifies WHICH algorithm/hyperparameters produced this
    # log (see build_run_metadata()) -- put it first so it's the first thing
    # anyone opening the file sees, e.g. when digging back through old runs.
    if run_metadata:
        log_payload = {"run_metadata": run_metadata, **log_payload}

    if llm_logs:
        log_payload["llm_call_logs"] = llm_logs

    # See save_iteration_logs()'s identical block -- same per-cell "cost to
    # first success" data, carried into the comprehensive/FINAL log too.
    if cell_first_success:
        log_payload["cell_first_success"] = cell_first_success

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