
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import json
import logging
import sys
import os

from typing import TypeVar, List
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


def load_txt(file_path: str) -> List[str]:
    """
    Load text file and return non-empty lines.

    Args:
        file_path (str): Path to the text file

    Returns:
        List[str]: List of stripped, non-empty lines
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

import random
def load_json(
    file_path: str,
    field: str,
    num_samples: int = -1,
    shuffle: bool = False,
    seed: int = 15,  
) -> List[str]:
    """
    Chuẩn hóa hàm load_json để khớp hoàn toàn với cơ chế của JailbreakDataset.
    """
    # 1. Load dataset từ file (giống JailbreakDataset)
    raw_dataset = load_dataset("json", data_files=file_path, split="train")

    # 2. Chuyển đổi thành List các Dictionary (Tương đương với List[Object])
    # Bước này quan trọng vì random.shuffle chỉ hoạt động chính xác trên List Python
    data_list = [sample for sample in raw_dataset]

    # 3. Thực hiện Shuffle bằng thư viện random của Python
    if shuffle:
        # Thiết lập seed ngay trước khi shuffle để đảm bảo tính tái lập
        random.seed(seed)
        random.shuffle(data_list)

    # 4. Lấy số lượng mẫu yêu cầu
    sample_count = len(data_list) if num_samples == -1 else min(num_samples, len(data_list))
    sampled_data = [sample[field] for sample in data_list[:sample_count]]

    return sampled_data

def _normalize_call_history(call_history, iteration=None, llm_type="target"):
    """
    Lọc và chuẩn hóa lịch sử gọi API dựa trên vòng lặp.
    """
    if not call_history:
        return []
    
    filtered = []
    for call in call_history:
        # Nếu iteration được chỉ định cụ thể (khác None và khác -1 cho global) thì tiến hành lọc
        if iteration is not None and iteration != -1 and call.get('run_iteration') != iteration:
            continue
        
        call_copy = dict(call)
        # Tạo chuỗi call_location dễ đọc
        call_copy['call_location'] = (
            f"{call.get('call_site', 'unknown')} | "
            f"run_iteration={call.get('run_iteration', '')}"
        )
        filtered.append(call_copy)
    
    return filtered

def _build_llm_log(llm_type, llm_call_history, model_name, llm_type_display):
    """
    Đóng gói dữ liệu lịch sử gọi API thành một block JSON chuẩn.
    """
    return {
        "llm_type": llm_type_display,
        "model_name": model_name,
        "total_calls": len(llm_call_history),
        "calls_in_log": len(llm_call_history),
        "calls": llm_call_history
    }
def save_iteration_log(
    log_dir, adv_prompts, responses, scores, iters, timestamp, iteration=-1, max_iters=None,
    target_call_history=None, attack_call_history=None, eval_call_history=None,
    target_model_name="", attack_model_name="", eval_model_name=""
):
    """
    Save log of current iteration's results (Cập nhật lưu LLM logs).
    """
    if iteration == -1:
        log_path = log_dir / f"rainbowplus_log_{timestamp}.json"
    else:
        log_path = log_dir / f"rainbowplus_log_{timestamp}_epoch_{iteration+1}.json"

    # Xử lý LLM Call Logs (Chỉ lấy của vòng lặp hiện tại, trừ khi là global: iteration=-1)
    target_call_history = target_call_history or []
    attack_call_history = attack_call_history or []
    eval_call_history = eval_call_history or []
    
    normalized_target = _normalize_call_history(target_call_history, iteration=iteration, llm_type="target")
    normalized_attack = _normalize_call_history(attack_call_history, iteration=iteration, llm_type="attack")
    normalized_eval = _normalize_call_history(eval_call_history, iteration=iteration, llm_type="eval")
    
    llm_logs = {}
    if normalized_target or target_model_name:
        llm_logs["target_llm"] = _build_llm_log("target", normalized_target, target_model_name, "Target Model")
    if normalized_attack or attack_model_name:
        llm_logs["attack_llm"] = _build_llm_log("attack", normalized_attack, attack_model_name, "Attack/Mutator Model")
    if normalized_eval or eval_model_name:
        llm_logs["eval_llm"] = _build_llm_log("eval", normalized_eval, eval_model_name, "Evaluation/Judge Model")

    # Đóng gói Payload
    log_payload = {
        "max_iters": max_iters,
        "adv_prompts": {str(key): value for key, value in adv_prompts._archive.items()},
        "responses": {str(key): value for key, value in responses._archive.items()},
        "scores": {str(key): value for key, value in scores._archive.items()},
        "iters": {str(key): value for key, value in iters._archive.items()},
    }
    
    if llm_logs:
        log_payload["llm_call_logs"] = llm_logs

    with open(log_path, "w", encoding='utf-8') as f:
        json.dump(log_payload, f, indent=2, ensure_ascii=False)

    logger.info(f"Log saved to {log_path}")

def calculate_lexical_diversity_from_archives(all_prompts):
    flat_prompts = []
    for key, prompts in all_prompts._archive.items():
        flat_prompts.extend(prompts)
    total = len(flat_prompts)
    unique = len(set(flat_prompts))
    diversity_score = unique / total if total > 0 else 0
    return {
        'total_prompts': total,
        'unique_prompts': unique,
        'diversity_score': diversity_score
    }


def calculate_embedding_diversity_from_archives(all_prompts):
    flat_prompts = []
    for key, prompts in all_prompts._archive.items():
        flat_prompts.extend(prompts)
    if len(flat_prompts) < 2:
        return 0.0
    model = SentenceTransformer('all-MiniLM-L6-v2', device = 'cpu')
    embeddings = model.encode(flat_prompts, show_progress_bar=False)
    # Compute pairwise cosine distances
    dists = cosine_distances(embeddings)
    # Only take upper triangle, excluding diagonal
    n = len(flat_prompts)
    triu_indices = np.triu_indices(n, k=1)
    avg_dist = dists[triu_indices].mean() if len(triu_indices[0]) > 0 else 0.0
    return float(avg_dist)

def save_comprehensive_log(
    log_dir, 
    all_prompts, 
    all_responses, 
    all_scores, 
    all_similarities,
    rejection_reasons,
    timestamp, 
    iteration=-1,
    all_prompt_ids=None,
    all_parent_ids=None,
    all_seed_ids=None,
    max_iters=None,
    target_call_history=None, attack_call_history=None, eval_call_history=None,
    target_model_name="", attack_model_name="", eval_model_name=""
):
    """
    Save comprehensive log of all generated prompts (Cập nhật lưu TOÀN BỘ LLM logs).
    """
    if iteration == -1:
        log_path = log_dir / f"comprehensive_log_{timestamp}.json"
    else:
        log_path = log_dir / f"comprehensive_log_{timestamp}_epoch_{iteration+1}.json"

    # Xử lý LLM Call Logs (Truyền iteration=None để VÉT SẠCH toàn bộ lịch sử)
    target_call_history = target_call_history or []
    attack_call_history = attack_call_history or []
    eval_call_history = eval_call_history or []
    
    normalized_target = _normalize_call_history(target_call_history, iteration=None, llm_type="target")
    normalized_attack = _normalize_call_history(attack_call_history, iteration=None, llm_type="attack")
    normalized_eval = _normalize_call_history(eval_call_history, iteration=None, llm_type="eval")
    
    llm_logs = {}
    if normalized_target or target_model_name:
        llm_logs["target_llm"] = _build_llm_log("target", normalized_target, target_model_name, "Target Model")
    if normalized_attack or attack_model_name:
        llm_logs["attack_llm"] = _build_llm_log("attack", normalized_attack, attack_model_name, "Attack/Mutator Model")
    if normalized_eval or eval_model_name:
        llm_logs["eval_llm"] = _build_llm_log("eval", normalized_eval, eval_model_name, "Evaluation/Judge Model")

    # Tính toán Diversity
    diversity = calculate_lexical_diversity_from_archives(all_prompts)
    embedding_diversity = calculate_embedding_diversity_from_archives(all_prompts)

    log_data = {
        "max_iters": max_iters,
        "all_prompts": {str(key): value for key, value in all_prompts._archive.items()},
        "all_responses": {str(key): value for key, value in all_responses._archive.items()},
        "all_scores": {str(key): value for key, value in all_scores._archive.items()},
        "all_similarities": {str(key): value for key, value in all_similarities._archive.items()},
        "rejection_reasons": {str(key): value for key, value in rejection_reasons._archive.items()},
        "diversity": diversity,
        "embedding_diversity": embedding_diversity,
    }
    
    if all_prompt_ids is not None:
        log_data["all_prompt_ids"] = {str(key): value for key, value in all_prompt_ids._archive.items()}
    if all_parent_ids is not None:
        log_data["all_parent_ids"] = {str(key): value for key, value in all_parent_ids._archive.items()}
    if all_seed_ids is not None:
        log_data["all_seed_ids"] = {str(key): value for key, value in all_seed_ids._archive.items()}

    # Chèn LLM Call logs vào nếu có
    if llm_logs:
        log_data["llm_call_logs"] = llm_logs

    with open(log_path, "w", encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Comprehensive log saved to {log_path}")
    

def initialize_language_models(config: ConfigurationLoader):
    """
    Initialize language models from configuration.

    Args:
        config: Configuration object containing model settings

    Returns:
        Dictionary of initialized language models
    """
    # Extract model configurations
    model_configs = [
        config.target_llm,
        config.mutator_llm,
    ]

    # Create unique language model switchers
    llm_switchers = {}
    seen_model_configs = set()

    for model_config in model_configs:
        # Create a hashable representation of model kwargs
        config_key = tuple(sorted(model_config.model_kwargs.items()))

        # Only create a new LLM switcher if this configuration hasn't been seen before
        if config_key not in seen_model_configs:
            try:
                llm_switcher = LLMSwitcher(model_config)
                model_name = model_config.model_kwargs.get("model", "unnamed_model")
                llm_switchers[model_name] = llm_switcher
                seen_model_configs.add(config_key)
            except ValueError as e:
                logger.error(f"Error initializing model {model_config}: {e}")

    return llm_switchers

