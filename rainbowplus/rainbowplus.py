#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import os
import sys
import argparse
import random
import json
import time
import logging
import re
import uuid  # Required for ID generation
from pathlib import Path
import yaml
import numpy as np

try:
    from dotenv import load_dotenv

    load_dotenv()  # picks up a .env in CWD, e.g. OPENAI_API_KEY/BASE_URL/MODEL_ID
except ImportError:
    pass  # python-dotenv not installed -- fall back to whatever's already in the shell env
from typing import List, Optional, Any, Dict, Tuple
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass, field
from copy import deepcopy
from vllm import SamplingParams
# --- Imports ---
from rainbowplus.scores import BleuScoreNLTK, OpenAIGuard, LlamaGuard
from rainbowplus.utils import (
    load_txt,
    load_json,
    initialize_language_models,
    save_comprehensive_logs,
    save_iteration_logs,
    format_example,
    build_run_metadata,
)

# Identifies which algorithm produced a given log -- written into every saved
# log's run_metadata (see build_run_metadata()) so a log file can be traced
# back to its run later without relying on directory naming conventions.
ALGORITHM_NAME = "gap-beam-search"
from rainbowplus.archive import Archive
from rainbowplus.configs import ConfigurationLoader
from rainbowplus.prompts import system_prompt_mutator, TARGET_PROMPT, MUTATOR_PROMPT_V1, SELECTOR_PROMPT, ANALYZER_PROMPT



# Set fixed random seed
RANDOM_SEED = 15
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)





def parse_arguments():
    parser = argparse.ArgumentParser(description="Adversarial Prompt Generation")
    parser.add_argument(
        "--num_samples", type=int, default=50, help="Number of initial seed prompts"
    )
    parser.add_argument(
        "--max_iters", type=int, default=200, help="Maximum number of iteration steps"
    )
    parser.add_argument(
        "--tree_depth", type=int, default=2, help="Depth of the attack tree"
    )
    
    parser.add_argument(
        "--sim_threshold",
        type=float,
        default=0.6,
        help="Similarity threshold for prompt mutation",
    )
    parser.add_argument(
        "--fitness_threshold",
        type=float,
        default=0.5,
        help="Minimum fitness score to add prompt to archive",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="./configs/base-opensource.yml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--log_dir", type=str, default="./logs", help="Directory for storing logs"
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=20,
        help="Number of iterations between log saves",
    )
    parser.add_argument(
        "--dataset", type=str, default="./data/formatted_harmbench.json", help="Dataset name"
    )
    parser.add_argument(
        "--shuffle", type=bool, default=True, help="Shuffle seed prompts"
    )
    parser.add_argument(
        "--target_query_budget",
        type=int,
        default=0,
        help="Stop the main loop once cumulative target-model queries reach this many "
        "(0 = disabled, run all --max_iters as before)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="RNG seed for reproducibility. Default reproduces today's fixed "
        f"RANDOM_SEED={RANDOM_SEED}. Vary it (--seed 0,1,2,...) for independent "
        "runs / multi-seed statistics.",
    )
    return parser.parse_args()




def apply_api_override(config):
    """
    If OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL_ID are all set (typically
    via .env, same var names as the data-agent proxy config so the same .env
    can be reused as-is), point target_llm/mutator_llm/fitness_llm at that one
    API instead of loading 3 separate local vLLM models. sampling_params
    (temperature/top_p/max_tokens per role) are left untouched -- those are
    algorithm hyperparameters, not infra identifiers.

    No silent partial fallback: if only some of the 3 vars are set, fail loudly
    now rather than mixing local models with a half-configured API client.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    model_id = os.environ.get("OPENAI_MODEL_ID")
    present = {"OPENAI_API_KEY": bool(api_key), "OPENAI_BASE_URL": bool(base_url), "OPENAI_MODEL_ID": bool(model_id)}

    if not any(present.values()):
        return config  # unchanged: local vLLM models, as before

    missing = [name for name, ok in present.items() if not ok]
    if missing:
        raise ValueError(
            f"apply_api_override: {missing} missing from env/.env while the others are set "
            f"({present}) -- set all 3 (OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL_ID) or none."
        )

    logger.info(
        f"apply_api_override: routing target_llm, mutator_llm and fitness_llm through "
        f"model='{model_id}' @ base_url='{base_url}' (local vLLM loading skipped for all 3)."
    )
    for role in ("target_llm", "mutator_llm", "fitness_llm"):
        llm_cfg = getattr(config, role)
        llm_cfg.type_ = "openai"
        llm_cfg.api_key = api_key
        llm_cfg.base_url = base_url
        llm_cfg.model_kwargs = {"model": model_id}
    return config


def load_descriptors(config):
    descriptors = {}
    for path, descriptor in zip(config.archive["path"], config.archive["descriptor"]):
        descriptors[descriptor] = load_txt(path)
    return descriptors

def update_archive(archive_obj: Archive, key: Tuple, data: List) -> None:
    """Helper to update archive with data."""
    if not archive_obj.exists(key):
        archive_obj.add(key, data)
    else:
        archive_obj.extend(key, data)

import copy
from tqdm import tqdm
import easyjailbreak
from easyjailbreak.attacker import AttackerBase
from easyjailbreak.datasets import JailbreakDataset
from easyjailbreak.loggers.logger import Logger
from easyjailbreak.selector.SelectBasedOnScores import SelectBasedOnScores
from rainbowplus.graph_easyjailbreak.vllm_model import VLLMModel, from_pretrained
from easyjailbreak.datasets import JailbreakDataset, Instance

from rainbowplus.graph_easyjailbreak.model_utils import count_calls, count_batch_calls
#from gap_easyjailbreak.models.bedrock_mistral_model import MistralModel
#from gap_easyjailbreak.models.bedrock_claude_model import ClaudeModel
from rainbowplus.graph_easyjailbreak.IntrospectGenerationAggregateMaxConcatenate import IntrospectGenerationAggregateMaxConcatenate as IntrospectGeneration
from rainbowplus.graph_easyjailbreak.GAP import GAP
from rainbowplus.graph_easyjailbreak.openai_compat_model import OpenAICompatModel


def run_rainbowplus(args, config, seed_prompts=[], attack_model=None, target_model=None, fitness_fn=None, similarity_fn=None):
    """Main RainbowPlus algorithm."""
    
    descriptors = load_descriptors(config)

    # Setup logging
    log_dir = Path(args.log_dir) / target_model.model_name / args.dataset.split('/')[-1].split('.')[0]
    log_dir.mkdir(parents=True, exist_ok=True)

    # Which algorithm + hyperparameters + model backend produced every log
    # saved below -- see build_run_metadata().
    run_metadata = build_run_metadata(ALGORITHM_NAME, args, config)

    # Prepare seed metadata
    seed = JailbreakDataset(dataset=args.dataset, local_file_type='json')
    for i in range(15):
        print(seed._dataset[i].query)
    # Uses args.seed (default RANDOM_SEED) so multi-seed runs get independent
    # shuffles + descriptor/cell/parent choices.
    random.seed(args.seed)
    np.random.seed(args.seed)
    seed.shuffle() # Shuffle seed prompts to ensure randomness
    for i in range(15):
        print(seed._dataset[i].query)
    seed_prompts = seed[:args.num_samples] # Take only the required number of samples
    global_context = []
    attacker = GAP (attack_model=attack_model,
                        target_model=target_model,
                        eval_model=fitness_fn,
                        jailbreak_datasets=seed,
                        tree_width=4,
                        tree_depth=args.tree_depth,
                        root_num=1,
                        branching_factor=4,
                        keep_last_n=3,
                        max_n_attack_attempts=2,
                        conversation_history=global_context,
                        attack_max_n_tokens = config.mutator_llm.sampling_params.get('max_tokens', 500),
                        attack_temperature=config.mutator_llm.sampling_params.get('temperature', 1),
                        attack_top_p = config.mutator_llm.sampling_params.get('top_p', 0.9),
                        target_max_n_tokens= config.target_llm.sampling_params.get('max_tokens', 150),
                        target_temperature=config.target_llm.sampling_params.get('temperature', 1),
                        target_top_p=config.target_llm.sampling_params.get('top_p', 1),
                        judge_max_n_tokens=config.fitness_llm.sampling_params.get('max_tokens', 10),
                        judge_temperature=config.fitness_llm.sampling_params.get('temperature', 1))
    

    archive_by_depth = Archive[Instance]("map-elites")
    attacker.attack_model.generate = count_calls(attack_model.generate)
    attacker.target_model.generate = count_calls(target_model.generate)
    attacker.eval_model.generate = count_calls(eval_model.generate)
    # Note: batch_generate already has @count_batch_calls decorator on VLLMModel class
    for i, instance in enumerate(seed_prompts):
        inst = instance.copy()
        inst.jailbreak_prompt = inst.query
        inst.reference_responses = [""]
        seed_prompts._dataset[i] = inst

    for iteration in range(args.max_iters):
        print(f"\n\n=== Iteration {iteration} ===\n\n")
        attacker.current_iteration = iteration
        branchs = archive_by_depth.flatten_values()
        if iteration < len(seed_prompts):
            selected_branchs = seed_prompts._dataset[iteration]
        else:
            if not branchs:
                selected_branchs = random.choice(seed_prompts._dataset)
            else:
                selected_branchs = random.choice(branchs)

        descriptor = {key: random.choice(value) for key, value in descriptors.items()}
        key = tuple(descriptor.values())        
        descriptor_str = "\n".join([f"{k}: {v}" for k, v in descriptor.items()])
        
        temp_instance = selected_branchs.copy()
        temp_instance.query = descriptor_str
        temp_instance.reference_responses = [""]
        temp_instance.eval_results = []
        temp_instance.target_responses = []
        temp_instance.attack_attrs = {}
        
        temp_instance.attack_attrs['cell_key'] = key
 
        dataset = JailbreakDataset([temp_instance])
        attacker.jailbreak_datasets = dataset

        for instance in tqdm(dataset, desc=f"Processing instances in {iteration}"):
            new_dataset_obj = attacker.single_attack(instance)
            flattened_instances = []
            for ds in new_dataset_obj:
                flattened_instances.extend(list(ds))
               
            target_key = instance.attack_attrs['cell_key']
            update_archive(archive_by_depth, target_key, flattened_instances)
            current_list_data = archive_by_depth.get(target_key)
            archive_by_depth.delete(target_key)
            
            # Sort if multiple items exist
            if len(current_list_data) > 1:
                current_list_data.sort(
                    key=lambda x: x.eval_results[-1] if len(x.eval_results) > 0 else 0, 
                    reverse=True
                )
                # Keep top-k based on tree_width
                current_list_data = current_list_data[:attacker.tree_width]

            update_archive(archive_by_depth, target_key, current_list_data)

        if iteration % args.log_interval == 0:
            current_ts = time.strftime(r"%Y%m%d-%H%M%S")
            save_iteration_logs(
                log_dir=log_dir,
                archive=archive_by_depth,
                global_context=global_context,
                timestamp=current_ts,
                iteration=iteration,
                max_iters=args.max_iters,
                target_call_history=attacker.target_call_history,
                attack_call_history=attacker.attack_call_history,
                eval_call_history=attacker.eval_call_history,
                target_model_name=target_model.get_name() if hasattr(target_model, 'get_name') else getattr(target_model, 'model_name', ''),
                attack_model_name=attack_model.get_name() if hasattr(attack_model, 'get_name') else getattr(attack_model, 'model_name', ''),
                eval_model_name=eval_model.get_name() if hasattr(eval_model, 'get_name') else getattr(eval_model, 'model_name', ''),
                run_metadata=run_metadata,
            )

        # Target-query budget stop (--target_query_budget, 0 = disabled): caps a run by
        # actual target-model query cost instead of only by --max_iters, so runs across
        # algorithms/branches can be compared at equal target-query spend. target_call_history
        # entries already carry an accurate per-call "query_count" (GAP.py sets it to
        # len(new_dataset), not the generate/batch_generate call-count delta used for
        # attack/eval -- see the double-counting fix in openai_compat_model.py), so summing
        # it here gives the true cumulative target-query count with no further fixup needed.
        if args.target_query_budget > 0:
            cumulative_target_queries = sum(
                record.get("query_count", 0) for record in attacker.target_call_history
            )
            if cumulative_target_queries >= args.target_query_budget:
                logger.info(
                    f"target_query_budget reached ({cumulative_target_queries} >= "
                    f"{args.target_query_budget}) at iteration {iteration} -- stopping main loop early."
                )
                break
    current_ts = time.strftime(r"%Y%m%d-%H%M%S")
    
    save_comprehensive_logs(
        log_dir=log_dir,
        archive=archive_by_depth,
        global_context=global_context,
        timestamp=current_ts,
        iteration="FINAL",
        max_iters=args.max_iters,
        target_call_history=attacker.target_call_history,
        attack_call_history=attacker.attack_call_history,
        eval_call_history=attacker.eval_call_history,
        target_model_name=target_model.get_name() if hasattr(target_model, 'get_name') else getattr(target_model, 'model_name', ''),
        attack_model_name=attack_model.get_name() if hasattr(attack_model, 'get_name') else getattr(attack_model, 'model_name', ''),
        eval_model_name=eval_model.get_name() if hasattr(eval_model, 'get_name') else getattr(eval_model, 'model_name', ''),
        run_metadata=run_metadata,
    )
    print("Done!")


if __name__ == "__main__":
    args = parse_arguments()
    config = ConfigurationLoader.load(args.config_file)
    config = apply_api_override(config)

    # Build vLLM models — model_name is the FastChat template, model_name_or_path is the HF hub id
    # Any extra keys in model_kwargs (max_model_len, gpu_memory_utilization, enforce_eager …)
    # are forwarded directly to vllm.LLM.
    # When apply_api_override() set type_="openai" above, build an OpenAICompatModel
    # instead -- no local weights loaded, every call goes over HTTP to the same API.
    def _make_model(llm_cfg, fastchat_name: str):
        if llm_cfg.type_ == "openai":
            return OpenAICompatModel(
                model_name=llm_cfg.model_kwargs["model"],
                api_key=llm_cfg.api_key,
                base_url=llm_cfg.base_url,
                generation_config=llm_cfg.sampling_params,
            )
        model_kwargs = dict(llm_cfg.model_kwargs)  # shallow copy
        model_path = model_kwargs.pop("model")     # HF hub id / local path
        return from_pretrained(
            model_name=fastchat_name,
            model_name_or_path=model_path,
            **model_kwargs,
        )

    attack_model = _make_model(config.mutator_llm, fastchat_name="vicuna_v1.1")
    target_model = _make_model(config.target_llm,  fastchat_name="qwen-7b-chat")
    eval_model   = _make_model(config.fitness_llm, fastchat_name="qwen-7b-chat")
    similarity_fn = BleuScoreNLTK()

    print(config)

    run_rainbowplus(
        args,
        config,
        seed_prompts=[],
        attack_model=attack_model,
        target_model=target_model,
        fitness_fn=eval_model,
        similarity_fn=similarity_fn,
    )
