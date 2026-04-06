#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

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
    save_attack_memory_log,
    _repair_yaml_primitives
)
from rainbowplus.archive import Archive
from rainbowplus.configs import ConfigurationLoader
from rainbowplus.prompts import system_prompt_mutator, TARGET_PROMPT, MUTATOR_PROMPT_V1, SELECTOR_PROMPT, ANALYZER_PROMPT
from rainbowplus.configs.base import LLMConfig
from rainbowplus.mutators.persona import PersonaMutator
from rainbowplus.attack_memory import AttackMemory


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
        "--tree_depth", type=int, default=5, help="Depth of the attack tree"
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
        default=5,
        help="Number of iterations between log saves",
    )
    parser.add_argument(
        "--dataset", type=str, default="./data/do-not-answer.json", help="Dataset name"
    )
    parser.add_argument(
        "--shuffle", type=bool, default=True, help="Shuffle seed prompts"
    )
    return parser.parse_args()




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
from easyjailbreak.datasets.instance import Instance
from easyjailbreak.loggers.logger import Logger
from easyjailbreak.models.huggingface_model import HuggingfaceModel
from easyjailbreak.models.openai_model import OpenaiModel

####### 4 major components #######
from easyjailbreak.seed.seed_template import SeedTemplate
# from easyjailbreak.mutation.generation.IntrospectGeneration import IntrospectGeneration
from easyjailbreak.constraint.DeleteOffTopic import DeleteOffTopic
from easyjailbreak.metrics.Evaluator.Evaluator_GenerativeGetScore import EvaluatorGenerativeGetScore
from easyjailbreak.selector.SelectBasedOnScores import SelectBasedOnScores
from easyjailbreak.models.huggingface_model import from_pretrained
from easyjailbreak.datasets import JailbreakDataset, Instance

from rainbowplus.graph_easyjailbreak.model_utils import count_calls
#from gap_easyjailbreak.models.bedrock_mistral_model import MistralModel
#from gap_easyjailbreak.models.bedrock_claude_model import ClaudeModel
from rainbowplus.graph_easyjailbreak.IntrospectGenerationAggregateMaxConcatenate import IntrospectGenerationAggregateMaxConcatenate as IntrospectGeneration
from rainbowplus.graph_easyjailbreak.GAP import GAP


def run_rainbowplus(args, config, seed_prompts=[], attack_model=None, target_model=None, fitness_fn=None, similarity_fn=None):
    """Main RainbowPlus algorithm."""
    
    descriptors = load_descriptors(config)

    # Setup logging
    log_dir = Path(args.log_dir) / target_model.model_name 
    log_dir.mkdir(parents=True, exist_ok=True)

    global_context = []
    attacker = GAP (attack_model=attack_model,
                        target_model=target_model,
                        eval_model=fitness_fn,
                        jailbreak_datasets=seed,
                        tree_width=3,
                        tree_depth=args.tree_depth,
                        root_num=1,
                        branching_factor=2,
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
    
    seed = JailbreakDataset("AdvBench")
    seed.shuffle() # Shuffle seed prompts to ensure randomness
    seed_prompts = seed[:args.num_samples] # Take only the required number of samples

    archive_by_depth = Archive[Instance]("map-elites")
    # Prepare seed metadata
    for i, instance in enumerate(seed_prompts):
        inst = instance.deepcopy()
        desc = {key: random.choice(value) for key, value in descriptors.items()}
        key = tuple(desc.values())
        seed_query = "\n".join([f"{k}: {v}" for k, v in desc.items()])

        inst.jailbreak_prompt = inst.query
        inst.query = seed_query
        inst.reference_responses = [""]

        inst.attack_attrs['cell_key'] = tuple(desc.values())
        target_key = inst.attack_attrs['cell_key']

        seed_prompts[i] = inst

    
   
    for iteration in range(1, args.max_iters):
        print(f"\n\n=== Iteration {iteration} ===\n\n")
        branchs = archive_by_depth.flatten_values()
        if iteration < len(seed_prompts):
            selected_branchs = [seed_prompts[iteration]]
        else:
            selected_branchs = [random.choice(branchs)]

        batch_instances = []

        descriptor = {key: random.choice(value) for key, value in descriptors.items()}
        key = tuple(descriptor.values())
        
        descriptor_str = "\n".join([f"{k}: {v}" for k, v in descriptor.items()])
        

        best_instance = None
        best_score = -1
        for _ in range(attacker.max_n_attack_attempts):
            temp_instance = selected_branchs.deepcopy()
            temp_instance.query = descriptor_str
            temp_instance.reference_responses = [""]
            if 'conv' in temp_instance.attack_attrs:
                del temp_instance.attack_attrs['conv']
            attacker.mutator.system_prompt = MUTATOR_PROMPT_V1
            transfer_prompt = attacker.mutator(temp_instance, global_context)
            temp_instance.jailbreak_prompt = transfer_prompt
            attacker.mutator.system_prompt = system_prompt_mutator
            if 'conv' in temp_instance.attack_attrs:
                del temp_instance.attack_attrs['conv']
                
            target_response = attacker.target_model.generate(temp_instance.jailbreak_prompt)
            temp_instance.target_responses = [target_response]
            
            attacker.evaluator([temp_instance])
            current_score = temp_instance.eval_results[-1] if temp_instance.eval_results else 0
            if current_score > best_score:
                best_score = current_score
                best_instance = temp_instance
 
        best_instance.attack_attrs['cell_key'] = key
        batch_instances.append(best_instance)

        dataset = JailbreakDataset(batch_instances)
        attacker.jailbreak_datasets = dataset

        for instance in tqdm(attacker.jailbreak_datasets, desc=f"Processing instances in {iteration}"):
            new_dataset_obj = attacker.single_attack(instance)
                    
            target_key = instance.attack_attrs['cell_key']
            update_archive(archive_by_depth, target_key, list(new_dataset_obj))
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

            pruned_dataset_obj = JailbreakDataset(current_list_data)
            update_archive(archive_by_depth, target_key, list(pruned_dataset_obj))

        if iteration % args.log_interval == 0:
            current_ts = time.strftime(r"%Y%m%d-%H%M%S")
            save_iteration_logs(
                log_dir=log_dir,
                archive=archive_by_depth,
                global_context=global_context,
                timestamp=current_ts,
                iteration=iteration,
                max_iters=args.max_iters
            )
    current_ts = time.strftime(r"%Y%m%d-%H%M%S")
    
    save_comprehensive_logs(
        log_dir=log_dir,
        archive=archive_by_depth,
        global_context=global_context,
        timestamp=current_ts,
        iteration="FINAL", # Đánh dấu đây là file chốt cuối cùng
        max_iters=args.max_iters
    )
    print("Done!")


if __name__ == "__main__":
    args = parse_arguments()
    config = ConfigurationLoader.load(args.config_file)

    #llms = initialize_language_models(config)
    attack_model = from_pretrained(
        config.mutator_llm.model_kwargs["model"],
        model_name = "vicuna_v1.1"
    )
    target_model = from_pretrained(
        config.target_llm.model_kwargs["model"],
        model_name = "qwen-7b-chat"
    )
    eval_model = from_pretrained(
        config.fitness_llm.model_kwargs["model"],
        model_name = "llama-2"
    )
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
