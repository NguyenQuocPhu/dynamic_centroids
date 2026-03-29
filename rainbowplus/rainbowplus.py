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
    save_all_instances,
    format_example,
    save_attack_memory_log,
    _repair_yaml_primitives
)
from rainbowplus.archive import Archive
from rainbowplus.configs import ConfigurationLoader
from rainbowplus.prompts import MUTATOR_PROMPT, TARGET_PROMPT, MUTATOR_PROMPT_V1, SELECTOR_PROMPT, ANALYZER_PROMPT
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
        "--num_samples", type=int, default=5, help="Number of initial seed prompts"
    )
    parser.add_argument(
        "--max_iters", type=int, default=20, help="Maximum number of iteration steps"
    )
    parser.add_argument(
        "--total", type=int, default=10, help="Total number of attacks per leaf node"
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
        default=50,
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
import itertools
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
from rainbowplus.graph_easyjailbreak.Tree import Forest, MCTSNode

def run_rainbowplus(args, config, seed_prompts=[], attack_model=None, target_model=None, fitness_fn=None, similarity_fn=None):
    """Main RainbowPlus algorithm."""
    
    descriptors = load_descriptors(config)

    # Initialize archives
    all_prompts = Archive("all_prompts")
    all_responses = Archive("all_responses")
    all_scores = Archive("all_scores")
    all_similarities = Archive("all_similarities")
    rejection_reasons = Archive("rejection_reasons")
    archive_by_depth = Archive[Forest]("map-elites")
    keys = list(descriptors.keys())
    all_combinations = list(itertools.product(*[descriptors[k] for k in keys]))

    for combo in all_combinations:
        archive_by_depth.add(combo, [Forest()]) 

    # Setup logging
    dataset_name = Path(config.sample_prompts).stem
    log_dir = Path(args.log_dir) / target_model.model_name / dataset_name
    log_dir.mkdir(parents=True, exist_ok=True)

    seed = JailbreakDataset("AdvBench")

    global_context = []
    attacker = GAP (attack_model=attack_model,
                        target_model=target_model,
                        eval_model=fitness_fn,
                        jailbreak_datasets=seed,
                        tree_width=999,
                        tree_depth=1,
                        root_num=1,
                        branching_factor=3,
                        keep_last_n=3,
                        max_n_attack_attempts=2,
                        logging_filename=dataset_name,
                        conversation_history=global_context,
                        attack_max_n_tokens = config.mutator_llm.sampling_params.get('max_tokens', 500),
                        attack_temperature=config.mutator_llm.sampling_params.get('temperature', 1),
                        attack_top_p = config.mutator_llm.sampling_params.get('top_p', 0.9),
                        target_max_n_tokens= config.target_llm.sampling_params.get('max_tokens', 150),
                        target_temperature=config.target_llm.sampling_params.get('temperature', 1),
                        target_top_p=config.target_llm.sampling_params.get('top_p', 1),
                        judge_max_n_tokens=config.fitness_llm.sampling_params.get('max_tokens', 10),
                        judge_temperature=config.fitness_llm.sampling_params.get('temperature', 1))
    
    for iteration in range(1, args.max_iters):
        print(f"\n\n=== Iteration {iteration} ===\n\n")
        keys = archive_by_depth.keys()
        selected_keys = random.sample(keys, 5)
        nodes = []
        for key in selected_keys:
            root = archive_by_depth.get(key)[0]
            if len(root.dummy_root.children) == 0:
                instance = random.choice(seed)
                inst = instance.copy()
                inst.jailbreak_prompt = inst.query
                inst.reference_responses = [""]
                new_node = MCTSNode(instance=inst)
                nodes.append(new_node)
            else:
                nodes.append(root.select_leaf())

        batch_instances = []

        # 1. Gom tất cả prompt lại
        for node in nodes:
            descriptor = {key: random.choice(value) for key, value in descriptors.items()}
            key = tuple(descriptor.values())

            descriptor_str = "\n".join([f"{k}: {v}" for k, v in descriptor.items()])

            transfer_prompt = MUTATOR_PROMPT_V1.format(
                risk=key[0],
                style=key[1],
                prompt=node.instance.jailbreak_prompt
            )
            attacker.attack_model.set_system_message(transfer_prompt)

            K = 3 
            best_instance = None
            best_score = -1
            for _ in range(K):
                transferred_prompt = attacker.attack_model.generate(node.instance.jailbreak_prompt)
                
                temp_instance = node.instance.copy()
                temp_instance.query = descriptor_str
                temp_instance.jailbreak_prompt = transferred_prompt
                temp_instance.reference_responses = [""]
                
                target_response = attacker.target_model.generate(transferred_prompt)
                temp_instance.target_responses = [target_response]
                
                attacker.evaluator([temp_instance])
                
                current_score = temp_instance.eval_results[-1] if temp_instance.eval_results else 0
                
                if current_score > best_score:
                    best_score = current_score
                    best_instance = temp_instance

            attacker.attack_model.set_system_message("")

            if 'conv' in best_instance.attack_attrs:
                del best_instance.attack_attrs['conv']
            best_instance.attack_attrs['cell_key'] = key
            batch_instances.append(best_instance)

        dataset = JailbreakDataset(batch_instances)
        attacker.jailbreak_datasets = dataset

        for instance in tqdm(dataset, desc=f"Processing evolved instances in iteration {iteration}"):
            dummy_node = archive_by_depth.get(instance.attack_attrs['cell_key'])[0]
            dummy_node.add_migrant_tree(instance)
            for i in range(args.total):
                leaf = dummy_node.select_leaf()
                print(f"\n\n=== Selected leaf with prompt in step {i}: {leaf.instance.jailbreak_prompt} and score: {leaf.max_score * 9.0 + 1} ===\n\n")
                if getattr(leaf, 'is_terminal', False):
                    continue
                new_instances = attacker.single_attack(leaf.instance)
                new_children = []

                for data in new_instances: 
                    for inst in data:
                        child = MCTSNode(instance=inst, parent=leaf)
                        leaf.children.append(child) 
                        new_children.append(child)

                if new_children:
                    dummy_node.backpropagate(leaf, new_children)

        
    current_ts = time.strftime(r"%Y%m%d-%H%M%S")
    
    save_all_instances(
        archive=archive_by_depth,
        save_dir=log_dir,
        filename=current_ts
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