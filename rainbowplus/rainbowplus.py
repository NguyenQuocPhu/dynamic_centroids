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
    save_ga_iteration_log,      # New logging function
    save_ga_comprehensive_log,  # New logging function
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



@dataclass
class Individual:
    """Enhanced Individual with metadata tracking."""
    embedding: np.ndarray
    fitness: float
    prompt: str
    response: str = ""
    history: List[str] = field(default_factory=list)
    flag: int = 0
    pca_embedding: Optional[np.ndarray] = None
    
    # Lineage tracking
    prompt_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = None
    seed_id: Optional[str] = None
    
    def __post_init__(self):
        """Ensure embedding is numpy array."""
        self.embedding = np.array(self.embedding)
        if self.pca_embedding is not None:
            self.pca_embedding = np.array(self.pca_embedding)

class EmbeddingSpace:
    """Manages independent embedding space with PCA and clustering."""
    
    def __init__(
        self, 
        n_components: int, 
        n_clusters: int, 
        cluster_capacity: int = 10,
    ):
        self.n_components = n_components
        self.n_clusters = n_clusters
        self.cluster_capacity = cluster_capacity
        
        self.individuals: List[Individual] = []
        self.clusters: Dict[int, List[Individual]] = {i: [] for i in range(n_clusters)}
        
        # ML Models
        self.pca = PCA(n_components=n_components)
        self.kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
        self.is_pca_fitted = False

    def add_individual(self, individual: Individual) -> None:
        """Add individual to space and update clustering."""
        self.individuals.append(individual)
        self._update_space()

    def _update_space(self) -> None:
        """Execute PCA, K-means, and update cluster capacity."""
        if len(self.individuals) < self.n_clusters:
            # Not enough data for clustering
            self.clusters[0] = sorted(
                self.individuals, 
                key=lambda x: x.fitness, 
                reverse=True
            )[:self.cluster_capacity]
            return

        embeddings = np.array([ind.embedding for ind in self.individuals])

        # 1. Dimensionality reduction with PCA
        if not self.is_pca_fitted:
            pca_embs = self.pca.fit_transform(embeddings)
            self.is_pca_fitted = True
        else:
            pca_embs = self.pca.transform(embeddings)

        for i, ind in enumerate(self.individuals):
            ind.pca_embedding = pca_embs[i]

        # 2. K-means clustering
        self.kmeans.fit(pca_embs)
        labels = self.kmeans.labels_
        
        temp_clusters = {c: [] for c in range(self.n_clusters)}
        for i, ind in enumerate(self.individuals):
            temp_clusters[labels[i]].append(ind)

        # 3. Keep only top individuals per cluster
        self.clusters = {}
        for c in range(self.n_clusters):
            if not temp_clusters[c]:
                continue
            cluster_inds = sorted(
                temp_clusters[c], 
                key=lambda x: x.fitness, 
                reverse=True
            )
            self.clusters[c] = cluster_inds[:self.cluster_capacity]

        # Update main buffer
        self.individuals = []
        for c_list in self.clusters.values():
            self.individuals.extend(c_list)

    def select_random(self, k: int = 1) -> Optional[Individual]:
        """
        Select random individual from random cluster's top-k.
        
        Args:
            k: Select from top-k individuals (1 = best only)
        
        Returns:
            Selected Individual or None
        """
        active_clusters = [c for c in self.clusters.values() if len(c) > 0]
        if not active_clusters:
            return None
            
        selected_cluster = random.choice(active_clusters)
        actual_k = min(k, len(selected_cluster))
        top_k = selected_cluster[:actual_k]
        
        return random.choice(top_k)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Adversarial Prompt Generation")
    parser.add_argument("--num_samples", type=int, help="Number of initial seed prompts")
    parser.add_argument("--max_iters", type=int, help="Maximum number of iterations")
    parser.add_argument("--sim_threshold", type=float, help="Similarity threshold")
    parser.add_argument("--num_mutations", type=int, help="Mutations per iteration")
    parser.add_argument("--fitness_threshold", type=float, help="Minimum fitness score")
    parser.add_argument("--config_file", type=str, default="./configs/base.yml")
    parser.add_argument("--log_dir", type=str, help="Log directory")
    parser.add_argument("--log_interval", type=int, help="Iterations between logs")
    parser.add_argument("--seed_memory", type=str, help="Path to seed prompts")
    parser.add_argument("--dataset", type=str, help="Dataset name")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle seeds")
    parser.add_argument("--threshold_bot_memory", type=float, default=0.4)
    parser.add_argument("--threshold_top_memory", type=float, default=0.6)
    parser.add_argument("--number_example_prompts", type=int, help="Few-shot examples")
    
    parser.add_argument("--k_elites", type=int, help="Elites per cluster")
    parser.add_argument("--buffer_size", type=int, default=5, help="History buffer size")
    parser.add_argument("--selection_k", type=int, default=2, help="Top-k for selection")
    return parser.parse_args()


def merge_config_with_args(config, args):
    """Merge config file with CLI arguments."""
    merged = type('Args', (), {})()
    
    # Set defaults from config
    merged.num_samples = config.num_samples
    merged.max_iters = config.max_iters
    merged.sim_threshold = config.sim_threshold
    merged.num_mutations = config.num_mutations
    merged.fitness_threshold = config.fitness_threshold
    merged.log_dir = config.log_dir
    merged.log_interval = config.log_interval
    merged.shuffle = config.shuffle
    merged.threshold_bot_memory = config.threshold_bot_memory
    merged.threshold_top_memory = config.threshold_top_memory
    merged.seed_memory = config.seed_memory or "configs/seed_memory.yml"
    merged.number_example_prompts = config.number_example_prompts
    merged.k_elites = config.k_elites
    merged.dataset = config.sample_prompts or "./data/do-not-answer.json"
    merged.config_file = args.config_file
    merged.buffer_size = getattr(config, 'buffer_size', 5)
    merged.selection_k = getattr(config, 'selection_k', 2)
    
    # Override with CLI args
    for key in vars(args):
        val = getattr(args, key)
        if val is not None:
            setattr(merged, key, val)
    
    return merged

def load_descriptors(config):
    descriptors = {}
    for path, descriptor in zip(config.archive["path"], config.archive["descriptor"]):
        if path.endswith('.yml'):
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
                if descriptor == "Persona":
                    descriptors[descriptor] = list(data['personas'].keys())
        else:
            descriptors[descriptor] = load_txt(path)
    return descriptors

def update_archive(archive_obj: Archive, key: Tuple, data: List) -> None:
    """Helper to update archive with data."""
    if not archive_obj.exists(key):
        archive_obj.add(key, data)
    else:
        archive_obj.extend(key, data)


def prepare_few_shot_examples(attack_memory, num_examples: int) -> Tuple[str, str]:
    """Prepare few-shot examples from attack memory."""
    sample_top = attack_memory.get_prompts_top(num_examples)
    sample_bot = attack_memory.get_prompts_bot(num_examples)
    
    failed_examples = "\n".join([format_example(e) for e in sample_top])
    success_examples = "\n".join([format_example(e) for e in sample_bot])
    
    return failed_examples, success_examples

def run_rainbowplus(args, config, seed_prompts=[], llms=None, fitness_fn=None, similarity_fn=None):
    """Main RainbowPlus algorithm."""
    
    # Load seed prompts
    if not seed_prompts:
        seed_prompts = load_json(
            config.sample_prompts, 
            field="question", 
            num_samples=args.num_samples, 
            shuffle=args.shuffle
        )
    
    descriptors = load_descriptors(config)

    # Initialize archives
    all_prompts = Archive("all_prompts")
    all_responses = Archive("all_responses")
    all_scores = Archive("all_scores")
    all_similarities = Archive("all_similarities")
    rejection_reasons = Archive("rejection_reasons")
    
    all_prompt_ids = Archive("all_prompt_ids")
    all_parent_ids = Archive("all_parent_ids")
    all_seed_ids = Archive("all_seed_ids")

    # Initialize Embedding
    embedding_model = SentenceTransformer('BAAI/bge-m3')
    # Initialize Growing Archive
    behavior_dim = 1024
    adv_prompts = Archive[EmbeddingSpace]("adv_prompts")
    
    # Initialize attack memory
    attack_memory = AttackMemory(
        args.threshold_bot_memory, 
        args.threshold_top_memory
    )

    # Setup logging
    dataset_name = Path(config.sample_prompts).stem
    target_model_name = config.target_llm.model_kwargs["model"]
    log_dir = Path(args.log_dir) / target_model_name / dataset_name
    log_dir.mkdir(parents=True, exist_ok=True)

    # Prepare seed metadata
    seed_metadata = []
    for idx, prompt in enumerate(seed_prompts):
        pid = f"seed_{idx}"
        seed_metadata.append({
            "prompt": prompt,
            "seed_id": pid,
            "prompt_id": pid
        })

    # --- MAIN LOOP ---
    for iteration in range(args.max_iters):
        logger.warning(f"#####ITERATION: {iteration}")

        # --- Step 1: Selection ---
        current_parent_id = None
        current_seed_id = None
        mutation_flag = 0
        history_buffer = []
        temp_response = ""
        
        if iteration < len(seed_prompts):
            # Phase 1: Seeding
            selection = seed_metadata[iteration]
            base_prompt = selection["prompt"]
            current_parent_id = selection["prompt_id"]
            current_seed_id = selection["seed_id"]
            
        else:
            # Phase 2: Evolution from archive
            all_spaces = adv_prompts.flatten_values()
            if not all_spaces:
                logger.warning("No spaces in archive, using fallback seed")
                selection = random.choice(seed_metadata)
                base_prompt = selection["prompt"]
                current_parent_id = selection["prompt_id"]
                current_seed_id = selection["seed_id"]
            else:
                random_space = random.choice(all_spaces)
                parent_individual = random_space.select_random(k=args.selection_k)
                
                if parent_individual:
                    base_prompt = parent_individual.prompt
                    mutation_flag = parent_individual.flag
                    history_buffer = parent_individual.history[-args.buffer_size:]
                    temp_response = parent_individual.response
                    current_parent_id = parent_individual.prompt_id
                    current_seed_id = parent_individual.seed_id
                else:
                    logger.warning("Empty space selected, using fallback")
                    selection = random.choice(seed_metadata)
                    base_prompt = selection["prompt"]
                    current_parent_id = selection["prompt_id"]
                    current_seed_id = selection["seed_id"]


        descriptor = {key: random.choice(values) for key, values in descriptors.items()}
        log_key_tuple = (descriptor.get("Category"), descriptor.get("Style"))

        failed_ex, success_ex = prepare_few_shot_examples(
            attack_memory, 
            args.number_example_prompts
        )
        
        selective_prompt = SELECTOR_PROMPT.format(
            successful_examples_text=success_ex,
            prompt=base_prompt
        )
        
        selector_params = config.mutator_llm.sampling_params.copy()
        selector_params['temperature'] = 0.7
        
        cognitive_analysis_raw = llms[config.mutator_llm.model_kwargs["model"]].generate(
            selective_prompt, selector_params
        )
        cognitive_primitives = _repair_yaml_primitives(cognitive_analysis_raw)
        
        logger.info(f"Cognitive Analysis: {cognitive_primitives}")
        
        
        strategy_analysis = ""
        if history_buffer:
            history_text = "\n".join(
                f"Strategy {i+1}: {text}" for i, text in enumerate(history_buffer)
            )
            
            analysis_prompt = ANALYZER_PROMPT.format(
                history=history_text,
                prompt=base_prompt,
                response=temp_response
            )
            
            strategy_analysis = llms[config.mutator_llm.model_kwargs["model"]].generate(
                analysis_prompt, selector_params
            )

        if mutation_flag == 0:
            mutation_prompt = MUTATOR_PROMPT_V1.format(
                risk=descriptor["Category"],
                style=descriptor["Style"],
                prompt=base_prompt,
                strategy=strategy_analysis
            )
        else:
            mutation_prompt = MUTATOR_PROMPT.format(
                risk=descriptor["Category"],
                style=descriptor["Style"],
                prompt=base_prompt,
                primitives=cognitive_primitives.strip()
            )
        
        mutated_prompts = llms[config.mutator_llm.model_kwargs["model"]].batch_generate(
            [mutation_prompt] * args.num_mutations, config.mutator_llm.sampling_params
        )

        # Filter by similarity
        mutated_prompts = [
            p for p in mutated_prompts if similarity_fn.score(p, base_prompt) < args.sim_threshold
        ]

        if mutated_prompts:
            # Responses
            target_prompts = []
            for i, p in enumerate(mutated_prompts):
                if not p or not p.strip():
                    raise ValueError(f"Empty mutated_prompt at index {i}")
                target_prompts.append(TARGET_PROMPT.format(prompt=p.strip()))
            target_model = config.target_llm.model_kwargs["model"]
            candidates = llms[target_model].batch_generate(target_prompts, config.target_llm.sampling_params)

            # Scores
            fitness_scores = fitness_fn.batch_score(mutated_prompts, candidates, config.fitness_llm.sampling_params)
            sim_scores = [similarity_fn.score(p, prompt) for p in mutated_prompts]
            
            # Embeddings (All prompts)
            all_behaviors = embedding_model.model.encode(mutated_prompts)

            batch_prompt_ids = []
            batch_reasons = []
            
            # NOTE: We use log_key based on descriptors for archiving, not for GA logic
            log_key_tuple = (descriptor.get("Category"), descriptor.get("Style"))

            for idx, (m_prompt, response, fit, sim, behavior) in enumerate(zip(mutated_prompts, candidates, fitness_scores, sim_scores, all_behaviors)):
                new_pid = str(uuid.uuid4())
                batch_prompt_ids.append(new_pid)
                
                status = "accepted"
                if sim > args.sim_threshold:
                    status = "similarity_too_high"

                if status == "accepted":
                    # Create new individual
                    new_history = history_buffer + [strategy_analysis] if strategy_analysis else history_buffer
                    
                    ind = Individual(
                        embedding=behavior,
                        fitness=fit,
                        prompt=m_prompt,
                        response=response,
                        history=new_history,
                        flag=1 if fit >= args.fitness_threshold else 0,
                        prompt_id=new_pid,
                        parent_id=current_parent_id,
                        seed_id=current_seed_id
                    )
                    
                    # Add to archive
                    if not adv_prompts.exists(log_key_tuple):
                        new_space = EmbeddingSpace(
                            n_components=behavior_dim,
                            n_clusters=args.k_elites,
                            cluster_capacity=10
                        )
                        adv_prompts.add(log_key_tuple, [new_space])
                    
                    local_space = adv_prompts.get(log_key_tuple)[0]
                    local_space.add_individual(ind)
                    
                    # Add to attack memory
                    attack_memory.add({
                        "behavior": behavior,
                        "prompt": m_prompt,
                        "score": fit
                    })

                batch_reasons.append(status)

            # --- Step 4: Update Archives (using Archive Objects) ---
            def update_archive(archive_obj, k, data):
                if not archive_obj.exists(k): archive_obj.add(k, data)
                else: archive_obj.extend(k, data)

            # Update Content Archives
            update_archive(all_prompts, log_key_tuple, mutated_prompts)
            update_archive(all_responses, log_key_tuple, candidates)
            update_archive(all_scores, log_key_tuple, fitness_scores)
            update_archive(all_similarities, log_key_tuple, sim_scores)
            update_archive(rejection_reasons, log_key_tuple, batch_reasons)
            #update_archive(all_iters, log_key_tuple, [i] * len(mutated_prompts)
            
            # Update Lineage Archives (Pipeline of IDs)
            update_archive(all_prompt_ids, log_key_tuple, batch_prompt_ids)
            update_archive(all_parent_ids, log_key_tuple, [current_parent_id] * len(mutated_prompts))
            update_archive(all_seed_ids, log_key_tuple, [current_seed_id] * len(mutated_prompts))

            # Also update legacy archives for backward compatibility
            # (Optional: filter only successful ones here if needed, but let's keep simple)
            
            '''
            if (i + 1) % args.log_interval == 0 or (i + 1) == args.max_iters:
                timestamp = time.strftime(r"%Y%m%d-%H%M%S")
                
                # Save GA State (Centroids)
                save_ga_iteration_log(
                    log_dir, GA, timestamp, iteration=i, max_iters=args.max_iters
                )
                
                # Save Comprehensive Log (All History + Lineage + GA State)
            '''    
                
                

    # Final Save
    final_ts = time.strftime(r"%Y%m%d-%H%M%S")
    #save_ga_iteration_log(log_dir, GA, final_ts, iteration=-1, max_iters=args.max_iters)
    
    save_attack_memory_log(log_dir, attack_memory, final_ts)
    
    save_ga_comprehensive_log(
        log_dir, 
        adv_prompts,
        all_prompts, all_responses, all_scores, all_similarities, rejection_reasons,
        final_ts, iteration=-1,
        all_prompt_ids=all_prompt_ids,
        all_parent_ids=all_parent_ids,
        all_seed_ids=all_seed_ids,
        max_iters=args.max_iters
    )
    return adv_prompts

if __name__ == "__main__":
    args = parse_arguments()
    config = ConfigurationLoader.load(args.config_file)
    merged_args = merge_config_with_args(config, args)
    if not config.sample_prompts:
        config.sample_prompts = merged_args.dataset

    llms = initialize_language_models(config)
    fitness_fn = LlamaGuard(config.fitness_llm.model_kwargs)
    similarity_fn = BleuScoreNLTK()

    print(config)

    run_rainbowplus(
        merged_args,
        config,
        seed_prompts=[],
        llms=llms,
        fitness_fn=fitness_fn,
        similarity_fn=similarity_fn,
    )