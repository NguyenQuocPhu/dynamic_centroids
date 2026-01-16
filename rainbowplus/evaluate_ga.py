import os
import json
import argparse
import ast
from pathlib import Path
from typing import List, Dict, Any

from rainbowplus.archive import Archive
from rainbowplus.prompts import EVALUATE_PROMPT
from rainbowplus.configs import EvalConfigLoader, EvaluationConfig
from rainbowplus.llms.vllm import vLLM


class EvaluationProcessor:
    def __init__(
        self,
        config: EvaluationConfig,
        keyword: str
    ):
        """
        Initialize the evaluation processor.

        :param config: Configuration object
        :param keyword: Keyword of global config file name
        """
        self.config = config
        self.keyword = keyword

        # Initialize directories and parameters
        self.log_dir = Path(self.config.log_dir)
        self.method = self.config.method

        self.sampling_params = self.config.judge_llm.sampling_params
        self.llm = vLLM(self.config.judge_llm.model_kwargs)
        self.algo = self.config.algo

    def _save_eval_log(self, file_name: str, archive_instances: List[Archive]) -> None:
        """Save evaluation logs to a JSON file."""
        log_path = f"{file_name}_eval.json"

        log_data = {
            archive.name: {str(key): value for key, value in archive._archive.items()}
            for archive in archive_instances
        }

        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)

        print(f"Log saved to {log_path}")

    def _process(
        self, adv_prompts: Archive, personas: Archive, responses: Archive, scores: Archive, similarities: Archive, reasons: Archive
    ) -> Dict[str, Archive]:
        """Process evaluation using median or max method."""
        eval_prompts = Archive("eval_prompts")
        eval_persona = Archive("eval_persona")
        eval_responses = Archive("eval_responses")
        eval_scores = Archive("eval_scores")
        eval_similarities = Archive("eval_similarities")
        eval_reasons = Archive("eval_reasons")
        judge_responses = Archive("judge_responses")

            
        # Process all prompts
        for key in adv_prompts._archive.keys():
            chosen_prompts = adv_prompts.get(key)
            chosen_persona = personas.get(key)
            chosen_responses = responses.get(key)
            chosen_scores = scores.get(key)
            chosen_similarities = similarities.get(key)
            chosen_reasons = reasons.get(key)

            # Evaluate

            queries = [
                EVALUATE_PROMPT.format(prompt=prompt, response=response)
                for prompt, response in zip(chosen_prompts, chosen_responses)
            ]
            outputs = self.llm.batch_generate(queries, self.sampling_params)

            eval_prompts.add(key, chosen_prompts)
            eval_persona.add(key, chosen_persona)
            eval_responses.add(key, chosen_responses)
            eval_scores.add(key, chosen_scores)
            eval_similarities.add(key, chosen_similarities)
            eval_reasons.add(key, chosen_reasons)
            judge_responses.add(key, outputs)

        return {
            "eval_prompts": eval_prompts,
            "eval_persona": eval_persona,
            "eval_responses": eval_responses,
            "eval_scores": eval_scores,
            "eval_similarities": eval_similarities,
            "eval_reasons": eval_reasons,
            "judge_responses": judge_responses,
        }
    
    def process_evaluations(self, k: int):
        """Process evaluations across log files."""
        # Get sorted log files starting with the specified method
        log_files = sorted(
            [f for f in os.listdir(self.log_dir) if f.startswith(self.algo) and self.keyword in f]
        )

        for idx, log_file in enumerate(log_files):
            # Load log file
            path = os.path.join(self.log_dir, log_file)
            with open(path, "r") as f:
                log_data = json.load(f)

            # Create Archives from log data
            all_prompts = log_data.get('all_prompts', [])
            all_responses = log_data.get('all_responses', [])
            all_scores = log_data.get('all_scores', [])
            all_similarities = log_data.get('all_similarities', [])
            rejection_reasons = log_data.get('rejection_reasons', [])

            adv_prompts = Archive("all_prompts")
            personas = Archive("all_personas")
            responses = Archive("all_responses")
            scores = Archive("all_scores")
            similarities = Archive("all_similarities")
            reasons = Archive("rejection_reasons")

            for key, list_prompts in all_prompts.items():
                log_key = ast.literal_eval(key)
                risk = log_key[0]
                style = log_key[1]
                group_key = (risk, style)

                responses_list = all_responses.get(key, [])
                scores_list = all_scores.get(key, [])
                similarities_list = all_similarities.get(key, [])
                reasons_list = rejection_reasons.get(key, [])
                personas_list = [log_key[2]] * len(list_prompts)

                def update_archive(archive_obj, k, data):
                    if not archive_obj.exists(k): archive_obj.add(k, data)
                    else: archive_obj.extend(k, data)
                
                update_archive(adv_prompts, group_key, list_prompts)
                update_archive(personas, group_key, personas_list)
                update_archive(responses, group_key, responses_list)
                update_archive(scores, group_key, scores_list)
                update_archive(similarities, group_key, similarities_list)
                update_archive(reasons, group_key, reasons_list)
            
            final_prompts = Archive("all_prompts_topk")
            final_personas = Archive("all_personas_topk")
            final_responses = Archive("all_responses_topk")
            final_scores = Archive("all_scores_topk")
            final_similarities = Archive("all_similarities_topk")
            final_reasons = Archive("rejection_reasons_topk")

            for key in scores.keys():
                score = scores.get(key)
                # 3. Tìm danh sách index của K phần tử điểm cao nhất
                # enumerate giúp giữ lại index gốc: (index, score)
                # sorted với key=x[1] để sort theo score, reverse=True để giảm dần
                sorted_indices = sorted(enumerate(score), key=lambda x: x[1], reverse=True)
                top_indices = [idx for idx, val in sorted_indices[:k]]
                def extract_by_indices(source_archive):
                    data_list = source_archive.get(key)
                    if not data_list: return []
                    # Map index đã chọn sang dữ liệu tương ứng
                    return [data_list[i] for i in top_indices]
                final_scores.add(key, [score[i] for i in top_indices])
                
                final_prompts.add(key, extract_by_indices(adv_prompts))
                final_personas.add(key, extract_by_indices(personas))
                final_responses.add(key, extract_by_indices(responses))
                final_similarities.add(key, extract_by_indices(similarities))
                final_reasons.add(key, extract_by_indices(reasons))

            # Process based on method
            processed_archives = self._process(final_prompts, final_personas, final_responses, final_scores, final_similarities, final_reasons)

            # Save evaluation log
            self._save_eval_log(log_files[idx], list(processed_archives.values()))


def main():
    parser = argparse.ArgumentParser(description="Evaluation Processing Script")
    parser.add_argument(
        "-c", "--config", default="configs/eval.yml", help="Path to configuration file"
    )
    parser.add_argument("-l", "--log_dir", help="Override log directory from config")
    parser.add_argument("-a", "--algo", default="rainbowplus", help="Override algorithm from config")
    parser.add_argument("-m", "--method", default="log", help="Override evaluation method from config")
    parser.add_argument("-k", "--keyword", default="global", help="Keyword of global config file name")
    parser.add_argument("--top_k", type = int, default = 5, help = "Number of elites will be choosen in each cell")

    args = parser.parse_args()

    # Update config with command-line arguments
    config = EvalConfigLoader.load(args.config)
    config.log_dir = args.log_dir or config.log_dir
    config.algo = args.algo or config.algo
    config.method = args.method or config.method
    k = args.top_k or config.top_k

    processor = EvaluationProcessor(
        config=config,
        keyword=args.keyword
    )
    processor.process_evaluations(k)


if __name__ == "__main__":
    main()