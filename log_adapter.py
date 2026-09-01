"""
log_adapter.py
─────────────────────────────────────────────────────────────────────────────
`main` (baseline RainbowPlus, no tree search) saves its comprehensive /
iteration logs in a different shape than algo/beam-search and algo/mcts:

  main            {"all_prompts":       {cell_key: [prompt, ...]},
                    "all_responses":     {cell_key: [response, ...]},
                    "all_scores":        {cell_key: [score, ...]},
                    "rejection_reasons": {cell_key: [reason, ...]},
                    ...}
                   -- dict-of-parallel-lists per cell, straight out of
                   Archive._archive (see rainbowplus/utils.py:
                   save_comprehensive_log / save_iteration_log). Index i in
                   all_prompts[cell_key] lines up with index i in
                   all_responses[cell_key] / all_scores[cell_key] /
                   rejection_reasons[cell_key].

  beam / mcts     {"map_elites_archive": {cell_key: [{"query": ...,
                    "jailbreak_prompt": ..., "target_response": ...,
                    "score": ...}, ...]}, ...}
                   -- dict of cell -> list of per-instance dicts (see
                   rainbowplus/utils.py: save_comprehensive_logs /
                   save_iteration_logs on those branches).

fair_blind_eval.py's extract_top_k() (and everything downstream of it --
judging, ASR, Mann-Whitney U, Cohen's d, QD metrics, ...) only understands
the second, "map_elites_archive" shape. normalize_log() converts a
main-schema log into that shape so the rest of fair_blind_eval.py runs
completely unmodified -- this module only adapts the *reader*, it does not
touch any metric definition.

Only rejection_reasons == "accepted" items are carried over: those are the
mutations RainbowPlus actually kept as a cell's candidate, which is the
main-schema analogue of an entry already sitting in beam/mcts's
map_elites_archive (something the search process decided to keep, not every
mutation it tried and discarded for being too similar to its parent or too
low-fitness). Comparing "accepted" main items against beam/mcts's archived
items keeps the blind comparison apples-to-apples.

main's schema never records the original seed query per-item (only the
mutated jailbreak_prompt / target_response / score are kept per index), so
`query` is left as "" for main-derived items -- fair_blind_eval.py's judge
only ever scores on (jailbreak_prompt, target_response), so this has no
effect on any metric.
"""

from typing import Any, Dict, List


def is_map_elites_schema(log_data: Dict[str, Any]) -> bool:
    """True for the beam/mcts on-disk shape (already what extract_top_k wants)."""
    return "map_elites_archive" in log_data


def is_main_schema(log_data: Dict[str, Any]) -> bool:
    """True for main's dict-of-parallel-lists shape."""
    return "all_prompts" in log_data


def convert_main_schema_to_archive(log_data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Build a map_elites_archive-shaped dict out of main's parallel-list log."""
    all_prompts = log_data.get("all_prompts", {})
    all_responses = log_data.get("all_responses", {})
    all_scores = log_data.get("all_scores", {})
    rejection_reasons = log_data.get("rejection_reasons", {})

    archive: Dict[str, List[Dict[str, Any]]] = {}
    for cell_key, prompts in all_prompts.items():
        responses = all_responses.get(cell_key, [])
        scores = all_scores.get(cell_key, [])
        reasons = rejection_reasons.get(cell_key, [])

        instances = []
        for i, prompt in enumerate(prompts):
            # Only keep mutations RainbowPlus actually accepted into the
            # cell -- see module docstring for why.
            reason = reasons[i] if i < len(reasons) else None
            if reason != "accepted":
                continue

            response = responses[i] if i < len(responses) else ""
            score = scores[i] if i < len(scores) else 0.0
            if not prompt or not response:
                continue

            instances.append({
                "query": "",  # not recorded per-item in main's schema
                "jailbreak_prompt": prompt,
                "target_response": response,
                "score": score,
            })

        if instances:
            archive[cell_key] = instances

    return archive


def normalize_log(log_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a log dict guaranteed to have a "map_elites_archive" key, whatever
    schema it started in. Every other top-level key (llm_call_logs,
    run_metadata, ...) is passed through unchanged.
    """
    if is_map_elites_schema(log_data):
        return log_data  # already the beam/mcts shape -- nothing to do

    if not is_main_schema(log_data):
        raise ValueError(
            "Unrecognized log schema: expected either 'map_elites_archive' "
            "(beam/mcts logs) or 'all_prompts' (main logs) at the top level."
        )

    normalized = dict(log_data)
    normalized["map_elites_archive"] = convert_main_schema_to_archive(log_data)
    return normalized
