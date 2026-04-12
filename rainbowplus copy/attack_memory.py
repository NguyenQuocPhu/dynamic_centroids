import numpy as np
import random
from typing import List, Dict, Any


class AttackMemory:
    def __init__(self, threshold_bot: float, threshold_top: float):
        self.threshold_bot = threshold_bot
        self.threshold_top = threshold_top
        self.entries_top: list[dict] = []
        self.entries_bot: list[dict] = []

    def add(self, entry: Dict[str, Any]):
        score = entry.get("score", 0.0)

        clean_entry = {
            "behavior": entry.get("behavior", None),
            "score": score,
            "prompt": entry.get("prompt", ""),
            "persona": entry.get("persona", {}),
            "risk": entry.get("risk", ""),
            "style": entry.get("style", "")
        }

        if score >= self.threshold_top:
            self.entries_top.append(clean_entry)
        elif score <= self.threshold_bot:
            self.entries_bot.append(clean_entry)

    def add_list(self, new_entries_list: List[Dict[str, Any]]):
        for entry in new_entries_list:
            self.add(entry)


    def _select_diverse_subset(
        self,
        candidates: List[Dict[str, Any]],
        k: int
    ) -> List[Dict[str, Any]]:

        if not candidates or k <= 0:
            return []

        # 1. Lọc candidate hợp lệ
        valid = [
            c for c in candidates
            if c.get("behavior") is not None and len(c["behavior"]) > 0
        ]

        if len(valid) <= k:
            return valid

        # 2. Sort theo score (giữ semantics cũ)
        valid.sort(key=lambda x: x.get("score", 0.0), reverse=True)

        try:
            embeddings = np.asarray([c["behavior"] for c in valid], dtype=np.float32)
        except Exception:
            return random.sample(valid, k)

        # 3. Normalize embeddings MỘT LẦN
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
        embeddings = embeddings / norms

        n = len(valid)

        # 4. Khởi tạo
        selected_mask = np.zeros(n, dtype=bool)
        first_idx = np.random.randint(0, n)
        #first_idx = 0
        selected_mask[first_idx] = True

        # sum_dist[i] = tổng khoảng cách từ i tới tập đã chọn
        # Khởi tạo bằng khoảng cách tới first
        first_vec = embeddings[first_idx]
        sum_dist = 1.0 - embeddings @ first_vec

        # Không cho chọn lại chính nó
        sum_dist[first_idx] = -np.inf

        selected_indices = [first_idx]

        # 5. Greedy loop (O(Nk))
        for _ in range(1, k):
            # Chọn candidate có tổng distance lớn nhất
            next_idx = np.argmax(sum_dist)

            if sum_dist[next_idx] == -np.inf:
                break

            selected_indices.append(next_idx)
            selected_mask[next_idx] = True

            # Update sum_dist incrementally
            next_vec = embeddings[next_idx]
            sum_dist += 1.0 - embeddings @ next_vec

            # Disable already selected
            sum_dist[selected_mask] = -np.inf

        return [valid[i] for i in selected_indices]

    def get_prompts_top(self, sample_size: int) -> List[Dict[str, Any]]:
        return self._select_diverse_subset(self.entries_top, sample_size)

    def get_prompts_bot(self, sample_size: int) -> List[Dict[str, Any]]:
        return self._select_diverse_subset(self.entries_bot, sample_size)
