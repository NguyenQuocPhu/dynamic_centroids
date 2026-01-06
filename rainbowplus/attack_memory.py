import heapq
from typing import List, Dict, Any
import random
import numpy as np

class AttackMemory:
    def __init__(self, threshold_bot: float, threshold_top: float):
        self.threshold_bot = threshold_bot
        self.threshold_top = threshold_top
        self.entries_top: list[dict] = []
        self.entries_bot: list[dict] = []
    
    def add(self, entry: Dict[str, Any]):
        """
        Thêm một entry mới.
        """
        score = entry.get("score", 0)
        
        clean_entry = {
            "behavior": entry.get("behavior", None),
            "score": score,
            "prompt": entry.get("prompt", ""),
            "persona": entry.get("persona", {}),
            "risk": entry.get("risk", ""),
            "style": entry.get("style", "")
        }

        # --- LOGIC XỬ LÝ TOP (SUCCESSFUL) ---
        if score >= self.threshold_top:
            self.entries_top.append(clean_entry)

        # --- LOGIC XỬ LÝ BOT (FAILED) ---
        elif score <= self.threshold_bot:
            self.entries_bot.append(clean_entry)
    
    def add_list(self, new_entries_list: List[Dict[str, Any]]):
        for entry in new_entries_list:
            self.add(entry)


    def _select_diverse_subset(self, candidates: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        """
        Chọn k phần tử sử dụng thuật toán Greedy k-DPP (Determinantal Point Processes).
        
        Mục tiêu: Tối đa hóa log-determinant của ma trận kernel con.
        Công thức Kernel: L_ij = Quality_i * Quality_j * Similarity_ij
        - Quality: score (đảm bảo > 0)
        - Similarity: Cosine Similarity của behavior vector
        """
        N = len(candidates)
        
        # 1. Xử lý các trường hợp biên
        if N == 0:
            return []
        
        # Lọc các candidate hợp lệ (có behavior vector)
        valid_candidates = []
        embeddings_list = []
        scores_list = []

        for c in candidates:
            emb = c.get('behavior')
            if emb is not None and len(emb) > 0:
                valid_candidates.append(c)
                embeddings_list.append(emb)
                
                # Lưu ý quan trọng: Điểm chất lượng (Quality) phải dương.
                # Với entries_bot (failed), điểm thường thấp hoặc bằng 0. 
                # Ta cộng thêm một epsilon nhỏ để tránh ma trận L toàn 0 (khiến det = 0).
                raw_score = float(c.get("score", 0))
                scores_list.append(max(raw_score, 1e-4)) 
        
        num_valid = len(valid_candidates)
        if num_valid <= k:
            return valid_candidates

        # 2. Xây dựng Ma trận Kernel (L)
        try:
            # Chuyển đổi sang numpy array
            V = np.array(embeddings_list)  # Shape: (N, Embedding_Dim)
            scores = np.array(scores_list) # Shape: (N,)

            # Chuẩn hóa vector để tính Cosine Similarity (V_norm . V_norm.T)
            # Thêm 1e-9 để tránh chia cho 0
            norms = np.linalg.norm(V, axis=1, keepdims=True)
            V_norm = V / (norms + 1e-9)
            
            # Ma trận tương đồng (Similarity Matrix) S
            # S_ij chạy từ -1 đến 1 (Cosine)
            S = np.dot(V_norm, V_norm.T) 

            # Ma trận Kernel L
            # L_ij = q_i * q_j * S_ij
            # Dùng broadcasting để nhân scores
            Q = scores.reshape(-1, 1) * scores.reshape(1, -1)
            L = Q * S  # Element-wise multiplication
            
            # Để đảm bảo tính ổn định số học (L nên là Positive Semi-Definite),
            # ta có thể cộng thêm một chút vào đường chéo (regularization)
            L[np.diag_indices_from(L)] += 1e-6

        except Exception as e:
            # Fallback nếu tính toán ma trận lỗi (tràn bộ nhớ, v.v.)
            print(f"Error in DPP calculation: {e}, falling back to random.")
            return random.sample(valid_candidates, k)

        # 3. Thuật toán Greedy k-DPP
        # Tại mỗi bước, chọn phần tử làm tăng định thức của ma trận con nhiều nhất
        
        selected_indices = []
        remaining_indices = list(range(num_valid))

        for _ in range(k):
            best_idx = -1
            max_log_det = -float('inf')

            # Duyệt qua các phần tử còn lại để tìm ứng viên tốt nhất
            for idx in remaining_indices:
                # Tạo tập index giả định: [đã chọn] + [ứng viên mới]
                current_subset = selected_indices + [idx]
                
                # Trích xuất ma trận con (Sub-kernel matrix)
                # np.ix_ tạo lưới chỉ số để cắt ma trận
                L_sub = L[np.ix_(current_subset, current_subset)]
                
                # Tính định thức
                # Sử dụng slogdet để an toàn hơn với số quá nhỏ (underflow) hoặc quá lớn
                sign, log_det = np.linalg.slogdet(L_sub)
                
                # Chỉ xét nếu định thức dương (sign > 0)
                if sign > 0 and log_det > max_log_det:
                    max_log_det = log_det
                    best_idx = idx
            
            # Nếu tìm được ứng viên (best_idx != -1), thêm vào danh sách
            if best_idx != -1:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
            else:
                # Trường hợp hiếm: Không tìm được ứng viên nào làm det > 0
                # (Ví dụ các vector còn lại phụ thuộc tuyến tính hoàn toàn)
                # Ta chọn đại một cái còn lại để lấp đầy k
                if remaining_indices:
                    picked = remaining_indices[0]
                    selected_indices.append(picked)
                    remaining_indices.remove(picked)

        # Trả về kết quả
        return [valid_candidates[i] for i in selected_indices]

    def get_prompts_top(self, sample_size: int) -> List[Dict[str, Any]]:
        return self._select_diverse_subset(self.entries_top, sample_size)

    def get_prompts_bot(self, sample_size: int) -> List[Dict[str, Any]]:
        return self._select_diverse_subset(self.entries_bot, sample_size)