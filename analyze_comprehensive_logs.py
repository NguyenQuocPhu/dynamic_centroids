import random
import numpy as np
import argparse
import json
import glob
import os
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_distances
from sentence_transformers import SentenceTransformer
# --- Helper Functions ---

def load_comprehensive_log(log_path):
    """Load a comprehensive log file."""
    with open(log_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_self_bleu(prompts_list, n=4):
    """Tính điểm Self-BLEU. Lower score = More diverse."""
    if len(prompts_list) < 2:
        return 0.0

    try:
        tokenized_prompts = [word_tokenize(p.lower()) for p in prompts_list]
    except (LookupError, NameError):
        tokenized_prompts = [p.lower().split() for p in prompts_list]

    smoother = SmoothingFunction().method1
    scores = []
    weights = tuple([1.0/n] * n)

    for i, candidate in enumerate(tokenized_prompts):
        references = tokenized_prompts[:i] + tokenized_prompts[i+1:]
        if not references: continue
        
        score = sentence_bleu(references, candidate, weights=weights, smoothing_function=smoother)
        scores.append(score)

    return np.mean(scores) if scores else 0.0

def calculate_distinct_n(prompts_list, n=2):
    """
    (Bổ sung) Tính Distinct-N: Tỷ lệ N-gram độc nhất.
    Metric này bổ trợ rất tốt cho Self-BLEU.
    Cao = Tốt (Từ vựng phong phú).
    """
    if not prompts_list: return 0.0
    ngrams_all = []
    for p in prompts_list:
        words = p.lower().split()
        if len(words) < n: continue
        ngrams_all.extend([tuple(words[i:i+n]) for i in range(len(words)-n+1)])
    
    if not ngrams_all: return 0.0
    return len(set(ngrams_all)) / len(ngrams_all)

def calculate_embedding_diversity(embeddings=None):
    """
    Tính Embedding Diversity (Cosine Distance trung bình).
    
    Args:
        prompts (list): Danh sách các câu text (nếu chưa có embeddings).
        embeddings (numpy array): Mảng vector (nếu đã encode rồi thì truyền vào đây cho nhanh).
        
    Returns:
        float: Giá trị từ 0.0 đến 2.0. Càng CAO càng TỐT (đa dạng).
    """
    
    if len(embeddings) < 2:
        return 0.0

    # 2. Tính ma trận khoảng cách Cosine (Cosine Distance Matrix)
    # cosine_distance = 1 - cosine_similarity
    # Kết quả trả về ma trận NxN
    dist_matrix = cosine_distances(embeddings)
    
    # 3. Tính trung bình (chỉ lấy phần tam giác trên để tránh trùng lặp và đường chéo)
    # Lấy các chỉ số của tam giác trên (upper triangle), k=1 để bỏ đường chéo chính (dist=0)
    triu_indices = np.triu_indices(len(embeddings), k=1)
    
    # Tính trung bình các khoảng cách
    diversity_score = np.mean(dist_matrix[triu_indices])
    
    return float(diversity_score)

from sklearn.cluster import DBSCAN
def calculate_distinct_clusters(embeddings=None, eps=0.4, min_samples=2):
    """
    Đếm số lượng cụm phân biệt sử dụng thuật toán DBSCAN.
    DBSCAN tốt hơn K-Means ở chỗ không cần quy định trước số cụm (K).
    
    Args:
        eps (float): Khoảng cách tối đa để coi 2 điểm là cùng 1 cụm. 
                     Thấp (vd 0.1) -> Cần rất giống nhau mới chung cụm -> Ra nhiều cụm.
                     Cao (vd 0.5) -> Hơi giống cũng chung cụm -> Ra ít cụm.
                     0.3 là ngưỡng trung bình thường dùng cho SBERT.
        min_samples (int): Số lượng điểm tối thiểu để tạo thành 1 cụm.
        
    Returns:
        int: Số lượng cụm tìm thấy.
    """
        
    if len(embeddings) < min_samples:
        return 1 # Coi như 1 nhóm nhỏ

    # 2. Chạy thuật toán DBSCAN
    # metric='cosine' để gom cụm dựa trên góc thay vì khoảng cách Ơ-clit
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(embeddings)
    
    # 3. Đếm số nhãn (labels)
    # labels_ chứa danh sách cụm (0, 1, 2...). 
    # Label -1 nghĩa là Nhiễu (Noise) - các điểm không thuộc cụm nào (quá khác biệt).
    labels = clustering.labels_
    
    # Đếm số lượng cụm (loại bỏ -1 nếu có)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    # (Tùy chọn) Nếu bạn muốn coi các điểm Nhiễu (Outliers) cũng là các chiến thuật độc nhất
    #n_clusters += list(labels).count(-1) 
    
    return n_clusters
def calculate_vendi_score(X):
    """
    Tính Vendi Score (Effective Number of Elements) cho tập embeddings.
    
    Công thức: Vendi(X) = exp(ShannonEntropy(eigenvalues(K)/N))
    Trong đó K là ma trận tương đồng Cosine: K = X_norm . X_norm^T
    
    Args:
        X (np.ndarray hoặc list): Ma trận đặc trưng kích thước (N, D).
                                  N: Số lượng mẫu.
                                  D: Số chiều embedding.
    
    Returns:
        float: Giá trị Vendi Score (từ 1.0 đến N).
               - 1.0: Tất cả giống hệt nhau.
               - N: Tất cả khác nhau hoàn toàn (vuông góc).
    """
    # 1. Chuyển đổi sang numpy array và kiểm tra đầu vào
    X = np.array(X)
    
    # Trường hợp rỗng hoặc 1 phần tử
    if X.shape[0] <= 1:
        return float(X.shape[0])
        
    # 2. Chuẩn hóa vector (L2 Norm) để tính Cosine Similarity
    # Thêm 1e-12 để tránh chia cho 0 nếu có vector toàn số 0
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    X_normalized = X / (norm + 1e-12)
    
    # 3. Tính Ma trận Kernel (Gram Matrix)
    # K[i, j] chính là Cosine Similarity giữa vector i và j
    K = np.dot(X_normalized, X_normalized.T)
    
    # 4. Phân rã Trị riêng (Eigendecomposition)
    # Dùng eigh hoặc eigvalsh vì K là ma trận đối xứng (nhanh và ổn định hơn eig)
    eigenvalues = np.linalg.eigvalsh(K)
    
    # 5. Xử lý trị riêng
    # Do sai số máy tính, trị riêng có thể ra số âm cực nhỏ (vd: -1e-15), cần gán về 0
    eigenvalues = np.maximum(eigenvalues, 0)
    
    # Chuẩn hóa trị riêng thành phân phối xác suất
    # Tổng các trị riêng của ma trận correlation luôn bằng N (Trace của ma trận)
    weight = eigenvalues / np.sum(eigenvalues)
    
    # 6. Tính Shannon Entropy
    # Chỉ tính log với các giá trị > 0 để tránh log(0)
    entropy = -np.sum(weight[weight > 0] * np.log(weight[weight > 0]))
    
    # 7. Tính Vendi Score
    return np.exp(entropy)

def get_top_k_data(evaluations, k=100, strategy='highest'):
    """Lấy Top-K theo fitness."""
    if not evaluations: return []
    k = min(int(k), len(evaluations))
    
    if strategy == 'highest':
        # Sort giảm dần và lấy K cái đầu
        selected = sorted(evaluations, key=lambda x: x.get('fitness', 0), reverse=True)[:k]
    elif strategy == 'random':
        # (Ít dùng cho Self-BLEU Top-K nhưng vẫn giữ)
        selected = random.sample(evaluations, k)
    else:
        return []
        
    return selected


import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA

def analyze_convex_hull_simple(embeddings, method_name="Method"):
    """
    1. Tính thể tích ước lượng (dựa trên PCA 3D).
    2. Vẽ hình bao lồi 2D (dựa trên PCA 2D).
    """
    # --- 1. TÍNH TOÁN (Metric) ---
    # Giảm xuống 3 chiều để tính thể tích "đại diện"
    if len(embeddings) >= 4:
        pca_3d = PCA(n_components=3)
        emb_3d = pca_3d.fit_transform(embeddings)
        volume = ConvexHull(emb_3d).volume
    else:
        volume = 0.0
    
    print(f"[{method_name}] Pseudo-Volume (3D PCA): {volume:.4f}")

    # --- 2. VẼ HÌNH 2D (Visualization) ---
    # Giảm xuống 2 chiều để vẽ lên giấy
    if len(embeddings) >= 3:
        pca_2d = PCA(n_components=2)
        points_2d = pca_2d.fit_transform(embeddings)
        hull = ConvexHull(points_2d)

        # Vẽ các điểm
        plt.figure(figsize=(8, 6))
        plt.plot(points_2d[:,0], points_2d[:,1], 'o', color='blue', alpha=0.3, label='Prompts')

        # Vẽ đường bao lồi (Sợi dây chun)
        for simplex in hull.simplices:
            plt.plot(points_2d[simplex, 0], points_2d[simplex, 1], 'r-', lw=2) # Đường màu đỏ
        
        # Tô màu vùng bên trong (Option)
        plt.fill(points_2d[hull.vertices,0], points_2d[hull.vertices,1], 'r', alpha=0.1)

        plt.title(f'{method_name} - 2D Convex Hull Coverage')
        plt.xlabel('PCA Dim 1')
        plt.ylabel('PCA Dim 2')
        plt.grid(True, linestyle=':', alpha=0.5)
        
        # Lưu file
        filename = f"hull_2d_{method_name}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✅ Đã lưu ảnh 2D đơn giản: {filename}")
        plt.close()
    else:
        print("Không đủ điểm dữ liệu để vẽ Hull.")

    return volume

from scipy.spatial import Voronoi, voronoi_plot_2d

def visualize_structure_and_fitness(centroids_emb, elites_emb, elites_scores, method_name="GA_Structure"):
    """
    Vẽ cấu trúc không gian với ranh giới VORONOI:
    1. Voronoi Lines: Ranh giới phân chia lãnh thổ giữa các Centroids.
    2. Elites: Các điểm nhỏ, màu theo Fitness.
    3. Centroids: Các điểm to, đánh dấu X.
    """
    # Gộp dữ liệu để chạy PCA chung
    num_centroids = len(centroids_emb)
    
    # Stack: [Centroids trước ... Elites sau]
    all_data = np.vstack([centroids_emb, elites_emb])
    
    # --- 1. GIẢM CHIỀU (PCA) ---
    if len(all_data) >= 3:
        # PCA 2D
        pca_2d = PCA(n_components=2)
        all_2d = pca_2d.fit_transform(all_data)
        
        # Tách lại
        centroids_2d = all_2d[:num_centroids]
        elites_2d = all_2d[num_centroids:]

        # --- 2. VẼ HÌNH (VISUALIZATION) ---
        # Sử dụng plt.subplots để dễ quản lý các layer
        fig, ax = plt.subplots(figsize=(12, 10))

        # --- LAYER 0: VORONOI BOUNDARIES (NỀN) ---
        # Chỉ vẽ nếu có đủ điểm để tạo Voronoi (>= 4 điểm để an toàn)
        if num_centroids >= 4:
            try:
                vor = Voronoi(centroids_2d)
                # voronoi_plot_2d vẽ các đường ranh giới
                # show_vertices=False: Ẩn các điểm giao cắt thừa
                # show_points=False: Ẩn điểm gốc (vì ta sẽ tự vẽ Centroids cho đẹp)
                voronoi_plot_2d(vor, ax=ax, show_vertices=False, show_points=False, 
                                line_colors='gray', line_width=1, line_alpha=0.3)
            except Exception as e:
                print(f"Warning: Could not draw Voronoi ({e})")

        # --- LAYER 1: ELITES (DÂN CƯ) ---
        sc = ax.scatter(elites_2d[:, 0], elites_2d[:, 1],
                        c=elites_scores,       # Tô màu theo Fitness
                        cmap='viridis',        # Tím -> Vàng
                        s=30,                  # Kích thước nhỏ
                        alpha=0.7,             # Hơi trong suốt
                        edgecolors='none',
                        label='Elites (Populations)',
                        zorder=2)              # Vẽ đè lên Voronoi

        # --- LAYER 2: CENTROIDS (LÃNH CHÚA) ---
        ax.scatter(centroids_2d[:, 0], centroids_2d[:, 1],
                   c='black',                    # Màu đỏ                 # Hình chữ X
                   s=70,                      # Kích thước lớn
                   edgecolors='none',         # Viền trắng
                   linewidth=1.5,
                   label='Centroids (Centers)',
                   zorder=3)                   # Vẽ trên cùng

        # --- LAYER 3: CONVEX HULL (BIÊN GIỚI TỔNG) ---

        # Trang trí
        # Tắt giới hạn trục mặc định của Voronoi (đôi khi nó zoom quá xa)
        # Ta set giới hạn dựa trên dữ liệu thực tế + chút lề
        margin = 0.1
        x_min, x_max = all_2d[:, 0].min(), all_2d[:, 0].max()
        y_min, y_max = all_2d[:, 1].min(), all_2d[:, 1].max()
        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(y_min - margin, y_max + margin)

        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('Elite Fitness Score', rotation=270, labelpad=15)
        
        ax.set_title(f'{method_name}: Voronoi Tessellation & Fitness Map', fontsize=15)
        ax.set_xlabel('PCA Dim 1')
        ax.set_ylabel('PCA Dim 2')
        ax.legend(loc='upper right')
        ax.grid(True, linestyle=':', alpha=0.3)
        
        filename = f"voronoi_{method_name}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✅ Đã lưu biểu đồ Voronoi: {filename}")
        plt.close()
        
    else:
        print("Không đủ dữ liệu để vẽ.")
def plot_fitness_distribution(scores_ga, scores_pt, output_filename="fitness_dist.png"):

    """
    Vẽ biểu đồ phân bố điểm số (Fitness) của 2 phương pháp.
    
    Args:
        scores_ga (list): Danh sách điểm số của Growing Archive (VD: [0.9, 0.8, 1.0...])
        scores_pt (list): Danh sách điểm số của Persona Teaming
    """
    print("\n[Viz] Đang vẽ biểu đồ phân bố Fitness...")

    # BƯỚC 1: Chuẩn bị dữ liệu cho "ngăn nắp"
    # Chúng ta tạo một cái bảng (DataFrame) để máy dễ vẽ
    # Cột 1: Phương pháp (Tên lớp học)
    # Cột 2: Điểm số (Fitness)
    data = []
    
    for s in scores_ga:
        data.append({"Method": "GrowingArchive", "Fitness": s})
    
    for s in scores_pt:
        data.append({"Method": "PersonaTeaming", "Fitness": s})
        
    df = pd.DataFrame(data)

    # BƯỚC 2: Vẽ biểu đồ
    plt.figure(figsize=(10, 6)) # Tạo khung tranh kích thước 10x6
    
    # Dùng hàm histplot (Histogram Plot)
    # x="Fitness": Trục ngang là điểm số
    # hue="Method": Tô màu khác nhau cho từng phương pháp
    # kde=True: Vẽ thêm đường cong mềm mại để dễ nhìn xu hướng
    # bins=20: Chia điểm số thành 20 cái giỏ nhỏ
    sns.histplot(
        data=df, 
        x="Fitness", 
        hue="Method", 
        kde=True, 
        bins=20,
        palette={'PersonaTeaming': '#FF4B4B', 'GrowingArchive': '#1F9D89'}, # Màu Đỏ vs Xanh
        alpha=0.6 # Độ trong suốt (để nếu chồng lên nhau vẫn nhìn thấy)
    )
    
    # Trang trí cho đẹp
    plt.title("So sánh Phân Bố Điểm Số (Fitness Distribution)", fontsize=15)
    plt.xlabel("Điểm Fitness (Càng cao càng tốt)", fontsize=12)
    plt.ylabel("Số lượng Prompt (Tần suất)", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.3) # Kẻ dòng mờ mờ cho dễ dóng
    
    # BƯỚC 3: Lưu thành file ảnh (để xem trên Kaggle)
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"✅ Đã lưu biểu đồ vào file: {output_filename}")
    plt.close() # Đóng lại cho đỡ tốn RAM


import ast
from collections import defaultdict
def build_archive_risk_style(log_data):
    grouped_data = defaultdict(lambda:{
        "prompts": [],
        "personas": [],
        "responses": [],
        "scores": [],
        "similarity_scores": [],
        "reasons": [],
    })
    all_prompts = log_data.get('all_prompts', {})
    all_responses = log_data.get('all_responses', {})
    all_scores = log_data.get('all_scores', {})
    all_similarities = log_data.get('all_similarities', {})
    rejection_reasons = log_data.get('rejection_reasons', {})
    for key, prompts_list in all_prompts.items():
        log_key = ast.literal_eval(key)
        risk = log_key[0]
        style = log_key[1]
        group_key = (risk, style)
        
        responses_list = all_responses.get(key, [])
        scores_list = all_scores.get(key, [])
        similarities_list = all_similarities.get(key, [])
        reasons_list = rejection_reasons.get(key, [])

        
        for i, prompt in enumerate(prompts_list):
            grouped_data[group_key]["prompts"].append(prompt)

            resp = responses_list[i] if i < len(responses_list) else ""
            score = scores_list[i] if i < len(scores_list) else 0.0
            similar = similarities_list[i] if i < len(similarities_list) else 0.0
            reason = reasons_list[i] if i < len(reasons_list) else ""
            
            grouped_data[group_key]["responses"].append(resp)
            grouped_data[group_key]["scores"].append(score)
            grouped_data[group_key]["similarity_scores"].append(similar)
            grouped_data[group_key]["reasons"].append(reason)


    return grouped_data

def plot_fixed_grid_heatmap(data_frame, metric='max_fitness', k=1):
    df_grouped = []
    qd_score = 0
    for key, elite in data_frame.items():
        risk_category = key[0]
        attack_style = key[1]
        scores = elite.get('scores', [])
        sorted_scores = sorted(scores, reverse=True)
        current_sum = sorted_scores[:k]
        df_grouped.append({
            'Risk Category': risk_category,
            'Attack Style': attack_style,
            'Value': sum(current_sum)
        })
        qd_score += sum(current_sum)
    
    df_grouped = pd.DataFrame(df_grouped)
    if df_grouped.empty:
        print(f"Warning: No data available for Heatmap ({metric}). Skipping.")
        return
    
    heatmap_data = df_grouped.pivot(index='Risk Category', columns='Attack Style', values='Value')
    heatmap_data = heatmap_data.fillna(0)

    # Tính vmin/vmax
    data_values = heatmap_data.values.flatten()
    non_zero_values = data_values[data_values > 0]
    vmin = 0
    vmax = non_zero_values.max() if len(non_zero_values) > 0 else 1

    # Tăng figsize và điều chỉnh spacing
    fig, ax = plt.subplots(figsize=(16, 10))
    
    sns.heatmap(
        heatmap_data,
        cmap='viridis',
        annot=True,
        fmt='.1f',
        annot_kws={'size': 9, 'weight': 'bold'},
        linewidths=1.5,
        linecolor='white',
        square=True,
        cbar_kws={
            'label': f'{metric.replace("_", " ").title()}',
            'shrink': 0.75,
            'pad': 0.02  # Khoảng cách colorbar
        },
        vmin=vmin,
        vmax=vmax,
        ax=ax
    )

    # Di chuyển xlabel lên trên
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_xlabel('Attack Style', fontsize=12, weight='bold', labelpad=10)
    ax.set_ylabel('Risk Category', fontsize=12, weight='bold', labelpad=10)
    
    # Xoay labels
    plt.xticks(rotation=45, ha='left', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    
    # Title với khoảng cách đủ lớn
    plt.title(
        f"Risk Category vs Attack Style ({metric}) | QD Score: {qd_score:.2f}",
        fontsize=14,
        weight='bold',
        pad=50  # QUAN TRỌNG: Tăng khoảng cách title
    )
    
    # Điều chỉnh layout để không bị cắt
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Để chỗ cho title
    
    filename = f"fixed_grid_heatmap_{metric}_top{k}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Đã lưu Heatmap: {filename}")
    plt.close()

def calculate_metrics(model, evaluation, method = 'GA', k = 5):
    # ASR
    prompts = []
    for elite in evaluation.values():
        p_list = elite.get('prompts', [])
        s_list = elite.get('scores', [])
        
        # Ghép cặp (prompt, score) để sort lấy top k trong từng cell
        # Xử lý an toàn nếu p_list chứa dict thay vì string
        clean_prompts = []
        for p in p_list:
            clean_prompts.append(p)
        
        zipped = sorted(zip(clean_prompts, s_list), key=lambda x: x[1], reverse=True)
        
        # Lấy top k prompt tốt nhất của cell này
        top_k_cell = [x[0] for x in zipped[:k]]
        prompts.extend(top_k_cell)
    
    # ASR_among_pass = num_unsafe / total_num_queries
    total_num_queries = 0
    num_unsafe = 0
    for key, elite in evaluation.items():
        num_queries = 0
        unsafe = 0

        reasons = elite.get('reasons', [])
        num_queries = sum(1 for r in reasons if r != 'similarity_too_high')
        if method == 'GA':
            unsafe = num_queries - sum(1 for r in reasons if r == 'rejected_fitness_low')   
        else:
            unsafe = sum(1 for r in reasons if r == 'accepted')
            
        total_num_queries += num_queries
        num_unsafe += unsafe
    print(f"Num_unsafe: {num_unsafe}")
    print(f"Total_num_queries: {total_num_queries}")
    print(f"ASR: {num_unsafe / total_num_queries:.4f}")
    # Convergence
    plot_fixed_grid_heatmap(evaluation, metric = method , k = k)

    # Diversity
    self_bleu = calculate_self_bleu(prompts, 1)
    distinct_n = calculate_distinct_n(prompts)
    embedding_diversity = calculate_embedding_diversity(model.encode(prompts))
    vendi_score = calculate_vendi_score(model.encode(prompts))
    print(f"Self-BLEU Score {method}: {self_bleu:.4f}")
    print(f"Distinct N-Grams {method}: {distinct_n:.4f}")
    print(f"Embedding Diversity {method}: {embedding_diversity:.4f}")
    print(f"Vendi Score {method}: {vendi_score:.4f}")
    


def find_log_files(directory):
    directory = Path(directory)
    
    # --- 1. Find GA Log ---
    comprehensive_ga_log = None
    ga_global = directory / "comprehensive_ga_log_global.json"
    
    if ga_global.exists():
        comprehensive_ga_log = str(ga_global)
    else:
        # Check patterns if global file doesn't exist
        ga_pattern = directory / "comprehensive_ga_log_*.json"
        ga_candidates = sorted(list(glob.glob(str(ga_pattern))))
        if ga_candidates:
            comprehensive_ga_log = ga_candidates[-1]

    # --- 2. Find Standard Log ---
    comprehensive_log = None
    std_global = directory / "comprehensive_log_global.json"
    
    if std_global.exists():
        comprehensive_log = str(std_global)
    else:
        # Check patterns
        std_pattern = directory / "comprehensive_log_*.json"
        std_candidates = sorted(list(glob.glob(str(std_pattern))))
        if std_candidates:
            comprehensive_log = std_candidates[-1]

    if comprehensive_log is None:
        std_global = directory / "rainbowplus_log_global.json"
        if std_global.exists():
            comprehensive_log = str(std_global)
        else:
            std_pattern = directory / "rainbowplus_log_*.json"
            std_candidates = sorted(list(glob.glob(str(std_pattern))))
            if std_candidates:
                comprehensive_log = std_candidates[-1]

    attack_memory_log = None
    attack_memory_pattern = directory / "attack_memory_*.yml"
    attack_memory_candidates = sorted(list(glob.glob(str(attack_memory_pattern))))
    if attack_memory_candidates:
        attack_memory_log = attack_memory_candidates[-1]

    return comprehensive_ga_log, comprehensive_log, attack_memory_log

def main():
    parser = argparse.ArgumentParser(description="Analyze comprehensive RainbowPlus logs")
    # SỬA: Thêm type=int
    parser.add_argument("--top-k", type=int, default=5, help="Number of top-k to compare 2 methods")
    parser.add_argument("directory", help="Directory containing log files")
    parser.add_argument("--output", help="Output file for analysis results")
    args = parser.parse_args()

    try:
        comprehensive_ga_log, comprehensive_log, memory_log = find_log_files(args.directory)
        print(f"Standard Log: {comprehensive_log}")
        print(f"GA Log:       {comprehensive_ga_log}")
        print(f"Attack Memory: {memory_log}")
        print("-" * 40)
        
    except Exception as e:
        print(f"Error finding logs: {e}")
        return
    
    centroids = []
    elites_emb = []
    elites_scores = []

    # 3. Compute Embeddings
    print("\n[AI] Computing Embeddings (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    #print("\n[AI] Computing Embeddings (gte-large-en-v1.5)...")
    #model = SentenceTransformer('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True)
    ga_prompts = []
    ga_scores = []
    if comprehensive_ga_log:
        try:
            # SỬA: Load trực tiếp file, không load chồng chéo
            ga_data = load_comprehensive_log(comprehensive_ga_log)
            ga = build_archive_risk_style(ga_data)
            calculate_metrics(model, ga, 'GA', args.top_k)
            
            ga_state = ga_data.get('state', {})
            print(f"#######################")
            elites = ga_state.get('elites', {})
            
            ga_evaluations = []
            for key, list_of_elite in elites.items():
                if list_of_elite:
                    max_evaluation = max(list_of_elite, key=lambda x: x.get('fitness', 0))
                    ga_evaluations.append(max_evaluation)
                    # Take the highest fitness in each cells
                    for elite in list_of_elite:
                        ga_prompts.append(elite['prompt'])
                        ga_scores.append(elite['fitness'])
                        elites_scores.append(elite['fitness'])
                        ga_evaluations.append(elite)
            
            # print(f"[GrowingArchive] Loaded {len(ga_prompts)} prompts.")
            # stats = get_top_k_data(ga_evaluations, args.top_k, 'highest')
            # scores = [e['fitness'] for e in stats]
            # print(f"GA Top-{args.top_k} Fitness: {float(np.mean(scores)):.4f} (±{float(np.std(scores)):.4f}")
            # flat_prompts = [e['prompt'] for e in stats]
            # diversity = calculate_self_bleu(flat_prompts, 1)
            # print(f"GA Self-BLEU: {diversity:.4f}")
            # distinct_n = calculate_distinct_n(flat_prompts)
            # print(f"GA Distinct-2: {distinct_n:.4f}")
            # ga_emb = model.encode(flat_prompts)
            # embedding_diversity = calculate_embedding_diversity(ga_emb)
            # print(f"GA Embedding diversity: {embedding_diversity:.4f}")
            # cluster = calculate_distinct_clusters(ga_emb)
            #print(f"GA Distinct Clusters: {cluster:.4f}")
            #analyze_convex_hull_simple(ga_emb, "GrowingArchive")
            #elites_emb = model.encode(flat_prompts)
            
        except Exception as e:
            print(f"Error processing GA log: {e}")

    # --- PROCESS STANDARD LOGS ---
    persona_prompts = []
    persona_scores = []
    if comprehensive_log:
        try:
            std_data = load_comprehensive_log(comprehensive_log)
            persona = build_archive_risk_style(std_data)
            calculate_metrics(model, persona, 'PersonaTeaming', args.top_k)
            
            all_prompts = std_data.get('all_prompts', {})
            all_scores = std_data.get('all_scores', {})
            
            persona_evaluations = []
            for category, prompts in all_prompts.items():
                persona_prompts.extend(prompts)
                scores = all_scores.get(category, [])
                
                for prompt, score in zip(prompts, scores):
                    persona_evaluations.append({
                        "prompt": prompt,
                        "fitness": score,
                    })
                    persona_scores.append(score)
            
            # print(f"\n[PersonaTeaming] Loaded {len(persona_prompts)} prompts.")
            
            # Highest Stats
            # stats_high = get_top_k_data(persona_evaluations, args.top_k, 'highest')
            # scores = [e['fitness'] for e in stats_high]
            # print(f"Persona Highest Top-{args.top_k} Fitness: {float(np.mean(scores)):.4f} (±{float(np.std(scores)):.4f}")
            # flat_prompts = [e['prompt'] for e in stats_high]
            # diversity = calculate_self_bleu(flat_prompts, 1)
            # print(f"Persona Self-BLEU: {diversity:.4f}")
            # distinct_n = calculate_distinct_n(flat_prompts)
            # print(f"Persona Distinct-2: {distinct_n:.4f}")
            # pt_emb = model.encode(flat_prompts)
            # embedding_diversity = calculate_embedding_diversity(pt_emb)
            #print(f"Persona Embedding diversity: {embedding_diversity}")
            #cluster = calculate_distinct_clusters(pt_emb)
            #print(f"Persona Distinct Clusters: {cluster:.4f}")
            #analyze_convex_hull_simple(pt_emb, "PersonaTeaming")
            '''
            # Random Stats
            stats_rand = get_top_k_data(persona_evaluations, args.top_k, 'random')
            print(f"Persona Random Top-{args.top_k} Fitness:  {stats_rand['mean']:.4f} (±{stats_rand['std']:.4f})")
            '''
        except Exception as e:
            print(f"Error processing Standard log: {e}")
    
    print("-" * 40)

    plot_fitness_distribution(ga_scores, persona_scores)
    #analyze_convex_hull_simple(centroids, "Centroids")
    #visualize_structure_and_fitness(centroids, elites_emb, elites_scores)


if __name__ == "__main__":
    main()
    