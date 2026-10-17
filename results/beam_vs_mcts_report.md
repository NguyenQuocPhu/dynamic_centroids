# Beam vs MCTS — kết quả đo

_Sinh tự động 2026-09-04 từ `results/runs.csv` (170 run). Không gọi API._

**Cấu hình so sánh** — cùng ngân sách target query, độ đo = điểm GAP judge 1-10,
AQD = trung bình điểm-tốt-nhất mỗi cell MAP-Elites (chuẩn hoá theo cell để beam,
vốn cắt mỗi cell còn `tree_width` survivor, so được với MCTS vốn giữ mọi node cây).

```
beam : --tree_depth 2                                        (mặc định)
mcts : --total 2 --tree_width 3 --uct_c 0.1 --terminal_score 0  (đã tune)
```

## 1. Bảy dataset @ budget 1500

| dataset | beam AQD (cell≥7) | mcts AQD (cell≥7) | gap | Cohen d | p |
|---|---|---|---|---|---|
| harmfulQA | 5.74±1.23 (54) n=3 | **8.40±0.68 (86)** n=3 | +2.67 | 2.69 | 0.001*** |
| harmbench | 3.16±1.82 (21) n=10 | **6.30±1.72 (63)** n=10 | +3.14 | 1.77 | 0.000*** |
| DangerousQA | 1.82±0.72 (2) n=3 | **4.00±1.19 (17)** n=3 | +2.18 | 2.21 | 0.007** |
| AdversarialQA | 3.01±1.10 (20) n=6 | **5.15±2.34 (36)** n=6 | +2.14 | 1.17 | 0.043* |
| do-not-answer | 2.87±0.62 (21) n=2 | **4.89±2.31 (25)** n=3 | +2.03 | 1.20 | 0.230 |
| CategoricalHarmfulQA | 4.37±1.84 (42) n=4 | **5.99±1.99 (62)** n=4 | +1.62 | 0.85 | 0.231 |
| BeaverTails | 5.68±0.13 (50) n=3 | **6.21±2.66 (52)** n=3 | +0.53 | 0.28 | 0.731 |

**MCTS thắng 7/7 dataset, gap trung bình +2.04, sign test p=0.0156.**
Bốn dataset đạt p<0.05 riêng lẻ; ba cái còn lại cùng chiều nhưng n=2-4 nên thiếu lực thống kê.

## 2. Đường cong hiệu quả truy vấn (harmbench)

| budget | beam AQD (cell≥7) | mcts AQD (cell≥7) | gap |
|---|---|---|---|
| 1500 | 3.16±1.82 (21) | 6.30±1.72 (63) | +3.14 |
| 2000 | 3.55±2.22 (24) | 4.98±3.48 (34) | +1.43 |
| 3000 | 6.63±2.23 (62) | 8.21±1.27 (82) | +1.59 |
| 5000 | 9.75±0.31 (98) | 9.99±0.01 (100) | +0.25 |

Cả hai bão hoà (~10/10) ở budget 5000; khác biệt nằm ở **tốc độ leo**, tức hiệu quả truy vấn.

## 3. Tuned-vs-tuned: tune cả beam có đuổi kịp không?

| dataset | beam d1 | beam d2 | beam d3 | beam d4 | mcts |
|---|---|---|---|---|---|
| AdversarialQA | — | 3.01 | — | — | **5.15** |
| harmbench | — | 3.16 | — | — | **6.30** |

Tune beam có giúp thật (d3/d4 hơn mặc định d2), nhưng **MCTS vẫn dẫn** kể cả khi
chọn depth tốt nhất riêng cho từng dataset — tức đã ưu ái beam tối đa.

## 4. Cảnh báo trung thực

- Một số dataset chỉ n=2-4 → p chưa đạt dù cùng chiều. Cần thêm lặp khi API ổn định.
- Độ đo ở đây là **GAP judge in-loop** (dùng để tune). Bảng 3-way chính thức phải chấm lại
  bằng `fair_blind_eval.py --api_judge` (StrongREJECT) — chưa chạy vì server đang 502/504.
- Baseline **chưa có số liệu** ở budget này; các run bị 504 giết giữa chừng.
- Đo sơ bộ bằng StrongREJECT trên 1 cặp harmbench: beam ASR 57% vs mcts 98% (chi2 p=1.3e-11).