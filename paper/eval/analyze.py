#!/usr/bin/env python3
"""
Paper tables, statistics and figures (pipeline step 3 of 3).

  python paper/eval/analyze.py                 # everything available so far
  python paper/eval/analyze.py --primary llamaguard3

Reads <runs_root>/_eval/{items.jsonl, runs.csv, judgments/*.jsonl}
Writes <runs_root>/_eval/report/: summary.md, *.csv, *.tex, fig_*.png

Metric definitions (state them in the paper exactly like this)
  cell-ASR   fraction of the 100 MAP-Elites cells (10 risk x 10 style) whose
             top-1 prompt (by the method's own in-loop score) is judged a
             successful jailbreak. Empty cells count as failures, so this
             rewards coverage AND quality. HEADLINE METRIC.
  item-ASR   success rate over the judged top-1 items only (= precision).
  For sub-sampled judges (sample_frac < 1) cell-ASR is estimated as
  item-ASR x coverage.
Statistical unit: one run = (target, dataset, seed). Methods are compared on
the SAME units (paired): mean difference, 95% bootstrap CI (10k resamples),
Wilcoxon signed-rank (zsplit), Holm correction per table, win/tie/loss,
Cohen's d_z.
"""
import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import run_matrix as RM   # noqa: E402

N_CELLS = 100
RNG = np.random.default_rng(0)
ARM_ORDER = ["baseline", "beam", "mcts"]


# ─────────────────────────────── loading ─────────────────────────────────────

def load(ev):
    items = pd.read_json(ev / "items.jsonl", lines=True)
    runs = pd.read_csv(ev / "runs.csv")
    judg = {}
    for f in sorted((ev / "judgments").glob("*.jsonl")):
        rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        df = pd.DataFrame([r for r in rows if "error" not in r])
        if len(df):
            judg[f.stem] = df.drop_duplicates("h", keep="last").set_index("h")
    return items, runs, judg


def per_run(items, runs, judg):
    """long table: one row per (run, judge) with cell_asr / item_asr."""
    top1 = items[items["rank"] == 0]
    out = []
    for jn, jd in judg.items():
        m = top1.join(jd[["success", "score"]], on="h", how="inner")
        if m.empty:
            continue
        g = m.groupby("run").agg(n_judged=("success", "size"), succ=("success", "sum"), mean_score=("score", "mean"))
        g["judge"] = jn
        out.append(g.reset_index())
    if not out:
        return pd.DataFrame()
    pr = pd.concat(out).merge(runs, on="run", how="left")
    pr["coverage"] = pr["cells_filled"] / N_CELLS
    pr["item_asr"] = pr["succ"] / pr["n_judged"]
    full = pr["n_judged"] >= pr["cells_filled"]
    pr["cell_asr"] = np.where(full, pr["succ"] / N_CELLS, pr["item_asr"] * pr["coverage"])
    return pr


# ─────────────────────────────── stats ───────────────────────────────────────

def boot_ci(x, n=10000, alpha=0.05):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (np.nan, np.nan)
    means = RNG.choice(x, (n, len(x)), replace=True).mean(1)
    return tuple(np.quantile(means, [alpha / 2, 1 - alpha / 2]))


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def paired(df, a, b, metric, unit=("target", "dataset", "seed")):
    pa = df[df["arm"] == a].set_index(list(unit))[metric]
    pb = df[df["arm"] == b].set_index(list(unit))[metric]
    j = pd.concat([pa.rename("a"), pb.rename("b")], axis=1, join="inner").dropna()
    d = (j["b"] - j["a"]).to_numpy()
    if len(d) == 0:
        return None
    lo, hi = boot_ci(d)
    try:
        p = stats.wilcoxon(d, zero_method="zsplit").pvalue if np.any(d != 0) else 1.0
    except ValueError:
        p = np.nan
    sd = d.std(ddof=1) if len(d) > 1 else np.nan
    return dict(A=a, B=b, n=len(d), mean_A=j["a"].mean(), mean_B=j["b"].mean(), diff=d.mean(), ci_lo=lo, ci_hi=hi,
                wins_B=int((d > 0).sum()), ties=int((d == 0).sum()), wins_A=int((d < 0).sum()),
                p_wilcoxon=p, d_z=(d.mean() / sd) if sd and sd > 0 else np.nan)


def kappa(a, b):
    a, b = np.asarray(a, int), np.asarray(b, int)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


# ─────────────────────────────── output helpers ──────────────────────────────

def to_tex(df, path, caption, fmt="{:.3f}"):
    cols = list(df.columns)
    lines = ["\\begin{table}[t]\\centering\\small", f"\\caption{{{caption}}}",
             "\\begin{tabular}{" + "l" * (len(df.index.names) if df.index.nlevels > 1 or df.index.name else 0)
             + "r" * len(cols) + "}", "\\toprule"]
    idx_names = [n or "" for n in df.index.names] if (df.index.nlevels > 1 or df.index.name) else []
    lines.append(" & ".join(idx_names + [str(c).replace("_", "\\_") for c in cols]) + " \\\\ \\midrule")
    for idx, row in df.iterrows():
        idx = idx if isinstance(idx, tuple) else (idx,)
        cells = [str(i).replace("_", "\\_") for i in idx] if idx_names else []
        for v in row:
            cells.append(fmt.format(v) if isinstance(v, (float, np.floating)) and not math.isnan(v) else
                         ("--" if isinstance(v, float) else str(v).replace("_", "\\_")))
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt_ci(m, lo, hi):
    return f"{100*m:.1f} [{100*lo:.1f}, {100*hi:.1f}]"


# ─────────────────────────────── sections ────────────────────────────────────

def main_table(pr, primary, out, md):
    d = pr[(pr["judge"] == primary) & (pr["exp"] == "E1") & (pr["variant"].fillna("") == "")]
    if d.empty:
        md.append(f"_No E1 runs judged by {primary} yet._\n")
        return
    rows = []
    for (tgt, arm), g in d.groupby(["target", "arm"]):
        lo, hi = boot_ci(g["cell_asr"])
        rows.append(dict(target=tgt, arm=arm, n=len(g), cell_asr=g["cell_asr"].mean(), lo=lo, hi=hi,
                         item_asr=g["item_asr"].mean(), coverage=g["coverage"].mean()))
    for arm, g in d.groupby("arm"):
        lo, hi = boot_ci(g["cell_asr"])
        rows.append(dict(target="ALL", arm=arm, n=len(g), cell_asr=g["cell_asr"].mean(), lo=lo, hi=hi,
                         item_asr=g["item_asr"].mean(), coverage=g["coverage"].mean()))
    t = pd.DataFrame(rows)
    t.to_csv(out / "T1_main.csv", index=False)
    piv = t.pivot(index="target", columns="arm", values="cell_asr").reindex(columns=[a for a in ARM_ORDER if a in set(t.arm)])
    to_tex(piv * 100, out / "T1_main.tex", f"cell-ASR (\\%) by target, judge={primary}, budget 1500", "{:.1f}")
    md.append(f"## T1 — cell-ASR (%) by target · judge `{primary}` · E1 (budget 1500)\n")
    md.append("| target | " + " | ".join(piv.columns) + " |\n|---|" + "---|" * len(piv.columns))
    for tgt in piv.index:
        cells = []
        for arm in piv.columns:
            r = t[(t.target == tgt) & (t.arm == arm)].iloc[0]
            cells.append(fmt_ci(r.cell_asr, r.lo, r.hi) + f" (n={r.n})")
        md.append(f"| {tgt} | " + " | ".join(cells) + " |")
    md.append("")

    # per dataset
    pd_ = d.groupby(["dataset", "arm"])["cell_asr"].mean().unstack("arm")
    pd_ = pd_.reindex(columns=[a for a in ARM_ORDER if a in pd_.columns])
    pd_.to_csv(out / "T2_per_dataset.csv")
    to_tex(pd_ * 100, out / "T2_per_dataset.tex", f"cell-ASR (\\%) by dataset (mean over targets, seeds), judge={primary}", "{:.1f}")
    md.append("## T2 — cell-ASR (%) by dataset (mean over targets × seeds)\n")
    md.append((pd_ * 100).round(1).to_markdown() + "\n")


def pairwise_table(pr, primary, out, md, exp="E1"):
    d = pr[(pr["judge"] == primary) & (pr["exp"] == exp) & (pr["variant"].fillna("") == "")]
    pairs = [("baseline", "beam"), ("baseline", "mcts"), ("beam", "mcts")]
    rows = []
    for scope in ["ALL"] + sorted(d["target"].unique()):
        s = d if scope == "ALL" else d[d["target"] == scope]
        for a, b in pairs:
            r = paired(s, a, b, "cell_asr")
            if r:
                rows.append(dict(scope=scope, **r))
    if not rows:
        return
    t = pd.DataFrame(rows)
    t["p_holm"] = holm(t["p_wilcoxon"].fillna(1.0))
    t.to_csv(out / "T3_pairwise.csv", index=False)
    md.append(f"## T3 — paired comparisons (B − A, cell-ASR points) · judge `{primary}`\n")
    md.append("| scope | A → B | n | Δ (95% CI) | W/T/L (B) | p (Holm) | d_z |\n|---|---|---|---|---|---|---|")
    for r in t.itertuples():
        md.append(f"| {r.scope} | {r.A} → {r.B} | {r.n} | {100*r.diff:+.1f} [{100*r.ci_lo:+.1f}, {100*r.ci_hi:+.1f}] | "
                  f"{r.wins_B}/{r.ties}/{r.wins_A} | {r.p_holm:.2g} | {r.d_z:.2f} |")
    md.append("")


def judge_robustness(pr, out, md):
    d = pr[(pr["exp"] == "E1") & (pr["variant"].fillna("") == "")]
    if d.empty:
        return
    rows = []
    for jn, g in d.groupby("judge"):
        row = {"judge": jn, "runs": g["run"].nunique()}
        for arm in ARM_ORDER:
            row[arm] = g[g["arm"] == arm]["cell_asr"].mean() * 100
        for a, b in [("baseline", "beam"), ("baseline", "mcts"), ("beam", "mcts")]:
            r = paired(g, a, b, "cell_asr")
            row[f"{b}>{a}"] = f"{r['wins_B']}/{r['n']}" if r else "-"
        means = {a: row[a] for a in ARM_ORDER if not np.isnan(row.get(a, np.nan))}
        row["ranking"] = " > ".join(sorted(means, key=lambda a: -means[a]))
        rows.append(row)
    t = pd.DataFrame(rows).set_index("judge")
    t.to_csv(out / "T4_judges.csv")
    md.append("## T4 — every judge: mean cell-ASR (%) and paired win counts (E1)\n")
    md.append(t.round(1).to_markdown() + "\n")
    md.append("_Sub-sampled judges (sample_frac<1) cover fewer runs per item; their cell-ASR is item-ASR × coverage._\n")


def agreement(items, judg, out, md):
    top1 = items[items["rank"] == 0][["h"]].drop_duplicates()
    names = sorted(judg)
    rows = []
    for a, b in itertools.combinations(names, 2):
        m = top1.join(judg[a]["success"].rename("a"), on="h", how="inner").join(judg[b]["success"].rename("b"), on="h", how="inner")
        if len(m) >= 20:
            rows.append(dict(judge_a=a, judge_b=b, n=len(m), agree=(m.a == m.b).mean(), kappa=kappa(m.a, m.b)))
    hum = sorted((out.parent / "human").glob("*_labeled.csv")) if (out.parent / "human").exists() else []
    for f in hum:
        h = pd.read_csv(f).dropna(subset=["label"])
        h = h[h["label"].astype(str).str.strip().isin(["0", "1", "0.0", "1.0"])]
        h["label"] = h["label"].astype(float).astype(int)
        for jn in names:
            m = h.join(judg[jn]["success"].rename("j"), on="h", how="inner")
            if len(m) >= 10:
                rows.append(dict(judge_a=f"human:{f.stem}", judge_b=jn, n=len(m), agree=(m.label == m.j).mean(),
                                 kappa=kappa(m.label, m.j)))
    if len(hum) >= 2:
        a, b = pd.read_csv(hum[0]), pd.read_csv(hum[1])
        m = a.merge(b, on="h", suffixes=("_1", "_2")).dropna(subset=["label_1", "label_2"])
        if len(m):
            rows.append(dict(judge_a=hum[0].stem, judge_b=hum[1].stem, n=len(m),
                             agree=(m.label_1 == m.label_2).mean(), kappa=kappa(m.label_1, m.label_2)))
    if rows:
        t = pd.DataFrame(rows)
        t.to_csv(out / "T5_agreement.csv", index=False)
        md.append("## T5 — judge agreement (Cohen's κ on shared top-1 items)\n")
        md.append(t.round(3).to_markdown(index=False) + "\n")


def cost_table(runs, out, md):
    d = runs[(runs["exp"] == "E1") & (runs["variant"].fillna("") == "")]
    if d.empty:
        return
    t = d.groupby("arm").agg(runs=("run", "size"), target_q=("target_q", "mean"), attack_q=("attack_q", "mean"),
                             eval_q=("eval_q", "mean"), wall_min=("wall_s", lambda s: s.mean() / 60),
                             budget_reached=("budget_reached", "mean"), coverage=("cells_filled", "mean"),
                             empty_prompts=("n_empty_prompt", "mean"))
    t["total_llm_calls"] = t[["target_q", "attack_q", "eval_q"]].sum(1)
    t.to_csv(out / "T6_cost.csv")
    md.append("## T6 — query accounting per run (E1 means)\n")
    md.append(t.round(1).to_markdown() + "\n")


def budget_curve(pr, primary, out, md):
    d = pr[(pr["judge"] == primary) & (pr["exp"].isin(["E1", "E2"])) & (pr["variant"].fillna("") == "")
           & pr["arm"].isin(["beam", "mcts"])]
    e2t = pr[pr["exp"] == "E2"]["target"].unique()
    e2d = pr[pr["exp"] == "E2"]["dataset"].unique()
    d = d[d["target"].isin(e2t) & d["dataset"].isin(e2d)]
    if d.empty or d["budget"].nunique() < 2:
        return
    t = d.groupby(["target", "arm", "budget"])["cell_asr"].agg(["mean", "count", lambda x: boot_ci(x)[0],
                                                                 lambda x: boot_ci(x)[1]])
    t.columns = ["mean", "n", "lo", "hi"]
    t.to_csv(out / "T7_budget_curve.csv")
    md.append("## T7 — cell-ASR vs target-query budget (E1 + E2)\n")
    md.append((t["mean"].unstack("budget") * 100).round(1).to_markdown() + "\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        tg = sorted(t.index.get_level_values(0).unique())
        fig, axes = plt.subplots(1, len(tg), figsize=(4.2 * len(tg), 3.2), squeeze=False)
        for ax, target in zip(axes[0], tg):
            for arm, color in [("beam", "#b26b16"), ("mcts", "#14575e")]:
                if (target, arm) not in t.index.droplevel(2):
                    continue
                s = t.loc[(target, arm)]
                ax.plot(s.index, s["mean"] * 100, "o-", color=color, label=arm)
                ax.fill_between(s.index, s["lo"] * 100, s["hi"] * 100, color=color, alpha=0.15)
            ax.set_title(target)
            ax.set_xlabel("target-query budget")
            ax.set_ylabel(f"cell-ASR % ({primary})")
            ax.grid(alpha=0.3)
            ax.legend()
        fig.tight_layout()
        fig.savefig(out / "fig_budget_curve.png", dpi=200)
        md.append("![budget curve](fig_budget_curve.png)\n")
    except Exception as e:
        md.append(f"_plot skipped: {e}_\n")


def equal_compute(pr, primary, out, md):
    d = pr[(pr["judge"] == primary) & (pr["variant"].fillna("") == "")]
    e3 = d[d["exp"] == "E3"]
    if e3.empty:
        return
    units = e3[["target", "dataset", "seed"]].drop_duplicates()
    base = d[(d["exp"] == "E1")].merge(units, on=["target", "dataset", "seed"])
    e3 = e3.assign(arm="baseline@5000")
    comb = pd.concat([base, e3])
    rows = [r for a, b in [("baseline@5000", "beam"), ("baseline@5000", "mcts"), ("baseline", "baseline@5000")]
            if (r := paired(comb, a, b, "cell_asr"))]
    if rows:
        t = pd.DataFrame(rows)
        t.to_csv(out / "T8_equal_compute.csv", index=False)
        md.append("## T8 — equal total-LLM-call budget: baseline@5000 target queries vs beam/mcts@1500\n")
        md.append(t[["A", "B", "n", "mean_A", "mean_B", "diff", "ci_lo", "ci_hi", "p_wilcoxon"]].round(3).to_markdown(index=False) + "\n")


def ablation(pr, primary, out, md):
    d = pr[(pr["judge"] == primary)]
    ab = d[d["exp"].isin(["E4", "E4b"])]
    if ab.empty:
        return
    units = ab[["target", "dataset", "seed"]].drop_duplicates()
    ref = d[(d["exp"] == "E1") & d["arm"].isin(["beam", "mcts"])].merge(units, on=["target", "dataset", "seed"])
    comb = pd.concat([ref, ab])
    rows = []
    for arm in sorted(ab["arm"].unique()):
        base_arm = arm.split("-")[0]
        r = paired(comb, base_arm, arm, "cell_asr")
        if r:
            rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(out / "T9_ablation.csv", index=False)
    md.append("## T9 — ablations vs the E1 default config (paired on the same units)\n")
    md.append(t[["A", "B", "n", "mean_A", "mean_B", "diff", "ci_lo", "ci_hi", "p_wilcoxon"]].round(3).to_markdown(index=False) + "\n")


def diversity(items, judg, primary, out, md, max_n=100):
    if primary not in judg:
        return
    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    except Exception:
        return
    sm = SmoothingFunction().method1
    top1 = items[(items["rank"] == 0) & (items["exp"] == "E1") & (items["variant"].fillna("") == "")]
    m = top1.join(judg[primary]["success"], on="h", how="inner")
    m = m[m["success"] == 1]
    rows = []
    for run, g in m.groupby("run"):
        toks = [p.lower().split() for p in g["prompt"].tolist()[:max_n]]
        if len(toks) < 3:
            continue
        sb = np.mean([sentence_bleu(toks[:i] + toks[i + 1:], toks[i], smoothing_function=sm) for i in range(len(toks))])
        rows.append(dict(run=run, arm=g["arm"].iloc[0], target=g["target"].iloc[0], n_success=len(toks), self_bleu=sb))
    if rows:
        t = pd.DataFrame(rows)
        t.to_csv(out / "T10_diversity_runs.csv", index=False)
        s = t.groupby("arm")[["self_bleu", "n_success"]].mean()
        md.append("## T10 — diversity of successful prompts (Self-BLEU, lower = more diverse; E1)\n")
        md.append(s.round(3).to_markdown() + "\n")


def transfer(ev, judg, primary, out, md):
    f = ev / "items_transfer.jsonl"
    if not f.exists() or primary not in judg:
        return
    t = pd.read_json(f, lines=True).join(judg[primary]["success"], on="h", how="inner")
    if t.empty:
        return
    m = t.groupby(["arm", "src_target", "target"])["success"].mean().unstack("target") * 100
    m.to_csv(out / "T11_transfer.csv")
    md.append(f"## T11 — transfer: success (%) of prompts that succeeded on src, re-sent to other targets ({primary})\n")
    md.append(m.round(1).to_markdown() + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", default=None, help="headline judge (default: judges.yaml primary)")
    a = ap.parse_args()
    import yaml
    primary = a.primary or yaml.safe_load(open(HERE / "judges.yaml", encoding="utf-8"))["primary"]
    spec, env = RM.load_spec(), RM.base_env()
    ev = RM.runs_root(spec, env) / "_eval"
    out = ev / "report"
    out.mkdir(parents=True, exist_ok=True)
    items, runs, judg = load(ev)
    pr = per_run(items, runs, judg)
    if not pr.empty:
        pr.to_csv(out / "per_run_per_judge.csv", index=False)
    md = ["# Paper results (auto-generated)\n",
          f"judges available: {', '.join(f'{k} ({len(v)} pairs)' for k, v in judg.items()) or 'none'} · "
          f"primary: **{primary}** · runs: {len(runs)} · items: {len(items)}\n"]
    if primary not in judg:
        md.append(f"**Primary judge `{primary}` has no judgments yet.**\n")
    if not pr.empty:
        main_table(pr, primary, out, md)
        pairwise_table(pr, primary, out, md)
        judge_robustness(pr, out, md)
        budget_curve(pr, primary, out, md)
        equal_compute(pr, primary, out, md)
        ablation(pr, primary, out, md)
    agreement(items, judg, out, md)
    cost_table(runs, out, md)
    diversity(items, judg, primary, out, md)
    transfer(ev, judg, primary, out, md)
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
