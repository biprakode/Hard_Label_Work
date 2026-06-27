#!/usr/bin/env python3
"""
Aggregate per-victim ablation JSONs (written by ablation_harness.py) into the
single ABLATION_RESULTS.md deliverable: 6 per-victim tables (stages 0-4 +
distillation), per-stage deltas, per-victim largest-lift notes, a stage-4
sanity-check section, and a reproducibility config block.
"""

import os
import glob
import json
import argparse
import datetime

# canonical victim ordering (fast-first, relu before leaky within a tier)
ORDER = [
    ('tiniest', 'relu'), ('tiniest', 'leakyrelu'),
    ('tinier',  'relu'), ('tinier',  'leakyrelu'),
    ('tiny',    'relu'), ('tiny',    'leakyrelu'),
]
STAGE_LABELS = {
    '0': "Stage 0 — RAW (Phase 1+2 load)",
    '1': "Stage 1 — + BIAS recovery",
    '2': "Stage 2 — + LR FIT (fc5)",
    '3': "Stage 3 — + SIGN SEARCH (SA+margin)",
    '4': "Stage 4 — + FROZEN REFINE (full pipeline)",
}
COMPONENT_OF_STAGE = {
    '1': "bias recovery", '2': "fc5 LR fit",
    '3': "SA+margin sign search", '4': "frozen-row refinement",
}
SANITY_TOL_PP = 2.0   # flag a stage-4 vs headline gap larger than this (percentage points)


def pct(x):
    return "n/a" if x is None else f"{100 * x:.2f}%"


def eqs(x):
    return "n/a" if x is None else f"{x:.1f}"


def signed_pp(x):
    return "n/a" if x is None else f"{100 * x:+.2f}"


def signed_pts(x):
    return "n/a" if x is None else f"{x:+.1f}"


def load_results(results_dir):
    data = {}
    for path in glob.glob(os.path.join(results_dir, "*.json")):
        try:
            j = json.load(open(path))
        except Exception:
            continue
        key = (j.get('_arch_alias') or _alias(j), j.get('activation'))
        data[key] = j
    return data


def _alias(j):
    # harness stores arch_key (tiniest/tinier/makeblobs); report uses tiny for makeblobs
    ak = j.get('arch_key')
    return 'tiny' if ak == 'makeblobs' else ak


def victim_table(j):
    """Return (markdown_lines, largest_lift_note) for one victim."""
    stages = j['stages']
    L = []
    L.append("| Stage | Agreement | Ext acc | Sign acc | EQS | ΔAgreement | ΔEQS |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")

    prev_ag = prev_eqs = None
    lifts = []  # (stage_id, d_agree, d_eqs)
    for sid in ['0', '1', '2', '3', '4']:
        s = stages.get(sid)
        if s is None:
            L.append(f"| {STAGE_LABELS[sid]} | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        ag, ea, sa, eq = s['agreement'], s.get('ext_acc'), s.get('sign_acc'), s.get('eqs')
        if sid == '0':
            d_ag = d_eq = None
        else:
            d_ag = None if (ag is None or prev_ag is None) else ag - prev_ag
            d_eq = None if (eq is None or prev_eqs is None) else eq - prev_eqs
            lifts.append((sid, d_ag if d_ag is not None else -1e9,
                          d_eq if d_eq is not None else -1e9))
        L.append(f"| {STAGE_LABELS[sid]} | {pct(ag)} | {pct(ea)} | {pct(sa)} | "
                 f"{eqs(eq)} | {signed_pp(d_ag)} | {signed_pts(d_eq)} |")
        prev_ag, prev_eqs = ag, eq

    d = j['distillation']
    L.append(f"| Distillation baseline (non-staged) | {pct(d['agreement'])} | "
             f"{pct(d.get('ext_acc'))} | n/a | {eqs(d.get('eqs'))} | — | — |")

    # largest single-component lift, by agreement (tie-broken by EQS)
    note = "No staged lift recorded."
    if lifts:
        best = max(lifts, key=lambda t: (t[1], t[2]))
        sid = best[0]
        note = (f"Largest lift carried by **{COMPONENT_OF_STAGE[sid]}** "
                f"(Stage {sid}): ΔAgreement {signed_pp(best[1] if best[1] > -1e8 else None)} pp, "
                f"ΔEQS {signed_pts(best[2] if best[2] > -1e8 else None)}.")
    return L, note


def sanity_rows(j):
    """Compare harness stage-4 against the driver headline references."""
    s4 = j['stages'].get('4', {})
    ag4 = s4.get('agreement')
    eqs4 = s4.get('eqs')
    href = j.get('headline_reference', {})
    rows = []
    flags = []

    dm = href.get('driver_extraction_metrics') or {}
    if dm.get('prediction_agreement') is not None:
        h = dm['prediction_agreement']
        gap = None if ag4 is None else (ag4 - h) * 100
        rows.append((f"driver run_extraction.py prediction_agreement ({dm.get('eval_tag','?')})",
                     pct(h), pct(ag4),
                     "n/a" if gap is None else f"{gap:+.2f} pp"))
        if gap is not None and abs(gap) > SANITY_TOL_PP:
            flags.append(f"agreement vs driver Δ={gap:+.2f} pp (> {SANITY_TOL_PP} pp)")

    sc = href.get('driver_scorecard') or {}
    if sc.get('fidelity') is not None:
        h = sc['fidelity']
        gap = None if ag4 is None else (ag4 - h) * 100
        rows.append((f"driver step-9 scorecard fidelity ({sc.get('eval_tag','?')})",
                     pct(h), pct(ag4),
                     "n/a" if gap is None else f"{gap:+.2f} pp"))
        if gap is not None and abs(gap) > SANITY_TOL_PP:
            flags.append(f"agreement vs scorecard Δ={gap:+.2f} pp (> {SANITY_TOL_PP} pp)")
    if sc.get('eqs_structural') is not None:
        h = sc['eqs_structural']
        gap = None if eqs4 is None else eqs4 - h
        rows.append(("driver step-9 scorecard EQS (structural)",
                     eqs(h), eqs(eqs4),
                     "n/a" if gap is None else f"{gap:+.1f}"))
    return rows, flags


def build(results_dir, out_path):
    data = load_results(results_dir)
    today = datetime.date.today().isoformat()

    L = []
    L.append("# Additive Phase-3 Ablation — make_blobs victims")
    L.append("")
    L.append(f"_Generated {today}. Produced by `ablation_tiny/run_ablation.sh` → "
             "`ablation_harness.py` (read-only on all pipeline/method code) → "
             "`aggregate_ablation.py`._")
    L.append("")
    L.append("Every number is an actual evaluation on the held-out **X_test3** "
             "(never queried; training pool = X_test ∪ X_test2). Stages are "
             "**cumulative**: stage *k* includes all components of stages < *k*. "
             "EQS is the structural-variant composite (C1=22, C2=26, C3=17, S=20).")
    L.append("")
    present = [(a, c) for (a, c) in ORDER if (a, c) in data]
    missing = [(a, c) for (a, c) in ORDER if (a, c) not in data]
    if missing:
        L.append("> **Pending victims (no results JSON found):** "
                 + ", ".join(f"`{a}_{c}`" for a, c in missing) + ".")
        L.append("")

    # ---- per-victim tables ----
    L.append("## 1. Per-victim ablation tables")
    L.append("")
    all_notes = []
    for (a, c) in present:
        j = data[(a, c)]
        rs = j.get('recovery_stats', {})
        L.append(f"### {a}_{c}")
        L.append("")
        L.append(f"_Victim `{os.path.basename(j.get('victim_path',''))}` · "
                 f"oracle acc (X_test3) = {pct(j.get('oracle_acc_test3'))} · "
                 f"recovered {rs.get('recovered_neurons','?')}/{rs.get('total_neurons','?')} "
                 f"neurons · LeakyReLU α={j.get('leaky_alpha')}._")
        L.append("")
        tbl, note = victim_table(j)
        L.extend(tbl)
        L.append("")
        L.append(f"_{note}_")
        L.append("")
        all_notes.append((f"{a}_{c}", note))

    # ---- per-victim largest-lift summary ----
    L.append("## 2. Per-victim largest-lift summary")
    L.append("")
    for tag, note in all_notes:
        L.append(f"- **{tag}** — {note}")
    L.append("")

    # ---- sanity check ----
    L.append("## 3. Stage-4 sanity check (vs headline)")
    L.append("")
    L.append("Stage 4 is the full pipeline and should reproduce the headline "
             "numbers produced by the canonical driver on the same Phase-1/2 "
             "artifacts. Phase-1 dual search is stochastic, so small deviations "
             f"are expected; gaps > {SANITY_TOL_PP} pp are flagged.")
    L.append("")
    any_flag = False
    for (a, c) in present:
        j = data[(a, c)]
        rows, flags = sanity_rows(j)
        L.append(f"**{a}_{c}**")
        L.append("")
        if rows:
            L.append("| Headline reference | Headline | Harness Stage 4 | Δ |")
            L.append("|---|---:|---:|---:|")
            for r in rows:
                L.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |")
        else:
            L.append("_No headline reference on disk (driver outputs not found)._")
        L.append("")
        if flags:
            any_flag = True
            L.append("> ⚠ " + "; ".join(flags))
            L.append("")
    if not any_flag and present:
        L.append("_No stage-4 discrepancies beyond tolerance / rounding._")
        L.append("")

    # ---- config block ----
    L.append("## 4. Config (reproducibility)")
    L.append("")
    if present:
        any_j = data[present[0]]
        cfg = any_j.get('config', {})
        L.append("Identical canonical SA+margin configuration for every victim "
                 "(per-arch only `refine_epochs` differs):")
        L.append("")
        L.append("| Knob | Value |")
        L.append("|---|---|")
        L.append(f"| sign-search method / objective | {cfg.get('sign_search_method')} / {cfg.get('sign_search_objective')} |")
        L.append(f"| sign pair-lookahead K | {cfg.get('sign_pair_lookahead')} |")
        L.append(f"| sign refine cycles / mini-epochs | {cfg.get('sign_refine_cycles')} / {cfg.get('sign_refine_mini_epochs')} |")
        L.append(f"| refine optimiser | AdamW, weight_decay={cfg.get('refine_weight_decay')}, cosine_lr={cfg.get('refine_cosine_lr')} |")
        L.append(f"| refine lr | {cfg.get('refine_lr')} |")
        L.append(f"| refine epochs (tiniest/tinier/tiny) | 300 / 500 / 500 |")
        L.append(f"| early-stop patience / eval-every | {cfg.get('early_stop_patience')} / {cfg.get('eval_every')} |")
        L.append(f"| train pool / eval set | {cfg.get('train_pool')} / {cfg.get('eval_set')} |")
        L.append(f"| query budget | {cfg.get('query_budget')} |")
        L.append(f"| EQS variant | {cfg.get('eqs_variant')} |")
        L.append(f"| reconstruct seed / scorecard seed | {cfg.get('reconstruct_seed')} / {cfg.get('scorecard_seed')} |")
        L.append(f"| margin-proxy subsample (n_boundary) | {cfg.get('n_boundary')} |")
        L.append("")
        L.append("**Victim checkpoints** (`tiny_stuff/`):")
        L.append("")
        for (a, c) in present:
            j = data[(a, c)]
            L.append(f"- `{a}_{c}` → `{j.get('victim_path')}`")
        L.append("")
    L.append("_Distillation baseline: all hidden rows Kaiming-initialised and "
             "trainable (`--refine-unfreeze`), same query budget + refinement "
             "settings; sign acc n/a (no recovered rows), EQS structural S-block = 0._")
    L.append("")

    with open(out_path, 'w') as f:
        f.write("\n".join(L))
    print(f"Wrote {out_path}  ({len(present)}/{len(ORDER)} victims present)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    build(a.results_dir, a.out)


if __name__ == '__main__':
    main()
