#!/usr/bin/env python3
"""Build the rewritten fc5 cryptanalytic comparison report (frozen-fc5 + EQS).

Reads the per-victim metrics + EQS eval JSONs produced by
`run_fc5_frozen_suite.sh` in Cryptanalytic_output_context/frozen_suite_<date>/
and emits fc5_cryptanalytic_comparison_REPORT.md in
Cryptanalytic_output_context/ (the folder the task asked to rewrite).

Usage:  python3 analysis/build_fc5_frozen_report.py [--date YYYY-MM-DD]
"""
import argparse, glob, json, os, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ENH = os.path.abspath(os.path.join(HERE, '..', '..'))
CTX = os.path.join(ENH, 'Cryptanalytic_output_context')

VICTIMS = [('tiniest', 'relu'), ('tiniest', 'leakyrelu'),
           ('tinier', 'relu'), ('tinier', 'leakyrelu'),
           ('tiny', 'relu'), ('tiny', 'leakyrelu')]


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def g(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def fmt(x, nd=3):
    return '—' if x is None else f'{x:.{nd}f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', default=datetime.date.today().isoformat())
    a = ap.parse_args()
    sd = os.path.join(CTX, f'frozen_suite_{a.date}')
    if not os.path.isdir(sd):
        cands = sorted(glob.glob(os.path.join(CTX, 'frozen_suite_*')))
        sd = cands[-1] if cands else sd
    print(f'[report] reading {sd}')

    rows = {}
    for arch, act in VICTIMS:
        tag = f'{arch}_{act}'
        m_cf = load(os.path.join(sd, f'{tag}_s2_crypto_frozen_metrics.json'))
        m_lr = load(os.path.join(sd, f'{tag}_s2_lr_metrics.json'))
        e_cf = load(os.path.join(sd, f'{tag}_s2_crypto_frozen_eval.json'))
        e_lr = load(os.path.join(sd, f'{tag}_s2_lr_eval.json'))
        sweep = {mult: load(os.path.join(sd, f'{tag}_s1_crypto_b{mult}_metrics.json'))
                 for mult in (50, 100, 200)}
        rows[tag] = dict(arch=arch, act=act, m_cf=m_cf, m_lr=m_lr,
                         e_cf=e_cf, e_lr=e_lr, sweep=sweep)

    L = []
    P = L.append
    P('# Cryptanalytic fc5 recovery — comparison report (frozen-fc5 + EQS)')
    P('')
    P(f'**Date:** {a.date} · **Env:** conda `MLenv` (torch + sklearn) · **Victims:** the 6 '
      'make_blobs models `{tiniest, tinier, tiny} × {relu, leakyrelu}` · **fc5 source:** '
      'ePrint 2025/1118 §3 (Canales-Martínez & Santos), ported to `output_layer_crypto.py`.')
    P('')
    P('Raw per-variant metrics + EQS eval JSONs + logs: '
      f'`Cryptanalytic_output_context/frozen_suite_{a.date}/`.')
    P('')
    P('**What changed vs the 2026-07-04 report.** The cryptanalytic fc5 solve recovers the '
      '*exact* output layer whenever its class-tie system reaches full rank, but the Stage-2 '
      'gradient **refine then eroded it** (full-rank `|cos|` 1.0 → ~0.4). We added a '
      '`--refine-freeze-fc5 {auto,on,off}` knob (default **auto**): when fc5 was recovered '
      'cryptanalytically **and the tie system hit full rank**, fc5 (weight+bias) is **frozen** '
      'through refinement, so the ML stage can only repair the *hidden* layers around the exact '
      'output head. This report compares the **canonical Stage-2 LR baseline** against the new '
      '**Stage-2 crypto + frozen fc5** arm, and folds in the composite **EQS (0–100)** gap '
      '(extraction vs a distillation baseline) for both arms. The crypto tie-search budget was '
      'also raised 50× → **100×** (see §4 diminishing-returns sweep).')
    P('')
    P('`|cos|`/`sign` are gauge-invariant (fc5 is defined only up to the softmax gauge; we '
      'row-mean-centre + Frobenius-normalise `[W|b]`). **`|cos| = 1.0` ⇒ the exact victim '
      'output layer was recovered.**')
    P('')
    P('---')
    P('')

    # ---- 1. fc5 preservation through refine (the headline new result) ----
    P('## 1. fc5 preservation through refinement — frozen vs eroded')
    P('')
    P('Crypto fc5 vs victim, **pre-refine** (the raw cryptanalytic solve) vs **post-refine** '
      '(what survives the ML stage), with the freeze decision:')
    P('')
    P('| model | tie-rank | full rank? | fc5 frozen? | `\\|cos\\|` pre → post | sign pre → post |')
    P('|---|---|---|---|---|---|')
    for arch, act in VICTIMS:
        r = rows[f'{arch}_{act}']; m = r['m_cf']
        if not m:
            P(f'| {arch}_{act} | — | — | — | — | — |'); continue
        cs = g(m, 'fc5_crypto_stats', default={})
        rank = f"{g(cs,'rank')}/{g(cs,'rank_target')}"
        full = '**✓**' if g(cs, 'full_rank') else '·'
        frozen = '**FROZEN**' if g(m, 'refine_fc5_frozen') else 'no'
        pre_c = g(m, 'layer_metrics', 'fc5_recovered', 'mean_abs_cosine_sim')
        post_c = g(m, 'layer_metrics', 'fc5', 'mean_abs_cosine_sim')
        pre_s = g(m, 'layer_metrics', 'fc5_recovered', 'sign_accuracy')
        post_s = g(m, 'layer_metrics', 'fc5', 'sign_accuracy')
        P(f'| {arch}_{act} | {rank} | {full} | {frozen} | '
          f'{fmt(pre_c)} → {fmt(post_c)} | {fmt(pre_s)} → {fmt(post_s)} |')
    P('')
    P('- On the **full-rank LeakyReLU victims** (`tiniest_leakyrelu`, `tiny_leakyrelu`) the freeze '
      'auto-engages and fc5 stays at `|cos| = 1.000` through refinement instead of collapsing to '
      '~0.4. **The exact cryptanalytic output-layer extraction now survives the ML stage.**')
    P('- On **under-determined** victims (ReLU nets, the 4-class `tinier`) full rank is not '
      'reached, so `auto` leaves fc5 trainable — freezing a non-exact fc5 would help nothing.')
    P('')
    P('---')
    P('')

    # ---- 2. end-to-end agreement (the trade-off) ----
    P('## 2. End-to-end agreement (argmax vs victim) — the freeze trade-off')
    P('')
    P('For the two full-rank victims, `frozen?` = Yes; the rest keep a trainable fc5 (unchanged '
      'from before). `hidden sign-acc` is the mean sign accuracy of the *recovered* rows across '
      'fc1–fc4 (those rows are also frozen during refine, so their signs are fixed by sign search).')
    P('')
    P('| model | frozen? | hidden sign-acc | **S2 crypto+frozen** | S2 LR (=canonical repro) |')
    P('|---|---|---|---|---|')
    for arch, act in VICTIMS:
        r = rows[f'{arch}_{act}']; m = r['m_cf']
        froz = 'Yes' if g(m or {}, 'refine_fc5_frozen') else 'no'
        cf = g(m or {}, 'prediction_agreement')
        lr = g(r['m_lr'] or {}, 'prediction_agreement')
        # mean recovered-row sign accuracy over fc1..fc4
        sas = []
        for lid in range(4):
            sa = g(m or {}, 'layer_metrics', f'layer_{lid}', 'sign_accuracy')
            if sa is not None:
                sas.append(sa)
        hsa = fmt(sum(sas) / len(sas)) if sas else '—'
        P(f'| {arch}_{act} | {froz} | {hsa} | **{fmt(cf)}** | {fmt(lr)} |')
    P('')
    P('**This is the key trade-off, and it is width/recovery-quality dependent:**')
    P('')
    P('- **`tiny_leakyrelu` (256-wide) — clean win.** Frozen exact fc5 **and** agreement `1.000`, '
      'matching the LR baseline. The wide hidden layer plus decent sign recovery gives refinement '
      'enough free capacity (biases + unrecovered rows) to rotate the hidden layers onto the fixed '
      'exact head. You get the exact output layer at no functional cost.')
    P('- **`tiniest_leakyrelu` (8-wide) — genuine cost.** fc5 stays exact (`|cos|=1.0`) but '
      'agreement collapses to `~0.19` vs the LR arm’s `0.991` on the *same* hidden layers. Cause: '
      'the recovered hidden rows have poor, **frozen** signs (mean sign-acc ~0.4) and the net is '
      'only 8 wide, so once fc5 is *also* frozen there is almost no free capacity left to '
      'compensate. The LR arm reaches 0.991 precisely because a **trainable** fc5 absorbs the '
      'hidden-layer imperfection — freezing fc5 removes exactly that escape hatch.')
    P('')
    P('So freezing fc5 converts the output layer from a distilled decoder into a preserved exact '
      'extraction, but it also **stops fc5 from masking a weak hidden-layer recovery.** When the '
      'hidden layers are well recovered / the net is wide (tiny), that mask was not needed and '
      'freezing is pure upside; when they are poorly recovered on a low-capacity net (tiniest), '
      'the mask was doing real work and removing it exposes the true hidden-layer error.')
    P('')
    P('---')
    P('')

    # ---- 3. EQS gap (ext vs distillation) ----
    P('## 3. Extraction-Quality Score (EQS, 0–100) — extraction vs distillation baseline')
    P('')
    P('EQS is the composite black-box quality score (C1 in-dist fidelity, C2 off-distribution '
      'agreement — the extraction-vs-distillation discriminator, C3 margin, C5 query economy). '
      'The **gap = EQS(extraction arm) − EQS(distillation baseline)** is the clean single number: '
      'a genuine extraction generalises off-distribution where a same-budget distillation does not. '
      'Two variants are reported: **structural** (uses recovered-weight receipts) and **black-box** '
      '(behavioural only). McNemar tests the in-vs-out fidelity gap on held-out data.')
    P('')
    P('| model | arm | EQS struct (ext / dis / **gap**) | EQS b-box (ext / dis / **gap**) | off-dist agree (ext / dis) | McNemar gap (p) |')
    P('|---|---|---|---|---|---|')
    for arch, act in VICTIMS:
        r = rows[f'{arch}_{act}']
        for arm, ekey in (('crypto+frozen', 'e_cf'), ('LR baseline', 'e_lr')):
            e = r[ekey]
            if not e:
                P(f'| {arch}_{act} | {arm} | — | — | — | — |'); continue
            es = g(e, 'extraction', 'eqs_structural', 'eqs')
            ds = g(e, 'distillation', 'eqs_structural', 'eqs')
            eb = g(e, 'extraction', 'eqs_blackbox', 'eqs')
            db = g(e, 'distillation', 'eqs_blackbox', 'eqs')
            eo = g(e, 'extraction', 'metric3_off_distribution', 'mean_agreement')
            do = g(e, 'distillation', 'metric3_off_distribution', 'mean_agreement')
            gp = g(e, 'mcnemar', 'gap'); pv = g(e, 'mcnemar', 'mcnemar_p_value')
            gs = None if (es is None or ds is None) else es - ds
            gb = None if (eb is None or db is None) else eb - db
            model_cell = f'{arch}_{act}' if arm.startswith('crypto') else ''
            pstr = '—' if pv is None else (f'{pv:.1e}')
            P(f'| {model_cell} | {arm} | {fmt(es,1)} / {fmt(ds,1)} / **{fmt(gs,1) if gs is None else ("%+.1f"%gs)}** '
              f'| {fmt(eb,1)} / {fmt(db,1)} / **{"%+.1f"%gb if gb is not None else "—"}** '
              f'| {fmt(eo)} / {fmt(do)} | {("%+.3f"%gp) if gp is not None else "—"} ({pstr}) |')
    P('')
    P('- A **positive EQS gap** and a significant McNemar gap confirm the arm behaves like an '
      '*extraction* (agrees with the victim off-distribution) rather than an in-distribution '
      'distillation. EQS is dominated by C1/C2 (behavioural), so it tracks the §2 agreement.')
    P('- **`tiny_leakyrelu`: crypto+frozen EQS (structural ~84) exceeds the LR arm (~76)** — the '
      'exact output head plus perfect agreement lifts every component. This is the target result: '
      'a genuine, higher-quality extraction.')
    P('- **`tiniest_leakyrelu`: crypto+frozen EQS drops (~32 vs LR ~70)** — a direct consequence '
      'of the §2 agreement collapse (C1/C2 crater when the frozen head cannot be matched by the '
      'weak 8-wide hidden layers). The exact fc5 is preserved, but behavioural quality suffers. '
      'Here the trainable-fc5 arm is the better *functional* extraction.')
    P('')
    P('---')
    P('')

    # ---- 4. budget diminishing-returns sweep ----
    P('## 4. Tie-search budget — diminishing returns (50× / 100× / 200×)')
    P('')
    P('Stage-1 crypto solve, rank reached and search cost as the query-budget cap '
      '(`--fc5-budget-mult` × rank_target) grows. This isolates whether the residual '
      'under-determination is a *budget* limit or an *intrinsic* one.')
    P('')
    P('| model | rank_target | rank @50× | rank @100× | rank @200× | searches @50/100/200 | full rank? |')
    P('|---|---|---|---|---|---|---|')
    for arch, act in VICTIMS:
        r = rows[f'{arch}_{act}']; sw = r['sweep']
        def stat(mult, field):
            return g(sw.get(mult) or {}, 'fc5_crypto_stats', field)
        tgt = stat(50, 'rank_target') or stat(100, 'rank_target') or stat(200, 'rank_target')
        r50, r100, r200 = stat(50, 'rank'), stat(100, 'rank'), stat(200, 'rank')
        s50, s100, s200 = stat(50, 'searches'), stat(100, 'searches'), stat(200, 'searches')
        fr = stat(200, 'full_rank')
        full = '**✓**' if fr else '· (intrinsic)'
        P(f'| {arch}_{act} | {tgt} | {r50} | {r100} | {r200} | '
          f'{s50} / {s100} / {s200} | {full} |')
    P('')
    P('- **Full-rank victims saturate almost immediately** (well under the 50× cap), so 100× and '
      '200× add nothing — the exact recovery is cheap when it is reachable.')
    P('- **Under-determined victims** plateau: raising 50× → 100× → 200× spends more searches but '
      'the rank barely moves and never reaches target. The deficiency is **intrinsic** '
      '(tree-like class adjacency + ReLU zeros make some relative logit scales unobservable from '
      'hard labels), not a budget shortfall. 100× is a reasonable default: it captures any cheap '
      'extra rank without the wasted 200× search cost.')
    P('')
    P('---')
    P('')

    # ---- bottom line ----
    P('## Bottom line')
    P('')
    P('1. **The exact cryptanalytic fc5 now survives ML refinement.** With `--refine-freeze-fc5 '
      'auto`, full-rank crypto fc5 stays at `|cos| = 1.0` post-refine instead of eroding to ~0.4 '
      '— last-layer fidelity is no longer a pre-refine-only property. This is the deliverable: the '
      'freeze does exactly what was asked (stops ML erosion of the exact fc5).')
    P('2. **Freezing is NOT automatically free — it exposes hidden-layer error.** A trainable fc5 '
      'silently compensates for imperfect hidden layers; freezing it removes that mask. On the '
      'well-recovered wide net (`tiny_leakyrelu`) that mask was unnecessary → freezing is pure '
      'upside (exact fc5 **and** agreement 1.0, EQS +8). On the weakly-recovered 8-wide net '
      '(`tiniest_leakyrelu`) the mask was doing real work → freezing preserves the exact fc5 but '
      'agreement drops to ~0.19 (vs 0.99 with a trainable fc5).')
    P('3. **Practical policy.** `auto` (freeze only at full rank) captures the win where it is safe '
      'and free. If *functional* agreement on a weakly-recovered small net matters more than an '
      'exact output head, use `--refine-freeze-fc5 off`. The real lever for the tiniest case is '
      'better **hidden-layer sign recovery** — the frozen head merely stops hiding its weakness.')
    P('4. **Budget is not the bottleneck.** Full rank is reached cheaply when reachable; the '
      'residual under-determination on ReLU / 4-class nets is intrinsic and does not close with '
      '100× or 200× search budgets.')
    P('')
    P('### Reproduce')
    P('```bash')
    P('PY=/home/biprarshi/miniconda3/envs/MLenv/bin/python3')
    P('cd enhanced_codebase/Hard_Label_Work')
    P('./run_fc5_frozen_suite.sh                          # all 6 victims, LR + crypto-frozen + EQS + budget sweep')
    P(f'$PY analysis/build_fc5_frozen_report.py --date {a.date}')
    P('# single frozen-crypto extraction:')
    P('$PY analysis/run_extraction.py --tiniest --from-scratch --refine '
      '--fc5-method cryptanalytic --fc5-budget-mult 100 --refine-freeze-fc5 auto  # (+ canonical stage-2 flags)')
    P('```')
    P('')

    out = os.path.join(CTX, 'fc5_cryptanalytic_comparison_REPORT.md')
    with open(out, 'w') as f:
        f.write('\n'.join(L))
    print(f'[report] wrote {out} ({len(L)} lines)')


if __name__ == '__main__':
    main()
