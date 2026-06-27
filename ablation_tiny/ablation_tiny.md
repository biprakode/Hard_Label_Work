TASK: Run an additive ablation of the Phase-3 reconstruction pipeline on the 6 make_blobs
victims and write results to a markdown file. Read-only on all method code. Do NOT modify
the pipeline, the victims, or any hyperparameter. This measures the EXISTING pipeline staged,
not a new procedure.

SCOPE — exactly these 6 victims, no CIFAR:
  tiniest_relu, tiniest_leakyrelu, tinier_relu, tinier_leakyrelu, tiny_relu, tiny_leakyrelu
Use the canonical 2026-06-21 victim checkpoints and the SA+margin sign-search configuration
already used for the headline results. Use the same three-tier dataset contract: train on
X_test ∪ X_test2, evaluate every number on the held-out X_test3.

THE ABLATION IS ADDITIVE. For each victim, evaluate the reconstructed model at five
cumulative stages, building the pipeline up one component at a time, and record metrics on
X_test3 at EACH stage:

  Stage 0  RAW            Phase 1+2 load only. Recovered directions in place, biases = 0,
                          output layer = random init, signs as Phase-2 produced them.
  Stage 1  + BIAS         add bias recovery from dual points (Section 6.2).
  Stage 2  + LR FIT       add the multinomial logistic-regression output-layer fit
                          (Section 6.3). NOTE: the LR fit MUST precede sign search, since
                          sign search needs a non-random decoder for a meaningful agreement
                          signal. Respect this order; do not move sign search before the fit.
  Stage 3  + SIGN SEARCH  add SA+margin metaheuristic sign search (Section 5).
  Stage 4  + FROZEN REFINE add frozen-row refinement (Section 6.4). This is the full pipeline.

  Each stage is CUMULATIVE: stage k includes everything from stages < k. Stage 4 must
  reproduce the headline per-victim numbers already reported (sanity check — flag any victim
  where stage 4 disagrees with the published Table 1 / Table 2 values by more than rounding).

METRICS — at every stage, for every victim, on X_test3:
  - Agreement   (fraction of held-out inputs where reconstructed argmax == oracle argmax)
  - Ext acc     (reconstructed accuracy against true labels)
  - Sign acc    (sign-recovery accuracy on recovered rows vs ground-truth weights; report
                 'n/a' at stages where signs are not yet resolved if that is the honest state,
                 but if Phase-2 signs exist at stage 0 report them)
  - EQS         (structural-variant composite, the same 22/26/17/20 weighting used elsewhere)

DISTILLATION BASELINE — exactly ONE row per victim, NOT staged:
  Train the full distillation baseline (all 832/256/... hidden rows random and trainable, same
  query budget, same refinement settings) and report its X_test3 Agreement, Ext acc, EQS.
  Sign acc is n/a for distillation (no recovered rows). This is the contrast row, it gets no
  additive chain.

CONSTRAINTS:
  - Do NOT invent or interpolate any number. Every cell comes from an actual evaluation run.
  - Do NOT change sign-search method, refinement epochs, weight decay, LR schedule, or the
    EQS weights. Use the canonical config.
  - If a stage is ill-defined for a victim (e.g. a metric cannot be computed), write the cell
    as 'n/a' with a one-line reason, do not fabricate.
  - Each metric is on X_test3 only. Never report a training-pool number.

OUTPUT — write ABLATION_RESULTS.md with:
  1. One table PER VICTIM (6 tables), rows = stages 0–4 + distillation baseline,
     columns = Agreement, Ext acc, Sign acc, EQS.
  2. A per-stage DELTA column or a short note under each table giving the lift each component
     adds (stage k minus stage k-1) for Agreement and EQS, so the marginal contribution of
     each component is visible.
  3. A one-paragraph PER-VICTIM note flagging which single component carried the largest lift
     on that victim.
  4. A SANITY-CHECK section confirming stage-4 numbers match the published headline table,
     listing any discrepancy.
  5. A CONFIG block recording: victim checkpoint paths, sign-search method, refinement epochs/
     optimizer/LR schedule/weight decay, EQS weights, dataset seeds, and the eval set
     (X_test3), so the run is reproducible.

Do NOT write any paper prose. Produce ABLATION_RESULTS.md only. I will hand it back for the
section draft.
