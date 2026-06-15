"""
Oracle-label refinement (frozen-rows distillation).

Polish the reconstructed model against oracle hard labels:
    * with `freeze_recovered_weights=True` (default), gradients on
      signature-recovered weight rows are zeroed — only biases, fc5, and
      rows of random-init (unrecovered) neurons update. This keeps the
      attack's extracted identity intact while letting non-extracted
      components absorb oracle-label information.
    * with `freeze_recovered_weights=False`, all params train (full
      distillation, drifts furthest from "extraction" but pushes accuracy
      closer to 100%).

Fix B (overfit-prevention knobs, all default-off):
    * `X_eval`: optional held-out tensor — agreement is measured on it every
      `eval_every` epochs to drive early-stop watchdog + best-checkpoint
      restore.
    * `early_stop` (with `patience`): stop if `patience` consecutive eval
      windows show no held-out improvement.
    * `weight_decay`: switches Adam→AdamW with given decay.
    * `use_cosine_lr`: wraps optimiser in CosineAnnealingLR(T_max=n_epochs).

Backward compat: all new args default to the values that reproduce the
legacy 1000-epoch Adam(lr=5e-3) no-early-stop run.
"""

import copy
import torch


def oracle_label_refinement(reconstructed_model, oracle_model, X_train,
                             recovered_masks, n_epochs=300, lr=5e-3,
                             freeze_recovered_weights=True, verbose=True,
                             # ----- Fix B additions -----
                             X_eval=None, eval_every=10, patience=5,
                             early_stop=False, weight_decay=0.0,
                             use_cosine_lr=False, eval_sample=1024):
    """Train against oracle(X_train).argmax for up to n_epochs.

    Args:
        X_eval: optional torch.Tensor — if provided AND early_stop, used as the
            watchdog signal. A sub-sample of up to `eval_sample` rows is taken
            (deterministic, first N) to keep cost low.
        eval_every: epochs between watchdog evaluations.
        patience: consecutive non-improving watchdog evals before stopping.
        early_stop: master toggle. When False, watchdog is informational only.
        weight_decay: AdamW weight_decay. 0 → plain Adam (legacy).
        use_cosine_lr: wrap optimiser in cosine annealing schedule.
    """
    reconstructed_model.train()
    oracle_model.eval()
    with torch.no_grad():
        oracle_labels = oracle_model(X_train).argmax(dim=1)

    hidden_layers = [reconstructed_model.fc1, reconstructed_model.fc2,
                     reconstructed_model.fc3, reconstructed_model.fc4]

    freeze_row_masks = {}
    if freeze_recovered_weights:
        for lid, mask in recovered_masks.items():
            freeze_row_masks[lid] = torch.tensor(mask, dtype=torch.bool)

    if weight_decay > 0:
        optimizer = torch.optim.AdamW(reconstructed_model.parameters(),
                                      lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(reconstructed_model.parameters(), lr=lr)

    scheduler = None
    if use_cosine_lr:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    loss_fn = torch.nn.CrossEntropyLoss()

    n_loggings = 10
    log_every = max(1, n_epochs // n_loggings)

    # ----- watchdog state -----
    use_watchdog = X_eval is not None
    if use_watchdog:
        X_watch = X_eval[:eval_sample]
        with torch.no_grad():
            watch_oracle_labels = oracle_model(X_watch).argmax(dim=1)
    best_eval_agree = -1.0
    best_state = None
    bad_evals = 0
    stopped_epoch = n_epochs

    with torch.no_grad():
        preds0 = reconstructed_model(X_train).argmax(dim=1)
        start_agree = (preds0 == oracle_labels).float().mean().item()
        start_watch = None
        if use_watchdog:
            watch_preds0 = reconstructed_model(X_watch).argmax(dim=1)
            start_watch = (watch_preds0 == watch_oracle_labels).float().mean().item()

    if verbose:
        opt_tag = f"AdamW(wd={weight_decay})" if weight_decay > 0 else "Adam"
        sched_tag = " +cosine" if use_cosine_lr else ""
        watch_tag = (f", watchdog on X_eval[:{eval_sample}] start={start_watch:.4f}"
                     if use_watchdog else "")
        es_tag = f", early_stop(patience={patience}, every={eval_every})" if (use_watchdog and early_stop) else ""
        print(f"  [refine] start agreement={start_agree:.4f}, "
              f"{'frozen recovered weights' if freeze_recovered_weights else 'all params trainable'}, "
              f"n_epochs={n_epochs}, lr={lr}, {opt_tag}{sched_tag}{watch_tag}{es_tag}")

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        preds = reconstructed_model(X_train)
        loss = loss_fn(preds, oracle_labels)
        loss.backward()

        if freeze_recovered_weights:
            for lid, layer in enumerate(hidden_layers):
                row_mask = freeze_row_masks.get(lid)
                if row_mask is not None and layer.weight.grad is not None:
                    layer.weight.grad[row_mask] = 0.0

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        do_watch = use_watchdog and ((epoch + 1) % eval_every == 0 or epoch == n_epochs - 1)
        if do_watch:
            reconstructed_model.eval()
            with torch.no_grad():
                watch_preds = reconstructed_model(X_watch).argmax(dim=1)
                watch_agree = (watch_preds == watch_oracle_labels).float().mean().item()
            reconstructed_model.train()
            if watch_agree > best_eval_agree + 1e-6:
                best_eval_agree = watch_agree
                best_state = copy.deepcopy(reconstructed_model.state_dict())
                bad_evals = 0
            else:
                bad_evals += 1
            if early_stop and bad_evals >= patience:
                stopped_epoch = epoch + 1
                if verbose:
                    print(f"  [refine] early stop at epoch {stopped_epoch} "
                          f"(best held-out agreement={best_eval_agree:.4f})")
                break

        if verbose and (epoch == 0 or (epoch + 1) % log_every == 0 or epoch == n_epochs - 1):
            with torch.no_grad():
                preds_log = reconstructed_model(X_train).argmax(dim=1)
                agree = (preds_log == oracle_labels).float().mean().item()
            extra = f"  watch={best_eval_agree:.4f}" if use_watchdog and best_eval_agree >= 0 else ""
            print(f"  [refine] epoch {epoch+1}/{n_epochs}  loss={loss.item():.4f}  "
                  f"agreement={agree:.4f}{extra}")

    # Restore best checkpoint (only when early_stop was requested AND we have one).
    restored_best = False
    if use_watchdog and early_stop and best_state is not None:
        reconstructed_model.load_state_dict(best_state)
        restored_best = True
        if verbose:
            print(f"  [refine] restored best checkpoint (held-out agreement={best_eval_agree:.4f})")

    reconstructed_model.eval()
    with torch.no_grad():
        preds = reconstructed_model(X_train).argmax(dim=1)
        final_agree = (preds == oracle_labels).float().mean().item()
        final_watch = None
        if use_watchdog:
            watch_preds = reconstructed_model(X_watch).argmax(dim=1)
            final_watch = (watch_preds == watch_oracle_labels).float().mean().item()

    result = {
        'start_agreement': float(start_agree),
        'final_agreement': float(final_agree),
        'freeze_recovered_weights': bool(freeze_recovered_weights),
        'n_epochs': int(n_epochs),
        'stopped_epoch': int(stopped_epoch),
        'lr': float(lr),
        'weight_decay': float(weight_decay),
        'use_cosine_lr': bool(use_cosine_lr),
        'early_stop': bool(early_stop),
    }
    if use_watchdog:
        result.update({
            'start_watchdog_agreement': float(start_watch) if start_watch is not None else None,
            'best_watchdog_agreement': float(best_eval_agree) if best_eval_agree >= 0 else None,
            'final_watchdog_agreement': float(final_watch) if final_watch is not None else None,
            'restored_best_checkpoint': bool(restored_best),
        })
    return result
