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
"""

import torch


def oracle_label_refinement(reconstructed_model, oracle_model, X_train,
                             recovered_masks, n_epochs=300, lr=5e-3,
                             freeze_recovered_weights=True, verbose=True):
    """Train against oracle(X_train).argmax for n_epochs."""
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

    optimizer = torch.optim.Adam(reconstructed_model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    n_loggings = 10
    log_every = max(1, n_epochs // n_loggings)

    with torch.no_grad():
        preds0 = reconstructed_model(X_train).argmax(dim=1)
        start_agree = (preds0 == oracle_labels).float().mean().item()
    if verbose:
        print(f"  [refine] start agreement={start_agree:.4f}, "
              f"{'frozen recovered weights' if freeze_recovered_weights else 'all params trainable'}, "
              f"n_epochs={n_epochs}, lr={lr}")

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

        if verbose and (epoch == 0 or (epoch + 1) % log_every == 0 or epoch == n_epochs - 1):
            with torch.no_grad():
                preds = reconstructed_model(X_train).argmax(dim=1)
                agree = (preds == oracle_labels).float().mean().item()
            print(f"  [refine] epoch {epoch+1}/{n_epochs}  loss={loss.item():.4f}  agreement={agree:.4f}")

    reconstructed_model.eval()
    with torch.no_grad():
        preds = reconstructed_model(X_train).argmax(dim=1)
        final_agree = (preds == oracle_labels).float().mean().item()
    return {
        'start_agreement': float(start_agree),
        'final_agreement': float(final_agree),
        'freeze_recovered_weights': bool(freeze_recovered_weights),
        'n_epochs': int(n_epochs),
        'lr': float(lr),
    }
