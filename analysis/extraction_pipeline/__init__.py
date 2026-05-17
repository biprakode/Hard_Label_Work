"""
extraction_pipeline
===================

Phase-3 reconstruction pipeline split into named modules. Cosmetic refactor of
the original `analysis/test_extraction4.py` — same behaviour, same CLI args,
same outputs, just laid out as a package.

Module map
----------
config                   Paths, LEAKY_ALPHA toggle, _act/_act_suffix helpers.
architectures            TinyModel, TinierModel, TiniestModel, FullModel.
data_loading             X_test / X_test2 loaders + ground-truth model loader.
metrics                  Three-tier weight metrics, accuracy testing.
weight_assembly          Build a model from signature + sign recovery outputs
                         (load_unsigned_weights, combine_weights_and_signs,
                         reconstruct_model, save_reconstructed_model).
bias_recovery            Recover biases from dual points using the rebuilt prefix.
output_layer_recovery    Fit fc5 via multinomial LR on hard-label oracle queries.
sign_search              Brute-force / greedy oracle sign search.
refinement               Oracle-label refinement (frozen-rows distillation).
workflow                 main() — the end-to-end orchestration.
"""
