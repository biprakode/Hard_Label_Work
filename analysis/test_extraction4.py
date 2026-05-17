"""
Thin shim — kept for backward compatibility.

The Phase-3 reconstruction logic now lives in the `extraction_pipeline/`
package. This file simply forwards to `extraction_pipeline.workflow.main`,
so any existing invocation
    python3 analysis/test_extraction4.py [--tiniest | --tinier | ...]
continues to work unchanged.

For new work, prefer
    python3 analysis/run_extraction.py ...

Re-exports of the most commonly imported names are provided below so
external callers that did `from test_extraction4 import TiniestModel,
oracle_sign_search, ...` keep functioning.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Re-exports from the modular package (preserve the historical import surface).
from extraction_pipeline.config import (  # noqa: F401
    BASE_DIR, LEAKY_ALPHA, _act, _act_suffix,
    SIGNATURE_WEIGHTS_PATH, SIGN_RECOVERY_PATH, OUTPUT_PATH, DUAL_POINTS_DIR,
    TINY_MODEL_PTH, TINY_MODEL_KERAS, MAKEBLOBS_MODEL_PTH,
    TINIER_MODEL_PTH, TINIEST_MODEL_PTH, FULL_MODEL_PTH,
    X_TEST_PATH, X_TEST_MAKEBLOBS_PATH, Y_TEST_MAKEBLOBS_PATH,
    X_TEST_TINIER_PATH, Y_TEST_TINIER_PATH,
    X_TEST_TINIEST_PATH, Y_TEST_TINIEST_PATH,
    X_TEST2_TINIEST_PATH, Y_TEST2_TINIEST_PATH,
    X_TEST2_MAKEBLOBS_PATH, Y_TEST2_MAKEBLOBS_PATH,
)
from extraction_pipeline.architectures import (  # noqa: F401
    TinyModel, TinierModel, TiniestModel, FullModel,
)
from extraction_pipeline.data_loading import (  # noqa: F401
    load_test_data, load_test2_data, load_ground_truth_model,
)
from extraction_pipeline.metrics import (  # noqa: F401
    compute_weight_metrics_v2, test_model_accuracy,
)
from extraction_pipeline.weight_assembly import (  # noqa: F401
    load_unsigned_weights, load_signs, combine_weights_and_signs,
    _kaiming_init, reconstruct_model, save_reconstructed_model,
)
from extraction_pipeline.bias_recovery import (  # noqa: F401
    _hidden_activations_up_to, recover_biases_from_duals,
)
from extraction_pipeline.output_layer_recovery import (  # noqa: F401
    recover_output_layer,
)
from extraction_pipeline.sign_search import (  # noqa: F401
    oracle_sign_search, greedy_oracle_sign_search,
    _run_one_pass, _greedy_sign_pass_layer,
)
from extraction_pipeline.refinement import (  # noqa: F401
    oracle_label_refinement,
)
from extraction_pipeline.workflow import main  # noqa: F401


if __name__ == '__main__':
    main()
