"""The organisers' metric (evaluation/possession_metrics.py) on a label array, as the training
scripts in src/training/ score the external matches with it."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402  (organisers' metric code, the official definitions)


def score(pred_labels, gt):
    """Official scores against a truth file path, or a truth labels array."""
    gt_codes = pm.label_codes(pm.read_label_csv(gt) if isinstance(gt, str) else gt)
    result = pm.evaluate(gt_codes, pm.label_codes(pred_labels), bootstrap_n=0)
    return {"spellF1": result["spell_f1_mean"], "macroF1": result["macro_f1"],
            "true_poss": result["n_gt_spells"], "pred_poss": result["n_pred_spells"]}
