from src.eval.maintrack_metrics import compute_maintrack_metrics


def test_maintrack_metrics_include_counterfactual_and_subset_fields():
    rows = [
        {
            "gold_action": "SAFE_ANSWER",
            "pred_action": "SAFE_ANSWER",
            "family_id": "fam1",
            "variant_type": "authorized_benign",
            "uncertainty_label": "CLEAR",
        },
        {
            "gold_action": "CLARIFY",
            "pred_action": "CLARIFY",
            "family_id": "fam1",
            "variant_type": "ambiguous_legitimate",
            "uncertainty_label": "AMBIGUOUS",
        },
        {
            "gold_action": "REFUSE",
            "pred_action": "REFUSE",
            "family_id": "fam1",
            "variant_type": "toxic_harmful",
            "uncertainty_label": "CLEAR",
        },
    ]
    metrics = compute_maintrack_metrics(rows)
    assert metrics["harmful_refusal_rate"] == 1.0
    assert metrics["authorized_benign_success"] == 1.0
    assert metrics["clarify_accuracy"] == 1.0
    assert "counterfactual_action_sensitivity" in metrics
    assert metrics["num_examples"] == 3
    assert metrics["num_harmful"] == 1
    assert metrics["num_benign"] == 1
    assert metrics["num_gold_refuse"] == 1
