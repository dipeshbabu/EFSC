from src.eval.metrics import compute_all_metrics


def test_compute_all_metrics_on_action_strings():
    rows = [
        {"gold_action": "ANSWER", "pred_action": "ANSWER", "family_id": "f1"},
        {"gold_action": "SAFE_ANSWER", "pred_action": "REFUSE", "family_id": "f1"},
        {"gold_action": "REFUSE", "pred_action": "REFUSE", "family_id": "f2"},
        {"gold_action": "CLARIFY", "pred_action": "ANSWER", "family_id": "f2"},
    ]
    metrics = compute_all_metrics(rows)
    assert metrics["accuracy"] == 0.5
    assert metrics["over_refusal_rate"] == 0.5
    assert metrics["harmful_refusal_rate"] == 1.0
