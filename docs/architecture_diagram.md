# QEFSC-FC Architecture Diagram

This is the upgraded architecture used by the current main-track implementation.

```text
Input prompt
   |
   v
Decoder LM backbone
Qwen 2.5 3B Instruct / Llama 3.1 8B Instruct
   |
   +-------------------------------+
   | hidden state from layer -12   |
   | hidden state from layer -8    |
   | hidden state from layer -4    |
   | hidden state from layer -1    |
   +-------------------------------+
   |
   v
Masked last-token pooling per selected layer
   |
   v
Multi-layer feature fusion
gated weighted sum / concat / mean
   |
   v
Shared safety representation z
   |
   +----------------------+----------------------+----------------------+
   |                      |                      |                      |
   v                      v                      v                      v
Harm head            Legitimacy head       Uncertainty head       Action head
LOW/MED/HIGH         LOW/MED/HIGH          CLEAR/AMBIGUOUS        ANSWER
                                                                 SAFE_ANSWER
                                                                 CLARIFY
                                                                 REFUSE
   |                      |                      |
   +---------- factor probabilities ------------+
                         |
                         v
          concatenate [z, p(harm), p(legit), p(uncertainty)]
                         |
                         v
                 conditioned action MLP
                         |
                         v
          {ANSWER, SAFE_ANSWER, CLARIFY, REFUSE}
```

## Deployment Path

```text
Trained QEFSC-FC model
   |
   v
Load decoder backbone in target precision
FP16 / BF16 / INT8 / INT4 NF4
   |
   v
Load trained LoRA adapters when enabled
   |
   v
Load controller and fusion weights
   |
   +------------------------------+
   | optional post-quantization   |
   | controller recalibration     |
   | backbone frozen              |
   +------------------------------+
   |
   v
Quantized inference
   |
   v
Main-track metrics + retention metrics
```

## Compared Methods

```text
Direct Policy baseline
final hidden state -> action head

Plain EFSC baseline
final hidden state -> bottleneck -> factor heads -> conditioned action head

QEFSC-FC proposed method
multi-layer hidden states -> feature fusion -> factor heads -> conditioned action head
```

## Implementation Map

- Proposed FP model: `src/models/decoder_qefsc_fc.py`
- Proposed quantized model: `src/models/decoder_qefsc_fc_quantized.py`
- Fusion modules: `src/models/feature_fusion.py`
- Plain EFSC baseline: `src/models/plain_efsc_decoder.py`
- Direct Policy baseline: `src/models/direct_policy_decoder.py`
- Quantized baselines: `src/models/plain_efsc_decoder_quantized.py`, `src/models/direct_policy_decoder_quantized.py`
- End-to-end QEFSC runner: `run_qefsc_quant_experiment.py`
- End-to-end baseline runner: `run_baseline_quant_experiment.py`
