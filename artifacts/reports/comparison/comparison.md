# Model comparison (auto-generated from evaluation outputs)

Test examples: 874

| Model | Accuracy (95% CI) | Macro-P | Macro-R | Macro-F1 (95% CI) | Weighted-F1 | F1 positive | F1 neutral | F1 negative | Train time (min) | Latency median (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| Text-only (XLM-R) | 0.6899 [0.659, 0.722] | 0.5418 | 0.5539 | 0.5449 [0.504, 0.589] | 0.6942 | 0.6365 | 0.2261 | 0.7722 | 4.4 | 16.8745 (mps) |
| Image-only (ResNet-50) | 0.6739 [0.641, 0.705] | 0.5200 | 0.5333 | 0.5251 [0.485, 0.563] | 0.6793 | 0.6237 | 0.1935 | 0.7581 | 4.0 | 11.1962 (mps) |
| Multimodal: concat | 0.7403 [0.711, 0.769] | 0.5775 | 0.5552 | 0.5640 [0.524, 0.605] | 0.7329 | 0.6731 | 0.2000 | 0.8187 | 5.8 | 19.4100 (mps) |
| Multimodal: cross-attention | 0.7449 [0.716, 0.776] | 0.6073 | 0.6123 | 0.6061 [0.564, 0.645] | 0.7477 | 0.6840 | 0.3088 | 0.8255 | 5.8 | 25.3443 (mps) |
| Baseline: majority class | 0.6259 [0.595, 0.659] | 0.2086 | 0.3333 | 0.2566 [0.249, 0.265] | 0.4818 | 0.0000 | 0.0000 | 0.7699 | – | – |
| Baseline: source-page prior* | 0.7826 [0.756, 0.810] | 0.5261 | 0.5295 | 0.5236 [0.503, 0.543] | 0.7516 | 0.7276 | 0.0000 | 0.8431 | – | – |

\* The source-page prior uses the meme's source page (metadata, not content) and is a shortcut reference, not a competing model.

## Paired bootstrap: macro-F1 difference (A − B)

| Comparison | Observed Δ | 95% CI | P(A not better) |
|---|---|---|---|
| multimodal_concat vs text | +0.0190 | [-0.030, 0.068] | 0.244 |
| multimodal_concat vs image | +0.0388 | [0.001, 0.082] | 0.022 |
| multimodal_attention vs text | +0.0612 | [0.018, 0.106] | 0.006 |
| multimodal_attention vs image | +0.0810 | [0.037, 0.124] | 0.000 |
| text vs baseline_source_prior | +0.0214 | [-0.021, 0.067] | 0.159 |
| image vs baseline_source_prior | +0.0015 | [-0.041, 0.039] | 0.494 |
| multimodal_concat vs baseline_source_prior | +0.0404 | [0.004, 0.080] | 0.015 |
| multimodal_attention vs baseline_source_prior | +0.0825 | [0.045, 0.122] | 0.000 |
| multimodal_attention vs multimodal_concat | +0.0422 | [-0.000, 0.083] | 0.029 |

## Modality ablation of the multimodal models

| Model | Condition | Accuracy | Macro-F1 | Predictions changed |
|---|---|---|---|---|
| Multimodal: concat | image_neutralised | 0.6396 | 0.4819 | 26.9% |
| Multimodal: concat | text_removed | 0.7277 | 0.4697 | 14.0% |
| Multimodal: cross-attention | image_neutralised | 0.6396 | 0.2953 | 33.3% |
| Multimodal: cross-attention | text_removed | 0.6693 | 0.3618 | 28.7% |
