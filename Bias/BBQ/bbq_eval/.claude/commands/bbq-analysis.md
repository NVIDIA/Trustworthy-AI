# BBQ result analysis

Use only scores and model metadata supplied for the current analysis. Do not infer
private model results or use hardcoded competitive reference bands.

Filter to ambiguous context when requested. Compare a candidate to a baseline
with matching dataset, prompt, parser, model size and quantization, and disclose
any mismatches. `acc_bias` is a signed score: examine magnitude and direction;
a positive value alone does not establish a better model. A mean can hide
opposing category effects, so present per-category results as well.

The CLI writes category results to `results/bbq_metrics.csv` and a pooled score
to `results/bbq_metrics_overall.csv`. The pooled score is computed from records;
it is not the unweighted mean of category scores or a mean across models.

Use user-provided model
results to compute reference bands and state their membership and sample sizes.
