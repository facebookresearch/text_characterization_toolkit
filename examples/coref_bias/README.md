## Bias Feature Analysis using [WinoBias dataset](https://uclanlp.github.io/corefBias/overview)
This example illustrates how to add a new feature related to gender bias and conduct analysis using our data characterization tool.

1) Following the [first step](https://github.com/fairinternal/data_characterization/tree/bias_analysis#1-extract-text-from-the-dataset), we add a text extraction script for extracting occupations and pronouns from each example, run the following command to extract:

```
python tools/text_extraction/winobias.py <PATH_TO_YOUR_WinoBias_DATA> examples/coref_bias/winobias_text_anti_features.jsonl
```
2)

Since we want to explore the correlation between gender bias and model predictions, we only focus on the gender bias feature. To do that, we add a new config file which provides a function to look up gender scores for occupations and pronouns. We then use this config file to compute gdner bias features as follows:
```
python ./tools/compute.py -i examples/coref_bias/winobias_text_features_anti.jsonl -o ./examples/coref_bias/winobias_text_features_anti.jsonl.tsv.tmp --config ./configs/winobias.json
```
3)We then obtain model predictions by running a [end-to-end](https://github.com/kentonl/e2e-coref) coreference resolution model on WinoBias data. And use the absolute gender difference of occupations and pronouns as gender bias and find that larger difference leads to worse performance. More details can be found in the Demo notebook or our paper.
