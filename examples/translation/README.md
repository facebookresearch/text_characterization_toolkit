# Translation Experiments


## Data Generation

Update the config.yaml to the languages and models you wish to compare, then run:

```bash
python examples/translation/generate_data_and_metrics.py

python tools/compute.py -i  examples/translation/translation_text_features.jsonl -o  examples/translation/translation_text_characteristics.tsv
```

## Analysis

Use the Demo.ipynb notebook to analyze the data. Note Hydra creates log directories based on time for the generated data so just update the paths in the notebook to the generated data.

