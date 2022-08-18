# Contributing to TCT
Please consider contriuting to this repository so that other members of the community can reuse ideas from your work. We aim to make contributing to this project as easy and transparent as possible.

## How you can contribute
Following the structure of this toolkit, we are looking for the following contributions:

### Datasets

The ```/tools/text_extraction``` directory contains scripts that convert publicly available datasets into the standard format of this toolkit. By adding converters to this folder we can expand the number of datasets that one can analyze off-the-shelf.

### Metrics

The main purpose of this repository is to collect all metrics one might want to look at when analyzing text data. If you found some metric valuable for analyzing your data or models, please consider adding a new ```MetricCollection``` to ```/lib/metrics.py``` so that others can use it too in the future. In case your metric relies on additional resources (e.g. word databases), make sure to add code to ```/scripts/download_resources.sh``` to so that other users can also download thet data.

### Analysis tools

If your work involves some data analysis that you think might be applicable to other datasets/models/metrics in the future, consider packaging it up into a standalone method and add it to ```/lib/analysis.py```. Thanks to the standardized data representation in the framework, others will be able to run the same analysis with a couple of lines of code in a notebook.

### Demos
Did you successfully use this toolkit go gain insights about your data or model? Please consider adding a new directory to ```/examples/``` with a notebook and any additional files to showcase your work. 


## Pull Requests
We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. If you haven't already, complete the Contributor License Agreement ("CLA").

## Contributor License Agreement ("CLA")
In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Meta's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Issues
We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

Meta has a [bounty program](https://www.facebook.com/whitehat/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.

## Testing
We currently have a very basic system in place to make sure changes don't break existing implementations of text characteristic. If you changed any code in the library, please run the following script and report any warnings in your PR:
```
test/check_for_metric_changes.sh
```

## License
By contributing to TCT, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.