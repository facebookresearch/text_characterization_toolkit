# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import collections
import os
import sys
import pandas as pd
from pathlib import Path
from statistics import mean, stdev

import numpy as np


# If we were to compute mean or stdev but don't have enough sentences,
# just return None so that we can filter that out if needed


def mean_or_none(v):
    v = list(v)
    return mean(v) if len(v) > 0 else None


def safe_stdev(v):
    v = list(v)
    return stdev(v) if len(v) > 1 else None


def safe_div(x, y):
    return x / y if y != 0 else 0


# Simple DP implementation of edit distance.
# TODO(danielsimig) upgrade this to some C library if too slow
def levenshtein_distance(a, b):
    m = len(a)
    n = len(b)

    d = np.ndarray(shape=(m + 1, n + 1))

    for i in range(m + 1):
        d[i][0] = i
    for j in range(n + 1):
        d[0][j] = j

    for j in range(1, n + 1):
        for i in range(1, m + 1):
            d[i][j] = np.min(
                [
                    d[i - 1][j] + 1,  # deletion
                    d[i][j - 1] + 1,  # insertion
                    d[i - 1][j - 1] + 1 - int(a[i - 1] == b[j - 1]),
                ]
            )

    return d[m][n] * 1.0 / max(n, m)


def parse_filename_macros(filename):
    base_dir = Path(os.path.join(os.getcwd(), sys.argv[0])).parent.parent
    return (
        filename
        .replace("<DATA_DIR>", os.path.join(base_dir, "data"))
        .replace("<CONFIG_DIR>", os.path.join(base_dir, "configs"))
    )


def find_ngrams(input_list, n_val):
    """Generate all ngrams given value n.

    Args:
        input_list: list of words, representing the original text
        n_val: value n of ngram

    Return:
        list of ngrams, ngram is represented as list of n words
    """
    return zip(*[input_list[i:] for i in range(n_val)])


def find_word_masks(str_list, ngram_list, n_val):
    """Find masks by searching at word level.

    Args:
        str_list: list of words, representing the original text
        ngram_list: list of ngram, ngram is represented as list of n words
        n_val: value n of ngram

    Return:
        masks: list of [start, end], start / end position (index starting from 1) of continuous words

    Steps:
        1. for all ngrams in ngram_list, find all extract matches in str_list,
            record word-level position (start, end), starting from 1 instead of 0.
        2. merge overlap positions into single position.
    Example:
        str_list = ['a', 'b', 'c', 'd', 'a', 'b']
        ngram_list = [['a', 'b'], ['b', 'c']]
        n_val = 2
        Result of step 1: [(1,2), (5,6), (2,3)]
        Result of step 2: [(1,3), (5,6)]
    """
    matched_list = []

    # Step 1: for all ngrams in ngram_list, find all extract matches in str_list
    for ngram_word_list in ngram_list:
        n_ngram = n_val
        # find all matches on first word
        word_inds = [i for i, x in enumerate(str_list) if x == ngram_word_list[0]]
        if word_inds:
            for ind in word_inds:
                # check if there are enough remaining words on text
                if ind + n_ngram <= len(str_list):
                    if all(
                        [
                            ngram_word_list[i] == str_list[ind + i]
                            for i in range(1, n_ngram)
                        ]
                    ):
                        matched_list.append((ind + 1, ind + n_ngram))

    # Step 2: merge overlap positions into single position
    # with the sorted matched_list, we don't need to compare new (begin, end) with all elements in existing matched_list,
    # instead, only need to check on the last element, which reduce complexity from O(n^2) to O(nlog(n)) + O(n).
    masks = []
    for begin, end in sorted(matched_list):
        if masks and masks[-1][1] >= begin - 1:
            masks[-1][1] = max(masks[-1][1], end)
        else:
            masks.append([begin, end])

    return masks


def mask_fraction(str_list, substr_list, str_len, n_val):
    """Find masks by searching at word level.

    Args:
        str_list: original list of token from text
        substr_list: list of substr (ngram)
        str_len: length of original string, which is reperented by str_list
        n_val: value n of ngram

    Return:
        fraction: fraction of all substr on str_list
    """
    word_masks = find_word_masks(str_list, substr_list, n_val)
    masks_len = 0
    for mask in word_masks:
        # add up character length of all words
        masks_len += sum([len(str_list[i]) for i in range(mask[0] - 1, mask[1])])
        # add up white spaces
        masks_len += mask[1] - mask[0]
    fraction = 1.0 * masks_len / str_len
    return fraction


def calculate_ngram_fraction(str_list, str_len, n_val, mode=None):
    """Calcualte repetition ngram fraction.
    - For each n in (2, …, 4), we calculate the fraction of characters contained within the most frequently-occurring n-gram;
    - For each n in (5, …, 10), we calculate the fraction of characters contained within all duplicate n-grams.

    Args:
        str_list: original list of token from text
        str_len: length of original string, which is reperented by str_list
        n_val: value n of ngram
        mode: {any|top} calculate fraction of top-occurance ngram or any duplicated ngram.

    Return:
        fraction: fraction of ngrams on given string
    """
    ngram_count = collections.Counter(find_ngrams(str_list, n_val))

    top_ngram = []

    if (mode == "any") or (mode is None and n_val >= 5):
        top_ngram = [k for k, v in ngram_count.items() if v > 1]
    elif (mode == "top") or (mode is None and n_val < 5):
        try:
            item_with_max_value = max(ngram_count.items(), key=lambda x: x[1])

            for k, v in ngram_count.items():
                if v == item_with_max_value[1] and v > 1:
                    top_ngram.append(k)
        except ValueError as e:
            return 0.0
    else:
        raise Exception(
            "Please set mode as 'any' or 'top' or None to run default setting"
        )

    try:
        if top_ngram:
            fraction = mask_fraction(str_list, top_ngram, str_len, n_val)
            return fraction
        else:
            return 0.0

    except Exception as e:
        return 0.0


def calculate_lp_fraction(str_list, mode=None):
    """Calcualte repetition line / paragraph fraction.
    - mode=cnt: count of duplicate lines / count of all lines
    - mode=len: character length of duplicate lines / character length of all lines

    Args:
        str_list: original list of line / paragraph from text
        mode: {len|cnt}

    Return:
        fraction: fraction of line / paragraph
    """
    if len(str_list) <= 1:
        return 0.0

    str_count = collections.Counter(str_list)

    repetition_count = {k: v for k, v in str_count.items() if v > 1}

    if mode == "len":
        repetition_len = sum(
            [len(entity) * count for entity, count in repetition_count.items()]
        )
        total_len = sum(
            [len(entity) * count for entity, count in str_count.items()]
        ) + (len(str_list) - 1)
        fraction = 1.0 * repetition_len / total_len
    elif mode == "cnt":
        fraction = 1.0 * sum(repetition_count.values()) / sum(str_count.values())
    else:
        raise Exception("Please set mode as 'len' or 'cnt'")

    return fraction

# Load the result of metric compuation indexed by id - ready for downstream analysis
# If a column has more than max_nan_ratio NaN values, drop it. Otherwise fill the NaN values
# with fill_na_val or the mean of the other values if fill_na_val is not specified.
def load_text_metrics(path, max_nan_ratio=0.2, text_keys=None, fill_na_val=None):
    metrics_df = pd.read_csv(path, sep="\t")
    if text_keys is not None:
        metrics_df = metrics_df[metrics_df["text_key"].isin(text_keys)]
    metrics_df.set_index("id", inplace = True)

    cols_to_drop = []
    for col in metrics_df.columns:
        if col in {"id", "text_key"}:
            continue
        nan_values = metrics_df[col].isna()
        pct_nan = sum(nan_values) * 1.0 / len(nan_values)
        if pct_nan > max_nan_ratio:
            print(
                f"Dropping column {col}!",
                f"It contains {pct_nan*100:.2f}% (>{max_nan_ratio*100:.2f}%) NaN values!"
            )
            cols_to_drop.append(col)
        else:
            non_na_mean = metrics_df[~nan_values][col].mean()
            fill_value = fill_na_val if fill_na_val is not None else non_na_mean
            metrics_df[col].fillna(value=fill_value, inplace=True)

    if len(cols_to_drop) > 0:
        metrics_df.drop(columns=cols_to_drop, inplace=True)

    return metrics_df