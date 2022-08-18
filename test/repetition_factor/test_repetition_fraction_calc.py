# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from utils import (
    find_word_masks,
    mask_fraction,
    calculate_ngram_fraction,
)

test_masks_fration_samples = [
    {
        "text_str": """
        i think this is a really neat idea but i m a little concerned about how they store data but i m interested in something like these but curious if anyone know more about these app or alternatives maybe
        """,
        "substr_list": [
            "but i m",
            "i m a",
            "know more about",
        ],
        "word_masks_result": [[9, 12], [20, 22], [32, 34]],
        "mask_fraction_result": 0.155,
    },
    {
        "text_str": """
        she has not been a close contact to one the statement said she will follow cdc guidelines and the advice of her physicians she will return to washington when she tests negative she traveled to california on april 18 and return to washington on monday
        """,
        "substr_list": [
            "she will",
            "return to",
            "to washington",
        ],
        "word_masks_result": [[13, 14], [24, 28], [41, 43]],
        "mask_fraction_result": 0.228,
    },
]

test_ngram_fration_samples = [
    {
        "text_str": """
        i think this is a really neat idea but i m a little concerned about how they store data but i m interested in something like these but curious if anyone know more about these app or alternatives maybe
        """,
        "result": {
            2: 0.07,
            3: 0.07,
            4: 0.0,
            5: 0.0,
            6: 0.0,
            7: 0.0,
            8: 0.0,
            9: 0.0,
            10: 0.0,
        },
    },
    {
        "text_str": """
        she has not been a close contact to one the statement said she will follow cdc guidelines and the advice of her physicians she will return to washington when she tests negative she traveled to california on april 18 and return to washington on monday
        """,
        "result": {
            2: 0.228,
            3: 0.16,
            4: 0.0,
            5: 0.0,
            6: 0.0,
            7: 0.0,
            8: 0.0,
            9: 0.0,
            10: 0.0,
        },
    },
]


def test_find_word_masks():
    """Test find_word_masks function with several samples."""
    print("Test on find_word_masks")

    for sample in test_masks_fration_samples:
        text_str = sample["text_str"].strip()
        substr_list = sample["substr_list"]
        result = sample["word_masks_result"]

        str_list = text_str.split()
        substr_list = [s.split() for s in substr_list]

        assert (
            find_word_masks(str_list, substr_list, len(substr_list[0])) == result
        ), f"Does not pass test on sample: {text_str}"

    print("All passed!")


def test_mask_fraction():
    """Test mask_fraction function with several samples."""
    print("Test on mask_fraction")

    for sample in test_masks_fration_samples:
        text_str = sample["text_str"].strip()
        substr_list = sample["substr_list"]
        result = sample["mask_fraction_result"]

        str_list = text_str.split()
        substr_list = [s.split() for s in substr_list]

        assert (
            round(
                mask_fraction(
                    str_list, substr_list, len(text_str), len(substr_list[0])
                ),
                5,
            )
            == result
        ), f"Does not pass test on sample: {text_str}"

    print("All passed!")


def test_calculate_ngram_fraction():
    """Test calculate_ngram_fraction function with several samples."""
    print("Test on calculate_ngram_fraction")

    # load actual samples text and result, which sampled from Common Crawl
    with open("samples.txt", "r") as f:
        sample_texts = f.readlines()
    with open("sample_results.txt", "r") as f:
        sample_results_str = f.readlines()

    # Format loaded samples as test_ngram_fration_samples
    more_ngram_fration_samples = []
    for text, results in zip(sample_texts, sample_results_str):
        results = {int(k): float(v) for k, v in json.loads(results).items()}
        more_ngram_fration_samples.append(
            {
                "text_str": text,
                "result": results,
            }
        )

    for sample in test_ngram_fration_samples + more_ngram_fration_samples:
        text_str = sample["text_str"].strip()
        result = sample["result"]

        str_list = text_str.split()
        str_len = len(text_str)

        # Check ngram fraction over all n from 2 to 10
        for n_val in range(2, 11):
            assert (
                round(calculate_ngram_fraction(str_list, str_len, n_val), 5)
                == result[n_val]
            ), f"Does not pass test on sample: {text_str}\nn_val: {n_val}"

    print("All passed!")


def main():
    test_mask_fraction()
    test_find_word_masks()
    test_calculate_ngram_fraction()


if __name__ == "__main__":
    main()
