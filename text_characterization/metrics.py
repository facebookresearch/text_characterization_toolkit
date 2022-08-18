# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import time
from collections import defaultdict
from statistics import mean

import numpy as np
import pandas as pd
from hyphenator import Hyphenator
from lexical_diversity.lex_div import (
    ttr,
    hdd,
    mtld,
)
from nltk.corpus import wordnet as wn

from .parser_backends import BasicSpacyBackend
from .utils import (
    mean_or_none,
    safe_div,
    safe_stdev,
    levenshtein_distance,
    parse_filename_macros,
    calculate_ngram_fraction,
    calculate_lp_fraction,
)


class MetricCollection:

    METRIC_REGISTRY = {}

    def __init__(self, config):
        self.config = config

    # Black magic for creating a registry of metrics automatically
    # This helps with initializing metrics from the config file easily
    def __init_subclass__(cls, /, **kwargs):
        super().__init_subclass__(**kwargs)
        super(cls, cls).METRIC_REGISTRY[cls.__name__] = cls

    @classmethod
    def from_config(cls, config):
        return cls.METRIC_REGISTRY[config["class"]](config.get("config"))

    # Add this to the constructor of the metric collection if the metric collection
    # will look for a specific field in its config.
    def require_config(self, name):
        assert name in self.config, f"{self.id} needs {name} config"

    # Add any resource intensive initialisation here
    def load_resources(self):
        pass

    # Return a list of <metric_category, metric_key, metric_description> tuples describing 
    # each metric the class is expected to return. This is used to pre-compile a header in 
    # batch processing mode as well as to produce automatic documentation for metrics.
    def get_metric_descriptions(self):
        raise NotImplementedError()

    # Return a list of all metric keys we expect to see from this collection.
    def get_all_metric_keys(self):
        return [x[1] for x in self.get_metric_descriptions()]

    # The implementation of the metric collection. Return a dict of metric keys
    # matching what's returned from get_all_metric_keys() an metric values (scalars)
    def compute(self, text, upstream_metrics, backend_results):
        raise NotImplementedError()


# Main class to compute all metrics defined in a config file
class MetricProcessor:
    def __init__(self, config, backend=None):
        super().__init__()

        self.metric_collections = [
            MetricCollection.from_config(metric_config) for metric_config in config
        ]
        self.backend = backend
        self.all_metrics = sum(
            [m.get_all_metric_keys() for m in self.metric_collections], []
        )

        # Make sure different metric collections all return unique keys
        assert len(set(self.all_metrics)) == len(self.all_metrics), str(self.all_metrics)


    def get_metric_descriptions(self):
        return sum([m.get_metric_descriptions() for m in self.metric_collections], [])

    # All the resource intensive stuff gets done here, call it only once
    def load_resources(self):
        for metric_collection in self.metric_collections:
            metric_collection.load_resources()

    # Call this once per text, relatively fast. Only runs backend pipeline once.
    def compute_metrics(self, text, return_list=False, log_time=False, allow_none=True):

        start_time = time.time()
        backend_results = self.backend(text)
        if log_time:
            print(f"SpaCy backend took { time.time() - start_time:.4f}s")

        metrics = {}
        if log_time:
            print("Time spent per metric class:")
        for metric_collection in self.metric_collections:
            start_time = time.time()
            metric_dict = metric_collection.compute(text, metrics, backend_results)
            if log_time:
                print(f"  {type(metric_collection).__name__}: {time.time() - start_time:.4f}s")
            metrics.update(metric_dict)

        # Certain use cases don't like None / NULL values, replace those with 0s for now.
        if not allow_none:
            metrics = {k: 0 if v is None else v for k, v in metrics.items()}

        if return_list:
            # For batch processing just return the numbers in a fixed order
            return [metrics.get(metric, None) for metric in self.all_metrics]
        else:
            return metrics


#####
# Each of the  following classes implement one section from the Coh-Metric suite.
# For documentation see https://fburl.com/0hfzh304 and the Coh-Metrix book.
#####


class DescriptiveIndices(MetricCollection):
    """
    Basic metrics about word / sentence / paragraph lengths and counts.
    If "hyphenator_dict" parameter is specified in the config, it'll also compute syllable-based stats.
    """

    def __init__(self, config):
        super().__init__(config)
        self.hyphenator = None

    METRICS = [
        ("DESPC", "Number of paragraphs"),
        ("DESSC", "Number of sentences"),
        ("DESWC", "Number of words"),
        ("DESPL", "Average number of sentences per paragraph"),
        ("DESPLd", "Standard deviation of paragraph lengths (in sentences)"),
        ("DESPLw", "Average number of words per paragraph"),
        ("DESSL", "Average number of words per sentence)"),
        ("DESSLd", "Standard deviation of sentence lengths (in words)"),
        ("DESWLsy", "Average word length (syllables)"),
        ("DESWLsyd", "Standard Deviation of word lengths (in syllables)"),
        ("DESWLlt", "Average word length (letters)"),
        ("DESWLltd", "Standard Deviation of word lengths (in letters)"),
    ]

    def get_metric_descriptions(self):
        return [("Descriptive", key, desc) for key, desc in self.METRICS]

    def load_resources(self):

        hyphenator_dict_file = self.config.get("hyphenator_dict", None)
        if hyphenator_dict_file is not None:
            self.hyphenator = Hyphenator(parse_filename_macros(hyphenator_dict_file))
        else:
            self.hyphenator = None

    def compute(self, text, upstream_metrics, backend_results):

        sentence_lenghts = backend_results.get_sentence_lengths()
        num_sentences_per_paragraph = backend_results.num_sentences_per_paragraph()
        num_words_per_paragraph = backend_results.num_words_per_paragraph()

        words = backend_results.get_words()
        num_letters_per_word = [len(word) for word in words]

        if self.hyphenator:
            num_syllables_per_word = [
                len(self.hyphenator.inserted(word).split("-")) for word in words
            ]
        else:
            num_syllables_per_word = None

        return {
            "DESPC": len(text.split("\n")),
            "DESSC": len(backend_results.get_sentences()),
            "DESWC": len(words),
            "DESPL": mean_or_none(num_sentences_per_paragraph),
            "DESPLd": safe_stdev(num_sentences_per_paragraph),
            "DESPLw": mean_or_none(num_words_per_paragraph),
            "DESSL": mean_or_none(sentence_lenghts),
            "DESSLd": safe_stdev(sentence_lenghts),
            "DESWLsy": mean_or_none(num_syllables_per_word)
            if num_syllables_per_word is not None
            else None,
            "DESWLsyd": safe_stdev(num_syllables_per_word)
            if num_syllables_per_word is not None
            else None,
            "DESWLlt": mean_or_none(num_letters_per_word),
            "DESWLltd": safe_stdev(num_letters_per_word),
        }


class LexicalDiversity(MetricCollection):

    METRICS = [
        ("LDTTRc", "Type-Token Ratio (TTR) computed over content words"),
        ("LDTTRa", "Type-Token Ratio (TTR) computed over all words"),
        ("LDMTLD", "Measure of Textual Lexical Diversity (MTLD)"),
        ("LDHDD", "HD-D lexical diversity index"),
    ]

    def get_metric_descriptions(self):
        return [
            ("Lexical Diversity", key, desc)
            for key, desc in self.METRICS
        ]

    def compute(self, text, upstream_metrics, backend_results):

        words = backend_results.get_words()
        content_words = backend_results.get_content_words()

        # Metrics based on https://link.springer.com/content/pdf/10.3758/BRM.42.2.381
        # VOCD has been replaced by HD-D as per the above paper.
        # Implementation from https://fburl.com/re1aa85e

        return {
            "LDTTRc": ttr(content_words),
            "LDTTRa": ttr(words),
            "LDMTLD": mtld(words),
            "LDHDD": hdd(words),
        }


class SyntacticComplexity(MetricCollection):

    METRICS = [
        ("SYNLE", "Left embeddedness: average words before main verb"),
        ("SYNNP", "Number of modifiers per noun phrase, mean"),
        ("SYNMEDpos", "Average edit distance between POS tags of consecutive sentences "),
        ("SYNMEDwrd", "Average edit distance between consecutive sentences"),
        ("SYNMEDlem", "Average edit distance between consecutive sentences (lemmatized)"),
        ("SYNSTRUTa", "Sentence syntax similarity, adjacent sentences, mean"),
        ("SYNSTRUTt", "Sentence syntax similarity, all combinations, mean"),
    ]

    def get_metric_descriptions(self):
        return [
            ("Syntactic Complexity", key, desc)
            for key, desc in self.METRICS
        ]

    def get_mean_pairwise_edit_distance(self, sentences):
        if len(sentences) < 2:
            return 0
        distances = []
        for i in range(1, len(sentences)):
            distances.append(levenshtein_distance(sentences[i], sentences[i - 1]))
        return mean_or_none(distances)

    def compute(self, text, upstream_metrics, backend_results):

        return {
            "SYNLE": backend_results.mean_words_before_main_verb(),
            "SYNNP": backend_results.mean_np_modifiers(),
            "SYNMEDpos": self.get_mean_pairwise_edit_distance(
                backend_results.get_sentence_pos_tags()
            ),
            "SYNMEDwrd": self.get_mean_pairwise_edit_distance(
                backend_results.get_sentence_words()
            ),
            "SYNMEDlem": self.get_mean_pairwise_edit_distance(
                backend_results.get_sentence_lemmas()
            ),
            "SYNSTRUTa": backend_results.mean_adjacent_syntatic_similarity(),
            "SYNSTRUTt": None,
        }


class Readability(MetricCollection):


    METRICS = [
        ("RDFRE", "Flesch Reading Ease"),
        ("READFKGL", "Flesch-Kincaid Grade Level"),
        # ("RDL2", "Coh-Metrix L2 Readability"),
    ]

    def get_metric_descriptions(self):
        return [("Readability", key, desc) for key, desc in self.METRICS]

    def compute(self, text, upstream_metrics, backend_results):
        asl = upstream_metrics["DESSL"]
        asw = upstream_metrics["DESWLsy"]

        if asl is None or asw is None:
            return {
                "RDFRE": None,
                "READFKGL": None,
                # "RDL2": None,
            }

        return {
            "RDFRE": 206.835 - 1.015 * asl - 84.6 * asw,
            "READFKGL": 0.39 * asl + 11.8 * asw - 15.59,
            # "RDL2": None,
        }


class NGramIncidenceScores(MetricCollection):
    """
    Compute the relative frequency of words sets defined by text files.
    Words are lowercased before counting.

    E.g. If the set of words is {"I", "you", "he", "she"} and the sentence is "I love you",
    the value of this metric will be 0.666
    """

    def __init__(self, config):
        super().__init__(config)
        self.word_set_keys = list(self.config["word_sets"].keys())
        # Map each word to a list of keys it belongs to
        self.word_set_map = None
        self.max_ngram_length = 0

    def get_metric_descriptions(self):
        return [
            ("Incidence Scores", f"WORD_SET_INCIDENCE_{key}", config['description'])
            for key, config in self.config["word_sets"].items()
        ]

    def normalize_string(self, s):
        return s.lower()

    def load_resources(self):

        self.word_set_map = defaultdict(list)

        for key, config in self.config["word_sets"].items():

            if "word_list_file" in config:

                # If a single string is given, we assume it's a filepath
                word_list = []
                with open(parse_filename_macros(config["word_list_file"])) as f:
                    for line in f:
                        word_list.append(line[:-1])

            elif "word_list" in config:
                # Alternatively, one can specify a list of words directly
                word_list = config["word_list"]

            else:
                raise Exception(f"Can't parse config for NGramIncidenceScores: {key}")

            for word in word_list:
                word = tuple(self.normalize_string(word).split(" "))
                self.word_set_map[word].append(key)
                if len(word) > self.max_ngram_length:
                    self.max_ngram_length = len(word)

    def compute(self, text, upstream_metrics, backend_results):
        words = [self.normalize_string(w) for w in backend_results.get_words()]
        num_words = len(words)
        if num_words == 0:
            return {set_key: 0.0 for set_key in self.word_set_keys}

        counts = defaultdict(int)
        for i in range(num_words):
            for j in range(i + 1, min(i + self.max_ngram_length + 1, num_words + 1)):
                for set_key in self.word_set_map.get(tuple(words[i:j]), []):
                    counts[set_key] += 1

        return {
            f"WORD_SET_INCIDENCE_{set_key}": safe_div(float(counts[set_key]), num_words)
            for set_key in self.word_set_keys
        }


class TokenAttributeRatios(MetricCollection):
    """
    Count the ratio of different type of tokens (words, numbers, symbols, urls, etc)
    High ratio of non-alphabetic tokens might indicate lower quality / noisy text.
    """

    def __init__(self, config):
        super().__init__(config)
        self.token_attributes = BasicSpacyBackend.TOKEN_ATTRIBUTES.values()

    def get_metric_descriptions(self):
        return [
            ("Incidence Scores", f"TOKEN_ATTRIBUTE_RATIO_{v[0]}", v[1])
            for v in self.token_attributes
        ]

    def compute(self, text, upstream_metrics, backend_results):
        token_attributes = backend_results.get_token_attributes()
        num_tokens = len(token_attributes)

        counts = defaultdict(int)
        for t in token_attributes:
            for attr in t:
                counts[attr] += 1

        return {
            f"TOKEN_ATTRIBUTE_RATIO_{attr}": safe_div(counts[attr] * 1.0, num_tokens)
            for attr, _ in self.token_attributes
        }


class WordProperty:
    # Base class for word properties that WordProperties metric collection can aggregate

    def __init__(self, config):
        self.config = config
        self.aggregation_method = config.get("aggregation_method", "mean")
        if "pos_filter" in config:
            self.pos_filter = set(config["pos_filter"].split(","))
        else:
            self.pos_filter = None

    def get_description(self):
        raise NotImplementedError

    def load_resources(self):
        pass

    def get_score(self, word, lemma, pos_tag):
        raise NotImplementedError

    def get_filtered_score(self, word, lemma, pos_tag):
        if self.pos_filter is not None and pos_tag not in self.pos_filter:
            return None
        return self.get_score(word, lemma, pos_tag)


class WordnetWordProperty(WordProperty):
    # Properties used by Coh-Metix based on WordNet

    METRIC_CATEGORY = "Word Property"

    def __init__(self, config):
        super().__init__(config)
        self.property = config["property"]
        self.description = config["description"]

    def get_description(self):
        return f"{self.aggregation_method.capitalize()} {self.description}"

    def hypernymy(self, word, pos_tag):
        synsets = []
        if pos_tag == "NOUN":
            synsets = wn.synsets(word, pos=wn.NOUN)
        elif pos_tag == "VERB":
            synsets = wn.synsets(word, pos=wn.VERB)
        return synsets[0].min_depth() if len(synsets) > 0 else None

    def polysemy(self, word):
        return len(wn.synsets(word))

    def get_score(self, word, lemma, pos_tag):
        if self.property == "hypernymy":
            score = self.hypernymy(word, pos_tag)
            if score is None:
                score = self.hypernymy(lemma, pos_tag)
            return score
        elif self.property == "polysemy":
            score = self.polysemy(word)
            if score == 0:
                score = self.polysemy(lemma)
            return score
        else:
            raise NotImplementedError(f"Unkown WordNet property {self.property}")


class POSWordProperty(WordProperty):

    METRIC_CATEGORY = "Incidence Scores"

    # Property that simply computes the incidence score of POS tags.
    # By overwriting get_filtered_score directly, the denominator becomes total number of words.
    def get_filtered_score(self, word, lemma, pos_tag):
        return float(pos_tag in self.pos_filter)

    def get_description(self):
        return f"Incidence score for POS tag {self.pos_filter}"


class LookupWordProperty(WordProperty):
    # Simple scalar lookup property. Load a word -> scalar mapping from one of the supported
    # file formats (csv, tsv, json) and apply this on the tokenized text

    METRIC_CATEGORY = "Word Property"

    def __init__(self, config):
        super().__init__(config)
        self.default_value = config.get("default_value", None)
        self.value_coalesce_method = config.get("value_coalesce_method", "any")
        self.description = config["description"]

    def get_description(self):
        return f"{self.aggregation_method.capitalize()} {self.description}"

    def load_from_csv(self, filename, sep):
        with open(filename) as f:
            df = pd.read_csv(
                f,
                sep=sep,
                usecols=[self.config["word_column"], self.config["property_column"]],
                converters={self.config["word_column"]: str},
            )

        return dict(
            zip(
                df[self.config["word_column"]],
                [
                    None if np.isnan(x) else x
                    for x in df[self.config["property_column"]]
                ],
            )
        )

    # If words appear repeatedly, make sure we're consistent with how we choose the value to use
    def coalesce_values(self, old, new):
        if old == None:
            return new
        if self.value_coalesce_method == "max":
            return max(old, new)
        elif self.value_coalesce_method == "min":
            return min(old, new)
        elif self.value_coalesce_method == "any":
            return old

    def load_resources(self):
        if "csv" in self.config:
            property_dict = self.load_from_csv(
                parse_filename_macros(self.config["csv"]), ","
            )
        elif "tsv" in self.config:
            property_dict = self.load_from_csv(
                parse_filename_macros(self.config["tsv"]), "\t"
            )
        elif "json" in self.config:
            with open(parse_filename_macros(self.config["json"])) as f:
                property_dict = json.load(f)
        else:
            raise NotImplementedError(
                f"Cannot parse word property config for {self.key}"
            )

        self.property_dict = {}
        for k, v in property_dict.items():
            self.property_dict[k] = self.coalesce_values(
                self.property_dict.get(k, None), v
            )

    def get_score(self, word, lemma, _):
        score = self.property_dict.get(
            word,
            self.property_dict.get(lemma, self.default_value))
        return score


class WordProperties(MetricCollection):
    # A generic metric class that looks up properties word by word then aggregates them as specified.

    key_prefix = "WORD_PROPERTY"

    aggr_method_map = {
        "mean": mean,
        "max": max,
        "min": min,
    }

    def __init__(self, config):
        super().__init__(config)

        self.need_lemmas = False
        self.word_properties = {}
        for key, word_property_config in config["word_property_configs"].items():
            if word_property_config["type"] == "lookup":
                word_property = LookupWordProperty(word_property_config)
                self.need_lemmas = True
            elif word_property_config["type"] == "wordnet":
                word_property = WordnetWordProperty(word_property_config)
                self.need_lemmas = True
            elif word_property_config["type"] == "pos_incidence":
                word_property = POSWordProperty(word_property_config)
            else:
                raise NotImplementedError(
                    f"Unknown word property type {word_property_config['type']}"
                )

            self.word_properties[key] = word_property

        self.aggr_methods = {
            k: p.aggregation_method for k, p in self.word_properties.items()
        }
        self.need_pos_tags = any(
            p.pos_filter is not None for p in self.word_properties.values()
        )

    def load_resources(self):
        for word_property in self.word_properties.values():
            word_property.load_resources()

    def get_metric_descriptions(self):
        return [
            (prop.METRIC_CATEGORY, f"{self.key_prefix}_{key}", prop.get_description())
            for key, prop in self.word_properties.items()
        ]

    def aggregate_values(self, key, values):

        if len(values) == 0:
            return None

        aggr_method = self.aggr_methods[key]
        assert (
            aggr_method in self.aggr_method_map
        ), f"Unknown aggregation method {aggregation_method}"
        return self.aggr_method_map[self.aggr_methods[key]](values)

    def compute(self, text, upstream_metrics, backend_results):
        words = backend_results.get_words()
        pos_tags = (
            backend_results.get_pos_tags()
            if self.need_pos_tags
            else [None] * len(words)
        )
        lemmas = (
            lem for snt in backend_results.get_sentence_lemmas() for lem in snt
        ) if self.need_lemmas else ([None] * len(words))

        property_values = {k: [] for k in self.word_properties.keys()}
        for word, lemma, pos_tag in zip(words, lemmas, pos_tags):
            for key, word_property in self.word_properties.items():
                score = word_property.get_filtered_score(word, lemma, pos_tag)
                if score is not None:
                    property_values[key].append(score)

        return {
            f"{self.key_prefix}_{key}": self.aggregate_values(key, values)
            for key, values in property_values.items()
        }


class RepetitionFractions(MetricCollection):

    key_prefix = "REPETITION_FRACTION"

    def __init__(self, config):
        super().__init__(config)
        self.ngrams = config.get("ngrams", range(2, 11)) if config else range(2, 11)
        assert min(self.ngrams) > 1, "n value of ngram is required to be greater than 1"
        assert len([type(n) == int for n in self.ngrams]) == len(self.ngrams), "ngrams is expected as list of int."
        assert len(set(self.ngrams)) == len(self.ngrams), "values in ngrams have to be unique."

    def get_all_metric_keys(self):
        metric_keys = [f"{self.key_prefix}_NGRAM_{i}_LEN" for i in self.ngrams] + [
            f"{self.key_prefix}_LINE_CNT",
            f"{self.key_prefix}_LINE_LEN",
            f"{self.key_prefix}_PARA_CNT",
            f"{self.key_prefix}_PARA_LEN",
        ]
        return metric_keys

    def compute(self, text, upstream_metrics, backend_results):
        # Calculate ngram repetition fractions
        word_list = [word.lower() for word in backend_results.get_words()]
        proxy_text_len = sum([len(w) for w in word_list]) + (len(word_list) - 1)

        metrics = {
            f"{self.key_prefix}_NGRAM_{i}_LEN": calculate_ngram_fraction(
                word_list, proxy_text_len, i
            )
            for i in self.ngrams
        }

        # Calculate line & paragraph repetition fractions
        line_list = backend_results.get_sentences_text()
        para_list = backend_results.get_paragraphs()

        metrics.update(
            {
                f"{self.key_prefix}_LINE_CNT": calculate_lp_fraction(line_list, "cnt"),
                f"{self.key_prefix}_LINE_LEN": calculate_lp_fraction(line_list, "len"),
                f"{self.key_prefix}_PARA_CNT": calculate_lp_fraction(para_list, "cnt"),
                f"{self.key_prefix}_PARA_LEN": calculate_lp_fraction(para_list, "len"),
            }
        )

        return metrics
