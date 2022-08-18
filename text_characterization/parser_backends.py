# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import spacy
from spacy.lang.en import English

from .utils import (
    mean_or_none,
)


class BasicSpacyBackend:
    """
    A computationally backend that only does tokenization and sentence splitting for the most basic type of metrics.
    Useful when processing large amounts of data.
    """

    processor = None

    TOKEN_ATTRIBUTES = {
        "is_alpha": ("ALHPA", "Alphanumerical tokens"),
        "is_digit": ("DIGIT", "Tokens consisting of digits"),
        "is_punct": ("PUNCT", "Punctuation tokens"),
        "like_url": ("URL", "URLs"),
        "like_email": ("EMAIL", "E-mail addresses"),
    }

    def _tag_token(cls, word):
        return [v[0] for k, v in cls.TOKEN_ATTRIBUTES.items() if word.__getattribute__(k)]

    @classmethod
    def load_resources(cls):
        cls.processor = English()
        cls.processor.add_pipe("sentencizer")
        # We can afford longer texts as we don't do anything complex with this backend
        cls.processor.max_length = 10000000

    def __init__(self, text):
        self.results = self.processor(text)

    # Returns a list of lists, each list representing tokens in a sentence
    def get_sentences(self):
        return [[x.text for x in sent] for sent in self.results.sents]

    # Returns a list of sentences
    def get_sentences_text(self):
        return [sent.text for sent in self.results.sents]

    # Return a list of paragraphs
    def get_paragraphs(self):
        return [
            paragraph
            for paragraph in self.results.text.split("\n")
            if len(paragraph.strip()) > 0
        ]

    # The tokenized text
    def get_words(self):
        return [x.text for x in self.results if not x.is_punct]

    # Returns a list of lists: for each token a list of attributes.
    def get_token_attributes(self):
        return [self._tag_token(w) for w in self.results]

    def get_sentence_lengths(self):
        return [sum(not w.is_punct for w in sent) for sent in self.results.sents]

    def get_sentence_words(self):
        return [[x.text for x in sent if not x.is_punct] for sent in self.results.sents]

    # Note that the expensive backend uses POS tags, here we just do a cheap filtering to alphabetic tokens.
    # The goal is to avoid messing up diversity metrics with whitespaces/punctuation/numbers etc.
    def get_content_words(self):
        return [x.text for x in self.results if x.is_alpha]

    def num_sentences_per_paragraph(self):
        paragraph_lengths = []
        current_len = 0
        # Count the number of full sentences between newlines. Note that we might have multiple newlines
        # even before the sentence is finished, those paragraphs will count as 0 sentence long.
        for s in self.results.sents:
            current_len += 1
            for w in s:
                if "\n" in w.text:
                    paragraph_lengths.append(current_len)
                    current_len = 0
        if current_len > 0:
            # If text ends without a newline
            paragraph_lengths.append(current_len)
        return paragraph_lengths

    def num_words_per_paragraph(self):
        paragraph_lengths = []
        current_len = 0
        for w in self.results:
            if not w.is_punct and not w.is_space:
                current_len += 1
            if "\n" in w.text:
                paragraph_lengths.append(current_len)
                current_len = 0

        if current_len > 0:
            paragraph_lengths.append(current_len)

        return paragraph_lengths


class FullSpacyBackend(BasicSpacyBackend):
    """
    A more expensive Spacy backend that computes various linguistic features such as POS tags.
    This is suitable for relatively smaller data (expect ~1 CPU second / doc) and supports more metrics.
    """

    content_pos_tags = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}
    np_modifiers = {"amod", "nmod", "advmod", "nummod", "quantmod"}

    @classmethod
    def load_resources(cls, model="en_core_web_md"):
        cls.processor = spacy.load(model)
        # In issue https://github.com/explosion/spaCy/issues/4577
        # import en_core_web_sm
        # cls.processor = en_core_web_sm.load()


    def get_pos_tags(self):
        return [x.pos_ for x in self.results if not x.is_punct]

    def get_sentence_pos_tags(self):
        return [[x.pos_ for x in sent if not x.is_punct] for sent in self.results.sents]

    def get_sentence_lemmas(self):
        return [
            [x.lemma_ for x in sent if not x.is_punct] for sent in self.results.sents
        ]

    def get_content_words(self):
        return [x.text for x in self.results if not x.is_punct and x.pos_ in self.content_pos_tags]

    def mean_np_modifiers(self):
        # Average number of modifiers per noun phrase
        # Note that the iterator used here does not consider nested structures
        # TODO(danielsimig) There's likely a more precise definition of what a modifier is
        num_modifiers = []
        for chunk in self.results.noun_chunks:
            num_modifiers.append(
                # sum(c.dep_ in self.np_modifiers for c in chunk.root.children)
                len(list(chunk.root.children))
            )
        return mean_or_none(num_modifiers)

    def mean_words_before_main_verb(self):
        return mean_or_none(sent.root.i - sent.start for sent in self.results.sents)

    def num_nodes_in_intersection(self, node1, node2):
        # Implements the tree intersection algorithm described in the Coh-Metrix book.
        # Note that the parse tree structure there is slightly different from what
        # parsers like spaCy return, e.g. there's no separate parent node for noun phrases,
        # just direct dependencies.
        # TODO(danielsimig) This is in need of more research to make it precise.

        children_1_deps = [x.tag for x in node1.children]
        children_2_deps = [x.tag for x in node2.children]
        if children_1_deps != children_2_deps:
            return 0

        intersection_size = len(children_1_deps)
        for c1, c2 in zip(node1.children, node2.children):
            intersection_size += self.num_nodes_in_intersection(c1, c2)
        return intersection_size

    def mean_adjacent_syntatic_similarity(self):

        sentences = list(self.results.sents)

        if len(sentences) < 2:
            return None

        similarities = []
        for i in range(1, len(sentences)):
            num_intersection = (
                self.num_nodes_in_intersection(sentences[i].root, sentences[i - 1].root)
                + 1
            )  # Root node is an intersection
            similarities.append(
                num_intersection
                * 1.0
                / (len(sentences[i]) + len(sentences[i - 1]) - num_intersection)
            )

        return mean_or_none(similarities)
