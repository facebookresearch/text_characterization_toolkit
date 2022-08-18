# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from datasets import load_dataset
import numpy as np
from nltk.translate.chrf_score import sentence_chrf
import spacy
  
nlp = spacy.load("en_core_web_sm")
  
import torch
import hydra
import json
from omegaconf import DictConfig, OmegaConf
from transformers import pipeline

def load_models_translate_and_get_effect_sizes(cfg, df):
    assert len(cfg.models) == 2 # For simplicity don't handle the model cross-walk yet
    model_scores = {}
    model_translations = {}
    for i, (model_tag, src) in enumerate(cfg.models):
        print(f"Getting translations for {model_tag}")
        if src == "hf":
            translator = pipeline("translation", src_lang=cfg.lang1, tgt_lang=cfg.lang2, model=model_tag, device=0, batch_size=32, max_length=512)
            if cfg.split_sentences:
                translations = []
                for doc_point in df[f"sentence_{cfg.lang1}"]:
                    doc = nlp(doc_point)
                    sents = [str(st) for st in doc.sents]
                    sent_translations = translator(sents)
                    translations.append(" ".join([x["translation_text"] for x in sent_translations]))
            else:
                translations = [x["translation_text"] for x in translator(df[f"sentence_{cfg.lang1}"])]
            model_scores[i] = []
            model_translations[i] = translations
            for translation, gt in zip(translations, df[f"sentence_{cfg.lang2}"]):
                model_scores[i].append(sentence_chrf(gt, translation))
        else:
            raise ValueError("Other model hubs aren't implemented yet. But in the future you could update this to include, say, Pytorch Hub models.")

    return [{"score1" : model_score[0], "translation1" : model_score[2], "translation2" : model_score[3], "score2" : model_score[1], "outcome" : model_score[1] - model_score[0]} for model_score in zip(model_scores[0], model_scores[1], model_translations[0], model_translations[1])]


@hydra.main(config_name="config")
def main(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    dataset = load_dataset("breakend/nllb-multi-domain", f"{cfg.lang1}-{cfg.lang2}", split="valid")

    s1_index = f"sentence_{cfg.lang1}"
    s2_index = f"sentence_{cfg.lang2}"

    with open("translation_text_features.jsonl", "w") as f:
        for i, (sentence1, sentence2) in enumerate(zip(dataset[s1_index], dataset[s2_index])):
            f.write(json.dumps({"id" : i, s1_index : sentence1, s2_index : sentence2}) + "\n")

    outcomes = load_models_translate_and_get_effect_sizes(cfg, dataset)
    with open("translation_outcomes.jsonl", "w") as f:
        for i, outcome in enumerate(outcomes):
            f.write(json.dumps({"id" : i, **outcome}) + "\n")

    
     

if __name__ == "__main__":
    main()

