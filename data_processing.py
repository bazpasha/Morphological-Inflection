import xml.etree.ElementTree as ET
from torch.utils.data import Sampler
import numpy as np
import random
import torch
import json
import math

def parse_and_filter(filename, filter_posts):
    result = []
    filter_posts = set(filter_posts)
    with open(filename, encoding="utf-8") as _in:
        tree = ET.parse(_in)
    root = tree.getroot()
    
    posts = root.findall("./grammemes/grammeme[@parent='POST']/name")
    posts = set(post.text for post in posts)
    
    lemmata = root.findall("lemmata")[0]
    for lemma in lemmata.getchildren():
        lemma_id = lemma.get("id")
        lemma_meta = lemma.findall("l")[0]
        lemma_features = [g.get("v") for g in lemma_meta.getchildren()]
        if not (set(lemma_features) & filter_posts):
            continue

        lemma_object = {
            "lemma_id": lemma_id,
            "lemma": lemma_meta.get("t"),
            "lemma_features": lemma_features,
            "forms": []
        }
        pos = ["NOUN" if "NOUN" in lemma_object["lemma_features"] else "VERB"]
        for form in lemma.findall("f"):
            text = form.get("t")
            form_features = [g.get("v") for g in form.getchildren()]
            lemma_object["forms"].append([text, form_features + pos])
        result.append(lemma_object)

    return result


def get_float_hash(s):
    MAGIC_PRIME_CONST = 1073676287
    small_hash = hash(s) % MAGIC_PRIME_CONST
    return float(small_hash) / MAGIC_PRIME_CONST


def train_val_test_split(corpora, train_part=0.8, val_part=0.1):
    assert train_part + val_part < 1, "No room for test part! train_part + val_part >= 1"
    train, val, test = [], [], []
    for lemma in corpora:
        h = get_float_hash(lemma["lemma_id"])
        if h < train_part:
            train.append(lemma)
        elif h < train_part + val_part:
            val.append(lemma)
        else:
            test.append(lemma)
    return train, val, test


class Mapper:
    def __init__(self, corpora=[]):
        possible_letters = set()
        possible_features = set()
        for lemma in corpora:
            possible_letters.update(lemma["lemma"])
            for form, features in lemma["forms"]:
                possible_letters.update(form)
                possible_features.update(features)

        self.features_mapping = {feature: i for i, feature in enumerate(possible_features)}
        self.letters_mapping = {letter: i + 3 for i, letter in enumerate(possible_letters)}
        self.letters_mapping.update({
            "PAD": 0,
            "BEG": 1,
            "END": 2,
        })
        self.inverse_letters_mapping = {v: k for k, v in self.letters_mapping.items()}
    
    def _map_word(self, word, max_len):
        mapped = [self.letters_mapping[c] for c in word]
        padding = [self.letters_mapping["PAD"]] * (max_len - len(word))
        return [self.letters_mapping["BEG"]] + mapped + [self.letters_mapping["END"]] + padding
    
    def map_words(self, words):
        max_len = max(len(word) for word in words)
        result = [self._map_word(word, max_len) for word in words]
        return torch.LongTensor(result)
    
    def map_features(self, features):
        tensor = np.zeros((len(features), self.n_features), dtype=int)
        for i, form_features in enumerate(features):
            for feature in form_features:
                tensor[i][self.features_mapping[feature]] = 1
        return torch.FloatTensor(tensor)
            
    @property
    def n_letters(self):
        return len(self.letters_mapping)
    
    @property
    def n_features(self):
        return len(self.features_mapping)
    
    def _parse_word(self, mapping):
        letters = []
        for i in mapping[1:]:
            if self.inverse_letters_mapping[i] == "END":
                break
            letters.append(self.inverse_letters_mapping[i])
        return "".join(letters)

    def parse_words(self, mappings):
        return [self._parse_word(mapping) for mapping in mappings]
    
    def to_json(self, filename):
        with open(filename, "w") as _out:
            json.dump({
                "features_mapping": self.features_mapping,
                "letters_mapping": self.letters_mapping
            }, _out)
    
    @classmethod
    def from_json(cls, filename):
        with open(filename, "r") as _in:
            data = json.load(_in)
        obj = cls()
        obj.features_mapping = data["features_mapping"]
        obj.letters_mapping = data["letters_mapping"]
        obj.inverse_letters_mapping = {v: k for k, v in obj.letters_mapping.items()}
        return obj


class InflectionSampler:
    def __init__(self, corpora, mapper, forms_per_lemma=3, batch_size=100):
        self.mapper = mapper
        self.forms_per_lemma = forms_per_lemma
        self.batch_size = batch_size
        self.data = []
        for lemma in corpora:
            for form, features in lemma["forms"]:
                self.data.append([
                    lemma["lemma"],
                    int("NOUN" in lemma["lemma_features"]),
                    form,
                    features
                ])
    
    def data_iter(self):
        n_iters = self.__len__()
        
        order = np.arange(len(self.data))
        np.random.shuffle(order)
        
        for i in range(n_iters):
            lemmas = []
            forms = []
            features = []
            is_noun = []
            for j in order[i * self.batch_size:(i + 1) * self.batch_size]:
                row = self.data[j]
                lemmas.append(row[0])
                is_noun.append(row[1])
                forms.append(row[2])
                features.append(row[3])
            
            yield (
                self.mapper.map_words(lemmas),
                torch.LongTensor(is_noun),
                self.mapper.map_words(forms),
                self.mapper.map_features(features),
            )
    
    def __len__(self):
        return math.ceil(len(self.data) / self.batch_size)