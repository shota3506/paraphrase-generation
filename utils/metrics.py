import math
import torch
import numpy as np
import nltk
import zss
from collections import Counter
from statistics import mean
from typing import Any, List, Set



########## Metrics ##########
def ngram(seq: List[Any], n: int = 1) -> List[List[Any]]:
    tokens = seq.split()
    return [tokens[i:i+n] for i in range(len(tokens)-n+1)]


def levenshtein_distance(seq1: List[Any], seq2: List[Any]) -> float:
    length1, length2 = len(seq1), len(seq2)

    table = np.zeros((length1 + 1, length2 + 1))
    # Base case
    for i1 in range(length1 + 1):
        table[i1, 0] = i1
    for i2 in range(length2 + 1):
        table[0, i2] = i2
    # General case
    for i1 in range(1, length1 + 1):
        for i2 in range(1, length2 + 1):
            if seq1[i1 - 1] == seq2[i2 - 1]:
                cost = 0.0
            else:
                cost = 1.0
            table[i1, i2] = min(table[i1-1, i2] + 1.0, # Insertion
                                table[i1, i2-1] + 1.0, # Deletion
                                table[i1-1, i2-1] + cost) # Replacement/Nothing
    return table[length1, length2]


def longest_common_subsequence(seq1: List[Any], seq2: List[Any]) -> int:
    n, m = len(seq1), len(seq2)
    table = [[0] * (m+1) for _ in range(n+1)]

    for i in range(1, n+1):
        for j in range(1, m+1):
            if seq1[i-1] == seq2[j-1]:
                table[i][j] = table[i-1][j-1] + 1
            else:
                table[i][j] = max(table[i-1][j], table[i][j-1])

    return table[n][m]


def bleu(refs: List[str], hyp: str, weights=(0.25, 0.25, 0.25, 0.25)) -> float:
    b = nltk.translate.bleu_score.sentence_bleu(
        [r.split() for r in refs], 
        hyp.split(), 
        weights=weights,
        smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method1)
    return b * 100


def ibleu(refs: List[str], hyp: str, src: str, alpha: float = 0.8) -> float:
    return alpha * bleu(refs, hyp) - (1 - alpha) * bleu([src], hyp)


def pairwise_bleu(hyps: List[str], weights=(0.25, 0.25, 0.25, 0.25)) -> float:
    n = len(hyps)
    if n <= 1:
        return None
    return mean(bleu(hyps[j:j+1], hyps[i], weights) for i in range(n) for j in range(n) if i != j)


def selfbleu(hyps: List[str], weights=(0.25, 0.25, 0.25, 0.25)) -> float:
    n = len(hyps)
    if n <= 1:
        return None
    return mean(bleu(hyps[:i] + hyps[i+1:], hyps[i], weights) for i in range(n))


def meteor(refs: List[str], hyp: str) -> float:
    m = nltk.translate.meteor_score.meteor_score(refs, hyp)
    return m * 100


def distn(hyps: List[str], n: int = 1) -> float:
    docs = [[' '.join(ng) for ng in ngram(h, n)] for h in hyps]
    
    st = set(t for doc in docs for t in doc)
    cnt = sum(len(doc) for doc in docs)
    return len(st) / cnt


def sentence_bert_similarity(refs: List[str], hyps: List[str], embedder):
    emb = embedder.encode(hyps + refs, convert_to_tensor=True, show_progress_bar=False)
    emb1, emb2 = emb[:len(hyps)], emb[len(hyps):]
    emb1_norm = emb1 / emb1.norm(dim=1, keepdim=True)
    emb2_norm = emb2 / emb2.norm(dim=1, keepdim=True)
    
    sim = torch.mm(emb1_norm, emb2_norm.transpose(0, 1))
    sim, _ = torch.max(sim, dim=1)
    sim = torch.mean(sim).item()
    return sim


def tree_edit_distance(parse1, parse2):
    def build_tree(s):
        old_t = nltk.Tree.fromstring(s)
        new_t = zss.Node("S")

        def create_tree(curr_t, t):
            if t.label() and t.label() != "S":
                new_t = zss.Node(t.label())
                curr_t.addkid(new_t)
            else:
                new_t = curr_t
                
            for i in t:
                if isinstance(i, nltk.Tree):
                    create_tree(new_t, i)
                    
        create_tree(new_t, old_t)
        return new_t

    return zss.simple_distance(
        build_tree(parse1), 
        build_tree(parse2), 
        label_dist=lambda x, y: 0 if x == y else 1
    )