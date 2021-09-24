
import json
import sys
import numpy as np
import json
import re
import math
import nltk
import argparse

from scipy.stats import kendalltau 
from tqdm import tqdm


def kendall_tau(order, ground_truth):
    """
    Computes the kendall's tau metric 
    between the predicted sentence order and true order

    Input: 
            order: list of ints denoting the predicted output order
            ground_truth: list of ints denoting the true sentence order
    
    Returns:
            kendall's tau - float
    """
    
    if len(ground_truth) == 1:
        if ground_truth[0] == order[0]:
            return 1.0
        
    reorder_dict = {}
        
    for i in range(len(ground_truth)):
        reorder_dict[ground_truth[i]] = i
        
    new_order = [0] * len(order)
    for i in range(len(new_order)):
        if order[i] in reorder_dict.keys():
            new_order[i] = reorder_dict[order[i]]
    
    corr, _ = kendalltau(new_order, list(range(len(order))))
    return corr

def lcs(X , Y): 
    """
    Computes the longest common subsequence between two sequences

    Input:
            X: list of ints
            Y: list of ints
    
    Returns:
            LCS: int
    """
    m = len(X) 
    n = len(Y) 

    L = [[None]*(n+1) for i in range(m+1)] 

    for i in range(m+1): 
        for j in range(n+1): 
            if i == 0 or j == 0 : 
                L[i][j] = 0
            elif X[i-1] == Y[j-1]: 
                L[i][j] = L[i-1][j-1]+1
            else: 
                L[i][j] = max(L[i-1][j] , L[i][j-1]) 

    return L[m][n] 


def skip_bigrams(arr):
    """
    Utility function for Rouge-S metric
    """
    bigrams = set()
    for i in range(len(arr)):
        for j in range(i+1, len(arr)):
            bigrams.add((arr[i], arr[j]))
    return bigrams

def rouge_s(gold, pred):
    """
    Rouge-S metric between two sequence

    Input:
            gold: list of ints
            pred: list of ints
    
    Returns:
            Rouge-S score
    """

    if len(gold) == 1 or len(pred) == 1:
        return int(gold[0] == pred[0])
    
    gold_bigrams = skip_bigrams(gold)
    pred_bigrams = skip_bigrams(pred)
    
    total = len(gold_bigrams)
    same = len(gold_bigrams.intersection(pred_bigrams))
    return (same / total)


def clean_output(gold, predictions):
    """
    Utility function to clean generated output from BART
    """

    label = gold.replace("<eos>", "").strip()
    labels = [int(id_[2:-1]) for id_ in label.split()]
    
    # handle cases when output is empty
    if len(predictions) == 0:
        return labels, []
    
    preds = []
    for p in predictions[0].split():
        pos = re.findall('\\d+', p)
        if len(pos) == 1:
            preds.append(int(pos[0]))
    return labels, preds


def evaluate(filename):
    """
    Evaluation iterator function. Generates all metrics 
    by calling the functions for every instance.

    Input:
            filename: file name of the generated output
    
    Returns: None
    """

    acc, PMR, kendall_score, LCS, rouge = 0, 0, 0, 0, 0
    total, total_sents = 0, 0
    
    err = 0
    
    with open(filename) as file:
        lines = file.readlines()
        for line in tqdm(lines):
            entry = json.loads(line.strip())
            gold, predictions = clean_output(entry["gold"], entry["predictions"])
            
            total += 1
            total_sents += len(gold)
            
            
            if len(predictions) == 0:
                err += 1
                continue
                
            LCS += lcs(gold, predictions)
            
            rouge += rouge_s(gold, predictions)
            
            if predictions == gold:
                PMR += 1

            tau_score = kendall_tau(predictions, gold)

            # handle cases of empty output
            if math.isnan(tau_score):
                err += 1
                tau_score = 0
            
            kendall_score += tau_score
                
            # Compute sentence level statistics
            for i in range(min(len(gold), len(predictions))):
                if gold[i] == predictions[i]:
                    acc += 1
                
    print(f" {err} sample(s) were not processed")
    print(" Accuracy: {:.6f}".format(acc / total_sents))
    print(" PMR: {:.6f}".format(PMR / total))
    print(" Kendall's Tau: {:.6f}".format(kendall_score / total))
    print(" LCS: {:.6f}".format(LCS / total_sents))
    print(" Rouge-S: {:.6f}".format(rouge / total))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_path", type = str, required=True
    )

    args = parser.parse_args()
    evaluate(args.output_path)

if __name__ == "__main__":
    main()