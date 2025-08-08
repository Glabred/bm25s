"""
Sometimes you might be interested in benchmarking BM25 on BEIR. bm25s makes this straightforward in Python.

To install:

```
pip install bm25s[core] beir
```

Now, run this script, you can modify the `run_benchmark()` part to use the datase you want to test on.
"""
import json
import os
from pathlib import Path
import time

import beir.util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
import numpy as np
from tqdm.auto import tqdm
import Stemmer

import bm25s
from bm25s.utils.benchmark import get_max_memory_usage, Timer
from bm25s.utils.beir import (
    BASE_URL,
    clean_results_keys,
)

def postprocess_results_for_eval(results, scores, query_ids):
    """
    Given the queried results and scores output by BM25S, postprocess them
    to be compatible with BEIR evaluation functions.
    query_ids is a list of query ids in the same order as the results.
    """

    results_record = [
        {"id": qid, "hits": results[i], "scores": list(scores[i])}
        for i, qid in enumerate(query_ids)
    ]

    result_dict_for_eval = {
        res["id"]: {
            docid: float(score) for docid, score in zip(res["hits"], res["scores"])
        }
        for res in results_record
    }

    return result_dict_for_eval

def run_benchmark(dataset, save_dir="datasets"):
    #### Download dataset and unzip the dataset
    data_path = beir.util.download_and_unzip(BASE_URL.format(dataset), save_dir)
    split = "test" if dataset != "msmarco" else "dev"
    
    # Define paths for saving/loading preprocessed data
    index_dir = Path(save_dir) / f"{dataset}_bm25s_index"
    corpus_tokens_file = Path(save_dir) / f"{dataset}_corpus_tokens.json"
    query_tokens_file = Path(save_dir) / f"{dataset}_query_tokens.json"
    corpus_ids_file = Path(save_dir) / f"{dataset}_corpus_ids.json"
    qids_file = Path(save_dir) / f"{dataset}_qids.json"
    
    timer = bm25s.utils.benchmark.Timer("[BM25S]")
    
    # Check if preprocessed data exists
    if (index_dir.exists() and corpus_tokens_file.exists() and 
        query_tokens_file.exists() and corpus_ids_file.exists() and qids_file.exists()):
        
        print(f"Loading preprocessed data for {dataset}...")
        
        # Load preprocessed corpus IDs and query IDs
        with open(corpus_ids_file, 'r') as f:
            corpus_ids = json.load(f)
        with open(qids_file, 'r') as f:
            qids = json.load(f)
            
        # Load preprocessed tokens
        print("Loading tokenized corpus...")
        with open(corpus_tokens_file, 'r') as f:
            corpus_data = json.load(f)
            # Reconstruct Tokenized object
            corpus_tokens = bm25s.tokenization.Tokenized(
                ids=corpus_data['ids'], 
                vocab=corpus_data['vocab']
            )
            
        print("Loading tokenized queries...")
        with open(query_tokens_file, 'r') as f:
            query_data = json.load(f)
            # Reconstruct Tokenized object
            query_tokens = bm25s.tokenization.Tokenized(
                ids=query_data['ids'],
                vocab=query_data['vocab']
            )
            
        # Load the BM25 model
        print("Loading BM25 index...")
        model = bm25s.BM25(method="lucene", k1=1.2, b=0.75, device="cuda")
        model = bm25s.BM25.load(index_dir, device="cuda")
        
        print(f"Successfully loaded preprocessed data for {dataset}")
        
    else:
        print(f"Preprocessing {dataset} dataset (first run)...")
        
        # Load raw data
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)

        corpus_ids, corpus_lst = [], []
        for key, val in corpus.items():
            corpus_ids.append(key)
            corpus_lst.append(val["title"] + " " + val["text"])
        del corpus

        qids, queries_lst = [], []
        for key, val in queries.items():
            qids.append(key)
            queries_lst.append(val)

        stemmer = Stemmer.Stemmer("english")
        
        print("Tokenizing corpus...")
        corpus_tokens = bm25s.tokenize(
            corpus_lst, stemmer=stemmer, leave=False
        )
        del corpus_lst

        print("Tokenizing queries...")
        query_tokens = bm25s.tokenize(
            queries_lst, stemmer=stemmer, leave=False
        )

        # Create and index the model
        model = bm25s.BM25(method="lucene", k1=1.2, b=0.75, device="cuda")
        t = timer.start("Indexing")
        model.index(corpus_tokens, leave_progress=False)
        timer.stop(t, show=True, n_total=len(corpus_tokens))
        
        # Save preprocessed data for future runs
        print("Saving preprocessed data...")
        
        # Save the BM25 index (handles GPU->CPU conversion automatically)
        model.save(index_dir)
        
        # Save tokenized data
        with open(corpus_tokens_file, 'w') as f:
            json.dump({
                'ids': corpus_tokens.ids,
                'vocab': corpus_tokens.vocab
            }, f)
        with open(query_tokens_file, 'w') as f:
            json.dump({
                'ids': query_tokens.ids,
                'vocab': query_tokens.vocab
            }, f)
            
        # Save IDs
        with open(corpus_ids_file, 'w') as f:
            json.dump(corpus_ids, f)
        with open(qids_file, 'w') as f:
            json.dump(qids, f)
            
        print(f"Saved preprocessed data to {save_dir}")
    
    mem_use = bm25s.utils.benchmark.get_max_memory_usage()
    print(f"Memory usage after loading the index: {mem_use:.2f} GB")
    
    # Load qrels for evaluation
    _, _, qrels = GenericDataLoader(data_folder=data_path).load(split=split)

    ############## BENCHMARKING BEIR HERE ##############
    t = timer.start("Retrieving")
    queried_results, queried_scores = model.retrieve(
        query_tokens, corpus=corpus_ids, k=100, n_threads=64
    )
    timer.stop(t, show=True, n_total=len(query_tokens))
    mem_use = bm25s.utils.benchmark.get_max_memory_usage()
    print(f"Memory usage after loading the index: {mem_use:.2f} GB")

    results_dict = postprocess_results_for_eval(queried_results, queried_scores, qids)
    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(
        qrels, results_dict, [1, 10, 100]
    )

    print(ndcg)
    print(recall)
    
    return ndcg, _map, recall, precision

if __name__ == "__main__":
    ndcg, _map, recall, precision = run_benchmark("quora")  # Change to dataset you want
