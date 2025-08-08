from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from functools import partial
from collections import Counter

import os
import logging
from pathlib import Path
import json
from typing import Any, Tuple, Dict, Iterable, List, NamedTuple, Union

import numpy as np
import torch
import time

from .utils import json_functions as json_functions

try:
    from .numba import selection as selection_jit
except ImportError:
    selection_jit = None

try:
    from .numba.retrieve_utils import _retrieve_numba_functional
except ImportError:
    _retrieve_numba_functional = None


def _faketqdm(iterable, *args, **kwargs):
    return iterable


if os.environ.get("DISABLE_TQDM", False):
    tqdm = _faketqdm
    # if can't import tqdm, use a fake tqdm
else:
    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = _faketqdm


from . import selection, utils, stopwords, scoring, tokenization
from .version import __version__
from .tokenization import tokenize
from .scoring import (
    _select_tfc_scorer,
    _select_idf_scorer,
    _build_scores_and_indices_for_matrix,
    _calculate_doc_freqs,
    _build_idf_array,
    _build_nonoccurrence_array,
)

logger = logging.getLogger("bm25s")
logger.setLevel(logging.DEBUG)


class Results(NamedTuple):
    """
    NamedTuple with two fields: documents and scores. The `documents` field contains the
    retrieved documents or indices, while the `scores` field contains the scores of the
    retrieved documents or indices.
    """

    documents: np.ndarray
    scores: np.ndarray

    def __len__(self):
        return len(self.documents)

    @classmethod
    def merge(cls, results: List["Results"]) -> "Results":
        """
        Merge a list of Results objects into a single Results object.
        """
        documents = np.concatenate([r.documents for r in results], axis=0)
        scores = np.concatenate([r.scores for r in results], axis=0)
        return cls(documents=documents, scores=scores)


def get_unique_tokens(
    corpus_tokens, show_progress=True, leave_progress=False, desc="Create Vocab"
):
    unique_tokens = set()
    for doc_tokens in tqdm(
        corpus_tokens, desc=desc, disable=not show_progress, leave=leave_progress
    ):
        unique_tokens.update(doc_tokens)
    return unique_tokens


def is_list_of_list_of_type(obj, type_=int):
    if not isinstance(obj, list):
        return False

    if len(obj) == 0:
        return False

    first_elem = obj[0]
    if not isinstance(first_elem, list):
        return False

    if len(first_elem) == 0:
        return False

    first_token = first_elem[0]
    if not isinstance(first_token, type_):
        return False

    return True


def _is_tuple_of_list_of_tokens(obj):
    if not isinstance(obj, tuple):
        return False

    if len(obj) == 0:
        return False

    first_elem = obj[0]
    if not isinstance(first_elem, list):
        return False

    if len(first_elem) == 0:
        return False

    first_token = first_elem[0]
    if not isinstance(first_token, str):
        return False

    return True


class BM25:
    def __init__(
        self,
        k1=1.5,
        b=0.75,
        delta=0.5,
        method="lucene",
        idf_method=None,
        dtype="float32",
        int_dtype="int32",
        corpus=None,
        backend="numpy",
        device="cpu",
    ):
        """
        BM25S initialization.

        Parameters
        ----------
        k1 : float
            The k1 parameter in the BM25 formula.

        b : float
            The b parameter in the BM25 formula.

        delta : float
            The delta parameter in the BM25L and BM25+ formulas; it is ignored for other methods.

        method : str
            The method to use for scoring term frequency. Choose from 'robertson', 'lucene', 'atire', 'bm25l', 'bm25+'.
            Note: GPU acceleration (device='cuda') only supports 'robertson', 'lucene', and 'atire'.

        idf_method : str
            The method to use for scoring inverse document frequency (same choices as `method`).
            If None, it will use the same method as `method`. If you are unsure, please do not
            change this parameter.
        dtype : str
            The data type of the BM25 scores.

        int_dtype : str
            The data type of the indices in the BM25 scores.

        corpus : Iterable[Dict]
            The corpus of documents. This is optional and is used for saving the corpus
            to the snapshot. We expect the corpus to be a list of dictionaries, where each
            dictionary represents a document.

        backend : str
            The backend used during retrieval. By default, it uses the numpy backend, which
            only requires numpy and scipy as dependencies. You can also select `backend="numba"`
            to use the numba backend, which requires the numba library. If you select `backend="auto"`,
            the function will use the numba backend if it is available, otherwise it will use the numpy
            backend.
            
        device : str
            The device to use for computation. Can be "cpu" or "cuda". If "cuda" is specified,
            the sparse matrix will be converted to PyTorch CSC tensors and moved to GPU memory
            for accelerated operations. If CUDA is not available, it will fall back to CPU.
        """
        self.k1 = k1
        self.b = b
        self.delta = delta
        self.dtype = dtype
        self.int_dtype = int_dtype
        self.method = method
        self.idf_method = idf_method if idf_method is not None else method
        self.methods_requiring_nonoccurrence = ("bm25l", "bm25+")
        self.corpus = corpus
        self._original_version = __version__
        
        # Handle device initialization with CUDA availability check
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            device = "cpu"
        self.device = torch.device(device)

        if backend == "auto":
            self.backend = "numba" if selection_jit is not None else "numpy"
        else:
            self.backend = backend
        print("self.backend: ", self.backend)

    @staticmethod
    def _infer_corpus_object(corpus):
        """
        Verifies if the corpus is a list of list of strings, an object with the `ids` and `vocab` attributes,
        or a tuple of two lists: first is list of list of ids, second is the vocab dictionary.
        """
        if hasattr(corpus, "ids") and hasattr(corpus, "vocab"):
            return "object"
        elif isinstance(corpus, tuple) and len(corpus) == 2:
            c1, c2 = corpus
            if isinstance(c1, list) and isinstance(c2, dict):
                return "tuple"
            else:
                raise ValueError(
                    "Corpus must be a list of list of tokens, an object with the `ids` and `vocab` attributes, or a tuple of two lists: the first list is the list of unique token IDs, and the second list is the list of token IDs for each document."
                )
        elif isinstance(corpus, Iterable):
            if is_list_of_list_of_type(corpus, type_=int):
                return "token_ids"
            else:
                return "tokens"
        else:
            raise ValueError(
                "Corpus must be a list of list of tokens, an object with the `ids` and `vocab` attributes, or a tuple of two lists: the first list is the list of unique token IDs, and the second list is the list of token IDs for each document."
            )

    def _convert_scores_to_torch_csc(self, scores):
        """
        Convert scipy sparse matrix components to PyTorch CSC tensor on GPU.
        
        Parameters
        ----------
        scores : dict
            Dictionary containing 'data', 'indices', 'indptr', and 'num_docs'
            from scipy.sparse.csc_matrix
            
        Returns
        -------
        dict
            Dictionary with PyTorch tensors for GPU computation
        """
        if not torch.cuda.is_available() and self.device.type == "cuda":
            raise RuntimeError("CUDA is not available but GPU device was specified")
        
        # Validate input scores dictionary
        required_keys = ["data", "indices", "indptr", "num_docs"]
        if not all(key in scores for key in required_keys):
            raise ValueError(f"scores dict must contain keys: {required_keys}")
        
        try:
            # Convert numpy arrays to PyTorch tensors and move to GPU
            data_tensor = torch.tensor(scores["data"], dtype=torch.float16, device=self.device)
            indices_tensor = torch.tensor(scores["indices"], dtype=torch.int64, device=self.device)
            indptr_tensor = torch.tensor(scores["indptr"], dtype=torch.int64, device=self.device)
            
            # Get matrix dimensions from original scores
            num_docs = scores["num_docs"]
            vocab_size = len(scores["indptr"]) - 1  # indptr has vocab_size + 1 elements
            
            # Create PyTorch sparse CSC tensor (for backward compatibility)
            sparse_csc_tensor = torch.sparse_csc_tensor(
                indptr_tensor,
                indices_tensor, 
                data_tensor,
                size=(num_docs, vocab_size),
                dtype=torch.float16,
                device=self.device
            )
            
            # Convert to COO format first, then to CSR for better SpMV performance
            sparse_coo = sparse_csc_tensor.to_sparse_coo()
            sparse_csr_tensor = sparse_coo.to_sparse_csr()
            
            # Return dict with PyTorch tensors for compatibility with existing code
            torch_scores = {
                "data": data_tensor,
                "indices": indices_tensor,
                "indptr": indptr_tensor,
                "num_docs": num_docs,
                "sparse_tensor": sparse_csc_tensor,  # Keep for backward compatibility
                "sparse_csr_tensor": sparse_csr_tensor,  # Optimized for SpMV
                "vocab_size": vocab_size
            }
            
            logger.debug(f"Successfully converted sparse matrix to PyTorch CSC tensor on {self.device}")
            logger.debug(f"Matrix shape: {num_docs} x {vocab_size}, nnz: {len(data_tensor)}")
            
            return torch_scores
            
        except Exception as e:
            logger.error(f"Failed to convert scores to PyTorch tensor: {e}")
            raise RuntimeError(f"Error converting scores to GPU tensors: {e}") from e

    def _compute_relevance_from_scores_gpu_mult(
        self,
        query_tokens_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        GPU-accelerated scoring using sparse matrix multiplication.
        
        This method creates a query vector and multiplies it with the sparse matrix
        to compute document scores. This approach can be more efficient for certain
        query patterns compared to index selection.
        
        Parameters
        ----------
        query_tokens_ids : torch.Tensor
            Tensor of token IDs to score on GPU
            
        Returns
        -------
        torch.Tensor
            Array of BM25 relevance scores for the query on GPU
        """
        # Ensure query tokens are on the same device and correct dtype
        query_tokens_ids = query_tokens_ids.to(self.device, dtype=torch.long)
        
        # Get the CSC sparse tensor components directly (like CPU version)
        data = self.scores["data"]        # Non-zero values
        indices = self.scores["indices"]  # Row indices for each value
        indptr = self.scores["indptr"]    # Column pointers
        num_docs = self.scores["num_docs"]
        
        # Use CPU-like approach: extract start/end positions for all query tokens at once
        indptr_starts = indptr[query_tokens_ids]
        indptr_ends = indptr[query_tokens_ids + 1]
        
        # Initialize document scores
        doc_scores = torch.zeros(num_docs, device=self.device, dtype=torch.float32)
        
        # Process each query token (mimicking CPU np.add.at behavior)
        for i in range(len(query_tokens_ids)):
            start, end = indptr_starts[i].item(), indptr_ends[i].item()
            if start < end:  # Token has non-zero scores
                doc_indices_for_token = indices[start:end]
                scores_for_token = data[start:end]
                
                # Use scatter_add for accumulation (equivalent to np.add.at)
                doc_scores.scatter_add_(0, doc_indices_for_token, scores_for_token)
        
        return doc_scores

    def _compute_relevance_from_scores_gpu_padded(
        self,
        query_tokens_ids_batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Optimized GPU scoring for batched queries using zero-padded tensors.
        
        Parameters
        ----------
        query_tokens_ids_batch : torch.Tensor
            Zero-padded tensor of shape (batch_size, max_query_length) where 
            shorter queries are padded with zeros.
            
        Returns
        -------
        torch.Tensor
            Document scores of shape (batch_size, num_docs)
        """
        query_tokens_ids_batch = query_tokens_ids_batch.to(self.device, dtype=torch.long)
        batch_size, max_query_length = query_tokens_ids_batch.shape
        num_docs = self.scores["num_docs"]
        
        # Initialize output tensor
        all_doc_scores = torch.zeros((batch_size, num_docs), device=self.device, dtype=torch.float32)
        
        if max_query_length == 0:
            return all_doc_scores
            
        # Create mask for non-zero tokens (padding tokens are 0)
        # Assuming 0 is used as padding and is not a valid token ID
        valid_mask = query_tokens_ids_batch > 0
        
        if not valid_mask.any():
            return all_doc_scores
            
        # Get all valid token positions
        batch_indices, token_positions = torch.where(valid_mask)
        valid_tokens = query_tokens_ids_batch[batch_indices, token_positions]
        
        # Get sparse matrix components for valid tokens
        indptr_starts = self.scores["indptr"][valid_tokens]
        indptr_ends = self.scores["indptr"][valid_tokens + 1]
        lengths = indptr_ends - indptr_starts
        
        # Filter out tokens that don't appear in any documents
        token_mask = lengths > 0
        if not token_mask.any():
            return all_doc_scores
            
        # Filter to only tokens that have document matches
        filtered_batch_indices = batch_indices[token_mask]
        filtered_starts = indptr_starts[token_mask]
        filtered_lengths = lengths[token_mask]
        
        # Generate indices for all document-score pairs
        max_length = filtered_lengths.max().item()
        position_indices = torch.arange(max_length, device=self.device).unsqueeze(0)
        valid_positions = position_indices < filtered_lengths.unsqueeze(1)
        
        # Calculate flat indices into the sparse matrix data
        flat_indices = filtered_starts.unsqueeze(1) + position_indices
        flat_indices = flat_indices[valid_positions]
        
        if len(flat_indices) == 0:
            return all_doc_scores
            
        # Get document indices and scores from sparse matrix
        doc_indices = self.scores["indices"][flat_indices]
        scores = self.scores["data"][flat_indices]
        
        # Expand batch indices to match the flattened structure
        batch_indices_expanded = filtered_batch_indices.unsqueeze(1).expand(-1, max_length)[valid_positions]
        
        # Combine batch and document indices for scatter operation
        combined_indices = batch_indices_expanded * num_docs + doc_indices
        
        # Accumulate scores using scatter_add
        flat_scores = all_doc_scores.view(-1)
        flat_scores.scatter_add_(0, combined_indices, scores)
        all_doc_scores = flat_scores.view(batch_size, num_docs)
        
        return all_doc_scores

    def _compute_relevance_from_scores_gpu_sparse_mm(
        self,
        sparse_query_tensor: torch.sparse.Tensor,
    ) -> torch.Tensor:
        """
        GPU scoring using optimized sparse matrix multiplication.
        This should be significantly faster than manual indexing approaches.
        Now uses pre-computed sparse query tensor to avoid preprocessing overhead.
        
        Parameters
        ----------
        sparse_query_tensor : torch.sparse.Tensor
            Sparse CSR tensor of shape (vocab_size, batch_size) representing query frequencies
            
        Returns
        -------
        torch.Tensor
            Document scores of shape (num_docs, batch_size)
        """
        
        
        # Get the sparse CSR tensor for documents
        sparse_csr_tensor = self.scores.get("sparse_csr_tensor", self.scores["sparse_tensor"])
        
        # Log memory before sparse matrix multiplication
        if torch.cuda.is_available() and self.device.type == "cuda":
            mem_before = torch.cuda.memory_allocated(self.device) / (1024**3)
            print(f"      [SparseMM] GPU memory before: {mem_before:.2f}GB")
        
        # Perform sparse matrix multiplication
        # sparse_csr_tensor shape: (num_docs, vocab_size)  
        # sparse_query_tensor shape: (vocab_size, batch_size)
        # Result shape: (num_docs, batch_size)
        start = time.time()
        doc_scores_transposed = torch.sparse.mm(sparse_csr_tensor, sparse_query_tensor)
        print(f"      sparse mm taken: {time.time() - start} seconds")
        
        # Log memory after sparse matrix multiplication
        if torch.cuda.is_available() and self.device.type == "cuda":
            mem_after_mm = torch.cuda.memory_allocated(self.device) / (1024**3)
            print(f"      [SparseMM] GPU memory after MM: {mem_after_mm:.2f}GB (delta: {mem_after_mm - mem_before:.2f}GB)")
                
        start = time.time()
        # Fix: Ensure we get a proper dense tensor
        # The result has sparse layout but is_sparse=False, which can cause issues
        if hasattr(doc_scores_transposed, 'layout') and doc_scores_transposed.layout == torch.sparse_csr:
            doc_scores_transposed = doc_scores_transposed.to_dense()
        print(f"      time taken to convert to dense: {time.time() - start} seconds")
        
        # Log final memory after conversion
        if torch.cuda.is_available() and self.device.type == "cuda":
            mem_final = torch.cuda.memory_allocated(self.device) / (1024**3)
            print(f"      [SparseMM] GPU memory after dense conversion: {mem_final:.2f}GB (total delta: {mem_final - mem_before:.2f}GB)")
        
        return doc_scores_transposed

    def _compute_relevance_from_scores_gpu_index(
        self,
        query_tokens_ids_batch: torch.Tensor,
        batch_sizes: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Highly optimized GPU scoring using vectorized operations.
        
        This method processes multiple queries simultaneously on GPU.
        
        Parameters
        ----------
        query_tokens_ids_batch : torch.Tensor
            Flattened tensor of token IDs for all queries (1D tensor)
        batch_sizes : torch.Tensor, optional
            Tensor containing the number of tokens in each query. If None, treats as single query.
            
        Returns
        -------
        torch.Tensor
            2D tensor of BM25 scores: (batch_size, num_docs) if batched, (num_docs,) if single query
        """
        # Ensure tokens are on the correct device and dtype
        query_tokens_ids_batch = query_tokens_ids_batch.to(self.device, dtype=torch.long)
        
        # Handle single query case (backward compatibility)
        if batch_sizes is None:
            batch_sizes = torch.tensor([len(query_tokens_ids_batch)], device=self.device)
            single_query = True
        else:
            batch_sizes = batch_sizes.to(self.device, dtype=torch.long)
            single_query = False
        
        batch_size = len(batch_sizes)
        num_docs = self.scores["num_docs"]
        
        # Get the CSC sparse tensor components
        data = self.scores["data"]
        indices = self.scores["indices"] 
        indptr = self.scores["indptr"]
        
        # Initialize result tensor for all queries
        all_doc_scores = torch.zeros((batch_size, num_docs), device=self.device, dtype=torch.float32)
        
        # Process all queries at once using advanced indexing
        if len(query_tokens_ids_batch) == 0:
            return all_doc_scores.squeeze(0) if single_query else all_doc_scores
        
        # Get start/end positions for all query tokens at once
        indptr_starts = indptr[query_tokens_ids_batch]
        indptr_ends = indptr[query_tokens_ids_batch + 1]
        
        # Calculate lengths for each query token
        lengths = indptr_ends - indptr_starts
        
        # Create mask for tokens that have non-zero scores
        valid_tokens = lengths > 0
        
        if not valid_tokens.any():
            return all_doc_scores.squeeze(0) if single_query else all_doc_scores
        
        # Filter to only process tokens with non-zero scores
        valid_indices = torch.where(valid_tokens)[0]
        valid_starts = indptr_starts[valid_tokens]
        valid_lengths = lengths[valid_tokens]
        
        # Create query assignment for each valid token
        query_starts = torch.cumsum(torch.cat([torch.tensor([0], device=self.device), batch_sizes[:-1]]), dim=0)
        query_assignment = torch.searchsorted(query_starts, valid_indices, right=True) - 1
        
        # Create batch indices for all relevant entries
        max_length = valid_lengths.max().item() if len(valid_lengths) > 0 else 0
        
        if max_length == 0:
            return all_doc_scores.squeeze(0) if single_query else all_doc_scores
        
        # Create index offsets for batch processing
        position_indices = torch.arange(max_length, device=self.device).unsqueeze(0)
        valid_positions = position_indices < valid_lengths.unsqueeze(1)
        
        # Calculate flat indices into data/indices arrays
        flat_indices = valid_starts.unsqueeze(1) + position_indices
        flat_indices = flat_indices[valid_positions]
        
        if len(flat_indices) == 0:
            return all_doc_scores.squeeze(0) if single_query else all_doc_scores
        
        # Extract document indices and scores
        batch_doc_indices = indices[flat_indices]
        batch_scores = data[flat_indices]
        
        # Get query assignments for each score (expand valid_positions mask)
        query_indices_expanded = query_assignment.unsqueeze(1).expand(-1, max_length)[valid_positions]
        
        # Create combined indices for 2D scatter: (query_idx * num_docs + doc_idx)
        combined_indices = query_indices_expanded * num_docs + batch_doc_indices
        
        # Flatten the result tensor and use scatter_add
        flat_scores = all_doc_scores.view(-1)
        flat_scores.scatter_add_(0, combined_indices, batch_scores)
        
        # Reshape back to (batch_size, num_docs)
        all_doc_scores = flat_scores.view(batch_size, num_docs)
        
        return all_doc_scores.squeeze(0) if single_query else all_doc_scores

    @staticmethod
    def _compute_relevance_from_scores(
        data: np.ndarray,
        indptr: np.ndarray,
        indices: np.ndarray,
        num_docs: int,
        query_tokens_ids: np.ndarray,
        dtype: np.dtype,
    ) -> np.ndarray:
        """
        This internal static function calculates the relevance scores for a given query,
        by using the BM25 scores that have been precomputed in the BM25 eager index.
        It is used by the `get_scores_from_ids` method, which makes use of the precomputed
        scores assigned as attributes of the BM25 object.

        Parameters
        ----------
        data (np.ndarray)
            Data array of the BM25 index.
        indptr (np.ndarray)
            Index pointer array of the BM25 index.
        indices (np.ndarray)
            Indices array of the BM25 index.
        num_docs (int)
            Number of documents in the BM25 index.
        query_tokens_ids (np.ndarray)
            Array of token IDs to score.
        dtype (np.dtype)
            Data type for score calculation.

        Returns
        -------
        np.ndarray
            Array of BM25 relevance scores for a given query.

        Note
        ----
        This function was optimized by the baguetter library. The original implementation can be found at:
        https://github.com/mixedbread-ai/baguetter/blob/main/baguetter/indices/sparse/models/bm25/index.py
        """
        indptr_starts = indptr[query_tokens_ids]
        indptr_ends = indptr[query_tokens_ids + 1]

        scores = np.zeros(num_docs, dtype=dtype)
        for i in range(len(query_tokens_ids)):
            start, end = indptr_starts[i], indptr_ends[i]
            np.add.at(scores, indices[start:end], data[start:end])

            # # The following code is slower with numpy, but faster after JIT compilation
            # for j in range(start, end):
            #     scores[indices[j]] += data[j]

        return scores

    def build_index_from_ids(
        self,
        unique_token_ids: List[int],
        corpus_token_ids: List[List[int]],
        show_progress=True,
        leave_progress=False,
    ):
        """
        Low-level function to build the BM25 index from token IDs, used by the `index` method,
        as well as the `build_index_from_tokens` method.
        You can override this function if you want to build the index in a different way.

        Parameters
        ----------
        unique_token_ids : List[int]
            List of unique token IDs.

        corpus_token_ids : List[List[int]]
            List of list of token IDs for each document.

        leave_progress : bool
            If True, the progress bars will remain after the function completes.
        """
        import scipy.sparse as sp

        avg_doc_len = np.array([len(doc_ids) for doc_ids in corpus_token_ids]).mean()
        n_docs = len(corpus_token_ids)
        n_vocab = len(unique_token_ids)

        # Step 1: Calculate the number of documents containing each token
        doc_frequencies = _calculate_doc_freqs(
            corpus_tokens=corpus_token_ids,
            unique_tokens=unique_token_ids,
            show_progress=show_progress,
            leave_progress=leave_progress,
        )

        # preliminary: if the method is one of BM25L or BM25+, we need to calculate the non-occurrence array
        if self.method in self.methods_requiring_nonoccurrence:
            self.nonoccurrence_array = _build_nonoccurrence_array(
                doc_frequencies=doc_frequencies,
                n_docs=n_docs,
                compute_idf_fn=_select_idf_scorer(self.idf_method),
                calculate_tfc_fn=_select_tfc_scorer(self.method),
                l_d=avg_doc_len,
                l_avg=avg_doc_len,
                k1=self.k1,
                b=self.b,
                delta=self.delta,
                dtype=self.dtype,
            )
        else:
            self.nonoccurrence_array = None

        # Step 2: Calculate the idf for each token using the document frequencies
        idf_array = _build_idf_array(
            doc_frequencies=doc_frequencies,
            n_docs=n_docs,
            compute_idf_fn=_select_idf_scorer(self.idf_method),
            dtype=self.dtype,
        )

        # Step 3 Calculate the BM25 scores for each token in each document
        scores_flat, doc_idx, vocab_idx = _build_scores_and_indices_for_matrix(
            corpus_token_ids=corpus_token_ids,
            idf_array=idf_array,
            avg_doc_len=avg_doc_len,
            doc_frequencies=doc_frequencies,
            k1=self.k1,
            b=self.b,
            delta=self.delta,
            show_progress=show_progress,
            leave_progress=leave_progress,
            dtype=self.dtype,
            int_dtype=self.int_dtype,
            method=self.method,
            nonoccurrence_array=self.nonoccurrence_array,
        )

        # Now, we build the sparse matrix
        score_matrix = sp.csc_matrix(
            (scores_flat, (doc_idx, vocab_idx)),
            shape=(n_docs, n_vocab),
            dtype=self.dtype,
        )
        data = score_matrix.data
        indices = score_matrix.indices
        indptr = score_matrix.indptr

        scores = {
            "data": data,
            "indices": indices,
            "indptr": indptr,
            "num_docs": n_docs,
        }
        return scores

    def build_index_from_tokens(
        self, corpus_tokens, show_progress=True, leave_progress=False
    ):
        """
        Low-level function to build the BM25 index from tokens, used by the `index` method.
        You can override this function if you want to build the index in a different way.
        """
        unique_tokens = get_unique_tokens(
            corpus_tokens,
            show_progress=show_progress,
            leave_progress=leave_progress,
            desc="BM25S Create Vocab",
        )
        vocab_dict = {token: i for i, token in enumerate(unique_tokens)}
        unique_token_ids = [vocab_dict[token] for token in unique_tokens]

        corpus_token_ids = [
            [vocab_dict[token] for token in tokens]
            for tokens in tqdm(
                corpus_tokens,
                desc="BM25S Convert tokens to indices",
                leave=leave_progress,
                disable=not show_progress,
            )
        ]

        scores = self.build_index_from_ids(
            unique_token_ids=unique_token_ids,
            corpus_token_ids=corpus_token_ids,
            show_progress=show_progress,
            leave_progress=leave_progress,
        )

        return scores, vocab_dict

    def index(
        self,
        corpus: Union[Iterable, Tuple, tokenization.Tokenized],
        create_empty_token=True,
        show_progress=True,
        leave_progress=False,
    ):
        """
        Given a `corpus` of documents, create the BM25 index. The `corpus` can be either:
        - An iterable of documents, where each document is a list of tokens (strings).
        - A tuple of two elements: the first is the list of unique token IDs (int), and the second is the vocabulary dictionary.
        - An object with the `ids` and `vocab` attributes, which are the unique token IDs and the token IDs for each document, respectively.

        Given a list of list of tokens, create the BM25 index.

        You can provide either the `corpus_tokens` or the `corpus_token_ids`. If you provide the `corpus_token_ids`,
        you must also provide the `vocab_dict` dictionary. If you provide the `corpus_tokens`, the vocab_dict
        dictionary will be created from the tokens, so you do not need to provide it.

        The `vocab_dict` dictionary is a mapping from tokens to their index in the vocabulary. This is used to
        create the sparse matrix representation of the BM25 scores, as well as during query time to convert the
        tokens to their indices.

        Parameters
        ----------
        corpus : Iterable or Tuple or tokenization.Tokenized
            The corpus of documents. This can be either:
            - An iterable of documents, where each document is a list of tokens (strings).
            - A tuple of two elements: the first is the list of unique token IDs (int), and the second is the vocabulary dictionary.
            - An object with the `ids` and `vocab` attributes, which are the unique token IDs and the token IDs for each document, respectively.

        create_empty_token : bool
            If True, it will create an empty token, "",  in the vocabulary if it is not already present.
            This is added at the end of the vocabulary and is used to score documents that do not contain any tokens.
            If False, it will not create an empty token, which may lead to an error if a query does not contain any tokens.

        show_progress : bool
            If True, a progress bar will be shown. If False, no progress bar will be shown.

        leave_progress : bool
            If True, the progress bars will remain after the function completes.
        """
        inferred_corpus_obj = self._infer_corpus_object(corpus)

        if inferred_corpus_obj == "tokens":
            logger.debug(msg="Building index from tokens")
            scores, vocab_dict = self.build_index_from_tokens(
                corpus, leave_progress=leave_progress, show_progress=show_progress
            )
        else:
            if inferred_corpus_obj == "tuple":
                logger.debug(msg="Building index from IDs")
                corpus_token_ids, vocab_dict = corpus
            elif inferred_corpus_obj == "object":
                logger.debug(msg="Building index from IDs objects")
                corpus_token_ids = corpus.ids
                vocab_dict = corpus.vocab
            elif inferred_corpus_obj == "token_ids":
                # we need to create a vocab_dict from the unique token IDs
                logger.debug(msg="Building index from token IDs")
                corpus_token_ids = corpus
                unique_ids = set()
                for doc_ids in corpus_token_ids:
                    unique_ids.update(doc_ids)
                # if there's allowed empty token, we need to add it to the vocab_dict to either 0 or max+1
                if create_empty_token:
                    if 0 not in unique_ids:
                        unique_ids.add(0)
                    else:
                        unique_ids.add(max(unique_ids) + 1)
                
                # create the vocab_dict from the unique token IDs
                vocab_dict = {token_id: i for i, token_id in enumerate(unique_ids)}
                
            else:
                raise ValueError(
                    "Internal error: Found an invalid corpus object, indicating `_inferred_corpus_object` is not working correctly."
                )

            unique_token_ids = list(vocab_dict.values())
            scores = self.build_index_from_ids(
                unique_token_ids=unique_token_ids,
                corpus_token_ids=corpus_token_ids,
                leave_progress=leave_progress,
                show_progress=show_progress,
            )

        if create_empty_token:
            if inferred_corpus_obj != "token_ids" and "" not in vocab_dict:
                vocab_dict[""] = max(vocab_dict.values()) + 1

        if self.device.type == "cuda":
            # Convert scipy sparse matrix data to PyTorch CSC tensor and pin to GPU memory
            self.scores = self._convert_scores_to_torch_csc(scores)
            logger.debug(f"GPU acceleration enabled for BM25 method: {self.method}")
        else:
            self.scores = scores

        self.vocab_dict = vocab_dict

        # we create unique token IDs from the vocab_dict for faster lookup
        self.unique_token_ids_set = set(self.vocab_dict.values())

    def get_tokens_ids(self, query_tokens: List[str]) -> List[int]:
        """
        For a given list of tokens, return the list of token IDs, leaving out tokens
        that are not in the vocabulary.
        """
        return [
            self.vocab_dict[token] for token in query_tokens if token in self.vocab_dict
        ]

    def get_scores_from_ids(
        self, query_tokens_ids: List[int], weight_mask=None
    ) -> np.ndarray:
        data = self.scores["data"]
        indices = self.scores["indices"]
        indptr = self.scores["indptr"]
        num_docs = self.scores["num_docs"]

        dtype = np.dtype(self.dtype)
        int_dtype = np.dtype(self.int_dtype)
        query_tokens_ids: np.ndarray = np.asarray(query_tokens_ids, dtype=int_dtype)

        max_token_id = int(query_tokens_ids.max(initial=0))

        if max_token_id >= len(indptr) - 1:
            raise ValueError(
                f"The maximum token ID in the query ({max_token_id}) is higher than the number of tokens in the index."
                "This likely means that the query contains tokens that are not in the index."
            )

        scores = self._compute_relevance_from_scores(
            data=data,
            indptr=indptr,
            indices=indices,
            num_docs=num_docs,
            query_tokens_ids=query_tokens_ids,
            dtype=dtype,
        )

        if weight_mask is not None:
            # multiply the scores by the weight mask
            scores *= weight_mask

        # if there's a non-occurrence array, we need to add the non-occurrence score
        # back to the scores
        if self.nonoccurrence_array is not None:
            nonoccurrence_scores = self.nonoccurrence_array[query_tokens_ids].sum()
            scores += nonoccurrence_scores

        return scores

    def _create_sparse_query_tensor_cpu(self, query_tokens_batched: List[List[str]]) -> torch.sparse.Tensor:
        """
        Create sparse query tensor on CPU only (for pipeline efficiency).
        This allows CPU tensor creation to happen concurrently with GPU operations.
        """
        assert query_tokens_batched
            
        batch_size = len(query_tokens_batched)
        vocab_size = self.scores["vocab_size"]
        start = time.time()
        
        # Collect all (batch_idx, token_id) pairs and count them efficiently
        pairs = []
        for batch_idx, query_tokens in enumerate(query_tokens_batched):
            if len(query_tokens) == 0:
                continue
                
            if isinstance(query_tokens[0], str):
                query_ids = self.get_tokens_ids(query_tokens)
            else:
                query_ids = query_tokens
            
            # Add valid token pairs for this batch
            for token_id in query_ids:
                pairs.append((batch_idx, token_id))
        
        # Count frequencies using Counter (highly optimized C implementation)
        frequency_dict = Counter(pairs)
        print(f"        [CPU Tensor] Time to count frequencies: {time.time() - start}")
        
        if not frequency_dict:
            # No valid tokens found, return empty sparse tensor
            return torch.sparse_csr_tensor(
                torch.zeros((1,), dtype=torch.int64),
                torch.zeros((0,), dtype=torch.int64),
                torch.zeros((0,), dtype=torch.float16),
                size=(vocab_size, batch_size),
                device='cpu'
            )
        
        start = time.time()
        # Extract indices and values from frequency dictionary
        batch_indices = []
        token_indices = []
        frequencies = []
        
        for (batch_idx, token_id), freq in frequency_dict.items():
            batch_indices.append(batch_idx)
            token_indices.append(token_id)
            frequencies.append(freq)
        
        print(f"        [CPU Tensor] Time to extract indices: {time.time() - start}")
        
        start = time.time()

        # Sort by token_indices (rows) to organize data for CSR format - all on CPU
        sorted_data = sorted(zip(token_indices, batch_indices, frequencies))
        
        if not sorted_data:
            # Handle empty case - create CPU arrays first
            crow_indices_cpu = [0] * (vocab_size + 1)
            col_indices_cpu = []
            values_cpu = []
        else:
            sorted_token_indices, sorted_batch_indices, sorted_frequencies = zip(*sorted_data)
            
            # Keep as CPU lists/arrays for now
            col_indices_cpu = list(sorted_batch_indices)
            values_cpu = list(sorted_frequencies)
            
            # Create row pointers (crow_indices) for CSR format - on CPU
            crow_indices_cpu = [0] * (vocab_size + 1)
            
            # Count entries per row (token) - all CPU operations
            current_row = 0
            for i, token_idx in enumerate(sorted_token_indices):
                while current_row < token_idx:
                    crow_indices_cpu[current_row + 1] = i
                    current_row += 1
                crow_indices_cpu[token_idx + 1] = i + 1
            
            # Fill remaining entries
            while current_row < vocab_size:
                crow_indices_cpu[current_row + 1] = len(sorted_data)
                current_row += 1
                
        # Create sparse CSR tensor on CPU only
        print(f"        [CPU Tensor] Creating sparse tensor {vocab_size}x{batch_size} on CPU")
        sparse_query_T_cpu = torch.sparse_csr_tensor(
            crow_indices_cpu,
            col_indices_cpu, 
            values_cpu,
            size=(vocab_size, batch_size),
            device='cpu',
            dtype=torch.float16
        )
        
        print(f"        [CPU Tensor] Time to create sparse CSR tensor: {time.time() - start}")
        return sparse_query_T_cpu

    def create_query_tensor_padded(self, query_tokens_batched: List[List[str]]) -> torch.Tensor:
        """
        Convert a batch of queries to a zero-padded tensor format.
        
        Parameters
        ----------
        query_tokens_batched : List[List[str]]
            List of queries, where each query is a list of tokens
            
        Returns
        -------
        torch.Tensor
            Zero-padded tensor of shape (batch_size, max_query_length)
        """
        if not query_tokens_batched:
            return torch.zeros((0, 0), device=self.device, dtype=torch.long)
            
        # Convert string tokens to token IDs and find max length in single pass
        query_ids_batched = []
        max_length = 0
        
        for query_tokens in query_tokens_batched:
            if len(query_tokens) == 0:
                query_ids = []
            elif isinstance(query_tokens[0], str):
                query_ids = self.get_tokens_ids(query_tokens)
            else:
                query_ids = query_tokens
            query_ids_batched.append(query_ids)
            max_length = max(max_length, len(query_ids))
        
        batch_size = len(query_ids_batched)
        
        if max_length == 0:
            return torch.zeros((batch_size, 0), device=self.device, dtype=torch.long)
        
        # Create padded list directly (avoiding numpy array initialization)
        padded_data = []
        for query_ids in query_ids_batched:
            padded_query = query_ids + [0] * (max_length - len(query_ids))
            padded_data.extend(padded_query)
                
        # Single tensor creation and reshape
        return torch.tensor(padded_data, device=self.device, dtype=torch.long).view(batch_size, max_length)

    def create_query_tensor(self, query_tokens_batched: List[List[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        query_ids_batched = []
        batch_sizes = []
        
        for query_tokens in query_tokens_batched:
            if len(query_tokens) == 0:
                query_ids = []
            elif isinstance(query_tokens[0], str):
                query_ids = self.get_tokens_ids(query_tokens)
            else:
                query_ids = query_tokens  # Already token IDs
            query_ids_batched.extend(query_ids)
            batch_sizes.append(len(query_ids))
        
        if not query_ids_batched:
            # All queries are empty
            return np.zeros((len(query_tokens_batched), self.scores["num_docs"]), dtype=np.float32)
        
        # Convert to tensors
        query_tokens_tensor = torch.tensor(query_ids_batched, device=self.device, dtype=torch.long)
        batch_sizes_tensor = torch.tensor(batch_sizes, device=self.device, dtype=torch.long)
        return query_tokens_tensor, batch_sizes_tensor
    
    def get_scores(
        self, query_tokens_single: List[str], weight_mask=None
    ) -> np.ndarray:
        if not isinstance(query_tokens_single, list):
            raise ValueError("The query_tokens must be a list of tokens.")

        if isinstance(query_tokens_single[0], str):
            query_tokens_ids = self.get_tokens_ids(query_tokens_single)
        elif isinstance(query_tokens_single[0], int):
            # already are token IDs, no need to convert
            query_tokens_ids = query_tokens_single
        else:
            raise ValueError(
                "The query_tokens must be a list of tokens or a list of token IDs."
            )

        return self.get_scores_from_ids(query_tokens_ids, weight_mask=weight_mask)

    def _log_gpu_memory(self, stage: str, batch_idx: int = None):
        """Log current GPU memory usage with stage information"""
        if torch.cuda.is_available() and self.device.type == "cuda":
            try:
                allocated = torch.cuda.memory_allocated(self.device) / (1024**3)
                cached = torch.cuda.memory_reserved(self.device)
                cached_gb = cached / (1024**3)
                
                if batch_idx is not None:
                    print(f"[GPU Memory] {stage} (batch {batch_idx}): {allocated:.2f}GB allocated, {cached_gb:.2f}GB cached")
                else:
                    print(f"[GPU Memory] {stage}: {allocated:.2f}GB allocated, {cached_gb:.2f}GB cached")
            except Exception as e:
                print(f"[GPU Memory] Failed to get memory stats at {stage}: {e}")

    @torch.inference_mode()
    def _get_top_k_results_batched(
        self,
        query_tokens_batched: List[List[str]],
        k: int = 1000,
        backend="auto",
        sorted: bool = False,
        weight_mask: np.ndarray = None,
        micro_batch_size: int = 2048,
    ):
        """
        Get top-k results for a batch of queries using GPU acceleration with micro-batching.
        
        This method processes queries in micro-batches to manage GPU memory effectively.
        Each micro-batch is completely independent and GPU memory is freed between batches.
        
        Parameters
        ----------
        query_tokens_batched : List[List[str]]
            List of queries, where each query is a list of tokens
        k : int
            Number of top documents to retrieve for each query
        backend : str
            Backend to use (ignored, uses GPU)
        sorted : bool
            Whether to sort the results by score
        weight_mask : np.ndarray, optional
            Weight mask to apply to document scores
        micro_batch_size : int
            Number of queries to process in each micro-batch
            
        Returns
        -------
        tuple
            (batch_scores, batch_indices) where each is np.ndarray of shape (batch_size, k)
        """
        if not query_tokens_batched:
            return np.array([]), np.array([])
            
        batch_size = len(query_tokens_batched)
        self._log_gpu_memory(f"Start batched processing ({batch_size} queries, micro_batch_size={micro_batch_size})")
        
        # If batch is small enough, process directly
        if batch_size <= micro_batch_size:
            print(f"[Pipeline] Processing single batch of {batch_size} queries")
            sparse_query_tensor = self._create_sparse_query_tensor_cpu(query_tokens_batched)
            if self.device.type == "cuda":
                sparse_query_tensor = sparse_query_tensor.to(self.device)
            
            # Get scores
            scores_tensor = self._compute_relevance_from_scores_gpu_sparse_mm(sparse_query_tensor)
            del sparse_query_tensor
            
            # Compute top-k on GPU
            topk_scores, topk_indices = self._compute_topk_on_gpu(scores_tensor, k, sorted)
            
            # Transfer to CPU
            batch_scores, batch_indices = self._transfer_topk_to_cpu(topk_scores, topk_indices)
            
            self._log_gpu_memory("End single batch processing")
            return batch_scores, batch_indices
        
        # Process in micro-batches
        num_micro_batches = (batch_size + micro_batch_size - 1) // micro_batch_size
        print(f"[Pipeline] Processing {batch_size} queries in {num_micro_batches} micro-batches of size {micro_batch_size}")
        print(f"[Pipeline] Using CPU/GPU interleaving to hide sparse tensor creation latency")
        logger.debug(f"Processing {batch_size} queries in micro batches of size {micro_batch_size}")
        
        all_batch_scores = []
        all_batch_indices = []
        
        # Pre-create sparse tensor for first micro-batch
        first_micro_batch = query_tokens_batched[0:min(micro_batch_size, batch_size)]
        print(f"[Pipeline] Pre-creating sparse tensor for first micro-batch")
        first_sparse_tensor_cpu = self._create_sparse_query_tensor_cpu(first_micro_batch)
        
        # Transfer first tensor to GPU
        if self.device.type == "cuda":
            current_sparse_tensor = first_sparse_tensor_cpu.to(self.device)
            del first_sparse_tensor_cpu
        else:
            current_sparse_tensor = first_sparse_tensor_cpu
            
        self._log_gpu_memory("After first sparse tensor creation")
        
        # Keep track of pending transfers from previous iteration
        pending_transfer = None
        
        for i, start_idx in enumerate(range(0, batch_size, micro_batch_size)):
            end_idx = min(start_idx + micro_batch_size, batch_size)
            micro_batch_actual_size = end_idx - start_idx
            
            print(f"[Pipeline {i+1}/{num_micro_batches}] Processing queries {start_idx}-{end_idx-1} ({micro_batch_actual_size} queries)")
            self._log_gpu_memory("Before micro-batch processing", i+1)

            # Start async sparse matrix multiplication on GPU
            print(f"  [Pipeline] Starting async sparse MM for current batch")
            start_time = time.time()
            scores_tensor = self._compute_relevance_from_scores_gpu_sparse_mm(
                current_sparse_tensor
            )
            print(f"  [Pipeline] Sparse MM dispatched (async): {time.time() - start_time:.3f}s")
            self._log_gpu_memory("After sparse MM dispatch")
            
            # While GPU is working on sparse MM, create next sparse tensor on CPU
            next_sparse_tensor_cpu = None
            if i + 1 < num_micro_batches:
                next_start_idx = end_idx
                next_end_idx = min(next_start_idx + micro_batch_size, batch_size)
                next_micro_batch = query_tokens_batched[next_start_idx:next_end_idx]
                
                print(f"  [Pipeline] Creating next sparse tensor on CPU while GPU processes current batch")
                cpu_start_time = time.time()
                next_sparse_tensor_cpu = self._create_sparse_query_tensor_cpu(next_micro_batch)
                print(f"  [Pipeline] CPU tensor creation: {time.time() - cpu_start_time:.3f}s")
            
            # Compute top-k on GPU (this forces sync with sparse MM)
            print(f"  [Pipeline] Computing top-k (this will sync GPU)")
            topk_scores, topk_indices = self._compute_topk_on_gpu(scores_tensor, k, sorted)
            
            # Clean up current sparse tensor
            del current_sparse_tensor
            
            # Complete any pending CPU transfer from previous iteration while setting up next batch
            if pending_transfer is not None:
                prev_topk_scores, prev_topk_indices, prev_batch_idx = pending_transfer
                print(f"  [Pipeline] Completing previous batch transfer while preparing next batch")
                prev_scores, prev_indices = self._transfer_topk_to_cpu(prev_topk_scores, prev_topk_indices)
                all_batch_scores.append(prev_scores)
                all_batch_indices.append(prev_indices)
                pending_transfer = None
            
            # Transfer next tensor to GPU for next iteration
            if next_sparse_tensor_cpu is not None:
                print(f"  [Pipeline] Transferring next tensor to GPU")
                if self.device.type == "cuda":
                    current_sparse_tensor = next_sparse_tensor_cpu.to(self.device)
                    del next_sparse_tensor_cpu
                else:
                    current_sparse_tensor = next_sparse_tensor_cpu
            
            # For all iterations except the last, queue the transfer for background processing
            if i < num_micro_batches - 1:
                print(f"  [Pipeline] Queuing GPU-to-CPU transfer for background processing")
                pending_transfer = (topk_scores, topk_indices, i)
            else:
                # Last iteration - do transfer immediately
                print(f"  [Pipeline] Final batch - transferring immediately")
                micro_scores, micro_indices = self._transfer_topk_to_cpu(topk_scores, topk_indices)
                all_batch_scores.append(micro_scores)
                all_batch_indices.append(micro_indices)
            
            self._log_gpu_memory("After micro-batch processing", i+1)
            
            # Free GPU memory between micro-batches
            if self.device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                self._log_gpu_memory("After GPU cache clear", i+1)
        
        # Complete any remaining pending transfer
        if pending_transfer is not None:
            prev_topk_scores, prev_topk_indices, prev_batch_idx = pending_transfer
            print(f"  [Pipeline] Completing final pending transfer")
            prev_scores, prev_indices = self._transfer_topk_to_cpu(prev_topk_scores, prev_topk_indices)
            all_batch_scores.append(prev_scores)
            all_batch_indices.append(prev_indices)
                
        print(f"[Micro-batch] Concatenating results from {num_micro_batches} micro-batches")
        self._log_gpu_memory("Before concatenation")
        
        # Concatenate all results
        final_scores = np.concatenate(all_batch_scores, axis=0)
        final_indices = np.concatenate(all_batch_indices, axis=0)
        
        self._log_gpu_memory("End batched processing")
        return final_scores, final_indices
    
    def _compute_topk_on_gpu(
        self,
        scores_tensor: torch.Tensor,
        k: int,
        sorted: bool = False,
    ):
        """Compute top-k on GPU, returns GPU tensors"""
        batch_size = scores_tensor.shape[1]
        print(f"  [TopK] Processing {batch_size} queries for top-k selection")
        
        start = time.time()
        # Get top-k for all queries simultaneously
        if sorted:
            # Get top-k with sorting (descending order)
            topk_scores, topk_indices = torch.topk(scores_tensor, k, dim=0, sorted=True, largest=True)
        else:
            # Get top-k without sorting (faster)
            topk_scores, topk_indices = torch.topk(scores_tensor, k, dim=0, sorted=False, largest=True)
        print(f"  [TopK] Time taken for top-k: {time.time() - start:.3f}s")     
        self._log_gpu_memory("After top-k computation")
        
        # Clean up scores tensor immediately
        del scores_tensor
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return topk_scores, topk_indices
    
    def _transfer_topk_to_cpu(
        self,
        topk_scores: torch.Tensor,
        topk_indices: torch.Tensor,
    ):
        """Transfer top-k results from GPU to CPU (can be async)"""
        start = time.time()
        
        # Start async transfer to CPU (these operations can overlap with next batch)
        batch_scores = topk_scores.T.cpu().numpy()
        batch_indices = topk_indices.T.cpu().numpy()
        
        print(f"  [Transfer] Time taken to transfer to CPU: {time.time() - start:.3f}s")
        self._log_gpu_memory("After CPU transfer")
        
        # Clean up GPU tensors
        del topk_scores, topk_indices
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._log_gpu_memory("After topk tensor cleanup")
        
        return batch_scores, batch_indices

    def _get_top_k_results(
        self,
        query_tokens_single: List[str],
        k: int = 1000,
        backend="auto",
        sorted: bool = False,
        weight_mask: np.ndarray = None,
    ):
        """
        This function is used to retrieve the top-k results for a single query.
        Since it's a hidden function, the user should not call it directly and
        may change in the future. Please use the `retrieve` function instead.
        """
        if len(query_tokens_single) == 0:
            logger.info(
                msg="The query is empty. This will result in a zero score for all documents."
            )
            scores_q = np.zeros(self.scores["num_docs"], dtype=self.dtype)
        else:
            scores_q = self.get_scores(query_tokens_single, weight_mask=weight_mask)

        if backend.startswith("numba"):
            if selection_jit is None:
                raise ImportError(
                    "Numba is not installed. Please install numba to use the numba backend."
                )
            topk_scores, topk_indices = selection_jit.topk(
                scores_q, k=k, sorted=sorted, backend=backend
            )
        else:
            topk_scores, topk_indices = selection.topk(
                scores_q, k=k, sorted=sorted, backend=backend
            )

        return topk_scores, topk_indices

    def retrieve(
        self,
        query_tokens: Union[List[List[str]], tokenization.Tokenized],
        corpus: List[Any] = None,
        k: int = 10,
        sorted: bool = True,
        return_as: str = "tuple",
        show_progress: bool = True,
        leave_progress: bool = False,
        n_threads: int = 0,
        chunksize: int = 50,
        backend_selection: str = "auto",
        weight_mask: np.ndarray = None,
    ):
        """
        Retrieve the top-k documents for each query (tokenized).

        Parameters
        ----------
        query_tokens : List[List[str]] or bm25s.tokenization.Tokenized
            List of list of tokens for each query. If a Tokenized object is provided,
            it will be converted to a list of list of tokens.

        corpus : List[str] or np.ndarray
            List of "documents" or a numpy array of documents. If provided, the function
            will return the documents instead of the indices. You do not have to provide
            the original documents (for example, you can provide the unique IDs of the
            documents here and then retrieve the actual documents from another source).

        k : int
            Number of documents to retrieve for each query.

        batch_size : int
            Number of queries to process in each batch. Internally, the function will
            process the queries in batches to speed up the computation.

        sorted : bool
            If True, the function will sort the results by score before returning them.

        return_as : str
            If return_as="tuple", a named tuple with two fields will be returned:
            `documents` and `scores`, which can be accessed as `result.documents` and
            `result.scores`, or by unpacking, e.g. `documents, scores = retrieve(...)`.
            If return_as="documents", only the retrieved documents (or indices if `corpus`
            is not provided) will be returned.

        show_progress : bool
            If True, a progress bar will be shown. If False, no progress bar will be shown.

        leave_progress : bool
            If True, the progress bars will remain after the function completes.

        n_threads : int
            Number of jobs to run in parallel. If -1, it will use all available CPUs.
            If 0, it will run the jobs sequentially, without using multiprocessing.

        chunksize : int
            Number of batches to process in each job in the multiprocessing pool.

        backend_selection : str
            The backend to use for the top-k retrieval. Choose from "auto", "numpy", "jax".
            If "auto", it will use JAX if it is available, otherwise it will use numpy.

        weight_mask : np.ndarray
            A weight mask to filter the documents. If provided, the scores for the masked
            documents will be set to 0 to avoid returning them in the results.

        Returns
        -------
        Results or np.ndarray
            If `return_as="tuple"`, a named tuple with two fields will be returned: `documents` and `scores`.
            If `return_as="documents"`, only the retrieved documents (or indices if `corpus` is not provided) will be returned.

        Raises
        ------
        ValueError
            If the `query_tokens` is not a list of list of tokens (str) or a tuple of two lists: the first list is the list of unique token IDs, and the second list is the list of token IDs for each document.

        ImportError
            If the numba backend is selected but numba is not installed.
        """
        num_docs = self.scores["num_docs"]
        if k > num_docs:
            raise ValueError(
                f"k of {k} is larger than the number of available scores"
                f", which is {num_docs} (corpus size should be larger than top-k)."
                f" Please set with a smaller k or increase the size of corpus."
            )
        allowed_return_as = ["tuple", "documents"]

        if return_as not in allowed_return_as:
            raise ValueError("`return_as` must be either 'tuple' or 'documents'")
        else:
            pass

        if n_threads == -1:
            n_threads = os.cpu_count()

        # if it's a list of list of tokens ids (int), we remove any integer not in the vocab_dict
        if is_list_of_list_of_type(query_tokens, type_=int):
            if not hasattr(self, "unique_token_ids_set") or self.unique_token_ids_set is None:
                raise ValueError(
                    "The unique_token_ids_set attribute is not found. Please run the `index` method first, or make sure"
                    "run retriever.load(load_vocab=True) before calling retrieve on list of list of token IDs (int)."
                )
            query_tokens_filtered = []
            for query in query_tokens:
                query_filtered = [
                    token_id
                    for token_id in query
                    if token_id in self.unique_token_ids_set
                ]
                if len(query_filtered) == 0:
                    if "" not in self.vocab_dict:
                        raise ValueError(
                            "The query does not contain any tokens that are in the vocabulary. "
                            "Please provide a query that contains at least one token that is in the vocabulary. "
                            "Alternatively, you can set `create_empty_token=True` when calling `index` to add an empty token to the vocabulary. "
                            "You can also manually add an empty token to the vocabulary by setting `retriever.vocab_dict[''] = max(retriever.vocab_dict.values()) + 1`. "
                            "Then, run `retriever.unique_token_ids_set = set(retriever.vocab_dict.values())` to update the unique token IDs."
                        )
                    query_filtered = [self.vocab_dict[""]]

                query_tokens_filtered.append(query_filtered)

            query_tokens = query_tokens_filtered

        if isinstance(query_tokens, tuple) and not _is_tuple_of_list_of_tokens(
            query_tokens
        ):
            if len(query_tokens) != 2:
                msg = (
                    "Expected a list of string or a tuple of two elements: the first element is the "
                    "list of unique token IDs, "
                    "and the second element is the list of token IDs for each document."
                    f"Found {len(query_tokens)} elements instead."
                )
                raise ValueError(msg)
            else:
                ids, vocab = query_tokens
                if not isinstance(ids, Iterable):
                    raise ValueError(
                        "The first element of the tuple passed to retrieve must be an iterable."
                    )
                if not isinstance(vocab, dict):
                    raise ValueError(
                        "The second element of the tuple passed to retrieve must be a dictionary."
                    )
                query_tokens = tokenization.Tokenized(ids=ids, vocab=vocab)

        if isinstance(query_tokens, tokenization.Tokenized):
            query_tokens = tokenization.convert_tokenized_to_string_list(query_tokens)

        corpus = corpus if corpus is not None else self.corpus

        if weight_mask is not None:
            if not isinstance(weight_mask, np.ndarray):
                raise ValueError("weight_mask must be a numpy array.")

            # check if weight_mask is a 1D array, if not raise an error
            if weight_mask.ndim != 1:
                raise ValueError("weight_mask must be a 1D array.")

            # check if the length of the weight_mask is the same as the length of the corpus
            if len(weight_mask) != self.scores["num_docs"]:
                raise ValueError(
                    "The length of the weight_mask must be the same as the length of the corpus."
                )
        if self.backend == "numba":
            start = time.time()
            if _retrieve_numba_functional is None:
                raise ImportError(
                    "Numba is not installed. Please install numba wiith `pip install numba` to use the numba backend."
                )

            backend_selection = (
                "numba" if backend_selection == "auto" else backend_selection
            )
            # if is list of list of int
            if is_list_of_list_of_type(query_tokens, type_=int):
                query_tokens_ids = query_tokens
            elif is_list_of_list_of_type(query_tokens, type_=str):
                query_tokens_ids = [self.get_tokens_ids(q) for q in query_tokens]
            else:
                raise ValueError(
                    "The query_tokens must be a list of list of tokens (str for stemmed words, int for token ids matching corpus) or a tuple of two lists: the first list is the list of unique token IDs, and the second list is the list of token IDs for each document."
                )

            res = _retrieve_numba_functional(
                query_tokens_ids=query_tokens_ids,
                scores=self.scores,
                corpus=corpus,
                k=k,
                sorted=sorted,
                return_as=return_as,
                show_progress=show_progress,
                leave_progress=leave_progress,
                n_threads=n_threads,
                chunksize=None,  # chunksize is ignored in the numba backend
                backend_selection=backend_selection,  # backend_selection is ignored in the numba backend
                dtype=self.dtype,
                int_dtype=self.int_dtype,
                nonoccurrence_array=self.nonoccurrence_array,
                weight_mask=weight_mask,
            )
            print(f"time taken numba: {time.time() - start}")        
            if return_as == "tuple":
                return Results(documents=res[0], scores=res[1])
            else:
                return res
        if self.device.type == "cuda":
            # Use batched GPU processing with micro-batching for better memory management
            batch_scores, batch_indices = self._get_top_k_results_batched(
                query_tokens_batched=query_tokens,
                k=k,
                backend=backend_selection,
                sorted=sorted,
                weight_mask=weight_mask,
            )
            
            start = time.time()
            # Post-process results to match expected format
            if return_as == "scores":
                print(f"time taken post process: {time.time() - start}")        
                return batch_scores
            
            retrieved_docs = self._retrieve_documents_from_corpus(batch_indices, corpus, return_as)
            
            print(f"time taken post process: {time.time() - start}")        
            if return_as == "tuple":
                return Results(documents=retrieved_docs, scores=batch_scores)
            elif return_as == "documents":
                return retrieved_docs
            else:
                raise ValueError("`return_as` must be either 'tuple' or 'documents'")

        tqdm_kwargs = {
            "total": len(query_tokens),
            "desc": "BM25S Retrieve",
            "leave": leave_progress,
            "disable": not show_progress,
        }
        start = time.time()
        topk_fn = partial(
            self._get_top_k_results,
            k=k,
            sorted=sorted,
            backend=backend_selection,
            weight_mask=weight_mask,
        )
        if n_threads == 0:
            # Use a simple map function to retrieve the results
            out = tqdm(map(topk_fn, query_tokens), **tqdm_kwargs)
        else:
            # Use concurrent.futures.ProcessPoolExecutor to parallelize the computation
            with ThreadPoolExecutor(max_workers=n_threads) as executor:
                process_map = executor.map(
                    topk_fn,
                    query_tokens,
                    chunksize=chunksize,
                )
                out = list(tqdm(process_map, **tqdm_kwargs))

        scores, indices = zip(*out)
        scores, indices = np.array(scores), np.array(indices)
        print(f"time taken topk_fn: {time.time() - start}")        
        start = time.time()
        if return_as == "scores":
            return scores
        print(f"time taken post process: {time.time() - start}")        
        retrieved_docs = self._retrieve_documents_from_corpus(indices, corpus, return_as)

        if return_as == "tuple":
            return Results(documents=retrieved_docs, scores=scores)
        elif return_as == "documents":
            return retrieved_docs
        else:
            raise ValueError("`return_as` must be either 'tuple' or 'documents'")

    def save(
        self,
        save_dir,
        corpus=None,
        data_name="data.csc.index.npy",
        indices_name="indices.csc.index.npy",
        indptr_name="indptr.csc.index.npy",
        vocab_name="vocab.index.json",
        params_name="params.index.json",
        nnoc_name="nonoccurrence_array.index.npy",
        corpus_name="corpus.jsonl",
        allow_pickle=False,
    ):
        """
        Save the BM25S index to the `save_dir` directory. This will save the scores array,
        the indices array, the indptr array, the vocab dictionary, and the parameters.

        Parameters
        ----------
        save_dir : str
            The directory where the BM25S index will be saved.

        corpus : List[Dict]
            The corpus of documents. If provided, it will be saved to the `corpus` file.

        corpus_name : str
            The name of the file that will contain the corpus.

        data_name : str
            The name of the file that will contain the data array.

        indices_name : str
            The name of the file that will contain the indices array.

        indptr_name : str
            The name of the file that will contain the indptr array.

        vocab_name : str
            The name of the file that will contain the vocab dictionary.

        params_name : str
            The name of the file that will contain the parameters.

        nnoc_name : str
            The name of the file that will contain the non-occurrence array.

        allow_pickle : bool
            If True, the arrays will be saved using pickle. If False, the arrays will be saved
            in a more efficient format, but they will not be readable by older versions of numpy.
        """
        # Save the self.vocab_dict and self.score_matrix to the save_dir
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save the scores arrays
        data_path = save_dir / data_name
        indices_path = save_dir / indices_name
        indptr_path = save_dir / indptr_name

        # Handle GPU tensors by converting to CPU numpy arrays first
        if self.device.type == "cuda" and "sparse_tensor" in self.scores:
            # GPU tensors - convert to CPU first
            print("Converting GPU tensors to CPU for saving...")
            data_cpu = self.scores["data"].cpu().numpy()
            indices_cpu = self.scores["indices"].cpu().numpy()
            indptr_cpu = self.scores["indptr"].cpu().numpy()
            
            np.save(data_path, data_cpu, allow_pickle=allow_pickle)
            np.save(indices_path, indices_cpu, allow_pickle=allow_pickle)
            np.save(indptr_path, indptr_cpu, allow_pickle=allow_pickle)
        else:
            # CPU arrays - save directly
            np.save(data_path, self.scores["data"], allow_pickle=allow_pickle)
            np.save(indices_path, self.scores["indices"], allow_pickle=allow_pickle)
            np.save(indptr_path, self.scores["indptr"], allow_pickle=allow_pickle)

        # save nonoccurrence array if it exists
        if self.nonoccurrence_array is not None:
            nnm_path = save_dir / nnoc_name
            np.save(nnm_path, self.nonoccurrence_array, allow_pickle=allow_pickle)

        # Save the vocab dictionary
        vocab_path = save_dir / vocab_name

        with open(vocab_path, "wt", encoding="utf-8") as f:
            f.write(json_functions.dumps(self.vocab_dict, ensure_ascii=False))

        # Save the parameters
        params_path = save_dir / params_name
        params = dict(
            k1=self.k1,
            b=self.b,
            delta=self.delta,
            method=self.method,
            idf_method=self.idf_method,
            dtype=self.dtype,
            int_dtype=self.int_dtype,
            num_docs=self.scores["num_docs"],
            version=__version__,
            backend=self.backend,
        )
        with open(params_path, "w") as f:
            json.dump(params, f, indent=4)

        corpus = corpus if corpus is not None else self.corpus

        if corpus is not None:
            with open(save_dir / corpus_name, "wt", encoding="utf-8") as f:
                # if it's not an iterable, we skip
                if not isinstance(corpus, Iterable):
                    logging.warning(
                        "The corpus is not an iterable. Skipping saving the corpus."
                    )

                for i, doc in enumerate(corpus):
                    if isinstance(doc, str):
                        doc = {"id": i, "text": doc}
                    elif isinstance(doc, (dict, list, tuple)):
                        doc = doc
                    else:
                        logging.warning(
                            f"Document at index {i} is not a string, dictionary, list or tuple. Skipping."
                        )
                        continue

                    try:
                        doc_str = json_functions.dumps(doc, ensure_ascii=False)
                    except Exception as e:
                        logging.warning(f"Error saving document at index {i}: {e}")
                    else:
                        f.write(doc_str + "\n")

            # also save corpus.mmindex
            mmidx = utils.corpus.find_newline_positions(save_dir / corpus_name)
            utils.corpus.save_mmindex(mmidx, path=save_dir / corpus_name)

    def load_scores(
        self,
        save_dir,
        data_name="data.csc.index.npy",
        indices_name="indices.csc.index.npy",
        indptr_name="indptr.csc.index.npy",
        num_docs=None,
        mmap=False,
        allow_pickle=False,
    ):
        """
        Load the scores arrays from the BM25 index. This is useful if you want to load
        the scores arrays separately from the vocab dictionary and the parameters.

        This is called internally by the `load` method, so you do not need to call it directly.

        Parameters
        ----------
        data_name : str
            The name of the file that contains the data array.

        indices_name : str
            The name of the file that contains the indices array.

        indptr_name : str
            The name of the file that contains the indptr array.

        mmap : bool
            Whether to use Memory-map for the np.load function. If false, the arrays will be loaded into memory.
            If True, the arrays will be memory-mapped, using 'r' mode. This is useful for very large arrays that
            do not fit into memory.

        allow_pickle : bool
            If True, the arrays will be loaded using pickle. If False, the arrays will be loaded
            in a more efficient format, but they will not be readable by older versions of numpy.
        """
        save_dir = Path(save_dir)

        data_path = save_dir / data_name
        indices_path = save_dir / indices_name
        indptr_path = save_dir / indptr_name

        mmap_mode = "r" if mmap else None
        data = np.load(data_path, allow_pickle=allow_pickle, mmap_mode=mmap_mode)
        indices = np.load(indices_path, allow_pickle=allow_pickle, mmap_mode=mmap_mode)
        indptr = np.load(indptr_path, allow_pickle=allow_pickle, mmap_mode=mmap_mode)

        scores = {}
        scores["data"] = data
        scores["indices"] = indices
        scores["indptr"] = indptr
        scores["num_docs"] = num_docs

        self.scores = scores

    @classmethod
    def load(
        cls,
        save_dir,
        data_name="data.csc.index.npy",
        indices_name="indices.csc.index.npy",
        indptr_name="indptr.csc.index.npy",
        vocab_name="vocab.index.json",
        params_name="params.index.json",
        nnoc_name="nonoccurrence_array.index.npy",
        corpus_name="corpus.jsonl",
        load_corpus=False,
        mmap=False,
        allow_pickle=False,
        load_vocab=True,
        device="cpu",
    ):
        """
        Load a BM25S index that was saved using the `save` method.
        This returns a BM25S object with the saved parameters and scores,
        which can be directly used for retrieval.

        Parameters
        ----------
        save_dir : str
            The directory where the BM25S index was saved.

        data_name : str
            The name of the file that contains the data array.

        indices_name : str
            The name of the file that contains the indices array.

        indptr_name : str
            The name of the file that contains the indptr array.

        vocab_name : str
            The name of the file that contains the vocab dictionary.

        params_name : str
            The name of the file that contains the parameters.

        nnoc_name : str
            The name of the file that contains the non-occurrence array.

        corpus_name : str
            The name of the file that contains the corpus.

        load_corpus : bool
            If True, the corpus will be loaded from the `corpus_name` file.

        mmap : bool
            Whether to use Memory-map for the np.load function. If false, the arrays will be loaded into memory.
            If True, the arrays will be memory-mapped, using 'r' mode. This is useful for very large arrays that
            do not fit into memory.

        allow_pickle : bool
            If True, the arrays will be loaded using pickle. If False, the arrays will be loaded
            in a more efficient format, but they will not be readable by older versions of numpy.

        load_vocab : bool
            If True, the vocab dictionary will be loaded from the `vocab_name` file. If False, the vocab dictionary
            will not be loaded, and the `vocab_dict` attribute of the BM25 object will be set to None.
        """
        if not isinstance(mmap, bool):
            raise ValueError("`mmap` must be a boolean")

        # Load the BM25 index from the save_dir
        save_dir = Path(save_dir)

        # Load the parameters
        params_path = save_dir / params_name
        with open(params_path, "r") as f:
            params: dict = json_functions.loads(f.read())

        # Load the vocab dictionary
        if load_vocab:
            vocab_path = save_dir / vocab_name
            with open(vocab_path, "r", encoding="utf-8") as f:
                vocab_dict: dict = json_functions.loads(f.read())
        else:
            vocab_dict = {}

        original_version = params.pop("version", None)
        num_docs = params.pop("num_docs", None)

        bm25_obj = cls(**params, device=device)
        bm25_obj.vocab_dict = vocab_dict
        bm25_obj._original_version = original_version
        bm25_obj.unique_token_ids_set = set(bm25_obj.vocab_dict.values())

        bm25_obj.load_scores(
            save_dir=save_dir,
            data_name=data_name,
            indices_name=indices_name,
            indptr_name=indptr_name,
            mmap=mmap,
            num_docs=num_docs,
            allow_pickle=allow_pickle,
        )
        
        # Convert scores to GPU tensors if device is CUDA
        if bm25_obj.device.type == "cuda":
            # Check if method requires nonoccurrence arrays (not supported for GPU yet)
            if bm25_obj.method in bm25_obj.methods_requiring_nonoccurrence:
                raise ValueError(
                    f"GPU acceleration is not supported for method '{bm25_obj.method}' which requires "
                    f"nonoccurrence arrays. Please use one of: 'robertson', 'lucene', 'atire' "
                    f"or set device='cpu' to use {bm25_obj.method}."
                )
            
            # Convert CPU scores to GPU tensors
            cpu_scores = bm25_obj.scores
            bm25_obj.scores = bm25_obj._convert_scores_to_torch_csc(cpu_scores)
            logger.debug(f"Converted loaded scores to GPU tensors for device: {bm25_obj.device}")

        if load_corpus:
            # load the model from the snapshot
            # if a corpus.jsonl file exists, load it
            corpus_file = save_dir / corpus_name
            if os.path.exists(corpus_file):
                if mmap is True:
                    corpus = utils.corpus.JsonlCorpus(corpus_file)
                else:
                    corpus = []
                    with open(corpus_file, "r", encoding="utf-8") as f:
                        for line in f:
                            doc = json_functions.loads(line)
                            corpus.append(doc)

                bm25_obj.corpus = corpus

        # if the method is one of BM25L or BM25+, we need to load the non-occurrence array
        # if it does not exist, we raise an error
        if bm25_obj.method in bm25_obj.methods_requiring_nonoccurrence:
            nnm_path = save_dir / nnoc_name
            if nnm_path.exists():
                bm25_obj.nonoccurrence_array = np.load(
                    nnm_path, allow_pickle=allow_pickle
                )
            else:
                raise FileNotFoundError(f"Non-occurrence array not found at {nnm_path}")
        else:
            bm25_obj.nonoccurrence_array = None

        return bm25_obj

    def _retrieve_documents_from_corpus(self, indices, corpus=None, return_as="tuple"):
        """
        Helper function to retrieve documents from corpus given indices.
        
        Parameters
        ----------
        indices : np.ndarray
            Document indices to retrieve
        corpus : list, np.ndarray, JsonlCorpus, optional
            Corpus to retrieve from. If None, uses self.corpus
        return_as : str
            Format to return results in
            
        Returns
        -------
        np.ndarray or tuple
            Retrieved documents in requested format
        """
        corpus = corpus if corpus is not None else self.corpus
        
        # Fast path: if no corpus needed, return indices directly
        if corpus is None or return_as == "indices":
            return indices
        elif return_as == "scores":
            raise ValueError("Cannot return scores without providing scores")
        
        # Handle corpus indexing
        if isinstance(corpus, utils.corpus.JsonlCorpus):
            retrieved_docs = corpus[indices]
        elif isinstance(corpus, np.ndarray):
            # NumPy arrays support direct fancy indexing
            retrieved_docs = corpus[indices]
        else:
            # Optimized corpus lookup for regular lists
            if indices.ndim == 1 or (indices.ndim == 2 and indices.shape[0] == 1):
                # Single query case - use the original simple logic
                index_flat = indices.flatten().tolist()
                results = [corpus[i] for i in index_flat]
                retrieved_docs = np.array(results).reshape(indices.shape)
            else:
                # Batch case - use optimized logic
                batch_size, k = indices.shape
                total_lookups = batch_size * k
                corpus_size = len(corpus)
                
                if corpus_size < 50000 and total_lookups > 100:
                    # Convert to numpy array for repeated access (amortize conversion cost)
                    if not hasattr(self, '_corpus_array_cache') or self._corpus_array_cache is None:
                        self._corpus_array_cache = np.array(corpus, dtype=object)
                    retrieved_docs = self._corpus_array_cache[indices]
                else:
                    # Use optimized batch processing to minimize Python overhead
                    retrieved_docs = np.empty(indices.shape, dtype=object)
                    
                    # Process in chunks to improve cache locality
                    chunk_size = min(1000, k)  # Process k documents at a time per query
                    for batch_idx in range(batch_size):
                        indices_for_batch = indices[batch_idx]
                        docs_for_batch = []
                        for i in range(0, k, chunk_size):
                            chunk_indices = indices_for_batch[i:i+chunk_size]
                            chunk_docs = [corpus[idx] for idx in chunk_indices]
                            docs_for_batch.extend(chunk_docs)
                        retrieved_docs[batch_idx] = docs_for_batch[:k]
        
        return retrieved_docs


    def free_gpu_memory(self):
        """
        Free GPU memory by moving tensors to CPU and clearing GPU cache.
        
        This method is useful for managing GPU memory when you're done with 
        GPU-accelerated operations or need to free up memory for other tasks.
        After calling this method, the BM25 object will use CPU computation.
        
        Note
        ----
        After calling this method, you can call `to_gpu()` to move tensors 
        back to GPU if needed.
        """
        if self.device.type == "cuda" and hasattr(self, 'scores') and isinstance(self.scores, dict):
            # Check if scores are already GPU tensors
            if "sparse_tensor" in self.scores:
                logger.debug("Converting GPU tensors back to CPU arrays")
                
                # Convert GPU tensors back to CPU numpy arrays
                cpu_scores = {
                    "data": self.scores["data"].cpu().numpy(),
                    "indices": self.scores["indices"].cpu().numpy(), 
                    "indptr": self.scores["indptr"].cpu().numpy(),
                    "num_docs": self.scores["num_docs"]
                }
                
                # Replace GPU tensors with CPU arrays
                self.scores = cpu_scores
                
                # Update device to CPU
                self.device = torch.device("cpu")
                
                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                logger.debug("GPU memory freed and device switched to CPU")
            else:
                logger.debug("Scores are already on CPU, only clearing GPU cache")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            logger.debug("BM25 object is already using CPU device")

    def to_gpu(self, device="cuda"):
        """
        Move BM25 tensors to GPU for accelerated computation.
        
        This method converts CPU-based scores to GPU tensors. Useful when 
        you want to switch from CPU to GPU computation or after calling 
        `free_gpu_memory()`.
        
        Parameters
        ----------
        device : str
            The GPU device to use (default: "cuda")
            
        Raises
        ------
        ValueError
            If the BM25 method is not supported on GPU
        RuntimeError
            If CUDA is not available
        """
        if device != "cpu" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
            
        if device == "cuda" and self.method in self.methods_requiring_nonoccurrence:
            raise ValueError(
                f"GPU acceleration is not supported for method '{self.method}' which requires "
                f"nonoccurrence arrays. Please use one of: 'robertson', 'lucene', 'atire'."
            )
        
        # Update device
        old_device = self.device
        self.device = torch.device(device)
        
        if device == "cuda" and hasattr(self, 'scores'):
            # Convert to GPU tensors if not already
            if not isinstance(self.scores, dict) or "sparse_tensor" not in self.scores:
                logger.debug(f"Converting scores from {old_device} to {self.device}")
                self.scores = self._convert_scores_to_torch_csc(self.scores)
                logger.debug("Successfully moved BM25 tensors to GPU")
            else:
                logger.debug("Scores are already GPU tensors")
        else:
            logger.debug(f"Device switched to {self.device}")

    def activate_numba_scorer(self):
        """
        Activate the Numba scorer for the BM25 index. This will apply the Numba JIT
        compilation to the `_compute_relevance_from_scores` function, which will speed
        up the scoring process. This will have an impact when you call the `retrieve`
        method and the `get_scores` method. The first time you call the `retrieve` method,
        it will be slower, as the function will be compiled on the spot. However, subsequent calls
        will be faster.

        This function requires the `numba` package to be installed. If it is not installed,
        an ImportError will be raised. You can install Numba with `pip install numba`.

        Behind the scenes, this will reassign the `_compute_relevance_from_scores` method
        to the JIT-compiled version of the function.
        """
        try:
            from numba import njit
        except ImportError:
            raise ImportError(
                "Numba is not installed. Please install Numba to compile the Numba scorer with `pip install numba`."
            )

        from .scoring import _compute_relevance_from_scores_jit_ready

        self._compute_relevance_from_scores = njit(
            _compute_relevance_from_scores_jit_ready
        )
