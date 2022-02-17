import torch
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction import _stop_words
import string
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import os
import warnings


class HFSearch:
    """ A HuggingFace Search Engine
    Supports the following methods:
    - bm25: BM25 ranking
    - retrieve: Retrieve the top-k passages using the bi-encoder
    - retrieve & rerank: Retrieve the top-k passages and re-rank them with the CrossEncoder
    """
    def __init__(self, hf_data_path="hf_data", embedding_path="embeddings/multi-qa-MiniLM-L6-cos-v1-embeddings.pt", max_seq_length=256, top_k=32):
        """
        Args:
            hf_data_path (str): Path to the HuggingFace data (must contain models.jsonl and passages.jsonl)
            embedding_path (str): Path to the HuggingFace embedding
            max_seq_length (int): Maximum sequence length for bi-encoder
            top_k (int): Top-k passages to retrieve by the bi-encoder
        """
        passages_path = os.path.join(hf_data_path, "passages.jsonl")
        models_path = os.path.join(hf_data_path, "models.jsonl")
        assert os.path.exists(passages_path), "Passages file not found at {}".format(passages_path)
        assert os.path.exists(models_path), "Models file not found at {}".format(models_path)
        assert os.path.exists(embedding_path), "Embedding file not found at {}".format(
            embedding_path
        )  # NOTE: embeddings and passages are linked

        models_df = pd.read_json(models_path, lines=True)
        passages_df = pd.read_json(passages_path, lines=True)
        models_df = models_df.reset_index().rename(columns={"index": "id"})
        self.models_df = passages_df[["id", "passage"]].merge(models_df, on="id", how="left").drop(columns=["id"])

        self.corpus_embeddings = torch.load(embedding_path, map_location=torch.device("cpu"))

        # We use the Bi-Encoder to encode all passages, so that we can use it with sematic search
        self.bi_encoder = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
        self.bi_encoder.max_seq_length = max_seq_length  # Truncate long passages to 256 tokens
        self.top_k = top_k  # Number of passages we want to retrieve with the bi-encoder

        # The bi-encoder will retrieve 32 documents. We use a cross-encoder, to re-rank the results list to improve the quality
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        # We lower case our text and remove stop-words from indexing
        tokenized_corpus = passages_df["passage"].apply(self._bm25_tokenizer).values.tolist()
        self.bm25 = BM25Okapi(tokenized_corpus)

    # This function will search all huggingface models which answer the query
    def search(self, query, method="bm25", limit=3, filters={}, sort=None, direction=None, verbose=False):
        """Search for a query in the huggingface models.

        Args:
            query (str): The query to search for.
            method (str): The method to use for search.
            limit (int): The number of results to return.
            filters (dict): A dictionary of filters to apply to the results.
            sort (str):
                The key with which to sort the resulting models. Possible values are the properties of the `ModelInfo`
                class.
            direction (:obj:`Literal[1]` or :obj:`int`):
                Direction in which to sort. The value `1` sorts by ascending order while all other values
                sort by descending order.
            verbose (bool): Whether to print the results.

        Usage:
            >>> search("bert", method="retrieve & rerank", limit=10, filters={"task["question-answering", "fill-mask"], "library_name": ["transformers"]})

        Returns:
            A list of results.
        """
        assert method in [
            "bm25",
            "retrieve",
            "retrieve & rerank",
        ], "Method must be one of 'bm25', 'retrieve' or 'retrieve & rerank'"
        assert isinstance(filters, dict), "Filters must be a dictionary"
        assert sort in [
            None,
            "lastModified",
            "likes",
            "downloads"], "Sort must be one of None, 'lastModified', 'likes' or 'downloads'"

        if verbose:
            print("Input question:", query)

        ##### BM25 search (lexical search) #####
        if method == "bm25":
            if len(filters) > 0:
                warnings.warn("Filters are not supported for bm25 search")

            bm25_scores = self.bm25.get_scores(self._bm25_tokenizer(query))
            top_n = np.argpartition(bm25_scores, -5)[-5:]
            bm25_hits = [{**self.models_df.iloc[idx].to_dict(), "score": bm25_scores[idx]} for idx in top_n]
            bm25_hits = self._sort_hits(bm25_hits, sort, direction)
            if verbose:
                print("\n-------------------------\n")
                print("Top-limit lexical search (BM25) hits")
                for hit in bm25_hits[0:limit]:
                    print("\t{:.3f}\t{}".format(hit["score"], hit["passage"].replace("\n", " ")))
            bm25_hits = bm25_hits[0:limit]
            return bm25_hits

        ##### Semantic Search #####
        # filter self.corpus_embeddings by filters
        filters = {"pipeline_tag": filters.get("task", []), "library_name": filters.get("library", [])}
        filters = {k: v for k, v in filters.items() if v is not None and len(v) > 0}
        if verbose:
            print("Filters:", filters)
        filt_models_df = self.models_df[
            self.models_df.apply(lambda x: all([x[k] in v for k, v in filters.items()]), axis=1)
        ]
        filt_corpus_embeddings = self.corpus_embeddings[filt_models_df.index.values, :]
        filt_models_df = filt_models_df.reset_index().drop(columns=["index"])

        if len(filt_models_df) == 0:
            if verbose:
                print("No models match the filters")
            return []

        # Encode the query using the bi-encoder and find potentially relevant passages
        question_embedding = self.bi_encoder.encode(query, convert_to_tensor=True, show_progress_bar=False)
        if torch.cuda.is_available():
            question_embedding = question_embedding.cuda()
        hits = util.semantic_search(question_embedding, filt_corpus_embeddings, top_k=self.top_k)
        hits = hits[0]  # Get the hits for the first query
        hits = [{**filt_models_df.iloc[hit["corpus_id"]].to_dict(), "score": hit["score"]} for hit in hits]

        if method == "retrieve":
            # Output of top-5 hits from bi-encoder
            hits = self._sort_hits(hits, sort, direction)
            if verbose:
                print("\n-------------------------\n")
                print("Top-limit Bi-Encoder Retrieval hits")
                for hit in hits[0:limit]:
                    print("\t{:.3f}\t{}".format(hit["score"], hit["passage"].replace("\n", " ")))
            semantic_hits = hits[0:limit]
            return semantic_hits

        ##### Re-Ranking #####
        if method == "retrieve & rerank":
            # Now, score all retrieved passages with the self.cross_encoder
            cross_inp = [[query, hit["passage"]] for hit in hits]
            cross_scores = self.cross_encoder.predict(cross_inp)

            # Sort results by the cross-encoder scores
            for idx in range(len(cross_scores)):
                hits[idx]["score"] = cross_scores[idx]

            # Output of top-5 hits from re-ranker
            hits = self._sort_hits(hits, sort, direction)
            if verbose:
                print("\n-------------------------\n")
                print("Top-limit Cross-Encoder Re-ranker hits")
                for hit in hits[0:limit]:
                    print("\t{:.3f}\t{}".format(hit["score"], hit["passage"].replace("\n", " ")))
            reranker_hits = hits[0:limit]
            return reranker_hits

    @staticmethod
    def _bm25_tokenizer(text):
        """lower case our text and remove stop-words from indexing"""
        tokenized_doc = []
        for token in text.lower().split():
            token = token.strip(string.punctuation)

            if len(token) > 0 and token not in _stop_words.ENGLISH_STOP_WORDS:
                tokenized_doc.append(token)
        return tokenized_doc

    @staticmethod
    def _sort_hits(hits, sort, direction):
        """Sort hits by the given sort and direction"""
        if sort is None:
            hits = sorted(hits, key=lambda x: x["score"], reverse=direction != 1)
        elif sort == "lastModified":
            hits = sorted(hits, key=lambda x: x["lastModified"], reverse=direction != 1)
        elif sort == "likes":
            hits = sorted(hits, key=lambda x: x["likes"], reverse=direction != 1)
        elif sort == "downloads":
            hits = sorted(hits, key=lambda x: x["downloads"], reverse=direction != 1)
        return hits