import warnings
warnings.filterwarnings("ignore")

import joblib
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction import _stop_words
import string
from tqdm.autonotebook import tqdm
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder, util

PASSAGES_PATH = "proc_data/passages.jsonl"
passages_df = pd.read_json(PASSAGES_PATH, lines=True)
passages = passages_df["passage"].values.tolist()

EMBEDDING_PATH = "embeddings/multi-qa-MiniLM-L6-cos-v1-embeddings.pkl"
corpus_embeddings = joblib.load(EMBEDDING_PATH)

#We use the Bi-Encoder to encode all passages, so that we can use it with sematic search
bi_encoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
bi_encoder.max_seq_length = 256     #Truncate long passages to 256 tokens
top_k = 32                          #Number of passages we want to retrieve with the bi-encoder

#The bi-encoder will retrieve 100 documents. We use a cross-encoder, to re-rank the results list to improve the quality
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# We lower case our text and remove stop-words from indexing
def bm25_tokenizer(text):
    tokenized_doc = []
    for token in text.lower().split():
        token = token.strip(string.punctuation)

        if len(token) > 0 and token not in _stop_words.ENGLISH_STOP_WORDS:
            tokenized_doc.append(token)
    return tokenized_doc

tokenized_corpus = []
for passage in passages:
    tokenized_corpus.append(bm25_tokenizer(passage))

bm25 = BM25Okapi(tokenized_corpus)

# This function will search all huggingface models which answer the query
def search(query, limit=6, verbose=False):
    print("Input question:", query)

    ##### BM25 search (lexical search) #####
    bm25_scores = bm25.get_scores(bm25_tokenizer(query))
    top_n = np.argpartition(bm25_scores, -5)[-5:]
    bm25_hits = [{"corpus_id": idx, "score": bm25_scores[idx]} for idx in top_n]
    bm25_hits = sorted(bm25_hits, key=lambda x: x["score"], reverse=True)
    if verbose:
        print("Top-limit lexical search (BM25) hits")
        for hit in bm25_hits[0:limit]:
            print("\t{:.3f}\t{}".format(hit["score"], passages[hit["corpus_id"]].replace("\n", " ")))
    bm25_hits = bm25_hits[0:limit]

    ##### Semantic Search #####
    # Encode the query using the bi-encoder and find potentially relevant passages
    question_embedding = bi_encoder.encode(query, convert_to_tensor=True)
    question_embedding = question_embedding.cuda()
    hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=top_k)
    hits = hits[0]  # Get the hits for the first query

    ##### Re-Ranking #####
    # Now, score all retrieved passages with the cross_encoder
    cross_inp = [[query, passages[hit["corpus_id"]]] for hit in hits]
    cross_scores = cross_encoder.predict(cross_inp)

    # Sort results by the cross-encoder scores
    for idx in range(len(cross_scores)):
        hits[idx]["cross-score"] = cross_scores[idx]

    # Output of top-5 hits from bi-encoder
    hits = sorted(hits, key=lambda x: x["score"], reverse=True)
    if verbose:
        print("\n-------------------------\n")
        print("Top-limit Bi-Encoder Retrieval hits")
        for hit in hits[0:limit]:
            print("\t{:.3f}\t{}".format(hit["score"], passages[hit["corpus_id"]].replace("\n", " ")))
    semantic_hits = hits[0:limit]

    # Output of top-5 hits from re-ranker
    hits = sorted(hits, key=lambda x: x["cross-score"], reverse=True)
    if verbose:
        print("\n-------------------------\n")
        print("Top-limit Cross-Encoder Re-ranker hits")
        for hit in hits[0:limit]:
            print("\t{:.3f}\t{}".format(hit['cross-score'], passages[hit['corpus_id']].replace("\n", " ")))
    reranker_hits = hits[0:limit]
    return {"bm25_hits": bm25_hits, "semantic_hits": semantic_hits, "reranker_hits": reranker_hits}

if __name__ == "__main__":
    search(query="What is the best way to get a good grade in CS?", limit=5, verbose=True)


