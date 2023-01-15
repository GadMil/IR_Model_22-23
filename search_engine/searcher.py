import csv
import math
import pickle
import re
from abc import abstractmethod
from collections import Counter, defaultdict

import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import hashlib
import heapq
import gzip

from inverted_index_gcp import InvertedIndex


def _hash(s):
    return hashlib.blake2b(bytes(s, encoding='utf8'), digest_size=5).hexdigest()


nltk.download('stopwords')


class PageViews:

    def __init__(self):
        self._page_views = 'views_ranks/pageviews-202108-user.pkl'
        self.pvCounter = {}

    def read_page_views(self):
        # read in the counter
        try:
            with open(self._page_views, 'rb') as file:
                self.pvCounter = pickle.loads(file.read())
        except OSError:
            return []

    def get_page_views(self, wiki_ids: list) -> list:
        return [self.pvCounter[wiki_id] if wiki_id in self.pvCounter else 0 for wiki_id in wiki_ids]


class PageRanks:

    def __init__(self):
        self._page_ranks = 'views_ranks/part-00000-562c3bda-ffab-4c73-944f-08aa804bf5db-c000.csv'
        self.prDict = None

    def read_page_ranks(self):
        # read in the rdd
        try:
            with open(self._page_ranks, 'r') as file:
                reader = csv.reader(file)
                data = list(reader)
                self.prDict = dict([(int(i), float(j)) for i, j in data])
        except OSError:
            self.prDict = {}

    def get_page_ranks(self, wiki_ids: list) -> list:
        pageranks = []
        for id in wiki_ids:
            try:
                pageranks.append(self.prDict[id])
            except:
                pageranks.append(0)
        return pageranks



class Tokenizer:
    def __init__(self):
        self.english_stopwords = frozenset(stopwords.words('english'))
        self.corpus_stopwords = ["category", "references", "also", "external", "links",
                                 "may", "first", "see", "history", "people", "one", "two",
                                 "part", "thumb", "including", "second", "following",
                                 "many", "however", "would", "became"]

        self.all_stopwords = self.english_stopwords.union(self.corpus_stopwords)
        self.RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

        self.NUM_BUCKETS = 124

    def token2bucket_id(self, token):
        return int(_hash(token), 16) % self.NUM_BUCKETS

    def tokenize(self, text):
        """
        This function aims in tokenize a text into a list of tokens. Moreover, it filter stopwords.
        Parameters:
        -----------
            text: string , representing the text to tokenize.
        Returns:
        -----------
            list of tokens (e.g., list of tokens).
        """
        list_of_tokens = [token.group() for token in self.RE_WORD.finditer(text.lower()) if
                          token.group() not in self.all_stopwords]
        return list_of_tokens


def get_posting_iter(index, query):
    """
    This function returning the iterator working with posting list.

    Parameters:
    ----------
    index: inverted index
    """
    words, pls = zip(*index.posting_lists_iter(query))
    return words, pls


def cosine_similarity(index, D, Q):
    """
    Calculate the cosine similarity for each candidate document in D and a given query (e.g., Q).
    Generate a dictionary of cosine similarity scores
    key: doc_id
    value: cosine similarity score

    Parameters:
    -----------
    D: DataFrame of tfidf scores.

    Q: vectorized query with tfidf scores

    Returns:
    -----------
    dictionary of cosine similarity score as follows:
                                                                key: document id (e.g., doc_id)
                                                                value: cosine similarly score.
    """
    # YOUR CODE HERE
    result = {}

    for doc_id, scores in D.iterrows():
        s = np.array(scores)
        result[doc_id] = np.dot(s, Q) / (index.doc_norm[doc_id] * np.linalg.norm(Q))

    return result


def get_top_n(sim_dict, N=100):
    """
    Sort and return the highest N documents according to the cosine similarity score.
    Generate a dictionary of cosine similarity scores

    Parameters:
    -----------
    sim_dict: a dictionary of similarity score as follows:
                                                                key: document id (e.g., doc_id)
                                                                value: similarity score. We keep up to 5 digits after the decimal point. (e.g., round(score,5))

    N: Integer (how many documents to retrieve). By default N = 3

    Returns:
    -----------
    a ranked list of pairs (doc_id, score) in the length of N.
    """
    try:
        return sorted([(doc_id, score) for doc_id, score in sim_dict.items()], key=lambda x: x[1], reverse=True)[
               :N]
    except:
        return []


class QuerySearcher:
    def __init__(self, index: InvertedIndex):
        self.index = index
        # self.words, self.pls = get_posting_iter(index, query)
        self.words, self.pls = None, None

    @abstractmethod
    def search_query(self, query_to_search):
        return


class BinaryQuerySearcher(QuerySearcher):

    def __init__(self, index):
        super().__init__(index)

    def search_query(self, query_to_search):
        # tokens = word_tokenize(query.lower())
        out = {}
        for token in query_to_search:
            try:
                res = self.index.read_posting_list(token)
                for doc_id, amount in res:
                    try:
                        out[doc_id] += 1
                    except:
                        out[doc_id] = 1
            except Exception as e:
                print("Index error, couldn't find term - ", e)

        return sorted(out, key=out.get, reverse=True)

    def get_candidate_documents_and_scores(self, query_to_search):
        candidates = {}

        for term in np.unique(query_to_search):
            if term in self.words:
                for doc_id, tf in self.pls[self.words.index(term)]:
                    if candidates.get(doc_id):
                        candidates[doc_id] += 1
                    else:
                        candidates[doc_id] = 1

        return candidates


class TfIdfQuerySearcher(QuerySearcher):
    def __init__(self, index):
        super().__init__(index)

    def search_query(self, query_to_search, N=100):
        """
        Generate a dictionary that gathers for every query its topN score.

        Parameters:
        -----------
        queries_to_search: a dictionary of queries as follows:
                                                            key: query_id
                                                            value: list of tokens.
        index:           inverted index loaded from the corresponding files.
        N: Integer. How many documents to retrieve. This argument is passed to the topN function. By default, N = 3.

        Returns:
        -----------
        return: a dictionary of queries and topN pairs as follows:
                                                        key: query_id
                                                        value: list of pairs in the following format:(doc_id, score).
        """
        # YOUR CODE HERE
        q_vector = []
        q_size = len(query_to_search)
        sim_dict = {}
        counter = Counter(query_to_search)

        for token, count in counter.items():
            if token in self.index.term_total.keys():  # avoid terms that do not appear in the index.
                q_vector.append(count * self.index.idf[token] / q_size)
                for doc_id, doc_tf in self.index.read_posting_list(token):
                    if doc_id == 0:
                        continue
                    tfidf = doc_tf * self.index.idf[token]
                    if doc_id in sim_dict.keys():
                        sim_dict[doc_id] += count * tfidf
                    else:
                        sim_dict[doc_id] = count * tfidf

        for doc_id in sim_dict.keys():
            sim_dict[doc_id] = sim_dict[doc_id] * (1/(self.index.doc_norm[doc_id] * np.linalg.norm(q_vector))) *\
                               (1/self.index.dl[doc_id])

        if len(sim_dict) < N:
            return get_top_n(sim_dict, N)
        heap = [(tfidf, doc_id) for doc_id, tfidf in sim_dict.items()]
        top_n = heapq.nlargest(N, heap)
        return [(doc_id, tfidf) for tfidf, doc_id in sorted(top_n, reverse=True)]


class BM25QuerySearcher(QuerySearcher):
    """
    Best Match 25.
    Parameters to tune:
    ----------
    k1 : float, default 1.5
    b : float, default 0.75
    Attributes:
    ----------
    tf_ : list[dict[str, int]]
        Term Frequency per document, normalized by document's length.
    doc_len_ : list[int]
        Number of terms per document.
    df_ : dict[str, int]
        Document Frequency per term. i.e. Number of documents in the
        corpus that contains the term.
    avg_dl_ : float
        Average number of terms for documents in the corpus.
    idf_ : dict[str, float]
        Inverse Document Frequency per term.
    """

    def __init__(self, index, k1=1.5, b=0.75):
        super().__init__(index)
        self.b = b
        self.k1 = k1


    # def calc_idf(self, query):
    #     """
    #     This function calculate the idf values according to the BM25 idf formula for each term in the query.
    #     Parameters:
    #     -----------
    #     query: list of token representing the query.
    #     """
    #     # YOUR CODE HERE
    #     for term in query:
    #         if term not in self.idf and self.index.df.get(term):
    #             freq = self.index.df[term]
    #             self.idf[term] = math.log((self.index.corpus_size - freq + 0.5) / (freq + 0.5))
    #         else:
    #             pass

    def search_query(self, query, N=200):
        """
         This function calculate the bm25 score for given query and document.
         We need to check only documents which are 'candidates' for a given query.
         This function return a dictionary of scores as the following:
                                                     key: query_id
                                                     value: a ranked list of pairs (doc_id, score) in the length of N.
         Parameters:
         -----------
         query: list of token representing the query. For example: ['look', 'blue', 'sky']
         doc_id: integer, document id.
         Returns:
         -----------
         score: float, bm25 score.
         """
        # YOUR CODE HERE
        q_vector = []
        q_size = len(query)
        sim_dict = {}
        counter = Counter(query)

        for token, count in counter.items():
            if token in self.index.term_total.keys():  # avoid terms that do not appear in the index.
                q_vector.append(count * self.index.idf[token] / q_size)
                for doc_id, doc_tf in self.index.read_posting_list(token):
                    if doc_id == 0:
                        continue
                    if doc_id in sim_dict.keys():
                        sim_dict[doc_id] += self._score(token, count, doc_id)
                    else:
                        sim_dict[doc_id] = self._score(token, count, doc_id)

        if len(sim_dict) < N:
            return get_top_n(sim_dict, N)
        heap = [(bm25, doc_id) for doc_id, bm25 in sim_dict.items()]
        top_n = heapq.nlargest(N, heap)
        return [(doc_id, bm25) for bm25, doc_id in sorted(top_n, reverse=True)]

    def _score(self, term, tf, doc_id):
        """
        This function calculate the bm25 score for given query and document.
        Parameters:
        -----------
        query: list of token representing the query.
        doc_id: integer, document id.
        Returns:
        -----------
        score: float, bm25 score.
        """
        # YOUR CODE HERE
        numerator = self.index.idf[term] * tf * (self.k1 + 1)
        denominator = tf + self.k1 * (1 - self.b + self.b * self.index.dl[doc_id] / self.index.avg_dl)
        score = (numerator / denominator)

        return score


def merge_results(title_scores, body_scores, title_weight=0.3, text_weight=0.7, N=200):
    """
    This function merge and sort documents retrieved by its weighted score (e.g., title and body).
    Parameters:
    -----------
    title_scores: a list of pairs in the following format: (doc_id,score) build upon the title index.
    body_scores: a dictionary list of pairs in the following format: (doc_id,score) build upon the body/text index.
    title_weight: float, for weighted average utilizing title and body scores
    text_weight: float, for weighted average utilizing title and body scores
    N: Integer. How many document to retrieve. This argument is passed to topN function. By default N = 3.
    Returns:
    -----------
    sorted dictionary topN pairs as follows:
                                            key: doc_id
                                            value: score
    """
    # YOUR CODE HERE
    merged_scores = {doc: score * title_weight for doc, score in title_scores}

    for doc, score in body_scores:
        if merged_scores.get(doc):
            merged_scores[doc] += score * text_weight
        else:
            merged_scores[doc] = score * text_weight

    return sorted(merged_scores.items(), key=lambda x: x[1], reverse=True)[:min(N, len(merged_scores))]


def get_similar_words(term, model):
    try:
        similars = model.most_similar(term, topn=3)
    except:
        similars = []
    return similars


def expand_query(tokens, model):
    query = tokens
    for tok in tokens:
        similars = get_similar_words(tok, model)
        for w in similars:
            query.append(w[0])
    return query
