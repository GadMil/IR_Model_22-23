import csv
import pickle
import re
from abc import abstractmethod
from collections import Counter

import nltk
import numpy as np
from nltk.corpus import stopwords
import hashlib
import heapq

from inverted_index_gcp import InvertedIndex


def _hash(s):
    return hashlib.blake2b(bytes(s, encoding='utf8'), digest_size=5).hexdigest()


nltk.download('stopwords')


class PageViews:
    """
    Stores a dictionary with the page views of wikipedia articles.
    """
    def __init__(self):
        self._page_views = 'views_ranks/pageviews-202108-user.pkl'
        self.pvCounter = {}

    def read_page_views(self):
        """
        Reads the page views file.
        """
        # read in the counter
        try:
            with open(self._page_views, 'rb') as file:
                self.pvCounter = pickle.loads(file.read())
        except OSError:
            self.pvCounter = Counter()

    def get_page_views(self, wiki_ids):
        """
        Gets the page views of wikipedia articles.
        Parameters:
        -----------
        wiki_ids: list of wikipedia article's ids.
        Returns:
        -----------
        list of page views in the same order as the ids received.
        """
        return [self.pvCounter[wiki_id] if wiki_id in self.pvCounter else 0 for wiki_id in wiki_ids]


class PageRanks:
    """
    Stores a dictionary with the page ranks of wikipedia articles.
    """
    def __init__(self):
        self._page_ranks = 'views_ranks/part-00000-562c3bda-ffab-4c73-944f-08aa804bf5db-c000.csv'
        self.prDict = None

    def read_page_ranks(self):
        """
        Reads the page ranks file.
        """
        try:
            with open(self._page_ranks, 'r') as file:
                reader = csv.reader(file)
                data = list(reader)
                self.prDict = dict([(int(i), float(j)) for i, j in data])
        except OSError:
            self.prDict = {}

    def get_page_ranks(self, wiki_ids):
        """
        Gets the page ranks of wikipedia articles.
        Parameters:
        -----------
        wiki_ids: list of wikipedia article's ids.
        Returns:
        -----------
        list of page ranks in the same order as the ids received.
        """
        pageranks = []
        for id in wiki_ids:
            try:
                pageranks.append(self.prDict[id])
            except:
                pageranks.append(0)
        return pageranks

    def get_page_ranks_with_id(self):
        """
        Gets the wikipedia articles with the maximum page rank score.
        Returns:
        -----------
        list of (wiki_id, rank) for the top page ranks, normalized by the maximum page rank.
        """
        top_ranks = sorted(self.prDict.items(), key=lambda x: x[1], reverse=True)
        max_pr = top_ranks[0][1]
        return [(pr[0], pr[1] / max_pr) for pr in top_ranks]


class Tokenizer:
    """
    Stores the tokenization functionality of query's terms.
    """
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
        This function aims in tokenize a text into a list of tokens. Moreover, it filters stopwords.
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
    """
    abstract class for all searching models to inherit.
    """
    def __init__(self, index: InvertedIndex):
        self.index = index

    @abstractmethod
    def search_query(self, query_to_search):
        return


class BinaryQuerySearcher(QuerySearcher):
    """
    Stores the functionality of binary search for a query.
    """
    def __init__(self, index):
        super().__init__(index)

    def search_query(self, query_to_search):
        """
        Search for relevant wikipedia articles for a given query.
        Parameters:
        -----------
        query_to_search: list of string, for the tokenized query.
        Returns:
        -----------
        The relevant wikipedia articles as a sorted list of ids.
        """
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


    def search_query_with_score(self, query_to_search):
        """
        Search for relevant wikipedia articles for a given query.
        Parameters:
        -----------
        query_to_search: list of string, for the tokenized query.
        Returns:
        -----------
        The relevant wikipedia articles as a sorted list of pairs (doc_id, score).
        """
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

        return get_top_n(out, 200)


class TfIdfQuerySearcher(QuerySearcher):
    """
    Stores the functionality of tfidf with cosine similarity search for a query.
    """
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
        index: inverted index loaded from the corresponding files.
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
        query_tf = Counter(query_to_search)

        for token, q_tf in query_tf.items():
            if token in self.index.term_total.keys():
                q_vector.append(q_tf * self.index.idf[token] / q_size)
                for doc_id, doc_tf in self.index.read_posting_list(token):
                    if doc_id != 0:
                        tfidf = doc_tf * self.index.idf[token]
                        if doc_id in sim_dict.keys():
                            sim_dict[doc_id] += q_tf * tfidf
                        else:
                            sim_dict[doc_id] = q_tf * tfidf

        for doc_id in sim_dict.keys():
            sim_dict[doc_id] = sim_dict[doc_id] / (self.index.doc_norm[doc_id] * np.linalg.norm(q_vector)) \
                               / self.index.dl[doc_id]

        if len(sim_dict) >= N:
            heap = [(tfidf, doc_id) for doc_id, tfidf in sim_dict.items()]
            top_n = heapq.nlargest(N, heap)
            result = [(doc_id, tfidf) for tfidf, doc_id in sorted(top_n, reverse=True)]
        else:
            result = get_top_n(sim_dict, N)

        return result


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

    def __init__(self, index, k1=3, b=0.25):
        super().__init__(index)
        self.b = b
        self.k1 = k1

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
        query_tf = Counter(query)

        for token, q_tf in query_tf.items():
            if token in self.index.term_total.keys():
                q_vector.append(q_tf * self.index.idf[token] / q_size)
                for doc_id, doc_tf in self.index.read_posting_list(token):
                    if doc_id == 0:
                        continue
                    if doc_id in sim_dict.keys():
                        sim_dict[doc_id] += self._score(token, q_tf, doc_id)
                    else:
                        sim_dict[doc_id] = self._score(token, q_tf, doc_id)

        if len(sim_dict) >= N:
            heap = [(bm25, doc_id) for doc_id, bm25 in sim_dict.items()]
            top_n = heapq.nlargest(N, heap)
            result = [(doc_id, bm25) for bm25, doc_id in sorted(top_n, reverse=True)]
        else:
            result = get_top_n(sim_dict, N)

        return result

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


def merge_results(title_scores, body_scores, anchor_scores, title_weight=0.8, text_weight=0.2, anchor_weight=0.4, N=200):
    """
    This function merge and sort documents retrieved by its weighted score (e.g., title and body).
    Parameters:
    -----------
    title_scores: a list of pairs in the following format: (doc_id,score) build upon the title index.
    body_scores: a dictionary list of pairs in the following format: (doc_id,score) build upon the title/text index.
    anchor_scores: a dictionary list of pairs in the following format: (doc_id,score) build upon the anchor/text index.
    title_weight: float, for weighted average utilizing body scores
    text_weight: float, for weighted average utilizing title scores
    anchor_weight: float, for weighted average utilizing anchor scores
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

    for doc, score in anchor_scores:
        if merged_scores.get(doc):
            merged_scores[doc] += score * anchor_weight
        else:
            pass

    return sorted(merged_scores.items(), key=lambda x: x[1], reverse=True)[:min(N, len(merged_scores))]
