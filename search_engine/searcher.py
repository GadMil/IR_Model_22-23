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
        self._page_ranks = 'views_ranks/part-00000-562c3bda-ffab-4c73-944f-08aa804bf5db-c000.csv.gz'
        self.prDF = None

    def read_page_ranks(self):
        # read in the rdd
        try:
            with open(self._page_ranks, 'rb') as file:
                self.prDF = pickle.loads(file.read())
        except OSError:
            return []

    def get_page_ranks(self, wiki_ids: list) -> list:
        relevant_ids = self.prDF[self.prDF[0].isin(wiki_ids)]
        relevant_ids_rank = {row[1][0]: row[1][1] for row in relevant_ids.itterrows()}
        return [relevant_ids_rank[wiki_id] if wiki_id in relevant_ids_rank else 0 for wiki_id in wiki_ids]


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

    return sorted([(doc_id, round(score, 5)) for doc_id, score in sim_dict.items()], key=lambda x: x[1], reverse=True)[
           :N]


class QuerySearcher:
    def __init__(self, index: InvertedIndex, query):
        self.index = index
        self.words, self.pls = get_posting_iter(index, query)

    @abstractmethod
    def search_query(self, query_to_search):
        return


class BinaryQuerySearcher(QuerySearcher):

    def __init__(self, index, query):
        super().__init__(index, query)

    def search_query(self, query_to_search):
        return sorted(self.get_candidate_documents_and_scores(query_to_search).items(), key=lambda x: x[1],
                      reverse=True)

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
    def __init__(self, index, query):
        super().__init__(index, query)

    def get_candidate_documents_and_scores(self, query_to_search):
        """
        Generate a dictionary representing a pool of candidate documents for a given query. This function will go
        through every token in query_to_search and fetch the corresponding information (e.g., term frequency,
        document frequency, etc.') needed to calculate TF-IDF from the posting list. Then it will populate the
        dictionary 'candidates.' For calculation of IDF, use log with base 10. tf will be normalized based on the
        length of the document.
        Parameters:
        -----------
        query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering
        stopwords, etc.').
        Returns:
        -----------
        dictionary of candidates. In the following format:
                                                                   key: pair (doc_id,term)
                                                                   value: tfidf score.
        """
        candidates = {}
        for term in np.unique(query_to_search):
            if term in self.words:
                list_of_doc = self.pls[self.words.index(term)]
                # normalized_tfidf = [(doc_freq[0], self.index.tfidf[doc_freq[0]][term]) for doc_freq in list_of_doc]
                normalized_tfidf = [(doc, freq / self.index.dl[doc] * self.index.idf[term]) for doc, freq in list_of_doc]

                for doc_id, tfidf in normalized_tfidf:
                    candidates[(doc_id, term)] = candidates.get((doc_id, term), 0) + tfidf

        return candidates

    def generate_query_tfidf_vector(self, query_to_search):
        """
        Generate a vector representing the query. Each entry within this vector represents a tfidf score.
        The terms representing the query will be the unique terms in the index.

        We will use tfidf on the query as well.
        For calculation of IDF, use log with base 10.
        tf will be normalized based on the length of the query.

        Parameters:
        -----------
        query_to_search: list of tokens (str).
                            This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').

        index:           inverted index loaded from the corresponding files.

        Returns:
        -----------
        vectorized query with tfidf scores
        """
        unique_q_terms = list(np.unique(query_to_search))
        q_size = len(query_to_search)
        Q = np.zeros(len(unique_q_terms))
        counter = Counter(query_to_search)

        for token in unique_q_terms:
            if token in self.index.term_total.keys():  # avoid terms that do not appear in the index.
                tf = counter[token] / q_size  # term frequency divided by the length of the query
                idf = self.index.idf[token]

                try:
                    ind = unique_q_terms.index(token)
                    Q[ind] = tf * idf
                except:
                    pass
        return Q

    def generate_document_tfidf_matrix(self, query_to_search):
        """
        Generate a DataFrame `D` of tfidf scores for a given query.
        Rows will be the documents candidates for a given query
        Columns will be the unique terms in the index.
        The value for a given document and term will be its tfidf score.

        Parameters:
        -----------
        query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g.,
        lower case, filtering stopwords, etc.').

        index:
        inverted index loaded from the corresponding files.

        words,pls: iterator for working with posting.

        Returns:
        -----------
        DataFrame of tfidf scores.
        """

        # No need to utilize all documents, only those having corresponding terms with the query.
        unique_q_terms = np.unique(query_to_search)
        candidates_scores = self.get_candidate_documents_and_scores(query_to_search)
        unique_candidates = np.unique([doc_id for doc_id, freq in candidates_scores.keys()])
        D = np.zeros((len(unique_candidates), len(unique_q_terms)))
        D = pd.DataFrame(D)

        D.index = unique_candidates
        D.columns = unique_q_terms

        for key in candidates_scores:
            tfidf = candidates_scores[key]
            doc_id, term = key
            D.loc[doc_id][term] = tfidf

        return D

    def search_query(self, queries_to_search, N=100):
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
        D = self.generate_document_tfidf_matrix(queries_to_search)
        Q = self.generate_query_tfidf_vector(queries_to_search)
        return get_top_n(cosine_similarity(self.index, D, Q), N)


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

    def __init__(self, index, query, tf=None, k1=1.5, b=0.75):
        super().__init__(index, query)
        self.b = b
        self.k1 = k1
        self.tf_ = tf
        self.idf = defaultdict()

    def calc_idf(self, query):
        """
        This function calculate the idf values according to the BM25 idf formula for each term in the query.
        Parameters:
        -----------
        query: list of token representing the query.
        """
        # YOUR CODE HERE
        for term in query:
            if term not in self.idf and self.index.df.get(term):
                freq = self.index.df[term]
                self.idf[term] = math.log((self.index.corpus_size - freq + 0.5) / (freq + 0.5))
            else:
                pass

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
        self.calc_idf(query)
        candidates = set()

        for term in np.unique(query):
            if term in self.words:
                candidates.update(doc_freq[0] for doc_freq in self.pls[self.words.index(term)])

        return sorted([(doc_id, self._score(query, doc_id)) for doc_id in candidates], key=lambda x: x[1],
                      reverse=True)[:min(N, self.index.corpus_size)]

    def _score(self, query, doc_id):
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
        score = 0.0

        for term in query:
            if term in self.words:
                term_frequencies = dict(self.pls[self.words.index(term)])

                if doc_id in term_frequencies.keys():
                    freq = term_frequencies[doc_id]
                    numerator = self.idf[term] * freq * (self.k1 + 1)
                    denominator = freq + self.k1 * (1 - self.b + self.b * self.index.dl[doc_id] / self.index.avg_dl)
                    score += (numerator / denominator)

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

    return sorted(merged_scores, key=lambda x: x[1], reverse=True)[:min(N, len(merged_scores))]
