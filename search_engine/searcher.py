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
        self._page_views = 'pageviews-202108-user.pkl'

    def get_page_views(self, wiki_ids: list) -> list:
        # read in the counter
        try:
            with open(self._page_views, 'rb') as file:
                pvCounter = pickle.loads(file.read())
        except OSError:
            return []

        return [pvCounter[wiki_id] if wiki_id in pvCounter else 0 for wiki_id in wiki_ids]


class PageRanks:

    def __init__(self):
        self._page_ranks = 'gs://information_retrieval_project/pr'

    def get_page_ranks(self, wiki_ids: list) -> list:
        # read in the rdd
        try:
            with open(self._page_ranks, 'rb') as file:
                prDF = pickle.loads(file.read())
        except OSError:
            return []

        relevant_ids = prDF[prDF[0].isin(wiki_ids)]
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


def get_posting_iter(index, directory):
    """
    This function returning the iterator working with posting list.

    Parameters:
    ----------
    index: inverted index
    """
    words, pls = zip(*index.posting_lists_iter(directory))
    return words, pls


def cosine_similarity(D, Q):
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
                                                                value: cosine similarty score.
    """
    # YOUR CODE HERE
    result = {}

    for doc_id, scores in D.iterrows():
        s = np.array(scores)
        result[doc_id] = np.dot(s, Q) / (np.linalg.norm(s) * np.linalg.norm(Q))

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
    def __init__(self, index: InvertedIndex, directory):
        self.index = index
        self.words, self.pls = get_posting_iter(index, directory)

    @abstractmethod
    def search_query(self, query_to_search):
        return


class BinaryQuerySearcher(QuerySearcher):

    def __init__(self, index, directory):
        super().__init__(index, directory)

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
    def __init__(self, index):
        super().__init__(index)

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
                # Changed (freq / self.index.dl[str(doc_id)])
                normalized_tfidf = [(doc_id, (freq / self.index.dl[doc_id]) *
                                     math.log(len(self.index.dl) / self.index.df[term], 10)) for doc_id, freq in
                                    list_of_doc]

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
        epsilon = .0000001
        total_vocab_size = len(self.index.term_total)
        Q = np.zeros(total_vocab_size)
        term_vector = list(self.index.term_total.keys())
        counter = Counter(query_to_search)

        for token in np.unique(query_to_search):
            if token in self.index.term_total.keys():  # avoid terms that do not appear in the index.
                tf = counter[token] / len(query_to_search)  # term frequency divided by the length of the query
                df = self.index.df[token]
                idf = math.log((len(self.index.dl)) / (df + epsilon), 10)  # smoothing

                try:
                    ind = term_vector.index(token)
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

        total_vocab_size = len(self.index.term_total)
        # No need to utilize all documents, only those having corresponding terms with the query.
        candidates_scores = self.get_candidate_documents_and_scores(query_to_search)
        unique_candidates = np.unique([doc_id for doc_id, freq in candidates_scores.keys()])
        D = np.zeros((len(unique_candidates), total_vocab_size))
        D = pd.DataFrame(D)

        D.index = unique_candidates
        D.columns = self.index.term_total.keys()

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
        result = {}

        D = self.generate_document_tfidf_matrix(queries_to_search)
        Q = self.generate_query_tfidf_vector(queries_to_search)
        return get_top_n(cosine_similarity(D, Q), N)


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

    def __init__(self, index, tf=None, k1=1.5, b=0.75):
        super().__init__(index)
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

    def search_query(self, query, N=100):
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


def merge_results(title_scores, body_scores, title_weight=0.5, text_weight=0.5, N=3):
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


page_views = PageViews()
page_ranks = PageRanks()


def get_page_views(wiki_ids: list):
    return page_views.get_page_views(wiki_ids)


def get_page_ranks(wiki_ids: list):
    return page_ranks.get_page_ranks(wiki_ids)
