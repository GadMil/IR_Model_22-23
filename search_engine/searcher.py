import math
import pickle
import re
from abc import abstractmethod
from collections import Counter

import nltk
import numpy as np
from nltk.corpus import stopwords
import hashlib


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


def get_posting_iter(index):
    """
    This function returning the iterator working with posting list.

    Parameters:
    ----------
    index: inverted index
    """
    words, pls = zip(*index.posting_lists_iter())
    return words, pls


class Ranking:
    def __init__(self, index):
        self.index = index
        self.words, self.pls = get_posting_iter(index)

    @abstractmethod
    def rank(self, query_to_search):
        return

    @abstractmethod
    def get_candidate_documents_and_scores(self, query_to_search):
        """
        Generate a dictionary representing a pool of candidate documents for a given query. This function will go through every token in query_to_search
        and fetch the corresponding information (e.g., term frequency, document frequency, etc.') needed to calculate TF-IDF from the posting list.
        Then it will populate the dictionary 'candidates.'
        For calculation of IDF, use log with base 10.
        tf will be normalized based on the length of the document.

        Parameters:
        -----------
        query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                         Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

        index:           inverted index loaded from the corresponding files.

        words,pls: iterator for working with posting.

        Returns:
        -----------
        dictionary of candidates. In the following format:
                                                                   key: pair (doc_id,term)
                                                                   value: tfidf score.
        """
        return


class BinaryRanking(Ranking):

    def __init__(self, index):
        super().__init__(index)

    def rank(self, query_to_search):
        return sorted(self.get_candidate_documents_and_scores(query_to_search), key=lambda x: x[1], reverse=True)

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


page_views = PageViews()
page_ranks = PageRanks()


def get_page_views(wiki_ids: list):
    return page_views.get_page_views(wiki_ids)


def get_page_ranks(wiki_ids: list):
    return page_ranks.get_page_ranks(wiki_ids)
