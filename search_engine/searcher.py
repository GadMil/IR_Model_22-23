import pickle


class PageViews:

    def __init__(self):
        self._page_views = None

    def set_page_views(self, pv: str):
        self._page_views = pv

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
        self._page_ranks = None

    def set_page_ranks(self, pr: str):
        self._page_ranks = pr

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


page_views = PageViews()
page_ranks = PageRanks()


def set_page_views(pv_temp: str):
    page_views.set_page_views(pv_temp)


def get_page_views(wiki_ids: list):
    return page_views.get_page_views(wiki_ids)


def set_page_ranks(pr: str):
    page_ranks.set_page_ranks(pr)


def get_page_ranks(wiki_ids: list):
    return page_ranks.get_page_ranks(wiki_ids)
