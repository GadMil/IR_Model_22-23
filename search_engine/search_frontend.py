from flask import Flask, request, jsonify
import searcher
from searcher import *
import inverted_index_gcp
import time
import os

from google.cloud import storage
import gensim.downloader

bucket_name = 'information_retrieval_project'
client = storage.Client('academic-ivy-370514')
bucket = client.bucket(bucket_name)

tokenizer = searcher.Tokenizer()


class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

exp = 1


def read_global_page_views(file_name):
    blob = storage.Blob(f'{file_name}.pkl', bucket)
    with open(f'./{file_name}.pkl', "wb") as file_obj:
        blob.download_to_file(file_obj)
    page_views.read_page_views()


def read_global_page_ranks(file_name):
    blob = storage.Blob(f'{file_name}.pkl', bucket)
    with open(f'./{file_name}.pkl', "wb") as file_obj:
        blob.download_to_file(file_obj)
    page_ranks.read_page_ranks()


def get_index(index_name, bins_folder):
    index = inverted_index_gcp.InvertedIndex.read_index('indices/', index_name)
    index.directory = bins_folder
    return index


body_index = get_index('body_index', 'body_bins')
title_index = get_index('title_index', 'title_bins')
anchor_index = get_index('anchor_index', 'anchor_bins')

page_views = PageViews()
page_ranks = PageRanks()

page_views.read_page_views()
page_ranks.read_page_ranks()

word2vec_glove = gensim.downloader.load('glove-wiki-gigaword-50')

@app.route("/search")
def search():
    """ Returns up to a 100 of your best search results for the query. This is
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    """
    start = time.time()
    global exp
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    query_tokens = tokenizer.tokenize(query)
    if query_tokens:
        # query_tokens = expand_query(query_tokens, word2vec_glove)
        body_ranks = BM25QuerySearcher(body_index).search_query(query_tokens)
        title_ranks = BM25QuerySearcher(title_index).search_query(query_tokens)
        merged_ranks = merge_results(title_ranks, body_ranks)
        for item in merged_ranks:
            res.append((int(item[0])))
        # anchor_ranks = BinaryQuerySearcher(anchor_index).search_query(query_tokens)
        # page_views_scores = page_views.get_page_views(list(merged_ranks.keys()))
        # page_ranks_scores = page_ranks.get_page_ranks(list(merged_ranks.keys()))
        #
        # doc_ranks = defaultdict(int)
        # i = 0
        # for doc_id, bm25Rank in merged_ranks:
        #     doc_ranks[doc_id] += (2 * bm25Rank * page_views.pvCounter[doc_id]) / (
        #                 bm25Rank + page_views.pvCounter[doc_id])
        #     doc_ranks[doc_id] *= 3 * anchor_ranks[i]
        #     i += 1
        #
        # doc_ranks = sorted([(doc_id, rank) for doc_id, rank in doc_ranks.items()], key=lambda x: x[1], reverse=True)[
        #             :100]
        #
        # for item in doc_ranks:
        #     res.append((int(item[0])))
    # END SOLUTION
    end = time.time()
    data = [[res, end-start]]
    df = pd.DataFrame(data, columns=["Results", "Time"])
    filename = f'{exp}.csv'
    out = os.path.join(os.getcwd(), 'Results')
    if not os.path.exists(out):
        os.mkdir(out)
    fullname = os.path.join(out, filename)
    df.to_csv(fullname)
    exp += 1
    return jsonify(res)


@app.route("/search_body")
def search_body():
    """ Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the
        staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    """
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    query_tokens = tokenizer.tokenize(query)
    if query_tokens:
        docs_ranks = TfIdfQuerySearcher(body_index).search_query(query_tokens)
        res = [(item[0], title_index.id_to_title.get(item[0], "")) for item in docs_ranks]
    # END SOLUTION
    return jsonify(res)


@app.route("/search_title")
def search_title():
    """ Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF
        DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords. For example, a document
        with a title that matches two distinct query words will be ranked before a
        document with a title that matches only one distinct query word,
        regardless of the number of times the term appeared in the title (or
        query).

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    """
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    query_tokens = tokenizer.tokenize(query)
    if query_tokens:
        docs_ranks = BinaryQuerySearcher(title_index).search_query(query_tokens)
        res = [(id, title_index.id_to_title.get(id, "")) for id in docs_ranks]
        # for item in docs_ranks:
        #     res.append((int(item[0]), title_index.id_to_title.get(item[0], "")))
    # END SOLUTION
    return jsonify(res)


@app.route("/search_anchor")
def search_anchor():
    """ Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE ANCHOR TEXT of articles, ordered in descending order of the
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page.
        DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment
        3 (GCP part) to do the tokenization and remove stopwords. For example,
        a document with an anchor text that matches two distinct query words will
        be ranked before a document with anchor text that matches only one
        distinct query word, regardless of the number of times the term appeared
        in the anchor text (or query).

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    """
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    query_tokens = tokenizer.tokenize(query)
    if query_tokens:
        docs_ranks = BinaryQuerySearcher(anchor_index).search_query(query_tokens)
        res = [(id, title_index.id_to_title.get(id, "")) for id in docs_ranks]
        # for item in docs_ranks:
        #     res.append((int(item[0]), title_index.id_to_title.get(item[0], "")))
    # END SOLUTION
    return jsonify(res)


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    """ Returns PageRank values for a list of provided wiki article IDs.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correspond to the provided article IDs.
    """
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    res = page_ranks.get_page_ranks(wiki_ids)
    # END SOLUTION
    return jsonify(res)


@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    """ Returns the number of page views that each of the provided wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correspond to the
          provided list article IDs.
    """
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    res = page_views.get_page_views(wiki_ids)
    # END SOLUTION
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=False)
