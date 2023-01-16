# IR_Model_22-23
# IR-Wikipedia-Search-Engine
# Creators 
Shaked Ben Aharon & Gad Michael Miller
# Summary
Implementation of an IR system that retrieves relevant articles from the English Wikipedia, with a corpus size of 6,348,910 articles.
Project for IR course in Ben-Gurion University.
# Pre-proccesing
The data is parsed and cleaned, lower cased and split into tokens, and finally stopwords were removed.
# Creating Inverted Index
Splitting our data to three -> Body (article's texts) index, Title index and Anchors index.
Each index stores:
* Posting lists- (term, [(tf, document id), ...])
* Total frequency per term
* All terms' df
* All terms' idf
* All Documents' length
* All documents' norms
* Average document's length
* The corpus's size
* The dictionary's size
 and more data that will simplify later calculation.
# Search
For each component we used diffrent search method:
* Title search -> Binary search method returnig all relevant results.
* Anchor search -> Binary search method returnig all relevant results.
* Body search -> terms are weighted with tf-idf and similarity is measured by cosine similarity function returnig 30 relevant results.
* We also support getting the articles' page rank and page views by their id (data is stored in advanced to make calculation fast).
* Main seatch action - BM25 score on the body index is merged with Binary search on the title index and page ranks.
in addition we used the documents page ranks and page views to preform re-ranking to the "better" documents from the results that we got from the search.
# Evaluation
We evaluated our results with:
* Average precision at k
* Recall
* Map at 40
