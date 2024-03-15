

# Vector Space Model

This exercise uses Pytorch to build Document term matix using Sparse Matrix with BM25 score as values for the matrix. Also, using pre-built embeddings vectors are used to compute query vectors and then finding their similarites using Cosine-Similarity.

# SETUP

Download the datasets and place it in root folder.
1. [movies-plot](http://ad-teaching.informatik.uni-freiburg.de/InformationRetrievalWS2324/datasets/movies-plots.tsv)
2. [embeddings-fasttext.pt](http://ad-teaching.informatik.uni-freiburg.de/InformationRetrievalWS2324/datasets/embeddings.fasttext.pt)

# TEST

```
python3 similarity_search.py movies-plots.tsv embeddings.fasttext.pt
```
