

# Web Applications 02

This exercise extends the code from exercise 08 and uses Javascript to request the server asynchronously and update the HTML content.

# SETUP

Download the datasets and place it in root folder.
1. [wikidata-entities](http://ad-teaching.informatik.uni-freiburg.de/InformationRetrievalWS2324/datasets/wikidata-entities.tsv")
2. [wikidata-properties](http://ad-teaching.informatik.uni-freiburg.de/InformationRetrievalWS2324/datasets/wikidata-properties.tsv")

# TEST

```
cat wikidata-complex.sql | sqlite3 wikidata-complex.db
python3 server.py wikidata-entities.tsv wikidata-complex.db 8080
```

# TEST
Server connection code in this exercise is used from the master solution.
