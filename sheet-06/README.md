
# Knowledge Graphs, SPARQL, SPARQL to SQL

This exercise implements executing SPARQL queries on RDF dataset by converting them to SQL using Python parsing.

# SETUP

Download the datasets and place it in root folder.

1. [Simple dataset](http://ad-teaching.informatik.uni-freiburg.de/InformationRetrievalWS2324/datasets/wikidata-simple.tsv")
2. [Complex dataset](http://ad-teaching.informatik.uni-freiburg.de/InformationRetrievalWS2324/datasets/wikidata-complex.tsv")

# TEST

**Run the following commands:**
1. Build Databases
```

cat wikidata-simple.sql | sqlite3 wikidata_simple.db
cat wikidata-complex.sql | sqlite3 wikidata_complex.db

```
2. Run SPARQL queries
```
python sparql_to_sql.py wikidata_complex.db ceos.sparql
python sparql_to_sql.py wikidata_complex.db monarchs.sparql
python sparql_to_sql.py example.db example.sparql
```

Trivia:
**The Acts of Union, passed by the English and Scottish Parliaments in 1707, led to the creation of a united kingdom to be called “Great Britain”**  on 1 May of that year. The UK Parliament met for the first time in October 1707.
