
# Database operations, Cost estimations and run sequences



This exercise implements GROUP BY operation in python and also writes different sequences and estimates the cost of operations such as SELECT, PROJECT, and JOIN (Hash-join) for analyses.

# SETUP

Download the dataset and extract in the root directory of the project.

[Download Movie Tables](http://ad-teaching.informatik.uni-freiburg.de/InformationRetrievalWS2324/datasets/movies-tables-new.tar.gz")

# TEST

```

python queries.py *.tsv --exercise 1
python queries.py *.tsv --exercise 2
cat queries.sql | sqlite3

```
