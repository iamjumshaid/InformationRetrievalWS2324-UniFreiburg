
# Ranking and Evaluation

This code implements:
This code builds on sheet-01. It further enhances the inverted_index by adding BM25 score on it.

### Usage:
    - Execute the script with a file argument to construct the inverted index.
    ```
    python inverted_index.py example.tsv
    ```
    - Evaluate the search engine with a training file argument.
    ```
    python evaluate.py example.tsv example-benchmark.tsv
    ```

### File Format:
- The expected format of the input file is one record per line.
- Each line should be in the format: `<title>TAB<description>`

### Example:
- After constructing the index, you can query keywords and get up to three matching records along with their titles and descriptions. The results are not more relevant than before due to ranking criteria.
