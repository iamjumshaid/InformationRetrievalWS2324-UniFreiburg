
# Building fast search

This code implements:
- **Building Inverted Index:** Reads records from a file and constructs an inverted index.
- **Keyword Query Processing:** Processes keyword queries and returns matching records.
- **Highlighting Keywords:** Highlights the queried keywords in the output records.

### Usage:
1. **Constructing Inverted Index:**
    - Execute the script with a file argument to construct the inverted index.
    ```
    python inverted_index.py example.tsv
    ```
2. **Keyword Querying:**
    - After constructing the index, the script waits for keyword queries.
    - Enter keywords separated by spaces to retrieve matching records.

### File Format:
- The expected format of the input file is one record per line.
- Each line should be in the format: `<title>TAB<description>`

### Example:
- After constructing the index, you can query keywords and get up to three matching records along with their titles and descriptions.
