"""
Copyright 2019, University of Freiburg
Chair of Algorithms and Data Structures.
Hannah Bast <bast@cs.uni-freiburg.de>
Claudius Korzen <korzen@cs.uni-freiburg.de>
Patrick Brosi <brosi@cs.uni-freiburg.de>
Natalie Prange <prange@cs.uni-freiburg.de>
Sebastian Walter <swalter@cs.uni-freiburg.de>
"""

import re
import argparse
import math
import readline  # noqa

import torch


class InvertedIndex:
    """

    A simple inverted index that uses BM25 scores.

    """

    def __init__(self) -> None:
        """

        Creates an empty inverted index.

        """
        # the inverted lists of tuples (doc id, score)
        self.inverted_lists: dict[str, list[tuple[int, float]]] = {}
        # the docs, a list of tuples (title, description)
        self.docs: list[tuple[str, str]] = []

        # a sparse matrix containing the same info as the inverted lists
        self.td_matrix: torch.Tensor = torch.empty((0, 0), dtype=torch.float)
        # mapping from terms to row indices in the td matrix
        self.term_indices: dict[str, int] = {}

    def get_keywords(self, query: str) -> list[str]:
        """

        Returns the keywords of the given query.

        """
        return re.findall(r"[A-Za-z]+", query.lower())

    def build_from_file(
        self,
        file_name: str,
        b: float,
        k: float
    ) -> None:
        """

        Construct the inverted index from the given file. The expected format
        of the file is one document per line, in the format
        <title>TAB<description>TAB<num_ratings>TAB<rating>TAB<num_sitelinks>
        Each entry in the inverted list associated to a word should contain a
        document id and a BM25 score. Compute the BM25 scores as follows:

        (1) In a first pass, compute the inverted lists with tf scores (that
            is the number of occurrences of the word within the <title> and the
            <description> of a document). Further, compute the document length
            (DL) for each document (that is the number of words in the <title>
            and the <description> of a document). Afterwards, compute the
            average document length (AVDL).
        (2) In a second pass, iterate over all inverted lists and replace the
            tf scores by BM25 scores, defined as:
            BM25 = tf * (k+1) / (k * (1 - b + b * DL/AVDL) + tf) * log2(N/df),
            where N is the total number of documents and df is the number of
            documents that contain the word.

        >>> ii = InvertedIndex()
        >>> ii.build_from_file("example.tsv", b=0.0, k=float("inf"))
        >>> inv_lists = sorted(ii.inverted_lists.items())
        >>> [(w, [(i, '%.3f' % tf) for i, tf in l]) for w, l in inv_lists]
        ... # doctest: +NORMALIZE_WHITESPACE
        [('animated', [(1, '0.415'), (2, '0.415'), (4, '0.415')]),
         ('animation', [(3, '2.000')]),
         ('film', [(2, '1.000'), (4, '1.000')]),
         ('movie', [(1, '0.000'), (2, '0.000'), (3, '0.000'), (4, '0.000')]),
         ('non', [(2, '2.000')]),
         ('short', [(3, '1.000'), (4, '2.000')])]

        >>> ii = InvertedIndex()
        >>> ii.build_from_file("example.tsv", b=0.75, k=1.75)
        >>> inv_lists = sorted(ii.inverted_lists.items())
        >>> [(w, [(i, '%.3f' % tf) for i, tf in l]) for w, l in inv_lists]
        ... # doctest: +NORMALIZE_WHITESPACE
        [('animated', [(1, '0.459'), (2, '0.402'), (4, '0.358')]),
         ('animation', [(3, '2.211')]),
         ('film', [(2, '0.969'), (4, '0.863')]),
         ('movie', [(1, '0.000'), (2, '0.000'), (3, '0.000'), (4, '0.000')]),
         ('non', [(2, '1.938')]),
         ('short', [(3, '1.106'), (4, '1.313')])]

        >>> ii = InvertedIndex()
        >>> ii.build_from_file("example.tsv", b=0.0, k=0.0)
        >>> inv_lists = sorted(ii.inverted_lists.items())
        >>> [(w, [(i, '%.3f' % tf) for i, tf in l]) for w, l in inv_lists]
        ... # doctest: +NORMALIZE_WHITESPACE
        [('animated', [(1, '0.415'), (2, '0.415'), (4, '0.415')]),
         ('animation', [(3, '2.000')]),
         ('film', [(2, '1.000'), (4, '1.000')]),
         ('movie', [(1, '0.000'), (2, '0.000'), (3, '0.000'), (4, '0.000')]),
         ('non', [(2, '2.000')]),
         ('short', [(3, '1.000'), (4, '1.000')])]
        """
        doc_lengths = []
        with open(file_name, "r", encoding="utf8") as file:
            doc_id = 0
            for line in file:
                doc_id += 1

                # store the doc as a tuple (title, description).
                title, desc, _ = line.split("\t", 2)
                self.docs.append((title, desc))

                keywords = self.get_keywords(title) + self.get_keywords(desc)
                doc_lengths.append(len(keywords))

                for word in keywords:
                    if word not in self.inverted_lists:
                        # the word is seen for first time, create a new list.
                        self.inverted_lists[word] = [(doc_id, 1.0)]
                        continue

                    # get the last inserted doc id and tf score for the word
                    last_doc_id, last_tf = self.inverted_lists[word][-1]
                    if last_doc_id != doc_id:
                        # doc was not yet seen, set tf to 1
                        self.inverted_lists[word].append((doc_id, 1.0))
                    else:
                        # doc was already seen, add 1 to tf
                        self.inverted_lists[word][-1] = (doc_id, last_tf + 1.0)

        # compute average document length
        n = len(doc_lengths)
        avdl = sum(doc_lengths) / max(1, n)
        # iterate the inverted lists and replace the tf scores by
        # BM25 scores, defined as follows:
        # BM25 = tf * (k + 1) / (k * (1 - b + b * DL / AVDL) + tf) * log2(N/df)
        for word, inverted_list in self.inverted_lists.items():
            for i, (doc_id, tf) in enumerate(inverted_list):
                # obtain the document length (dl) of the document.
                dl = doc_lengths[doc_id - 1]
                # compute alpha = (1 - b + b * DL / AVDL).
                alpha = 1 - b + (b * dl / avdl)
                # compute tf2 = tf * (k + 1) / (k * alpha + tf).
                tf2 = tf * (1 + (1 / k)) / (alpha + (tf / k)) if k > 0 else 1
                # compute df (that is the length of the inverted list).
                df = len(self.inverted_lists[word])
                # compute the BM25 score = tf2 * log2(N/df).
                inverted_list[i] = (doc_id, tf2 * math.log2(n / df))

    def build_td_matrix(self) -> None:
        """

        Builds a sparse term-document matrix and term-to-row mapping
        for the inverted lists.

        >>> torch.set_printoptions(precision=3)
        >>> ii = InvertedIndex()
        >>> ii.build_from_file("example.tsv", b=0, k=float("inf"))
        >>> ii.build_td_matrix()
        >>> print(torch.tensor(sorted(ii.td_matrix.to_dense().tolist())))
        tensor([[0.000, 0.000, 0.000, 0.000],
                [0.000, 0.000, 1.000, 2.000],
                [0.000, 0.000, 2.000, 0.000],
                [0.000, 1.000, 0.000, 1.000],
                [0.000, 2.000, 0.000, 0.000],
                [0.415, 0.415, 0.000, 0.415]])
        """
        # building a sparse matrix of BM25 scores from inverted lists
        values = []
        rows = []
        cols = []

        for index, (term, docs) in enumerate(self.inverted_lists.items()):
            # associate each word with row number
            self.term_indices[term] = index
            for doc_id, score in docs:
                values.append(score)
                rows.append(index)
                # because we stores doc_id with + 1 index
                cols.append(doc_id - 1)

        # build the td_matrix
        self.td_matrix = torch.sparse_coo_tensor(
                    [rows, cols], values, dtype=torch.float)


    def process_query(
        self,
        keywords: list[str],
    ) -> list[tuple[int, float]]:
        """

        Process the keyword query as in exercise sheet 2, but with
        linear algebra operations using the term-document matrix.

        >>> ii = InvertedIndex()
        >>> ii.inverted_lists = {
        ...   "foo": [(1, 0.2), (3, 0.6)],
        ...   "bar": [(1, 0.4), (2, 0.7), (3, 0.5)],
        ...   "baz": [(2, 0.1)]
        ... }
        >>> ii.build_td_matrix()
        >>> result = ii.process_query(["foo", "bar"])
        >>> [(id, "%.1f" % tf) for id, tf in result]
        [(3, '1.1'), (2, '0.7'), (1, '0.6')]
        >>> result = ii.process_query(["bar"])
        >>> [(id, "%.1f" % tf) for id, tf in result]
        [(2, '0.7'), (3, '0.5'), (1, '0.4')]
        >>> result = ii.process_query(["barb"])
        >>> [(id, "%.1f" % tf) for id, tf in result]
        []
        >>> result = ii.process_query(["foo", "bar", "baz"])
        >>> [(id, "%.1f" % tf) for id, tf in result]
        [(3, '1.1'), (2, '0.8'), (1, '0.6')]
        >>> result = ii.process_query([""])
        >>> [(id, "%.1f" % tf) for id, tf in result]
        []

        >>> ii = InvertedIndex()
        >>> ii.inverted_lists = {
        ...   "foo": [(1, 0.2), (3, 0.6)],
        ...   "bar": [(2, 0.4), (3, 0.1), (4, 0.8)]
        ... }
        >>> ii.build_td_matrix()
        >>> result = ii.process_query(["foo", "bar", "foo", "bar"])
        >>> [(id, "%.1f" % tf) for id, tf in result]
        [(4, '1.6'), (3, '1.4'), (2, '0.8'), (1, '0.4')]
        """

        # building the query vector
        # intialise with 0's and dimension
        # would be equal to the length of the words
        query_vector = torch.zeros(len(self.term_indices))
        # for each matching term/row make the query vector 1
        for term in keywords:
            if term not in self.term_indices:
                continue
            query_vector[self.term_indices[term]] += 1

        # check if tensor is all zero
        if torch.all(query_vector == 0):
            return []

        # this returns the score for the query
        # with our td matrix
        scores = query_vector @ self.td_matrix

        # now we return the result as (doc_id, bm25_score)
        results: list[tuple[int, float]] = []
        for index, score in enumerate(scores):
            results.append((index + 1, float(score)))

        return sorted(results, key=lambda result: result[1], reverse=True)

    def render_output(
        self,
        doc_ids: list[tuple[int, float]],
        keywords: list[str],
        k: int = 3
    ) -> str:
        """
        Renders the output for the top-k of the given doc_ids. Fetches the
        the titles and descriptions of the related records and highlights
        the occurrences of the given keywords in the output, using ANSI escape
        codes.
        """
        # compile a pattern to identify the given keywords in a string
        p = re.compile(
            r"\b(" + "|".join(re.escape(k) for k in keywords) + r")\b",
            flags=re.IGNORECASE
        )
        outputs = []
        # output at most k matching records
        for i in range(min(len(doc_ids), k)):
            doc_id, _ = doc_ids[i]

            # ids are 1-based
            title, desc = self.docs[doc_id - 1]

            # highlight the keywords in the title in bold and red
            title = re.sub(p, "\033[0m\033[1;31m\\1\033[0m\033[1m", title)

            # print the rest of the title in bold
            title = f"\033[1m{title}\033[0m"

            # highlight the keywords in the description in red
            desc = re.sub(p, "\033[31m\\1\033[0m", desc)

            # append formatted title and description to output list
            outputs.append(f"{title}\n{desc}")

        outputs.append(f"\n# total hits: {len(doc_ids):,}")
        return "\n".join(outputs)


def parse_args() -> argparse.Namespace:
    """
    Defines and parses command line arguments for this script.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "file",
        type=str,
        help="the file from which to construct the inverted index",
    )
    parser.add_argument(
        "-b",
        "--b-param",
        type=float,
        default=0.75,
        help="the b parameter for BM25",
    )
    parser.add_argument(
        "-k",
        "--k-param",
        type=float,
        default=1.75,
        help="the k parameter for BM25",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """
    Constructs an inverted index from a given text file, then asks the user in
    an infinite loop for keyword queries and outputs the title and description
    of up to three matching records.
    """
    # create a new inverted index from the given file
    print(f"Reading from file {args.file}")
    ii = InvertedIndex()
    ii.build_from_file(args.file, args.b_param, args.k_param)
    ii.build_td_matrix()

    while True:
        # ask the user for a keyword query
        query = input("\nYour keyword query: ")

        # split the query into keywords
        keywords = ii.get_keywords(query)

        # process the keywords
        doc_ids = ii.process_query(keywords)

        # render the output (with ANSI codes to highlight the keywords)
        output = ii.render_output(doc_ids, keywords)

        # print the output
        print(f"\n{output}")


if __name__ == "__main__":
    main(parse_args())
