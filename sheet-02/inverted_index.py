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
import json
import os

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
        # if inverted list is already calculated read from that
        # if __name__ == "__main__" or __name__ == 'inverted_index':
        new_file_name = f"{file_name.rsplit('.', 1)[0]}_inverted_list_{b}_{k}.json"
        new_file_name_2 = f"{file_name.rsplit('.', 1)[0]}_docs_{b}_{k}.json"
        if os.path.exists(new_file_name) and os.path.exists(new_file_name_2):
            with open(new_file_name, 'r') as f:
                self.inverted_lists = json.load(f)
            with open(new_file_name_2, 'r') as f:
                self.docs = json.load(f)
            return

        doc_id = 0
        # storing DL
        doc_lengths = []
        with open(file_name, "r", encoding="utf8") as file:
            for line in file:
                doc_id += 1

                # store the doc as a tuple (title, description).
                title, desc, _ = line.split("\t", 2)
                self.docs.append((title, desc))

                keywords = self.get_keywords(title) + self.get_keywords(desc)

                # calculating doc length
                # doc at 0 index is first document.
                doc_lengths.append(len(keywords))

                for word in keywords:
                    if word not in self.inverted_lists:
                        # the word is seen for first time, create a new list with 1 tf
                        self.inverted_lists[word] = [(doc_id, 1.0)]
                        continue

                    # get the last document id and term frequency for the word
                    last_doc_id, last_tf = self.inverted_lists[word][-1]

                    if doc_id == last_doc_id:
                        # that means the doc_id is already recorded for current word
                        # just have to increase the frequency
                        self.inverted_lists[word][-1] = (doc_id, last_tf + 1)
                    else:
                        # the doc is not recorded for the current word
                        self.inverted_lists[word].append((doc_id, 1.0))

                    # The below code is one solution but using the better approach above
                    # if word in self.inverted_lists and doc_id not in [t[0] for t in self.inverted_lists[word]]:
                    #     # the word is present, but found first time in a document so adding the doc_id with 1 tf
                    #     self.inverted_lists[word].append((doc_id, 1))
                    # else:
                    #     # if word is already present in an existing doc then increment existing tf
                    #     for index, posting in enumerate(self.inverted_lists[word]):
                    #         doc, tf = posting
                    #         if doc == doc_id:
                    #             self.inverted_lists[word][index] = (doc, tf+1)
                    #             break

        # calculating bm25 score for each document
        n = len(doc_lengths)
        avg_doc_len = sum(doc_lengths) / max(1, n)
        for word, inverted_list in self.inverted_lists.items():
            for index, posting in enumerate(inverted_list):
                doc, tf = posting
                dl = doc_lengths[doc - 1]
                alpha = (1 - b) + (b * dl / avg_doc_len)
                tf_plus = 0
                if k == 0:
                    tf_plus = 1
                elif k == float('inf'):
                    tf_plus = tf
                else:
                    tf_plus = tf * (k + 1) / (k * alpha + tf)
                df = len(inverted_list)
                bm25_score = tf_plus * math.log((n / df), 2)
                self.inverted_lists[word][index] = (doc, bm25_score)

        # store the final inverted list and doc to avoid repition of calculation
        # if __name__ == "__main__" or __name__ == 'inverted_index':
        with open(new_file_name, 'w') as f:
            json.dump(self.inverted_lists, f)
        with open(new_file_name_2, 'w') as f:
            json.dump(self.docs, f)

    def merge(
        self,
        list1: list[tuple[int, float]],
        list2: list[tuple[int, float]],
    ) -> list[tuple[int, float]]:
        """
        Compute the union of the two given inverted lists in linear time
        (linear in the total number of entries in the two lists), where the
        entries in the inverted lists are postings of form (doc_id, bm25_score)
        and are expected to be sorted by doc_id, in ascending order.

        >>> ii = InvertedIndex()
        >>> l1 = ii.merge([(1, 2.1), (5, 3.2)], [(1, 1.7), (2, 1.3), (6, 3.3)])
        >>> [(id, "%.1f" % tf) for id, tf in l1]
        [(1, '3.8'), (2, '1.3'), (5, '3.2'), (6, '3.3')]

        >>> l2 = ii.merge([(3, 1.7), (5, 3.2), (7, 4.1)], [(1, 2.3), (5, 1.3)])
        >>> [(id, "%.1f" % tf) for id, tf in l2]
        [(1, '2.3'), (3, '1.7'), (5, '4.5'), (7, '4.1')]

        >>> l2 = ii.merge([], [(1, 2.3), (5, 1.3)])
        >>> [(id, "%.1f" % tf) for id, tf in l2]
        [(1, '2.3'), (5, '1.3')]

        >>> l2 = ii.merge([(1, 2.3)], [])
        >>> [(id, "%.1f" % tf) for id, tf in l2]
        [(1, '2.3')]

        >>> l2 = ii.merge([], [])
        >>> [(id, "%.1f" % tf) for id, tf in l2]
        []
        """
        intersected_list = []
        i = j = 0
        while i < len(list1) and j < len(list2):
            if list1[i][0] < list2[j][0]:
                intersected_list.append(list1[i])
                i += 1
            elif list1[i][0] == list2[j][0] and list1[i][0]:
                intersected_list.append((list1[i][0], list1[i][1] + list2[j][1]))
                i += 1
                j += 1
            elif list2[j][0] < list1[i][0]:
                intersected_list.append(list2[j])
                j += 1

        intersected_list.extend(list1[i:])
        intersected_list.extend(list2[j:])

        return intersected_list

    def process_query(
        self,
        keywords: list[str],
        use_refinements: bool = False
    ) -> list[tuple[int, float]]:
        """
        Process the given keyword query as follows: Fetch the inverted list for
        each of the keywords in the query and compute the union of all lists.
        Sort the resulting list by BM25 scores in descending order.

        This method returns all results for the given query, not just the
        top 3!

        If you want to implement some ranking refinements, make these
        refinements optional (their use should be controllable via the
        use_refinements flag).

        >>> ii = InvertedIndex()
        >>> ii.inverted_lists = {
        ... "foo": [(1, 0.2), (3, 0.6)],
        ... "bar": [(1, 0.4), (2, 0.7), (3, 0.5)],
        ... "baz": [(2, 0.1)]}
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
        """
        # edge case
        if len(keywords) == 0:
            return []

        result = []
        # create the merge list
        for word in keywords:
            if word in self.inverted_lists:
                result = self.merge(result, self.inverted_lists[word])

        # sort the result by bmw score in descending order
        return sorted(result, key= lambda x: float(x[1]), reverse=True)

    # @@NOTE: This function is copied from master solution so credits to them.
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
                outputs.append(f"{i+1}: {title} - {desc}")

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
    parser.add_argument(
        "--use-refinements",
        action="store_true",
        help="whether to use refinements"
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
    print(ii.inverted_lists)
    # @@NOTE: This part is copied from master solution
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
