"""
Copyright 2019, University of Freiburg
Hannah Bast <bast@cs.uni-freiburg.de>
Claudius Korzen <korzen@cs.uni-freiburg.de>
Patrick Brosi <brosi@cs.uni-freiburg.de>
Natalie Prange <prange@cs.uni-freiburg.de>
Sebastian Walter <swalter@cs.uni-freiburg.de>
"""

import argparse
import re


class InvertedIndex:
    """
    A simple inverted index as explained in lecture 1.
    """

    def __init__(self) -> None:
        """
        Creates an empty inverted index.
        """
        # the inverted lists of record ids
        self.inverted_lists: dict[str, list[int]] = {}
        # the records, a list of tuples (title, description)
        self.records: list[tuple[str, str]] = []
        self.doc_with_record: dict[int, list[tuple[str, str]]] = {}

    def get_keywords(self, query: str) -> list[str]:
        """
        Returns the keywords of the given query.
        """
        return re.findall(r"[A-Za-z]+", query.lower())

    def build_from_file(self, file_name: str) -> None:
        """
        Constructs the inverted index from given file in linear time (linear in
        the number of words in the file). The expected format of the file is
        one record per line, in the format
        <title>TAB<description>TAB<num_ratings>TAB<rating>TAB<num_sitelinks>
        You can ignore the last three columns for now, they will become
        interesting for exercise sheet 2.

        >>> ii = InvertedIndex()
        >>> ii.build_from_file("example.tsv")
        >>> sorted(ii.inverted_lists.items())
        [('a', [1, 2]), ('doc', [1, 2, 3]), ('film', [2]), ('movie', [1, 3])]
        >>> ii.records # doctest: +NORMALIZE_WHITESPACE
        [('Doc 1', 'A movie movie.'), ('Doc 2', 'A film.'),
         ('Doc 3', 'Movie.')]
        """
        # TODO: make sure that each inverted list contains a particular record
        # id at most once, even if the respective word occurs multiple times in
        # the same record. also cache the titles and8 descriptions of the movies
        # in self.records to use them later for output.
        with open(file_name, "r") as file:
            record_id = 0
            for line in file:
                line = line.strip()
                record_id += 1

                # caching title and description
                title, desc, _ = line.split("\t", 2)
                self.records.append((title, desc))

                # store doc_id for later
                self.doc_with_record[record_id] = (title, desc)

                keywords = self.get_keywords(line)
                for word in keywords:
                    if word not in self.inverted_lists:
                        # the word is seen for first time, create a new list
                        self.inverted_lists[word] = []

                    # only adding record id if not present already for the word
                    if record_id not in self.inverted_lists[word]:
                        self.inverted_lists[word].append(record_id)

    def intersect(self, list1: list[int], list2: list[int]) -> list[int]:
        """
        Computes the intersection of the two given inverted lists in linear
        time (linear in the total number of elements in the two lists).

        >>> ii = InvertedIndex()
        >>> ii.intersect([1, 5, 7], [2, 4])
        []
        >>> ii.intersect([1, 2, 5, 7], [1, 3, 5, 6, 7, 9])
        [1, 5, 7]
        """
        intersected_list = []
        i = j = 0
        while i < len(list1) and j < len(list2):
            if list1[i] == list2[j]:
                intersected_list.append(list1[i])
                i += 1
                j += 1
            elif list1[i] < list2[j]:
                i += 1
            else:
                j += 1

        return intersected_list

    def process_query(self, keywords: list[str]) -> list[int]:
        """
        Processes the given keyword query as follows: Fetches the inverted list
        for each of the keywords in the given query and computes the
        intersection of all inverted lists (which is empty, if there is a
        keyword in the query which has no inverted list in the index).

        >>> ii = InvertedIndex()
        >>> ii.build_from_file("example.tsv")
        >>> ii.process_query([])
        []
        >>> ii.process_query(["doc"])
        [1, 2, 3]
        >>> ii.process_query(["doc", "movie"])
        [1, 3]
        >>> ii.process_query(["doc", "movie", "comedy"])
        []
        """
        # edge case
        if len(keywords) == 0:
            return []

        # Get all inverted lists for the given set of keywords
        set_of_inverted_lists = []
        for word in keywords:
            if word in self.inverted_lists:
                set_of_inverted_lists.append(self.inverted_lists[word])
            else:
                # if there is no inverted list for one word
                # the whole result would be empty
                return []

        # calculate intersect of inverted lists
        if len(set_of_inverted_lists) == 1:
            return set_of_inverted_lists[0]

        result = set_of_inverted_lists[0]
        i = 1

        while i < len(set_of_inverted_lists):
            result = self.intersect(result, set_of_inverted_lists[i])
            i += 1
        return result



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
    ii.build_from_file(args.file)

    while True:
        query = input('Enter keywords for query seperated by space:')
        keywords = ii.get_keywords(query=query)
        result = ii.process_query(keywords=keywords)

        for i in range(min(len(result), 3)):
            title, desc = ii.doc_with_record[result[i]]

            # highlighting the query word
            for word in keywords:
                title = title.replace(word, f"\033[31m{word}\033[0m")
                desc = desc.replace(word, f"\033[31m{word}\033[0m")

            print(f'{result[i]} - {title} - {desc}')


if __name__ == "__main__":
    main(parse_args())
