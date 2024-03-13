"""
Copyright 2019, University of Freiburg
Chair of Algorithms and Data Structures.
Hannah Bast <bast@cs.uni-freiburg.de>
Claudius Korzen <korzen@cs.uni-freiburg.de>
Patrick Brosi <brosi@cs.uni-freiburg.de>
Natalie Prange <prange@cs.uni-freiburg.de>
Sebastian Walter <swalter@cs.uni-freiburg.de>
"""

import readline  # noqa
import re
import argparse
import time

try:
    # try to import the ad_freiburg_qgram_utils package,
    # which contains faster Rust-based implementations of the ped and
    # merge_lists functions; install it via pip install ad-freiburg-qgram-utils
    from ad_freiburg_qgram_utils import (  # type: ignore
        ped,  # type: ignore
        sort_merge as merge_lists,  # type: ignore
    )
except ImportError:
    # fallback to the Python implementation in utils.py
    # if the ad_freiburg_qgram_utils is not installed
    from utils import ped, merge_lists


class QGramIndex:
    """

    A QGram-Index.

    """

    def __init__(self, q: int, use_syns: bool = False):
        """

        Creates an empty qgram index.

        """
        assert q > 0, "q must be greater than zero"
        self.q = q
        self.padding = "$" * (self.q - 1)
        # map from q-gram to list of (ID, frequency) tuples
        self.inverted_lists: dict[str, list[tuple[int, int]]] = {}
        self.word_records: list[tuple[str, int, list[str]]] = []

        # statistics for doctests and output, calculated in
        # find_matches

        # tuple of (
        #   number of actual PED computations,
        #   number of potential PED computations / length of merged list
        # )
        self.ped_stats = (0, 0)
        self.ped_time = 0.0

        # @NOTE: copied from master solution
        # tuple of (
        #   number of inverted lists merged,
        #   total number of items in inverted lists merged
        # )
        self.merge_stats = (0, 0)
         # time spent merging inverted lists
        self.merge_time = 0.0

    def build_from_file(self, file_name: str) -> None:
        """

        Builds the index from the given file.

        The file should contain one entity per line, in the following format:
            name\tscore\tsynonyms\tinfo1\tinfo2\t...

        Synonyms are separated by a semicolon.

        An example line:
            Albert Einstein\t275\tEinstein;A. Einstein\tGerman physicist\t..."

        The entity IDs are one-based (starting with one).

        >>> q = QGramIndex(3, False)
        >>> q.build_from_file("test.tsv")
        >>> sorted(q.inverted_lists.items())
        ... # doctest: +NORMALIZE_WHITESPACE
        [('$$b', [(2, 1)]), ('$$f', [(1, 1)]), ('$br', [(2, 1)]),
         ('$fr', [(1, 1)]), ('bre', [(2, 1)]), ('fre', [(1, 1)]),
         ('rei', [(1, 1), (2, 1)])]
        """
        with open(file_name, "r", encoding="utf8") as file:
            next(file)  # skipping first line column names
            word_id = 0
            for line in file:
                word_id += 1
                word, score, syn, *other_info = line.rstrip('\n').split('\t')
                self.word_records.append((word, int(score), other_info))
                word = self.normalize(word)  # normalise word
                # building the inverted list
                for qgram in self.compute_qgrams(word):
                    if qgram not in self.inverted_lists:
                        self.inverted_lists[qgram] = []
                    # the qgram is repeated in the word
                    if (
                        len(self.inverted_lists[qgram]) > 0
                        and self.inverted_lists[qgram][-1][0] == word_id
                    ):
                        current_frequency = self.inverted_lists[qgram][-1][1]
                        self.inverted_lists[qgram][-1] = (
                            word_id,
                            current_frequency + 1)
                    else:
                        self.inverted_lists[qgram].append((word_id, 1))

    def compute_qgrams(self, word: str) -> list[str]:
        """

        Compute q-grams for padded version of given string.

        >>> q = QGramIndex(3, False)
        >>> q.compute_qgrams("freiburg")
        ['$$f', '$fr', 'fre', 'rei', 'eib', 'ibu', 'bur', 'urg']
        >>> q.compute_qgrams("f")
        ['$$f']
        >>> q.compute_qgrams("")
        []
        """
        # add padding to the word
        padded_word = self.padding + word
        qgrams: list[str] = []
        for i in range(0, len(word)):
            qgrams.append(padded_word[i:i + self.q])
        return qgrams

    def find_matches(
        self,
        prefix: str,
        delta: int
    ) -> list[tuple[int, int]]:
        """

        Finds all entities y with PED(x, y) <= delta for a given integer delta
        and a given prefix x. The prefix should be normalized and non-empty.
        You can assume that the threshold for filtering PED computations
        (defined below) is greater than zero. That way, it suffices to only
        consider names which have at least one q-gram in common with prefix.

        Returns a list of (ID, PED) tuples ordered first by PED and then entity
        score. The IDs are one-based (starting with 1).

        Also calculates statistics about the PED computations and saves them in
        the attribute ped_stats.

        >>> q = QGramIndex(3, False)
        >>> q.build_from_file("test.tsv")
        >>> q.find_matches("frei", 0)
        [(1, 0)]
        >>> q.ped_stats
        (1, 2)
        >>> q.find_matches("frei", 1)
        [(1, 0), (2, 1)]
        >>> q.ped_stats
        (2, 2)
        >>> q.find_matches("freib", 1)
        [(1, 1)]
        >>> q.ped_stats
        (1, 2)
        """
        assert len(prefix) > 0, "prefix must not be empty"
        threshold = len(prefix) - (self.q * delta)
        assert threshold > 0, \
            "threshold must be positive, adjust delta accordingly"

        # Slide 26-27 Lecture 07
        # threshold is basically |Q2(x)| - (q * delta)
        # this holds the condition that we compute
        # PED of x and y only if y has at least threshold
        # number of q-grams in-common with x.

        # 1. computer the set of q-grams for prefix
        # 2. fetch inverted_list of each qgram
        q_prefix = []
        for qgram in self.compute_qgrams(self.normalize(prefix)):
            if qgram in self.inverted_lists:
                q_prefix.append(self.inverted_lists[qgram])

        start = time.perf_counter()
        # 3. computing union of the inverted list
        # this merged list will return
        # how many q_gram of prefix are present
        # in each document [(1, 4), (2, 1)]
        # doc 01 has 4 qgrams of prefix
        merged_q_prefix = merge_lists(q_prefix)

        # @NOTE: copied from master solution
        self.merge_stats = (len(q_prefix), sum(len(ls) for ls in q_prefix))
        self.merge_time = (time.perf_counter() - start) * 1000

        # now we calculate edit distance only
        # if | Q(x) intersection Q(y) | >= threshold
        # if a word appears k times that means it has
        # k q-grams common with prefix

        start = time.perf_counter()
        # computing edit distance of threshold passing words
        matched_words: list[tuple[int, int]] = []
        count = 0
        for word_id_y, common_qgrams_of_y in merged_q_prefix:
            if common_qgrams_of_y >= threshold:
                # fetch the word
                # -1 because we store id with 1
                y = self.word_records[word_id_y - 1][0]
                distance = ped(prefix, y, delta)
                count += 1
                if distance <= delta:
                    matched_words.append((word_id_y, distance))

        # ped_stats [0] ped caclulated actually
        # ped_stats [1] # of times ped could have been calculated
        # if threshold wasn't checked
        self.ped_stats = (count, len(merged_q_prefix))

        self.ped_time = (time.perf_counter() - start) * 1000

        # sort matched record first with asc ped
        # then with desc score
        matched_words = sorted(
            matched_words,
            key=lambda matched_word: (
                matched_word[1],
                -int(self.word_records[matched_word[0] - 1][1])
            )
        )

        return matched_words

    def get_infos(
        self,
        id: int
    ) -> tuple[str, str, int, list[str]] | None:
        """

        Returns the synonym, name, score and additional info for the given ID.
        If the index was built without synonyms, the synonym is always
        equal to the name. Returns None if the ID is invalid.

        >>> q = QGramIndex(3, False)
        >>> q.build_from_file("test.tsv")
        >>> q.get_infos(1)
        ('frei', 'frei', 3, ['first entity', 'used for doctests'])
        >>> q.get_infos(2)
        ('brei', 'brei', 2, ['second entity', 'also for doctests'])
        >>> q.get_infos(3)
        """
        if id < 1 or id > len(self.word_records):
            return None
        record_id = id - 1

        return (
            self.word_records[record_id][0], # name
            self.word_records[record_id][0], # setting synonym to name of entity
            self.word_records[record_id][1], # score
            self.word_records[record_id][2], # other_info
        )

    def normalize(self, word: str) -> str:
        """

        Normalize the given string (remove non-word characters and lower case).

        >>> q = QGramIndex(3, False)
        >>> q.normalize("freiburg")
        'freiburg'
        >>> q.normalize("Frei, burG !?!")
        'freiburg'
        >>> q.normalize("Frei14burg")
        'frei14burg'
        """
        return "".join(m.group(0).lower() for m in re.finditer(r"\w+", word))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "file",
        type=str,
        help="file to build q-gram index from"
    )
    parser.add_argument(
        "-q",
        "--q-grams",
        type=int,
        default=3,
        help="size of the q-grams"
    )
    parser.add_argument(
        "-s",
        "--use-synonyms",
        action="store_true",
        help="whether to use synonyms for the q-gram index"
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """

    Builds a qgram index from the given file and then, in an infinite loop,
    lets the user type a query and shows the result for the normalized query.

    """
    q_gram = QGramIndex(args.q_grams, False)
    start = time.perf_counter()
    q_gram.build_from_file(args.file)
    print(f"\nTime taken: {(time.perf_counter() - start) * 1000:.1f}ms.")

    while(True):
        query = q_gram.normalize(input("\nEnter query:"))
        print("Normalised Query:", query)

        start = time.perf_counter()

        # This delta ensures that a match has at-least
        # one qgram in common with query
        delta = len(query) // (args.q_grams + 1)

        matches = q_gram.find_matches(query, delta)

        # @Note: copied from master solution
        merged_lists, merged_items = q_gram.merge_stats
        ped_actual, ped_total = q_gram.ped_stats
        print(
            f"Got {len(matches)} result(s), merged {merged_lists} lists with "
            f"tot. {merged_items} elements ({q_gram.merge_time:.1f}ms), "
            f"{ped_actual}/{ped_total} ped computations "
            f"({q_gram.ped_time:.1f}ms), took "
            f"\033[1m{(time.perf_counter() - start) * 1000:.1f} ms"
            f"\033[0m total."
        )

        for word_id, ped in matches[:5]:
            infos = q_gram.get_infos(word_id)
            assert infos is not None, "invalid ID"
            syn, name, score, info = infos
            print(
                f"\n\033[1m{name}\033[0m (score={score}, ped={ped}, "
                f"qid={info[0]}, via '{syn}'):\n{info[1]}"
            )

if __name__ == "__main__":
    main(parse_args())
