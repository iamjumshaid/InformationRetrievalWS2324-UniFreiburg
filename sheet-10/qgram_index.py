"""
Copyright 2019, University of Freiburg
Chair of Algorithms and Data Structures.
Hannah Bast <bast@cs.uni-freiburg.de>
Claudius Korzen <korzen@cs.uni-freiburg.de>
Patrick Brosi <brosi@cs.uni-freiburg.de>
Natalie Prange <prange@cs.uni-freiburg.de>
Sebastian Walter <swalter@cs.uni-freiburg.de>
"""

import time
import readline  # noqa
import re


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
        self.infos: list[tuple[str, int, list[str]]] = []
        self.names: list[str] = []
        self.norm_names: list[str] = []
        self.syn_to_ent: list[int] = []
        self.use_syns = use_syns

        # statistics for doctests and output, calculated in
        # find_matches

        # tuple of (
        #   number of inverted lists merged,
        #   total number of items in inverted lists merged
        # )
        self.merge_stats = (0, 0)
        # time spent merging inverted lists
        self.merge_time = 0.0

        # tuple of (
        #   number of actual PED computations,
        #   number of potential PED computations / length of merged list
        # )
        self.ped_stats = (0, 0)
        # time spent calculating PEDs
        self.ped_time = 0.0

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
        with open(file_name, "r", encoding="utf8") as f:
            ent_id = 0
            syn_id = 0
            # skip first line
            next(f)

            for line in f:
                ent_id += 1
                line = line.rstrip("\r\n")
                name, score, synonyms, *info = line.split("\t")

                # cache the entity name, score and additional info
                self.infos.append((name, int(score), info))

                # all synonyms
                names = [name]
                if self.use_syns:
                    syns = [
                        s for s in synonyms.split(";")
                        if s != ""
                    ]
                    names.extend(syns)

                for n in names:
                    syn_id += 1
                    normed_name = self.normalize(n)
                    # cache the names, normalized names and entity id
                    self.norm_names.append(normed_name)
                    self.names.append(n)
                    self.syn_to_ent.append(ent_id)

                    for qgram in self.compute_qgrams(normed_name):
                        if qgram not in self.inverted_lists:
                            self.inverted_lists[qgram] = []
                        if (
                            len(self.inverted_lists[qgram]) > 0
                            and
                            self.inverted_lists[qgram][-1][0] == syn_id
                        ):
                            freq = self.inverted_lists[qgram][-1][1] + 1
                            self.inverted_lists[qgram][-1] = (syn_id, freq)
                        else:
                            self.inverted_lists[qgram].append((syn_id, 1))

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
        ret = []
        padded = self.padding + word
        for i in range(0, len(word)):
            ret.append(padded[i:i + self.q])
        return ret

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

        Returns a list of (ID, PED) tuples ordered first by PED and then
        entity score. The IDs are one-based (starting with 1).

        Also calculates statistics about the PED computations
        and saves them in the attribute ped_stats.

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

        lists = []
        for qgram in self.compute_qgrams(prefix):
            if qgram in self.inverted_lists:
                lists.append(self.inverted_lists[qgram])

        start = time.perf_counter()
        merged = merge_lists(lists)
        self.merge_stats = (len(lists), sum(len(ls) for ls in lists))
        self.merge_time = (time.perf_counter() - start) * 1000

        start = time.perf_counter()

        matches = []
        c = 0
        for syn_id, freq in merged:
            if freq < threshold:
                continue
            # ids are 1-based
            pedist = ped(prefix, self.norm_names[syn_id - 1], delta)
            c += 1
            if pedist <= delta:
                matches.append((syn_id, pedist))

        self.ped_stats = (c, len(merged))
        self.ped_time = (time.perf_counter() - start) * 1000

        if self.use_syns:
            # only one result per entity when using synonyms,
            # namely the best PED
            unique: list[tuple[int, int]] = []
            for match in sorted(
                matches,
                key=lambda match: (self.syn_to_ent[match[0] - 1], match[1])
            ):
                if len(unique) == 0:
                    unique.append(match)
                    continue

                ent_id = self.syn_to_ent[match[0] - 1]
                last_syn_id, _ = unique[-1]
                last_ent_id = self.syn_to_ent[last_syn_id - 1]
                if last_ent_id != ent_id:
                    unique.append(match)

            matches = unique

        # sort unique matches by PED and score
        matches = sorted(
            matches,
            key=lambda match: (
                match[1],
                -self.infos[self.syn_to_ent[match[0] - 1] - 1][1]
            )
        )
        return matches

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
        if id < 1 or id > len(self.names):
            return None
        syn = self.names[id - 1]
        ent_id = self.syn_to_ent[id - 1]
        name, score, info = self.infos[ent_id - 1]
        return syn, name, score, info

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
