"""
Copyright 2019, University of Freiburg
Chair of Algorithms and Data Structures.
Claudius Korzen <korzen@cs.uni-freiburg.de>
Patrick Brosi <brosi@cs.uni-freiburg.de>
Sebastian Walter <swalter@cs.uni-freiburg.de>
"""

import argparse

from inverted_index import InvertedIndex  # NOQA


class Evaluate:
    """
    Class for evaluating the InvertedIndex class against a benchmark.
    """

    def read_benchmark(self, file_name: str) -> dict[str, set[int]]:
        """
        Read a benchmark from the given file. The expected format of the file
        is one query per line, with the ids of all documents relevant for that
        query, like: <query>TAB<id1>WHITESPACE<id2>WHITESPACE<id3> ...

        >>> evaluate = Evaluate()
        >>> benchmark = evaluate.read_benchmark("example-benchmark.tsv")
        >>> sorted(benchmark.items())
        [('animated film', {1, 3, 4}), ('short film', {3, 4})]
        """
        result = {}
        with open(file_name, 'r') as file:
            for line in file:
                query, ids = line.strip().split('\t', 1)
                # convert string of ids into set of ints
                result[query] = set(map(int, ids.split()))
        return result


    def evaluate(
        self,
        ii: InvertedIndex,
        benchmark: dict[str, set[int]],
        use_refinements: bool = False
    ) -> tuple[float, float, float]:
        """
        Evaluate the given inverted index against the given benchmark as
        follows. Process each query in the benchmark with the given inverted
        index and compare the result list with the groundtruth in the
        benchmark. For each query, compute the measure P@3, P@R and AP as
        explained in the lecture. Aggregate the values to the three mean
        measures MP@3, MP@R and MAP and return them.

        Implement a parameter 'use_refinements' that controls the use of
        ranking refinements on calling the method process_query of your
        inverted index.

        >>> ii = InvertedIndex()
        >>> ii.build_from_file("example.tsv", b=0.75, k=1.75)
        >>> evaluator = Evaluate()
        >>> benchmark = evaluator.read_benchmark("example-benchmark.tsv")
        >>> measures = evaluator.evaluate(ii, benchmark, use_refinements=False)
        >>> [round(measure, 3) for measure in measures]
        [0.667, 0.833, 0.694]
        """
        mp_at_3 = 0.0
        mp_at_r = 0.0
        map = 0.0
        num_of_queries = max(1, len(benchmark))

        for query, relevant_ids in benchmark.items():
            keywords = ii.get_keywords(query)
            result_ids = [r[0] for r in ii.process_query(keywords)]

            mp_at_3 += self.precision_at_k(result_ids, relevant_ids, k=3)

            map += self.average_precision(result_ids, relevant_ids)

            mp_at_r += self.precision_at_k(result_ids, relevant_ids, len(relevant_ids))

        return (mp_at_3 / num_of_queries, mp_at_r / num_of_queries, map / num_of_queries)

    def precision_at_k(
        self,
        result_ids: list[int],
        relevant_ids: set[int],
        k: int
    ) -> float:
        """
        Compute the measure P@k for the given list of result ids as it was
        returned by the inverted index for a single query, and the given set of
        relevant document ids.

        >>> evaluator = Evaluate()
        >>> evaluator.precision_at_k([5, 3, 6, 1, 2], {1, 2, 5, 6, 7, 8}, k=0)
        0.0
        >>> evaluator.precision_at_k([5, 3, 6, 1, 2], {1, 2, 5, 6, 7, 8}, k=4)
        0.75
        >>> evaluator.precision_at_k([5, 3, 6, 1, 2], {1, 2, 5, 6, 7, 8}, k=8)
        0.5
        >>> evaluator.precision_at_k([], {1, 2, 5, 6, 7, 8}, k=5)
        0.0
        >>> evaluator.precision_at_k([3, 4], {1, 2, 5, 6, 7, 8}, k=5)
        0.0
        >>> evaluator.precision_at_k([3, 4], {3, 4}, k=2)
        1.0
        """
        # edge case
        if k == 0:
            return 0.0
        result_ids = result_ids[:k]
        intersection = [x for x in result_ids if x in relevant_ids]
        return len(intersection) / k


    def average_precision(
        self,
        result_ids: list[int],
        relevant_ids: set[int]
    ) -> float:
        """
        Compute the average precision (AP) for the given list of result ids as
        it was returned by the inverted index for a single query, and the given
        set of relevant document ids.

        >>> evaluator = Evaluate()
        >>> evaluator.average_precision([7, 17, 9, 42, 5], {5, 7, 12, 42})
        0.525
        >>> evaluator.average_precision([], {5, 7, 12, 42})
        0.0
        >>> evaluator.average_precision([1, 3], {5, 7, 12, 42})
        0.0
        >>> evaluator.average_precision([5, 7], {5, 7, 12, 42})
        0.5
        """
        # edge case
        if len(result_ids) == 0:
            return 0.0

        # creating sorted list of positions of the relevant documents in the result list
        r_list = [index + 1 for index, result in enumerate(result_ids) if result in relevant_ids]

        # calculating average precision
        total_sum = 0.0
        for r in r_list:
            # finding how many relvant doc at position r
            no_of_rel_doc = len([i for i in result_ids[:r] if i in relevant_ids])
            p_at_r = no_of_rel_doc / r
            total_sum += p_at_r
        ap = total_sum / len(relevant_ids)
        return ap

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
        "benchmark",
        type=str,
        help="the benchmark file to use for evaluation",
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
    Constructs an inverted index from the given dataset and evaluates the
    inverted index against the given benchmark.
    """
    # create a new inverted index from the given file
    print(f"Reading from file {args.file}")
    ii = InvertedIndex()
    ii.build_from_file(args.file, args.b_param, args.k_param)

    evaluator = Evaluate()
    benchmark = evaluator.read_benchmark(args.benchmark)
    measures = evaluator.evaluate(ii, benchmark, use_refinements=False)
    measures = [round(measure, 3) for measure in measures]
    print(f'MP@3: {measures[0]}, MP@R: {measures[1]}, MAP: {measures[2]}')


if __name__ == "__main__":
    main(parse_args())
