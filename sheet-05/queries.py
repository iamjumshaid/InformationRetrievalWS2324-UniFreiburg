import argparse
from timeit import repeat
from typing import Callable

from table import Table
from operations import (
    join,
    select,
    project,
    group_by,
    order_by,
    limit
)


def timeit(
    f: Callable[[dict[str, Table]], Table],
    tables: dict[str, Table],
    n: int
) -> tuple[Table, float]:
    """

    Runs function f on tables n times and returns the
    result table and average runtime in milliseconds.

    """
    assert n > 0, "n must be greater than 0"
    # we replaced our own timeit implementation with the
    # timeit module from the standard library, which returns
    # more accurate and reliable results
    runtimes = repeat(
        "f(tables)",
        globals=locals(),
        number=n,
        repeat=5
    )
    runtime = sum(runtimes) / len(runtimes)
    # get final table
    table = f(tables)
    return table, 1000.0 * runtime / n


def run_sequence_1(tables: dict[str, Table]) -> Table:
    """

    Runs the first sequence of relational operations on the tables
    for the given query and SQL.

    Query:
    All movies that won an Oscar (in any category)
    and were released in 2000, 2001, 2002, or 2003.

    SQL:
    SELECT DISTINCT m.title
    FROM movies m, awards a, award_names an
    WHERE m.movie_id = a.movie_id
    AND a.award_id = an.award_id
    AND an.award_name LIKE "Academy Award%"
    AND CAST(m.year AS INT) BETWEEN 2000 AND 2003;

    """
    # select movies from year 2000 and 2004
    movies = select(
        tables["movies"],
        lambda row: 2000 <= int(row[2] or 0) <= 2004
    )

    # select academy awards only
    award_names = select(
        tables["award_names"],
        lambda row: row[1].startswith('Academy Award')
    )
    awards = join(
        tables["awards"],
        award_names,
        2, 0
        )

    # select movie titles which won an award
    titles = project(join(
        movies,
        awards,
        0, 0
    ), [1], distinct=True)

    return titles


def calc_cost_1(tables: dict[str, Table]) -> float:
    """

    Calculates the cost of the first sequence of relational operations.
    If you did the cost calculation by hand, just
    return your final cost estimate here.

    """
    # select costs n * k, outputs (n / 2, k)
    # project costs n * k', output (n, k')
    # hash join costs n1 + n2 + min(n1, n2) * (k1 + k2),
    # outputs (min(n1, n2), k1 + k2)

    cost = 0.0
    # select movies from year 2000 and 2004
    nm, km = tables["movies"].shape
    cost += nm * km
    nm, km = nm / 2, km  # output of this result

    # select academy awards only oscars
    no, ko = tables["award_names"].shape
    cost += no * ko
    no, ko = no / 2, ko  # output of this result

    # join all awards with oscar awards
    na, ka = tables["awards"].shape
    cost += na + no + min(na, no) * (ko + ka)
    na, ka = min(na, no), (ko + ka)

    # join movies and awards
    cost += nm + na + min(nm, na) * (km + ka)
    nma, kma = min(nm, na), (km + ka)

    # select movie titles
    cost += nma * 1

    return cost


def run_sequence_2(tables: dict[str, Table]) -> Table:
    """

    Runs the second sequence of relational operations on the tables
    for the given query and SQL.

    Query:
    All movies that won an Oscar (in any category)
    and were released in 2000, 2001, 2002, or 2003.

    SQL:
    SELECT DISTINCT m.title
    FROM movies m, awards a, award_names an
    WHERE m.movie_id = a.movie_id
    AND a.award_id = an.award_id
    AND an.award_name LIKE "Academy Award%"
    AND CAST(m.year AS INT) BETWEEN 2000 AND 2003;

    """
    # select movies from year 2000 and 2004
    movies = select(
        tables["movies"],
        lambda row: 2000 <= int(row[2] or 0) <= 2004
    )

    # awards of this movie
    movies_awards = join(
        movies,
        tables["awards"],
        0, 0
    )

    # select academy awards only
    award_names = select(
        tables["award_names"],
        lambda row: row[1].startswith('Academy Award')
    )

    # getting only academy award
    movies_awards_award_names = join(
        movies_awards,
        award_names,
        movies_awards.shape[1] - 1,
        0
    )

    # extract movie titles
    titles = project(
        movies_awards_award_names,
        [1],
        distinct=True
    )

    return titles


def calc_cost_2(tables: dict[str, Table]) -> float:
    """

    Calculates the cost of the second sequence of relational operations.
    If you did the cost calculation by hand, just
    return your final cost estimate here.

    """
    # select costs n * k, outputs (n / 2, k)
    # project costs n * k', output (n, k')
    # hash join costs n1 + n2 + min(n1, n2) * (k1 + k2),
    # outputs (min(n1, n2), k1 + k2)

    cost = 0.0

    # select movies from year 2000 and 2004
    n, k = tables["movies"].shape
    cost += n * k
    n, k = n/2, k

    # awards of these movie
    n1, k1 = n, k
    n2, k2 = tables["awards"].shape
    cost += n1 + n2 + min(n1, n2) * (k1 + k2)
    nma, kma = min(n1, n2), (k1 + k2)

    # select academy awards only
    n, k = tables["award_names"].shape
    cost += n * k
    n, k = n/2, k

    # getting only academy award (oscars)
    n1, k1 = nma, kma
    n2, k2 = n, k
    cost += n1 + n2 + min(n1, n2) * (k1 + k2)
    no, ko = min(n1, n2), (k1 + k2)

    # extract movie titles
    cost += 1 * no

    return cost


def run_group_by_sequence(tables: dict[str, Table]) -> Table:
    """

    Runs a sequence of relational operations on the tables
    for the given query and SQL.

    Query:
    Top 10 directors by average IMDb score with at least 10 movies,
    considering only movies with at least 100,000 votes.

    SQL:
    SELECT
    p.name,
    COUNT(m.movie_id) AS num_movies,
    ROUND(AVG(m.imdb_score), 2) AS avg_score
    FROM movies m, persons p, directors d
    WHERE m.movie_id = d.movie_id
    AND p.person_id = d.person_id
    AND CAST(m.votes AS INT) >= 100000
    GROUP BY d.person_id
    HAVING num_movies >= 10
    ORDER BY avg_score DESC
    LIMIT 10;

    """
    # select movies with at least 100_000 votes
    movies = select(
        tables["movies"],
        lambda row: int(row[-1] or 0) >= 100000
    )

    # join movies and directors
    movies_directors = join(
        movies,
        tables["directors"],
        0,
        0
    )
    # join movies_directors and persons
    movies_directors_persons = join(
        movies_directors,
        tables["persons"],
        movies_directors.shape[1] - 1,
        0
    )
    # group by person_id
    grouped = group_by(
        movies_directors_persons,
        # group by person id and name
        [
            movies_directors.shape[1],
            movies_directors.shape[1] + 1
        ],
        [
            # num movies
            (
                0,
                lambda values: str(len(values))
            ),
            # average imdb score
            (
                4,
                lambda values: str(round(
                    sum(float(v or 0.0) for v in values)
                    / max(1, len(values)), 2
                ))
            )
        ]
    )
    # select only directors with at least 10 movies
    grouped = select(
        grouped,
        lambda row: int(row[2] or 0) >= 10
    )
    # order by avg_score
    ordered = order_by(
        grouped,
        3,
        ascending=False
    )
    # limit to 10 rows
    limited = limit(
        ordered,
        10
    )
    # get name, num movies and avg_score
    limited = project(
        limited,
        [1, 2, 3]
    )
    # rename columns (just for output purposes)
    limited.columns = ["name", "num_movies", "avg_score"]
    return limited


def run_improved_group_by_sequence(tables: dict[str, Table]) -> Table:
    """

    Runs an improved sequence of relational operations on the tables
    for the given query and SQL.

    Query:
    Top 10 directors by average IMDb score with at least 10 movies,
    considering only movies with at least 100,000 votes.

    SQL:
    SELECT
    p.name,
    COUNT(m.movie_id) AS num_movies,
    ROUND(AVG(m.imdb_score), 2) AS avg_score
    FROM movies m, persons p, directors d
    WHERE m.movie_id = d.movie_id
    AND p.person_id = d.person_id
    AND CAST(m.votes AS INT) >= 100000
    GROUP BY d.person_id
    HAVING num_movies >= 10
    ORDER BY avg_score DESC
    LIMIT 10;

    """
    # select movies with at least 100_000 votes
    # movie_id	title	year	desc	imdb_score	votes
    movies = select(
        tables["movies"],
        lambda row: int(row[-1] or 0) >= 100000
    )

    # get all directors for the movies
    # movie_id	title	year	desc	imdb_score	votes movie_id	person_id
    movies_directors = join(
        movies,
        tables["directors"],
        0, 0
    )

    # group by director_id having more than 10 movies
    # person_id | movie_id | imdb_score
    groups = group_by(
        movies_directors,
        [movies_directors.shape[1] - 1],
        [
            (0, lambda movie_ids: str(len(movie_ids))),  # count of movies
            (movies.shape[1] - 2,  # average of imdb_score
             lambda imdb_scores:
                str(round(
                    sum(float(score or 0.0) for score in imdb_scores)
                    / max(1, len(imdb_scores)), 2)))
         ]
    )

    # select group only having >= 10 movies
    filtered_groups = select(
        groups,
        lambda row: int(row[1] or 0) >= 10
    )

    # order by number of movies
    filtered_groups = order_by(
        filtered_groups,
        filtered_groups.shape[1] - 1,
        ascending=False
    )

    # limit by top 10 records
    # person_id | movie_id | imdb_score
    filtered_groups = limit(
        filtered_groups,
        10
    )

    # Join with persons to extract names of person_id
    # person_id | movie_id | imdb_score | person_id | name | birth_date
    top_10_directors = join(
        filtered_groups,
        tables["persons"],
        0, 0
    )

    # Select only ["name", "num_movies", "avg_score"]
    top_10_directors = project(
        top_10_directors,
        [top_10_directors.shape[1] - 2, 1, 2]
    )

    orderd_top_10_directors = order_by(
        top_10_directors,
        top_10_directors.shape[1] - 1,
        ascending=False
    )

    orderd_top_10_directors.columns = ["name", "num_movies", "avg_score"]
    return orderd_top_10_directors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "tables",
        nargs="+",
        type=str,
        help="paths to the tsv-files that will be read as tables"
    )
    parser.add_argument(
        "-e",
        "--exercise",
        choices=[1, 2],
        type=int,
        required=True,
        help="execute the code for the given exercise"
    )
    parser.add_argument(
        "-n",
        "--n-times",
        type=int,
        default=10,
        help="number of times each sequence will be executed "
        "to measure runtime"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="whether to print the full untruncated table"
    )
    return parser.parse_args()


def check_rows(first: Table, second: Table) -> None:
    assert (
        sorted(tuple(c or "" for c in r) for r in first.rows)
        == sorted(tuple(c or "" for c in r) for r in second.rows)
    ), "rows of the tables must be equal"


def main(args: argparse.Namespace) -> None:
    print("Loading tables from files...")
    tables = {}
    for file in args.tables:
        table = Table.build_from_file(file)
        assert table.name not in tables, \
            f"table with name {table.name} already exists"
        tables[table.name] = table

    if args.exercise == 1:
        cost_1 = calc_cost_1(tables)
        cost_2 = calc_cost_2(tables)

        result_1, runtime_1 = timeit(
            run_sequence_1,
            tables,
            args.n_times
        )
        result_2, runtime_2 = timeit(
            run_sequence_2,
            tables,
            args.n_times
        )

        check_rows(result_1, result_2)
        result_1.verbose = args.verbose
        print(result_1)

        print(f"\nCost of sequence 1: {cost_1:,.1f}")
        print(f"Cost of sequence 2: {cost_2:,.1f}")
        print(f"Cost ratio: {cost_1 / cost_2:.2f}")

        print(f"\nSequence 1 took {runtime_1:,.1f}ms")
        print(f"Sequence 2 took {runtime_2:,.1f}ms")
        print(f"Runtime ratio: {runtime_1 / runtime_2:.2f}")

        return

    result, runtime = timeit(
        run_group_by_sequence,
        tables,
        args.n_times
    )
    result_imp, runtime_imp = timeit(
        run_improved_group_by_sequence,
        tables,
        args.n_times
    )

    check_rows(result, result_imp)
    result.verbose = args.verbose
    print(result)

    print(f"\nSequence took {runtime:,.1f}ms")
    print(f"Improved sequence took {runtime_imp:,.1f}ms")
    print(f"Runtime ratio: {runtime / runtime_imp:.2f}")


if __name__ == "__main__":
    main(parse_args())
