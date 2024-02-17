import argparse
import time
from typing import Callable

from table import Table
from operations import (
    join,
    select,
    project
)


def timeit(
    f: Callable[[dict[str, Table]], Table],
    tables: dict[str, Table],
    n: int
) -> tuple[Table, int]:
    """

    Runs function f on tables n times and returns the
    result table and average runtime in milliseconds.

    """
    assert n > 0, "n must be greater than 0"
    # init empty table
    table = Table("dummy", [], [])
    start = time.perf_counter()
    for _ in range(n):
        table = f(tables)
    end = time.perf_counter()
    return table, round(1000 * (end - start) / n)


def run_example_sequence(tables: dict[str, Table]) -> Table:
    """

    Runs a sequence of relational operations on the tables
    for the given example query and SQL.

    Query:
    Names and birth dates of all actors in the movie Avatar.

    SQL:
    SELECT p.name, p.birth_date
    FROM movies m, roles r, persons p
    WHERE m.title = "Avatar"
    AND r.movie_id = m.movie_id
    AND r.person_id = p.person_id;

    """
    # get movies with title Avatar
    avatar = select(
        tables["movies"],
        lambda row: row[1] == "Avatar"
    )
    # join avatar movies with roles table on movie_id
    avatar_actors = join(
        avatar,
        tables["roles"],
        0,
        0
    )
    # join avatar actors with persons so we get the name and birth date
    # information
    avatar_actor_infos = join(
        avatar_actors,
        tables["persons"],
        avatar.shape[1] + 1,
        0
    )
    # only select name and birth date columns from final table
    _, num_cols = avatar_actor_infos.shape
    avatar_actor_names = project(
        avatar_actor_infos,
        [num_cols - 2, num_cols - 1]
    )
    return avatar_actor_names


def run_sequence_1(tables: dict[str, Table]) -> Table:
    """

    Runs the first sequence of relational operations on the tables
    for the given query and SQL.

    Query:
    Actors who won a Golden Globe for a movie with an IMDb
    score of 8.0 or higher since 2010, including the name of
    the movie, the role played, and the type of Golden Globe award.

    SQL:
    SELECT m.title, p.name, r.role, an.award_name
    FROM movies m, awards a, persons p, roles r, award_names an
    WHERE an.award_name LIKE "Golden Globe Award for Best Act%"
    AND a.movie_id = m.movie_id
    AND a.person_id = p.person_id
    AND a.award_id = an.award_id
    AND r.movie_id = m.movie_id
    AND r.person_id = p.person_id
    AND CAST(m.year AS INT) >= 2010
    AND CAST(m.imdb_score AS REAL) >= 8.0;

    """
    # TODO: code your first sequence of operations here;
    # see run_example_sequence for an example how that
    # should look like. make sure to add explanatory
    # comments before each operation

    # get movies that have imdb_score >= 8.0 and year >=2010
    movies_8_2010 = project(select(
        tables["movies"],
        lambda row: float(row[4]) >= 8.0 and
        (int(row[2]) if row[2] is not None else 0) >= 2010
    ), [0, 1], distinct=True)

    # all roles of movies_8_2010
    roles_movies_8_2010 = project(join(
        movies_8_2010,
        tables["roles"],
        0,
        0
    ), [0, 1, 3, 4])

    # all names of persons for those roles
    persons_roles_movies_8_2010 = project(join(
        roles_movies_8_2010,
        tables["persons"],
        2,
        0
    ), [0, 1, 2, 3, 5])

    # persons who got best actor awards
    best_actor_award = join(select(
        tables["award_names"],
        lambda row: 'Golden Globe Award for Best Act' in row[1]),
        tables["awards"],
        0,
        2)

    return project(join(join(persons_roles_movies_8_2010, best_actor_award, 0, 2),
                        best_actor_award, 2, 3),
                   [1, 4, 3, 6], distinct=True)


def run_sequence_2(tables: dict[str, Table]) -> Table:
    """

    Runs the second sequence of relational operations on the tables
    for the given query and SQL.

    Query:
    Actors who won a Golden Globe for a movie with an IMDb
    score of 8.0 or higher since 2010, including the name of
    the movie, the role played, and the type of Golden Globe award.

    SQL:
    SELECT m.title, p.name, r.role, an.award_name
    FROM movies m, awards a, persons p, roles r, award_names an
    WHERE an.award_name LIKE "Golden Globe Award for Best Act%"
    AND a.movie_id = m.movie_id
    AND a.person_id = p.person_id
    AND a.award_id = an.award_id
    AND r.movie_id = m.movie_id
    AND r.person_id = p.person_id
    AND CAST(m.year AS INT) >= 2010
    AND CAST(m.imdb_score AS REAL) >= 8.0;

    """

    # TODO: code your second sequence of operations here;
    # see run_example_sequence for an example how that
    # should look like. make sure to add explanatory
    # comments before each operation

    # @NOTE: The code in this sequence is not optimised nor written well
    # All Golden Globes in one table:
    # award_name| movie_id  | person_id | name
    golden_globes_best_act = project(
        join(
            join(
                select(tables["award_names"],
                       lambda row: row[1].startswith('Golden Globe Award for Best Act')),
                tables["awards"],
                0, 2),
            tables["persons"],
            3, 0
        ),
        [1, 2, 5, 6])

    # movies with rate higher 8.0 and year over 2010
    # id, name
    selected_movies = project(select(tables["movies"],
                                     lambda row: (int(row[2]) if row[2] is not None else 0) >= 2010 and (float(row[4]) if row[2] is not None else 0) >= 8.0),
                              [0, 1])
    # award_name | movie_id | person_id | name  | movie_id | title
    irgenwas = join(golden_globes_best_act, selected_movies, 1, 0)
    irgenwas2 = join(irgenwas, tables["roles"], 1, 0)
    irgenwas2 = select(irgenwas2, lambda row: row[2] == row[7])
    return project(
        irgenwas2,
        [5, 3, 8, 0]
        )


def run_sequence_3(tables: dict[str, Table]) -> Table:
    """

    Runs the third sequence of relational operations on the tables
    for the given query and SQL.

    Query:
    Actors who won a Golden Globe for a movie with an IMDb
    score of 8.0 or higher since 2010, including the name of
    the movie, the role played, and the type of Golden Globe award.

    SQL:
    SELECT m.title, p.name, r.role, an.award_name
    FROM movies m, awards a, persons p, roles r, award_names an
    WHERE an.award_name LIKE "Golden Globe Award for Best Act%"
    AND a.movie_id = m.movie_id
    AND a.person_id = p.person_id
    AND a.award_id = an.award_id
    AND r.movie_id = m.movie_id
    AND r.person_id = p.person_id
    AND CAST(m.year AS INT) >= 2010
    AND CAST(m.imdb_score AS REAL) >= 8.0;

    """
    # TODO: code your third sequence of operations here;
    # see run_example_sequence for an example how that
    # should look like. make sure to add explanatory
    # comments before each operation

    # since Golden Globe Award for Best Act has only 5 records,
    # starting with filtering award_names first.
    # award_id, name
    award_names = select(
        tables["award_names"],
        lambda row: row[1].startswith('Golden Globe Award for Best Act')
    )

    # finding awards for category Golden Globe Award for Best Act
    # (award_id, award_name), (moive_id, person_id, award_id)
    award_names_awards = join(
        award_names,
        tables["awards"],
        0,
        tables["awards"].shape[1] - 1
    )


    # finding role for each reward

    # first joining on the basis of movie_id(awards table) and movie_id(roles table)
    # (award_id, award_name), (moive_id, person_id, award_id), (movie_id, person_id, role)
    award_names_awards_roles = join(
        award_names_awards,
        tables["roles"],
        award_names.shape[1],
        0
    )

    # now joining on the basis of person_id(awards table) and person_id(roles table)
    # this gives us all the Golden Globe Award for Best Act won for each movie by each person
    # for a certain role
    # (award_id, award_name), (moive_id, person_id, award_id), (movie_id, person_id, role)
    award_names_awards_roles = select(
        award_names_awards_roles,
        lambda row: row[3] == row[6]
    )

    # finding movies that has rating >= 8.0 and year >=2010
    # id, title, year, desc, imdb_rating
    movies = select(
        tables["movies"],
        lambda row: (float(row[4]) if row[4] is not None else 0) >= 8.0 and (int(row[2]) if row[2] is not None else 0) >= 2010
    )

    # combining movies with award_names_awards_roles
    # (award_id, award_name), (moive_id, person_id, award_id), (movie_id, person_id, role), (movie_id, title, year, desc, imdb_rating)
    award_names_awards_roles_movies = join(
        award_names_awards_roles,
        movies,
        2,
        0
    )

    # finding the person names
    # (award_id, award_name), (moive_id, person_id, award_id), (movie_id, person_id, role), (movie_id, title, year, desc, imdb_rating), (person_id, name, dob)
    award_names_awards_roles_movies_persons = join(
        award_names_awards_roles_movies,
        tables["persons"],
        3,
        0
    )

    return project(
        award_names_awards_roles_movies_persons,
        [9, 14, 7, 1] # [title, person_name, role, award_name]
    )

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
        "--example",
        action="store_true",
        help="execute the example query only"
    )
    parser.add_argument(
        "-n",
        "--n-times",
        type=int,
        default=3,
        help="number of times each sequence will be executed "
        "to measure runtime"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="whether to print the full untruncated table"
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    print("Loading tables from files...")
    tables = {}
    for file in args.tables:
        table = Table.build_from_file(file)
        assert table.name not in tables, \
            f"table with name {table.name} already exists"
        tables[table.name] = table

    if args.example:
        # run example sequence and measure runtime
        result, runtime = timeit(run_example_sequence, tables, args.n_times)
        print(result)
        print(f"Example sequence took {runtime:,}ms")
        return

    # run all three sequences and measure runtimes
    glob = globals()
    runtimes = []
    results = []
    for i in range(3):
        result, runtime = timeit(
            glob[f"run_sequence_{i + 1}"],
            tables,
            args.n_times
        )
        runtimes.append(runtime)
        results.append(result)

    # make sure all three sequences return the same result
    assert all(
        sorted(tuple(c or "" for c in r) for r in result.rows)
        == sorted(tuple(c or "" for c in r) for r in results[0].rows)
        for result in results[1:]
    ), "results of all three sequences must be equal"

    # print result table and runtimes
    results[0].verbose = args.verbose
    print(results[0])
    for i, runtime in enumerate(runtimes):
        print(f"Sequence {i+1} took {runtime:,}ms")


if __name__ == "__main__":
    main(parse_args())
