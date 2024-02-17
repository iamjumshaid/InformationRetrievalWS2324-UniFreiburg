from typing import Callable

from table import Table, Row


def project(
    table: Table,
    columns: list[int],
    distinct: bool = False
) -> Table:
    """

    Selects the given columns by index from the table.
    If distinct is true, makes sure to return unique rows.

    >>> t = Table.build_from_file("persons.example.tsv")
    >>> project(t, [1, 2]) \
    # doctest: +NORMALIZE_WHITESPACE
    table: persons.example
    name  | age
    -----------
    John  | 29
    Mary  | 18
    Peter | 38
    Jane  | ?
    Mark  | 38
    >>> project(t, [2]) \
    # doctest: +NORMALIZE_WHITESPACE
    table: persons.example
    age
    ---
    29
    18
    38
    ?
    38
    >>> project(t, [2], distinct=True) \
    # doctest: +NORMALIZE_WHITESPACE
    table: persons.example
    age
    ---
    29
    18
    38
    ?
    """
    assert len(columns) > 0, "zero columns given"
    assert all(0 <= col < len(table.columns) for col in columns), \
        "at least one column out of range"
    # create new columns from existing columns
    new_columns = [table.columns[col] for col in columns]

    # create new rows from existing rows in the table list[(tuples)]
    new_rows = []
    for row in table.rows:
        new_row = tuple(row[col] for col in columns)
        # for distinct values add row only if it is not duplicated
        if not distinct or (distinct and new_row not in new_rows):
            new_rows.append(new_row)

    return Table(
        table.name,
        columns=new_columns,
        rows=new_rows)

def select(
    table: Table,
    predicate: Callable[[Row], bool]
) -> Table:
    """

    Selects the rows from the table where
    the predicate evaluates to true.

    >>> t = Table.build_from_file("persons.example.tsv")
    >>> select(t, lambda row: row[1] == "John") \
    # doctest: +NORMALIZE_WHITESPACE
    table: persons.example
    id | name | age | job_id
    ------------------------
    0  | John | 29  | 0
    >>> select(t, lambda row: int(row[2] or 0) > 30) \
    # doctest: +NORMALIZE_WHITESPACE
    table: persons.example
    id | name  | age | job_id
    -------------------------
    2  | Peter | 38  | 1
    4  | Mark  | 38  | 0
    """
    # apply the predicate on each row of table and return selected rows
    selected_rows = [row for row in table.rows if predicate(row)]
    return Table(
        table.name,
        columns=table.columns,
        rows=selected_rows)


def join(
    table: Table,
    other: Table,
    column: int,
    other_column: int
) -> Table:
    """

    Joins the two tables on the given column indices using
    a hash-based equi-join.

    >>> p = Table.build_from_file("persons.example.tsv")
    >>> p.name = "persons" # overwrite name of the table for brevity
    >>> j = Table.build_from_file("jobs.example.tsv")
    >>> j.name = "jobs" # overwrite name of the table for brevity
    >>> X = join(p, j, 3, 0)
    >>> # sort by person id to make output deterministic
    >>> X.rows = sorted(X.rows, key=lambda row: row[0])
    >>> X # doctest: +NORMALIZE_WHITESPACE
    table: persons X jobs
    id | name  | age | job_id | id | job_title
    --------------------------------------------------
    0  | John  | 29  | 0      | 0  | manager
    1  | Mary  | 18  | 2      | 2  | software engineer
    2  | Peter | 38  | 1      | 1  | secretary
    3  | Jane  | ?   | 1      | 1  | secretary
    4  | Mark  | 38  | 0      | 0  | manager
    """
    assert (
        0 <= column < len(table.columns)
        and 0 <= other_column < len(other.columns)
    ),  "at least one column out of range"

    # building hash map from first table
    # @@NOTE: missing optimisation = Use smaller table for hashing
    x: dict[int, list[int]] = {}
    for index, row in enumerate(table.rows):
        hash_key = int(row[column])

        if hash_key is None:
            continue

        if hash_key in x:
            x[hash_key].add(index)
        else:
            x[hash_key] = {index}

    # iterating each row of table y
    # and adding the matching rows
    # from the given columns
    result_rows = []
    for row in other.rows:
        # if matching column not in hashin index skip
        if int(row[other_column]) not in x:
            continue

        # for each matching row in hashing index
        # append the resulted rows
        for row_index in x[int(row[other_column])]:
            result_rows.append(table.rows[row_index] + row)

    return Table(
        name = f'{table.name} X {other.name}',
        columns= table.columns + other.columns,
        rows = result_rows
    )
