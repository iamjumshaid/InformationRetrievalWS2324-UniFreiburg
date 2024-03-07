from typing import Callable

from table import Table, Value, Row


def project(
    table: Table,
    columns: list[int],
    distinct: bool = False
) -> Table:
    """

    Selects the given columns by index from the table.
    If distinct is true, makes sure to return unique rows.

    >>> t = Table.build_from_file("persons.example.tsv")
    >>> project(t, [1, 2]) # doctest: +NORMALIZE_WHITESPACE
    table: persons.example
    name  | age
    -----------
    John  | 29
    Mary  | 18
    Peter | 38
    Jane  | ?
    Mark  | 38
    Lisa  | 20
    >>> project(t, [2]) # doctest: +NORMALIZE_WHITESPACE
    table: persons.example
    age
    ---
    29
    18
    38
    ?
    38
    20
    >>> project(t, [2], distinct=True) # doctest: +NORMALIZE_WHITESPACE
    table: persons.example
    age
    ---
    29
    18
    38
    ?
    20
    """
    assert len(columns) > 0, "zero columns given"
    assert all(0 <= col < len(table.columns) for col in columns), \
        "at least one column out of range"

    rows = []
    seen = set()
    for row in table.rows:
        sub_row = tuple(row[col] for col in columns)
        if not distinct:
            rows.append(sub_row)
        elif sub_row not in seen:
            seen.add(sub_row)
            rows.append(sub_row)

    return Table(
        table.name,
        [table.columns[col] for col in columns],
        rows
    )


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
    return Table(
        table.name,
        table.columns,
        [
            row
            for row in table.rows
            if predicate(row)
        ]
    )


def join(
    table: Table,
    other: Table,
    column: int,
    other_column: int,
    join_type: str = "inner"
) -> Table:
    """

    Joins the two tables on the given column indices using
    a hash-based equi-join.

    >>> p = Table.build_from_file("persons.example.tsv")
    >>> p.name = "persons" # overwrite name of the table for brevity
    >>> j = Table.build_from_file("jobs.example.tsv")
    >>> j.name = "jobs" # overwrite name of the table for brevity
    >>> X = join(p, j, 3, 0)
    >>> X.rows = sorted(X.rows) # sort for doctest
    >>> X # doctest: +NORMALIZE_WHITESPACE
    table: persons X jobs
    id | name  | age | job_id | id | job_title
    --------------------------------------------------
    0  | John  | 29  | 0      | 0  | manager
    1  | Mary  | 18  | 2      | 2  | software engineer
    2  | Peter | 38  | 1      | 1  | secretary
    3  | Jane  | ?   | 1      | 1  | secretary
    4  | Mark  | 38  | 0      | 0  | manager
    >>> X = join(p, j, 3, 0, "left_outer")
    >>> X.rows = sorted(X.rows) # sort for doctest
    >>> X # doctest: +NORMALIZE_WHITESPACE
    table: persons X jobs
    id | name  | age | job_id | id | job_title
    --------------------------------------------------
    0  | John  | 29  | 0      | 0  | manager
    1  | Mary  | 18  | 2      | 2  | software engineer
    2  | Peter | 38  | 1      | 1  | secretary
    3  | Jane  | ?   | 1      | 1  | secretary
    4  | Mark  | 38  | 0      | 0  | manager
    5  | Lisa  | 20  | 5      | ?  | ?
    >>> X = join(p, j, 3, 0, "right_outer")
    >>> # sort for doctest
    ... X.rows = sorted(tuple(str(c) for c in r) for r in X.rows)
    >>> X # doctest: +NORMALIZE_WHITESPACE
    table: persons X jobs
    id   | name  | age  | job_id | id | job_title
    -----------------------------------------------------
    0    | John  | 29   | 0      | 0  | manager
    1    | Mary  | 18   | 2      | 2  | software engineer
    2    | Peter | 38   | 1      | 1  | secretary
    3    | Jane  | None | 1      | 1  | secretary
    4    | Mark  | 38   | 0      | 0  | manager
    None | None  | None | None   | 3  | ceo
    """
    assert (
        0 <= column < len(table.columns)
        and 0 <= other_column < len(other.columns)
    ),  "at least one column out of range"
    assert join_type in {"inner", "left_outer", "right_outer"}, \
        "unknown join type"

    # optimization: decide
    # which table is hashed and which is iterated over,
    # in general it is faster to hash the smaller table
    # and iterate over the large one; this can only be
    # done for inner joins, for outer joins the order
    # of the tables matter
    other_rows, rows = other.rows, table.rows
    num_other_columns = len(other.columns)
    is_inner = join_type == "inner"
    reversed = (
        (is_inner and len(other_rows) > len(rows))
        or join_type == "right_outer"
    )
    if reversed:
        rows, other_rows = other_rows, rows
        column, other_column = other_column, column
        num_other_columns = len(table.columns)

    other_hashed: dict[str, list[int]] = {}
    for i, row in enumerate(other_rows):
        val = row[other_column]
        if val is None:
            continue

        if val not in other_hashed:
            other_hashed[val] = [i]
        else:
            other_hashed[val].append(i)

    joined_rows = []
    for row in rows:
        val = row[column]
        not_found = val not in other_hashed
        if is_inner and not_found:
            continue
        elif not_found:
            nulls = tuple(None for _ in range(num_other_columns))
            joined_rows.append(
                nulls + row if reversed else
                row + nulls
            )
            continue

        assert val is not None, "should not happen"
        for row_idx in other_hashed[val]:
            other_row = other_rows[row_idx]
            joined_rows.append(
                other_row + row if reversed else
                row + other_row
            )

    return Table(
        f"{table.name} X {other.name}",
        table.columns + other.columns,
        joined_rows
    )


def order_by(
    table: Table,
    column: int,
    ascending: bool = True
) -> Table:
    """

    Order the table by the given column in the given order.
    None values should come first when ascending is true,
    last if ascending is false.

    >>> t = Table.build_from_file("persons.example.tsv")
    >>> order_by(t, 2) # doctest: +NORMALIZE_WHITESPACE
    table: persons.example
    id | name  | age | job_id
    -------------------------
    3  | Jane  | ?   | 1
    1  | Mary  | 18  | 2
    5  | Lisa  | 20  | 5
    0  | John  | 29  | 0
    2  | Peter | 38  | 1
    4  | Mark  | 38  | 0
    >>> order_by(t, 2, ascending=False) # doctest: +NORMALIZE_WHITESPACE
    table: persons.example
    id | name  | age | job_id
    -------------------------
    2  | Peter | 38  | 1
    4  | Mark  | 38  | 0
    0  | John  | 29  | 0
    5  | Lisa  | 20  | 5
    1  | Mary  | 18  | 2
    3  | Jane  | ?   | 1
    """
    assert 0 <= column < len(table.columns), \
        "column out of range"
    return Table(
        table.name,
        table.columns,
        sorted(
            table.rows,
            key=lambda row: row[column] or "",
            reverse=not ascending
        )
    )


def limit(
    table: Table,
    limit: int
) -> Table:
    """

    Limits the number of rows of the table to the given limit.

    >>> t = Table.build_from_file("persons.example.tsv")
    >>> limit(t, 2) # doctest: +NORMALIZE_WHITESPACE
    table: persons.example
    id | name | age | job_id
    ------------------------
    0  | John | 29  | 0
    1  | Mary | 18  | 2
    """
    assert limit > 0, "limit must be positive"
    return Table(table.name, table.columns, table.rows[:limit])


AggregationFn = Callable[[list[Value]], Value]


def group_by(
    table: Table,
    columns: list[int],
    aggregations: list[tuple[int, AggregationFn]]
) -> Table:
    """

    Groups the table by the given columns and aggregates
    the columns left by the given aggregation functions.
    The aggregations should be given as a list of tuples
    of (column, aggregation_function).

    >>> from math import prod
    >>> p = Table.build_from_file("persons.example.tsv")
    >>> g = group_by(
    ...     p,
    ...     [2],
    ...     [(1, lambda names: ", ".join(names)),
    ...      (0, lambda ids: str(prod(int(id) for id in ids))),
    ...      (3, lambda ids: str(sum(int(id) for id in ids)))]
    ... ) # doctest: +NORMALIZE_WHITESPACE
    >>> # sort for doctest
    ... g.rows = sorted(tuple(str(c) for c in r) for r in g.rows)
    >>> g # doctest: +NORMALIZE_WHITESPACE
    table: persons.example
    age  | name        | id | job_id
    --------------------------------
    18   | Mary        | 1  | 2
    20   | Lisa        | 5  | 5
    29   | John        | 0  | 0
    38   | Peter, Mark | 8  | 1
    None | Jane        | 3  | 1
    """
    assert len(aggregations) > 0 and len(columns) > 0, \
        "zero columns or aggregations given"
    assert all(0 <= col < len(table.columns) for col in columns), \
        "at least one column out of range"
    column_set = set(columns)
    assert len(column_set) == len(columns), \
        "columns to group by are not unique"
    agg_set = set(col for col, _ in aggregations)
    assert all(0 <= col < len(table.columns) for col in agg_set), \
        "at least one column to aggregate out of range"
    assert column_set.isdisjoint(agg_set), \
        "none of the columns to aggregate should be in the columns grouped by"

    # column names and row for group by table
    new_columns = [table.columns[col] for col in columns]
    for column, func in aggregations:
        new_columns.append(table.columns[column])
    new_rows = []

    # building a new hash map for group_by column
    # @@NOTE: alternate solution: store x dict value as
    # dict[intcolumn_index: list_values]
    x: dict[tuple[str], list[str]] = {}
    for row in table.rows:
        group = tuple([row[col] for col in columns])

        # add group as hash key
        if group not in x:
            x[group] = []

        # add values to each group
        values = [row[column] for column, func in aggregations]
        x[group].append(values)

    # Now applying aggregate functions on the hashmap
    for group in x.keys():
        # first append group values
        new_values = [x for x in group]
        for index, aggregate in enumerate(aggregations):
            # apply aggegate functions on values
            values = [value[index] for value in x[group]]
            new_values.append(aggregate[1](values))
        # create the row
        new_rows.append(tuple(new_values))

    return Table(
        table.name,
        new_columns,
        new_rows
    )
