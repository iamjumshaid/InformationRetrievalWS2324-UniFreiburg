import os

Value = str | None
Row = tuple[Value, ...]


class Table:
    def __init__(
        self,
        name: str,
        columns: list[str],
        rows: list[Row]
    ) -> None:
        self.name = name
        self.columns = columns
        self.rows = rows
        # whether to print the full value of a column
        # or truncate it to 32 characters, internal use only
        self.verbose = False

    @staticmethod
    def build_from_file(file_name: str) -> "Table":
        """

        Reads a table from a file, splitting each line into
        columns by tabs.
        The first line of the table should contain the column names.

        >>> Table.build_from_file("persons.example.tsv") \
        # doctest: +NORMALIZE_WHITESPACE
        table: persons.example
        id | name  | age | job_id
        -------------------------
        0  | John  | 29  | 0
        1  | Mary  | 18  | 2
        2  | Peter | 38  | 1
        3  | Jane  | ?   | 1
        4  | Mark  | 38  | 0
        5  | Lisa  | 20  | 5
        >>> Table.build_from_file("jobs.example.tsv") \
        # doctest: +NORMALIZE_WHITESPACE
        table: jobs.example
        id | job_title
        ----------------------
        0  | manager
        1  | secretary
        2  | software engineer
        3  | ceo
        """
        name, _ = os.path.splitext(os.path.basename(file_name))
        with open(file_name, "r", encoding="utf8") as inf:
            columns = next(inf).strip().split("\t")
            rows = []
            for line in inf:
                row = tuple(
                    None if val == "" else val for val in
                    (val.strip() for val in line.split("\t"))
                )
                rows.append(row)

        # some checks for non-empty and unique column names
        assert all(col != "" for col in columns), \
            "column names cannot be empty"
        assert len(set(columns)) == len(columns), \
            "column names must be unique"
        assert all(len(row) == len(columns) for row in rows), \
            f"expected all rows to contain {len(columns)} columns"

        return Table(
            name,
            columns,
            rows
        )

    @property
    def shape(self) -> tuple[int, int]:
        """

        Returns the number of rows and columns
        of the table as a tuple.

        >>> t = Table.build_from_file("persons.example.tsv")
        >>> t.shape
        (6, 4)
        >>> t = Table.build_from_file("jobs.example.tsv")
        >>> t.shape
        (4, 2)
        """
        return len(self.rows), len(self.columns)

    def __repr__(self) -> str:
        """

        Allows printing of tables in a reasonably pretty
        format.

        """
        str_rows = [self.columns] + [
            tuple("?" if val is None else str(val) for val in row)
            for row in self.rows
        ]
        max_col_lengths = [
            max(
                min(len(row[i]) if self.verbose else 32, len(row[i]))
                for row in str_rows
            )
            for i in range(len(self.columns))
        ]
        str_rows = [
            " | ".join(
                col.ljust(ml)
                if len(col) <= ml or self.verbose
                else col[:29] + "..."
                for col, ml in zip(row, max_col_lengths)
            )
            for row in str_rows
        ]
        return (
            f"table: {self.name}\n"
            + str_rows[0]
            + "\n"
            + "-" * len(str_rows[0])
            + "\n"
            + "\n".join(str_rows[1:])
        )
