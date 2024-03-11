"""
Copyright 2023, University of Freiburg,
Chair of Algorithms and Data Structures.
Authors:
Patrick Brosi <brosi@cs.uni-freiburg.de>,
Claudius Korzen <korzen@cs.uni-freiburg.de>,
Natalie Prange <prange@cs.uni-freiburg.de>,
Sebastian Walter <swalter@cs.uni-freiburg.de>
"""

import argparse
import re
import sqlite3
import time


Triple = tuple[str, str, str]


class SPARQL:
    """ A simple SPARQL engine for a SQL backend. """

    def parse_sparql(
        self,
        sparql: str
    ) -> tuple[list[str], list[Triple], tuple[str, bool] | None, int | None]:
        """

        Parses a SPARQL query into its components, a tuple of
        (variables, triples, order by, limit).

        ORDER BY and LIMIT clauses are optional, but LIMIT
        should always come after ORDER BY if both are specified.

        >>> engine = SPARQL()
        >>> engine.parse_sparql(
        ...     "SELECT ?x ?y WHERE {"
        ...     "?x pred_1 some_obj . "
        ...     "?y pred_2 ?z "
        ...     "}"
        ... ) # doctest: +NORMALIZE_WHITESPACE
        (['?x', '?y'], [('?x', 'pred_1', 'some_obj'),
        ('?y', 'pred_2', '?z')], None, None)
        >>> engine.parse_sparql(
        ...     "SELECT ?x ?y WHERE {"
        ...     "?x pred_1 some_obj . "
        ...     "?y pred_2 ?z "
        ...     "} ORDER BY DESC(?x)"
        ... ) # doctest: +NORMALIZE_WHITESPACE
        (['?x', '?y'], [('?x', 'pred_1', 'some_obj'),
        ('?y', 'pred_2', '?z')], ('?x', False), None)
        >>> engine.parse_sparql(
        ...     "SELECT ?x ?y WHERE {"
        ...     "?x pred_1 some_obj . "
        ...     "?y pred_2 ?z "
        ...     "} ORDER BY ASC(?x)"
        ... ) # doctest: +NORMALIZE_WHITESPACE
        (['?x', '?y'], [('?x', 'pred_1', 'some_obj'),
        ('?y', 'pred_2', '?z')], ('?x', True), None)
        >>> engine.parse_sparql(
        ...     "SELECT ?x ?y WHERE {"
        ...     "?x pred_1 some_obj . "
        ...     "?y pred_2 ?z "
        ...     "} ORDER BY ASC(?x) LIMIT 25"
        ... ) # doctest: +NORMALIZE_WHITESPACE
        (['?x', '?y'], [('?x', 'pred_1', 'some_obj'),
        ('?y', 'pred_2', '?z')], ('?x', True), 25)
        """
        # format the SPARQL query into a single line for parsing
        sparql = " ".join(line.strip() for line in sparql.splitlines())

        # transform all letters to lower cases.
        sparqll = sparql.lower()

        # find all variables in the SPARQL between the SELECT and WHERE clause.
        select_start = sparqll.find("select ") + 7
        select_end = sparqll.find(" where", select_start)
        variables = sparql[select_start:select_end].split()

        # find all triples between "WHERE {" and "}"
        where_start = sparqll.find("{", select_end) + 1
        where_end = sparqll.rfind("}", where_start)
        where_text = sparql[where_start:where_end]
        triple_texts = where_text.split(".")
        triples = []
        for triple_text in triple_texts:
            subj, pred, obj = triple_text.strip().split(" ", 2)
            triples.append((subj, pred, obj))

        # find the (optional) ORDER BY clause
        order_by_start = sparqll.find(" order by ", where_end)
        if order_by_start > 0:
            search = sparqll[order_by_start + 10:]
            match = re.search(r"^(asc|desc)\((\?[^\s]+)\)", search)
            assert match is not None, \
                f"could not find order by direction or variable in {search}"
            order_by = (match.group(2).strip(), match.group(1) == "asc")
            assert order_by[0] in variables, \
                f"cannot order by, {order_by[0]} not in variables"
            order_by_end = order_by_start + 10 + len(match.group(0))
        else:
            order_by = None
            order_by_end = where_end

        # find the (optional) LIMIT clause
        limit_start = sparqll.find(" limit ", order_by_end)
        if limit_start > 0:
            limit = int(sparql[limit_start + 7:].split()[0])
        else:
            limit = None

        return variables, triples, order_by, limit

    def sparql_to_sql(self, sparql: str) -> str:
        """

        Translates the given SPARQL query to a corresponding SQL query.

        PLEASE NOTE: there are many ways to express the same SPARQL query in
        SQL. Stick to the implementation advice given in the lecture. Thus, in
        case your formatting, the name of your variables / columns or the
        ordering differs, feel free to adjust the syntax
        (but not the semantics) of the test case.

        The SPARQL query in the test below lists all german politicians whose
        spouses were born in the same birthplace.

        >>> engine = SPARQL()
        >>> engine.sparql_to_sql(
        ...     "SELECT ?x ?y WHERE {"
        ...     "?x occupation politician . "
        ...     "?x country_of_citizenship Germany . "
        ...     "?x spouse ?y . "
        ...     "?x place_of_birth ?z . "
        ...     "?y place_of_birth ?z "
        ...     "}"
        ... ) # doctest: +NORMALIZE_WHITESPACE
        'SELECT t0.subject, \
                t2.object \
         FROM   wikidata as t0, \
                wikidata as t1, \
                wikidata as t2, \
                wikidata as t3, \
                wikidata as t4 \
         WHERE  t0.object="politician" \
                AND t0.predicate="occupation" \
                AND t1.object="Germany" \
                AND t1.predicate="country_of_citizenship" \
                AND t2.predicate="spouse" \
                AND t3.predicate="place_of_birth" \
                AND t3.subject=t0.subject \
                AND t3.subject=t1.subject \
                AND t3.subject=t2.subject \
                AND t4.object=t3.object \
                AND t4.predicate="place_of_birth" \
                AND t4.subject=t2.object;'
        """
        # parse the SPARQL query into its components, might raise an exception
        # if the query is invalid
        variables, triples, order_by, limit = self.parse_sparql(sparql)

        # building from statemnt
        statment_from: list[str] = []

        # variable array for building where statement
        var_arr: dict[str, list[str]] = {}
        other_conditions: list[str] = []

        for index, (subject, predicate, object) in enumerate(triples):
            statment_from.append(f"wikidata as t{index}")
            # check subject, object, predicate for each triple
            if "?" in subject:
                if subject not in var_arr:
                    var_arr[subject] = []
                var_arr[subject].append(f"t{index}.subject")
            else:
                other_conditions.append(f't{index}.subject="{subject}"')

            if "?" in predicate:
                if predicate not in var_arr:
                    var_arr[predicate] = []
                var_arr[predicate].append(f"t{index}.predicate")
            else:
                other_conditions.append(f't{index}.predicate="{predicate}"')

            if "?" in object:
                if object not in var_arr:
                    var_arr[object] = []
                var_arr[object].append(f"t{index}.object")
            else:
                other_conditions.append(f't{index}.object="{object}"')

        # Build SQL statement string
        selected_variables = [var_arr[var][0] for var in variables]
        sql = f"SELECT {', '.join(selected_variables)}"
        sql += f" FROM {', '.join(statment_from)}"

        # building where conditions
        wheres: list[str] = []
        for variable in var_arr:
            values = var_arr[variable]
            for i in range(len(values) - 1):
                wheres.append(f"{values[-1]}={values[i]}")
        sql += f" WHERE {' AND '.join(sorted(other_conditions + wheres))}"

        if order_by is not None:
            sql += f" ORDER BY {var_arr[order_by[0]][0]} "
            sql += " ASC " if order_by[1] else " DESC "

        if limit is not None:
            sql += f" LIMIT {limit} "
        sql += ";"

        return sql

    def process_sql_query(
        self,
        db_name: str,
        sql: str
    ) -> list[tuple[str, ...]]:
        """

        Runs the given SQL query against the given instance of a SQLite3
        database and returns the result rows.

        >>> engine = SPARQL()
        >>> sql = engine.sparql_to_sql(
        ...     "SELECT ?x ?y WHERE {"
        ...     "?x occupation politician . "
        ...     "?x country_of_citizenship Germany . "
        ...     "?x spouse ?y . "
        ...     "?x place_of_birth ?z . "
        ...     "?y place_of_birth ?z "
        ...     "}"
        ... )
        >>> sorted(engine.process_sql_query("example.db", sql))
        ... # doctest: +NORMALIZE_WHITESPACE
        [('Fritz_Kuhn', 'Waltraud_Ulshöfer'), \
         ('Helmut_Schmidt', 'Loki_Schmidt'), \
         ('Karl-Theodor_zu_Guttenberg', 'Stephanie_zu_Guttenberg'), \
         ('Konrad_Adenauer', 'Auguste_Adenauer'), \
         ('Konrad_Adenauer', 'Emma_Adenauer'), \
         ('Konrad_Naumann', 'Vera_Oelschlegel'), \
         ('Waltraud_Ulshöfer', 'Fritz_Kuhn'), \
         ('Wolfgang_Schäuble', 'Ingeborg_Schäuble')]
        """
        db = sqlite3.connect(db_name)
        cursor = db.cursor()
        cursor.execute(sql)
        return cursor.fetchall()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "db",
        type=str,
        help="path to the SQLite3 database file"
    )
    parser.add_argument(
        "query",
        type=str,
        help="path to the file containing the SPARQL query"
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    # create a SPARQL engine
    engine = SPARQL()

    # load SPARQL query from file
    with open(args.query, "r", encoding="utf8") as qf:
        sparql = qf.read().strip()

    print(f"\nSPARQL query:\n\n{sparql}")

    try:
        sql = engine.sparql_to_sql(sparql)
    except Exception:
        print("\nInvalid SPARQL query\n")
        return

    print(f"\nSQL query:\n\n{sql}")

    # run the SQL query against the database
    start = time.perf_counter()
    result = engine.process_sql_query(args.db, sql)
    end = time.perf_counter()

    print("\nResult:\n")
    for row in result:
        print("\t".join(row))

    # copied from master solution
    print(f"\n#Rows: {len(result):,}")
    print(f"Runtime: {1000 * (end - start):.1f}ms\n")

if __name__ == "__main__":
    main(parse_args())
