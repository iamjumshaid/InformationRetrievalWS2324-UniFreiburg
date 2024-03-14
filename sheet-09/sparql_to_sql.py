"""
Copyright 2023, University of Freiburg,
Chair of Algorithms and Data Structures.
Authors:
Patrick Brosi <brosi@cs.uni-freiburg.de>,
Claudius Korzen <korzen@cs.uni-freiburg.de>,
Natalie Prange <prange@cs.uni-freiburg.de>,
Sebastian Walter <swalter@cs.uni-freiburg.de>
"""

import re
import sqlite3


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

        # plan the SQL query
        var_map: dict[str, list[str]] = {}
        tables = []
        wheres = []
        for i, (subj, pred, obj) in enumerate(triples):
            tables.append(f"t{i}")

            # process the subject
            if subj[0] == "?":
                if subj not in var_map:
                    var_map[subj] = []
                var_map[subj].append(f"{tables[-1]}.subject")
            else:
                wheres.append(f"{tables[-1]}.subject=\"{subj}\"")

            # process the predicate
            if pred[0] == "?":
                if pred not in var_map:
                    var_map[pred] = []
                var_map[pred].append(f"{tables[-1]}.predicate")
            else:
                wheres.append(f"{tables[-1]}.predicate=\"{pred}\"")

            # process the object
            if obj[0] == "?":
                if obj not in var_map:
                    var_map[obj] = []
                var_map[obj].append(f"{tables[-1]}.object")
            else:
                wheres.append(f"{tables[-1]}.object=\"{obj}\"")

        # build the elements of the WHERE clause
        for var in var_map:
            var_list = var_map[var]
            for i in range(len(var_list) - 1):
                wheres.append(f"{var_list[-1]}={var_list[i]}")

        # compose the SQL query
        select_vars = [var_map[var][0] for var in variables if var in var_map]
        sql = f"SELECT {', '.join(select_vars)}"
        sql += f" FROM {', '.join([f'wikidata as {t}' for t in tables])}"
        if len(wheres) > 0:
            sql += f" WHERE {' AND '.join(sorted(wheres))}"
        if order_by is not None:
            var, asc = order_by
            sql += f" ORDER BY {var_map[var][0]} {'ASC' if asc else 'DESC'}"
        if limit is not None:
            sql += f" LIMIT {limit}"
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
        connection = sqlite3.connect(db_name)
        cursor = connection.cursor()
        cursor.execute(sql)
        return cursor.fetchall()
