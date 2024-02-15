
# Ranking and Evaluation

This exercise is about Database design and practicing simple SQL queries with SQLITE3

### Schema:
1. movies (id, title, release_year, imdb_score)
2. persons (id, name)
3. movies_persons (id, movie_id, person_id, role)
4. awards (id, name)
5. winners (id, award_id, person_id)

# TEST
```
cat movies.sql queries.sql | sqlite3
```
