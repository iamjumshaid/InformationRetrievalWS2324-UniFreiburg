-- Output prettier tables and enable timing
.headers on
.timer on
.mode column
.separator '	'

.import movies.tsv movies
.import roles.tsv roles
.import directors.tsv directors
.import persons.tsv persons
.import awards.tsv awards
.import award_names.tsv award_names

-- Comment in the following lines to build id indices for the
-- movies and persons tables.
CREATE INDEX movie_id_index ON movies(movie_id);
CREATE INDEX person_id_index ON persons(person_id);

.print \nActors who won a Golden Globe for a movie with an IMDb score of 8.0 or higher since 2010, including the name of the movie, the role played, and the type of Golden Globe award.\n

.width 23 17 14 70
SELECT m.title, p.name, r.role, an.award_name
-- We assign shorter aliases to the tables here to make the query more readable
FROM movies m, awards a, persons p, roles r, award_names an
-- We use the LIKE predicate to match the award name, where
-- % is used as a wildcard for any number of characters.
WHERE an.award_name LIKE 'Golden Globe Award for Best Act%'
AND a.movie_id = m.movie_id
AND a.person_id = p.person_id
AND a.award_id = an.award_id
AND r.movie_id = m.movie_id
AND r.person_id = p.person_id
-- We use the CAST function to convert the year and IMDb score
-- to the same data type as the values we are comparing to.
-- This is due to all columns in the tables being stored as text, because
-- the corresponding Python code does also only use text.
AND CAST(m.year AS INT) >= 2010
AND CAST(m.imdb_score AS REAL) >= 8.0;
