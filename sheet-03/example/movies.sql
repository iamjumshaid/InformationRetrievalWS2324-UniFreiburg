.separator "\t"
CREATE TABLE movies(id TEXT PRIMARY KEY, title TEXT, year INTEGER, score REAL);
CREATE TABLE persons(id TEXT PRIMARY KEY, name TEXT, birth_date DATE);
CREATE TABLE roles(movie_id TEXT REFERENCES movies(id), person_id TEXT REFERENCES persons(id));
.import movies.tsv movies
.import persons.tsv persons
.import roles.tsv roles
SELECT * FROM movies, persons, roles;
