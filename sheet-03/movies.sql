-- TODO: Create the tables and import the data from the tsv files here.
-- For each file you should first create a table with an appropriate schema, e.g. using
--     CREATE TABLE test(some_id INTEGER PRIMARY KEY, some_text TEXT)
-- and afterwards populate the table with data from your tsv file using
--     .import test.tsv test
-- where test.tsv has two tab-separated columns, the first one containing integers,
-- and the second one containing text.
-- Don't forget to set the correct separator before importing your data.

.separator "\t"

CREATE TABLE movies(id INTEGER PRIMARY KEY, title TEXT, release_year INTEGER, imdb_score REAL);
.import movies.tsv movies

CREATE TABLE persons(id INTEGER PRIMARY KEY, name TEXT);
.import persons.tsv persons

CREATE TABLE movies_persons(id INTEGER PRIMARY KEY, movie_id INTEGER REFERENCES movies(id), person_id INTEGER REFERENCES persons(id), role TEXT);
.import movies_persons.tsv movies_persons

CREATE TABLE awards(id INTEGER PRIMARY KEY, name TEXT);
.import awards.tsv awards

CREATE TABLE winners(movie_id INTEGER REFERENCES movies(movie_id), person_id INTEGER REFERENCES persons(person_id), award_id INTEGER REFERENCES awards(id));
.import winners.tsv winners




