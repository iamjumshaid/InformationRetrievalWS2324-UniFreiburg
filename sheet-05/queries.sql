.separator '	'

.import movies.tsv movies
.import roles.tsv roles
.import directors.tsv directors
.import persons.tsv persons
.import awards.tsv awards
.import award_names.tsv award_names

-- Create indexes for faster queries
CREATE INDEX movies_id_index ON movies(movie_id);
CREATE INDEX persons_id_index ON persons(person_id);

-- Output prettier tables and enable timing
.headers on
.timer on
.mode column

-- .print \nAll movies that won an Oscar (in any category) and were released in 2000, 2001, 2002, or 2003.\n

-- SELECT DISTINCT m.title
-- FROM movies m, awards a, award_names an
-- WHERE m.movie_id = a.movie_id
-- AND a.award_id = an.award_id
-- AND an.award_name LIKE 'Academy Award%'
-- AND CAST(m.year AS INT) BETWEEN 2000 AND 2003;

-- .print \nTop 10 directors by average IMDb score with at least 10 movies, considering only movies with at least 100,000 votes.\n

-- SELECT
-- p.name,
-- COUNT(m.movie_id) AS num_movies,
-- ROUND(AVG(m.imdb_score), 2) AS avg_score
-- FROM movies m, persons p, directors d
-- WHERE m.movie_id = d.movie_id
-- AND p.person_id = d.person_id
-- AND CAST(m.votes AS INT) >= 100000
-- GROUP BY d.person_id
-- HAVING num_movies >= 10
-- ORDER BY avg_score DESC
-- LIMIT 10;

.print \nWhat are the movies with the highest IMDb score for each decade, considering only movies with at least 500,000 votes?\n

-- Expected output of the SQL query below:
-- movies                                                          score  decade
-- --------------------------------------------------------------  -----  ------
-- Casablanca                                                      8.5    1940
-- 12 Angry Men                                                    9.0    1950
-- The Good, the Bad and the Ugly                                  8.8    1960
-- The Godfather                                                   9.2    1970
-- Star Wars: Episode V – The Empire Strikes Back                  8.7    1980
-- The Shawshank Redemption                                        9.3    1990
-- The Dark Knight; The Lord of the Rings: The Return of the King  9.0    2000
-- Inception                                                       8.8    2010
-- The Kashmir Files                                               8.6    2020

.width 62 5 6
-- TODO: enter your SQL query to answer the question above here;
-- its output should match the table above; the expected widths of the
-- columns are already set for you

SELECT m.title as movies, MAX(m.imdb_score) as score, a.year as decade
FROM movies m,
      (SELECT movie_id, CAST(ROUND(CAST(year as INT) / 10) * 10 as INT) as year
      FROM movies
      WHERE CAST(votes AS INT) >= 500000
      ORDER BY year) a
WHERE m.movie_id = a.movie_id
GROUP BY a.year
ORDER BY a.year

