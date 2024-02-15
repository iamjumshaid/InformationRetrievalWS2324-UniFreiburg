-- make sqlite3 output prettier tables
.headers on
.mode column

.print \nIn which year was Titanic released and what rating on IMDb does it have?\n

SELECT release_year, imdb_score
FROM movies
WHERE title="Titanic";

.print \nWho directed Fargo?\n

SELECT p.name
FROM movies as m, persons as p, movies_persons as mp
WHERE m.title="Fargo"
  AND mp.role="director"
  AND mp.movie_id = m.id
  AND mp.person_id = p.id;

.print \nWhich actors won Oscars for which roles in which movies and in which categories?\n

SELECT p.name as actor_name, mp.role as role, m.title as movie_name, a.name as category
FROM movies as m, persons as p, movies_persons as mp, awards as a, winners as w
WHERE w.movie_id = m.id
  AND w.person_id = p.id
  AND w.award_id = a.id
  AND mp.role = 'actor'
  AND mp.person_id = w.person_id


-- OPTIONAL: add more questions and SQL queries here
