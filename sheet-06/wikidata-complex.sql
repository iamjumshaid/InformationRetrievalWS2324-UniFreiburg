CREATE TABLE wikidata(subject TEXT, predicate TEXT, object TEXT);

.separator '	'
.import wikidata-complex.tsv wikidata

CREATE INDEX subject_index ON wikidata(subject);
CREATE INDEX predicate_index ON wikidata(predicate);
CREATE INDEX object_index ON wikidata(object);
ANALYZE wikidata;
