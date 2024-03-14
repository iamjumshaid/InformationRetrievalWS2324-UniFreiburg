CREATE TABLE wikidata(subject TEXT, predicate TEXT, object TEXT);

.separator '	'
.import wikidata-complex.tsv wikidata
-- wikidata-properties.tsv provides triples with
-- labels and counts for properties
.import wikidata-properties.tsv wikidata

CREATE INDEX wikidata_subj ON wikidata(subject);
CREATE INDEX wikidata_pred ON wikidata(predicate);
CREATE INDEX wikidata_obj ON wikidata(object);
ANALYZE wikidata;
