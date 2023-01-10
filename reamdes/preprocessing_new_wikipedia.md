# Preprocessing a New Wikipedia


## Download the wikipedia

Right now the only location I know of that you can easily download from is [the wikipedia dump](https://dumps.wikimedia.org/enwiki/20220701/enwiki-20220701-pages-articles-multistream.xml.bz2).

Download this with:
```bash
# In: /scratch/ddr8143/wikipedia
wget https://dumps.wikimedia.org/enwiki/20220701/enwiki-20220701-pages-articles-multistream.xml.bz2
```

## Extract wikipedia

Then extract it with wikiextractor (this took ~1.5 hours with 47 processes):

```bash
pip install wikiextractor

python -m wikiextractor.WikiExtractor --templates enwiki_20220701_templates.out --json /scratch/ddr8143/wikipedia/enwiki-20220701-pages-articles-multistream.xml.bz2
```

## Postprocess the wikidata dump into full page index files

See `notebooks/process_new_wikipedia.ipynb` for example of how to use `wikipedia_utils.postprocess_wikipedia_to_page_index` to convert the wikidata dump into a postprocessed sequence of files in a single directory that can be used to create a pyserini index:

```python
wu.postprocess_wikipedia_to_page_index(
    input_wikipath="/scratch/ddr8143/wikipedia/wiki_20220701/raw_extracted",
    output_dir="/scratch/ddr8143/wikipedia/wiki_20220701/fullpage_preindex",
    verbose=True,
    force=False,
)
```

## Then Create an Index

Finally we can use pyserini to create the index:
```bash
# Note that this uses a different wikipedia dump then described above
python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input /scratch/ddr8143/wikipedia/qampari_wikipedia/postprocessed \
  --index /scratch/ddr8143/multiqa/indexes/full_page_qampari_wikidata_index \
  --generator DefaultLuceneDocumentGenerator \
  --threads 8 \
  --storePositions --storeDocvectors --storeRaw
```

And there are plenty of things we can do after this point (TBD in the future).
