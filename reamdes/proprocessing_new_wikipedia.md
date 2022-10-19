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

```
pip install wikiextractor

python -m wikiextractor.WikiExtractor --templates enwiki_20220701_templates.out --json /scratch/ddr8143/wikipedia/enwiki-20220701-pages-articles-multistream.xml.bz2
```
