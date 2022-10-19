# Finetune DPR Retriever on AmbigQA

## Setup and Download Data
First, download necessary resources using the `scripts/download_data.py` script with additional keys (line 391) for ambigqa data:
```bash
# Get the version of wikipedia used by NQ DPR training 
python scripts/download_data.py --resource data.wikipedia_split.psgs_w100

# Get the AmbigQA data (use light bc we don't need all annotations)
python scripts/download_data.py --resource data.ambigqa_light 

# Get the checkpoint trained on NQ to finetune from
python scripts/download_data.py --resource checkpoint.retriever.single.nq.bert-base-encoder
```

Then, setup pyserini to do the BM25 retrieval to get the positive and hard negative contexts:
```bash
# Following: https://github.com/castorini/pyserini/blob/master/docs/installation.md#development-installation
conda install -c conda-forge openjdk=11
conda install -c conda-forge pytorch faiss-cpu
conda install -c conda-forge maven

git clone https://github.com/castorini/pyserini.git --recurse-submodules

cd pyserini/
cd tools/eval && tar xvfz trec_eval.9.0.4.tar.gz
cd trec_eval.9.0.4 && make && cd ../../..
cd tools/eval/ndeval && make && cd ../../..

pip install -e .
python -m spacy download en_core_web_sm

# Then install answerini: https://github.com/castorini/anserini
git clone https://github.com/castorini/anserini.git --recurse-submodules
cd anserini
mvn clean package appassembler:assemble
# note a few tests failed, lets ignore for now
mvn clean package appassembler:assemble -Dmaven.test.skip=true

cd tools/eval && tar xvfz trec_eval.9.0.4.tar.gz
cd trec_eval.9.0.4 && make && cd ../../..
cd tools/eval/ndeval && make && cd ../../..

# Finally, finish pyserini setup
cd ../
cp anserini/target/anserini-0.15.0-SNAPSHOT-fatjar.jar pyserini/pyserini/resources/jars/
cd pyserini

# And test the untittests, all but 4 passed for me
python -m unittest
```

And install DPR:
```bash
git clone git@github.com:facebookresearch/DPR.git

# Update setup.py, find packages with:
#     packages=find_packages(exclude=["conf"]),
# and add find_packages import from setuptools

pip install -e .
```

## Make changes to underlying repos to make this work

For the DPR repo, if you use transformers<4.0.0 you'll need rust to install tokenizers<0.9.4.  Avoid this by using a new version of transformers, the reqs file for DPR includes transformers>= 4.3.0.  However, then you'll need to make a small update to `dpr/models/biencoder.py` to ignore the `position_ids` field which newer hf BERT models have but that aren't in the checkpoints:

```python
# Changes made to line 245, load_state()
# Remove: self.load_state_dict(saved_state.model_dict, strict=strict)
# Add:
        # Longterm HF compatibility fix:
        #   The actual BertModel ignores the "position_ids" key but since
        #   we're just directly loading the state into the module  then
        #   we're not hitting check so do it manually
        acceptable_missing = [
            'question_model.embeddings.position_ids',
            'ctx_model.embeddings.position_ids',
        ]
        missing, unexpected = self.load_state_dict(saved_state.model_dict, strict=False)
        if len(missing) > 0:
            print("Keys in local model but not in checkpoint:", missing)
            extra_missing = set(missing) - set(acceptable_missing)
            if len(extra_missing) > 0:
                raise Exception(f"Unacceptable missing keys: {extra_missing}")
        if len(unexpected) > 0:
            raise Exception(f"Keys in checkpoint not in local model: {unexpected}")
```

For the pyserini repo, for AmbigQA, make the following changes to add a query iterator to enable reading the slightly different dataset format:
```python
+++ b/pyserini/query_iterator.py
@@ -30,6 +30,7 @@ from urllib.error import HTTPError, URLError
 class TopicsFormat(Enum):
     DEFAULT = 'default'
     KILT = 'kilt'
+    QUESTION = 'question'


 class QueryIterator(ABC):
@@ -101,6 +102,24 @@ class DefaultQueryIterator(QueryIterator):
         order = QueryIterator.get_predefined_order(topics_path)
         return cls(topics, order)

+class QuestionQueryIterator(QueryIterator):
+    def get_query(self, id_):
+        return self.topics[id_].get("question")
+
+    @classmethod
+    def from_topics(cls, topics_path: str):
+        if os.path.exists(topics_path) and topics_path.endswith('.json'):
+            with open(topics_path, 'r') as f:
+                all_data = json.load(f)
+        else:
+            raise NotImplementedError(f"Not sure how to parse {topics_path}. Please specify the file extension.")
+
+        topics = {d["id"]: d for d in all_data}
+        order = [d["id"] for d in all_data]
+
+        if not topics:
+            raise FileNotFoundError(f'Topic {topics_path} Not Found')
+        return cls(topics, order)

 class KiltQueryIterator(QueryIterator):

@@ -152,5 +171,6 @@ def get_query_iterator(topics_path: str, topics_format: TopicsFormat):
     mapping = {
         TopicsFormat.DEFAULT: DefaultQueryIterator,
         TopicsFormat.KILT: KiltQueryIterator,
+        TopicsFormat.QUESTION: QuestionQueryIterator,
     }
     return mapping[topics_format].from_topics(topics_path)
```

## Preprocessing the Data

The AmbigQA is a subset of NaturalQuestions but with multiple annotations per question.  For training we'll take the question and answer from the dataset, as well as using the positive contexts from other questions as the negative contexts (which happens during loss calc).  However, for positive and hard negative contexts we need to create them ourselves.

We'll do this by querying the wikipedia index using BM25.  The positive contexts will be the first context which contains the correct answer in the top 100 returned when querying the index with the question.  The hard negative will be the top context retrieved that doesn't contain *any* of the correct answers out of the first 100.

Creating this dataset happens in two parts: 

(1) Do the query for 100 hits for each element in the dataset
```bash
# Repeat for all splits
SPLIT="dev"; HITS="100";
python -m pyserini.search.lucene \
  --index wikipedia-dpr \
  --topics /scratch/ddr8143/repos/DPR/downloads/data/ambigqa_light/${SPLIT}.json \
  --topics-format question \
  --hits ${HITS} \
  --batch-size 1000 \
  --threads 10 \
  --output runs/bm25.ambigqa_light.${SPLIT}.h${HITS}.trec
```

(2) Postprocess the results into a dataset
```bash
SPLIT="dev"; HITS="1000";
python -m pyserini.eval.convert_multiqa_trec_run_to_dpr_retrieval \
  --index wikipedia-dpr \
  --topics /scratch/ddr8143/repos/DPR/downloads/data/ambigqa_light/${SPLIT}.json \
  --input runs/bm25.ambigqa_light.${SPLIT}.h${HITS}.trec \
  --output runs/bm25.ambigqa_light.${SPLIT}.h${HITS}.json \
  --drop-no-pos
```

## Run training

Its important to use DDP instead of DataParallel if you want in batch negatives to work:
```bash
cd DPR/

# Note that modle_file is the checkpoint to initialize from
BS=48; WS=2; python -m torch.distributed.launch --nproc_per_node=${WS} train_dense_encoder.py \
  train=biencoder_nq \
  train_datasets=[/scratch/ddr8143/repos/pyserini/runs/bm25.ambigqa_light.train.h100.json] \
  dev_datasets=[/scratch/ddr8143/repos/pyserini/runs/bm25.ambigqa_light.dev.h100.json] \
  train=biencoder_nq \
  train.batch_size=48 \
  output_dir=/scratch/ddr8143/multiqa/baseline_runs_v0/ \
  checkpoint_file_name=ambigqa_bm25_100.from_nq.bs_${BS}.ws_${WS} \
  model_file=/scratch/ddr8143/repos/DPR/downloads/checkpoint/retriever/single/nq/bert-base-encoder.cp
```

## Identify best checkpoint

Turns out that the training is setup to switch validation metrics mid-training, so the first part of training uses NLL Validation and the second uses Avg Rank.  Unfortuantely this means that the "best checkpoint" updating doesn't quite work.  Look through each of these metrics manually to see what your actual best checkpoint is.

## Encode Wikipedia w/ best checkpoint

Here we will use `DPR/generate_dense_embeddings.py` to convert the wikipedia indx into dense embeddings.  I chose 250 shards, bs256, with each shard running in slightly less than 30mins on a single GPU (but needing 64GB of memory).
```bash
python generate_dense_embeddings.py \
	model_file=/scratch/ddr8143/multiqa/baseline_runs_v0/ambigqa_bm25_100.from_nq.bs_48.ws_4.t_0.s_0/best_checkpoint.8 \
	ctx_src=dpr_wiki \
	shard_id=0 num_shards=250 batch_size=256 \
	out_file=/scratch/ddr8143/multiqa/baseline_runs_v0/ambigqa_bm25_100.from_nq.bs_48.ws_4.t_0.s_0/dpr_wiki_encoded/shard
```

Make sure that the final line is that the file was written, sometimes you OOM if you didn't request enough memory and the file exists but wasn't finished writing.
