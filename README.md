## Don't Mess with Mister-in-Between: Improved Negative Search for Knowledge Graph Completion
Source code for our EACL 2023 Paper "Don't Mess with Mister-in-Between: Improved Negative Search for Knowledge Graph Completion".

## Setup

- pip -r install requirements.txt

## Hardware Requirements
- ~220G CPU RAM
- 4 40G A100 GPUs

#### Datasets
To download WN18RR, FB15k237, Wikidata5M datasets, please follow the instructions in [SimKGC](https://github.com/intfloat/SimKGC). DBPedia500 can be downloaded from [here](https://drive.google.com/drive/folders/1YBKw4nOnbscpDeTD_gWxfcpHRFG3MY20).
## Training & Evaluation
We provide instructions for training and evaluation on WN18RR and results for other datasets can be replicated similarly. Please refer to the [scripts](./scripts) for more details.
### WN18RR

#### Preprocess
- Put the WN18RR dataset under [here](./data/WN18RR).
- ```bash scripts/preprocess.sh WN18RR``` 


#### Training

1. Train a Vanilla Dual Encoder \
  ```OUTPUT_DIR=./checkpoint/wn18rr/ bash scripts/train_wn.sh```
2. Generate Hard Negatives \
  ```bash scripts/hard_negative_gen.sh ./checkpoint/wn18rr/model_best.mdl WN18RR``` For bm25 negatives, we use [pyserini](https://github.com/castorini/pyserini). Please refer to [transform_bm25.py](./transform_bm25.py) for more details.
3. Train a Final Model by using a specific type of hard negatives \
  ```OUTPUT_DIR=./checkpoint/wn18rr_tail_entity_2hop_neighbours_hard_negative_1pos1neg/ bash scripts/train_ann_hard_negative_wn.sh```


#### Evaluation

##### Single Model Evaluation
- ```bash scripts/eval.sh ./checkpoint/wn18rr_tail_entity_2hop_neighbours_hard_negative_1pos1neg/model_best.mdl WN18RR```

##### Model Ensembling
- ```bash scripts/embedding_fusion.sh WN18RR```
- ```bash scripts/rank_fusion.sh WN18RR```