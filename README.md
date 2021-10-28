## Is Everything in Order? A Simple Way to Order Sentences

This repo contains code for the EMNLP 2021 paper:

**Is Everything in Order? A Simple Way to Order Sentences**

*Somnath Basu Roy Chowdhury\*, Faeze Brahman\*, Snigdha Chaturvedi* EMNLP 2021

[Link to paper](https://arxiv.org/pdf/2104.07064.pdf)

### Pre-requisities

Please create a fresh conda env and run:

```
pip install -r requirements.txt
```

### Datasets

First, create the dataset splits and put them in `./data` folder.

Please find the links for the various datasets: [arXiv](https://drive.google.com/drive/folders/0B-mnK8kniGAiNVB6WTQ4bmdyamc), [Wiki Movie Plots](https://www.kaggle.com/jrobischon/wikipedia-movie-plots), [SIND](http://visionandlanguage.net/VIST/dataset.html), [NSF](https://archive.ics.uci.edu/ml/datasets/NSF+Research+Award+Abstracts+1990-2003), [ROCStories](https://www.cs.rochester.edu/nlp/rocstories/), [NeurIPS](https://www.kaggle.com/benhamner/nips-papers), [AAN](https://github.com/EagleW/ACL_titles_abstracts_dataset).

All datsets should be formatted in jsonl files where each line is a json containing two fields: `orig_sents`, and `shuf_sents`. `orig_sents` is a list of markers [y1, y2, ..., yN], which denotes the position of ith sentence of the corresponding ordered sequence in the shuffled input (`shuf_sents`). An example is provided for ROCStories in [here](https://drive.google.com/drive/folders/1bY7CvXF1q2kgpmtXWtD0NT3bFRfLHpV1?usp=sharing).

### Train the ReBART model:

To train the ReBART model run the following command:

```
bash train_rebart.sh
```
You can specify the hyper-parameters inside the bash script.

### Generate

To generate the outputs (position markers) using the trained model, run the following commands:

```
export DATA_DIR="data/arxiv-abs"
export MODEL_PATH="outputs/reorder_exp/bart-large_arxiv"
python source/generate.py --in_file $DATA_DIR/test.jsonl --out_file $MODEL_PATH/test_bart_greedy.jsonl --model_name_or_path $MODEL_PATH --beams 1 --max_length 40 --task index_with_sep --device 0
```

### Evaluate

To evaluate the model and get the performance metrics, run:

```
python eval/evaluation.py --output_path $MODEL_PATH/test_bart_greedy.jsonl
```


### Citation

If you used our work please cite us using:

```
@inproceedings{Basu-brahman-chaturvedi-rebart,
    title = "Is Everything in Order? A Simple Way to Order Sentences",
    author = "Somnath Basu Roy Chowdhury, Faeze Brahman and
      Snigdha Chaturvedi",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2021",
    publisher = "Association for Computational Linguistics",
}
```

