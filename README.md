# pytorch_example

Pytorch example for NLP.


## Task

Binary sentence classification: classify a sentence into {0:subjective, 1:objective}.  
Data is downloaded from: https://www.cs.cornell.edu/people/pabo/movie-review-data/


## Requirements

```
pip install -r requirements.txt

# Download FastText embeddings to your optional directory
wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip
unzip crawl-300d-2M-subword.zip
```
* See local notes for the pytorch version specification


## Model

FastText + BiLSTM + Attention.


## Sample commands

```
# Run on GPU (id=0)
python -u main.py --device 0 --emb_path PATH_TO_FASTTEXT_EMB

# Run in background to keep the process after SSH is disconnected
nohup python -u main.py --device 0 --emb_path PATH_TO_FASTTEXT_EMB > sample_log &

# Run and save a model
python -u main.py --device 0 --emb_path PATH_TO_FASTTEXT_EMB --save --model_path sample_model

# Run test on the saved model
python -u main.py --device 0 --emb_path PATH_TO_FASTTEXT_EMB --run_test --model_path sample_model
```


## Local notes

* Large files like FastText embeddings are recommended to store under `/cl/work/YOUR_DIRECTORY`
* Currently, CUDA driver on elm* is 10.1
    * If you installed torch with CUDA >= 10.2, reinstall torch that is compatible with CUDA < 10.2
    * See https://pytorch.org for CUDA specification
