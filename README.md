# pytorch_example

Pytorch example for NLP.


## Task

Binary sentence classification: classify a sentence into {0:subjective, 1:objective}.  
Data is downloaded from: https://www.cs.cornell.edu/people/pabo/movie-review-data/


## Requirements

```
pip install -r requirements.txt
```


## Model

BiLSTM + Attention.


## Sample commands

```
# Run on GPU (id=0)
python -u main.py --device 0

# Run in the background to keep the process running evan after SSH is disconnected
nohup python -u main.py --device 0 > sample_log &

# Run train and save a model
python -u main.py --device 0 --save --model_path sample_model

# Run test on the saved model
python -u main.py --device 0 --run_test --model_path sample_model
```
