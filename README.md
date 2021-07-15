# pytorch_example

Example code for [PyTorch Tutorial](https://docs.google.com/presentation/d/1JuVQPAM9JiEmP7iVFcCj6svneoip-_PNHnO0ygGfv0g/edit?usp=sharing).


## Requirements

```bash
pip install -r requirements.txt
```


## Task

Binary sentence classification: classify a sentence into {0:subjective, 1:objective}.  
Data is downloaded from: https://www.cs.cornell.edu/people/pabo/movie-review-data/


## Model

BiLSTM + Attention.


## Sample commands

```bash
# Run on CPU
python -u main.py

# Run on GPU (e.g., id=0)
CUDA_VISIBLE_DEVICES=0 python -u main.py

# Run in the background to keep the process running evan after SSH is disconnected
CUDA_VISIBLE_DEVICES=0 nohup python -u main.py > sample.log &

# Train and save a model
CUDA_VISIBLE_DEVICES=0 python -u main.py --save --model_path sample_model

# Test on the saved model
CUDA_VISIBLE_DEVICES=0 python -u main.py --run_test --model_path sample_model
```
