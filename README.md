This is the code for our paper: Distance-aware Learning for Inductive Link Prediction on Temporal Networks.

## Requirements

- Python 3.8
- PyTorch 1.8.1
- Cuda 11.1

## Quick Start
**Firstly**, download the data of MathOverflow, AskUbuntu and StackOverflow, and put them in the floder named "data" outside DEAL-Revise. 

**Secondly**, preprocess the data:
* MathOverflow dataset
```bash
python ./datasets/preprocess.py --data math
```
* AskUbuntu dataset
```bash
python ./datasets/preprocess.py --data ask
```
* StackOverflow dataset
```bash
python ./datasets/preprocess.py --data stack
```

**Then**, pretrain the node embeddings:
* MathOverflow dataset
```bash
python ./Pretrain/main.py --data math
```
* AskUbuntu dataset
```bash
python ./Pretrain/main.py --data ask
```
* StackOverflow dataset
```bash
python ./Pretrain/main.py --data stack
```

**Finally**, train DEAL to make predictions:
* MathOverflow dataset
```bash
python ./DEAL/main.py --data math
```
* AskUbuntu dataset
```bash
python ./DEAL/main.py --data ask
```
* StackOverflow dataset
```bash
python ./DEAL/main.py --data stack
```
