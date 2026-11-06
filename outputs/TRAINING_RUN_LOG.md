# Training run log

- Dataset: real anonymized ULB/Worldline card transactions
- Split: chronological by published `Time`
- Sample rows: 150,000 train; 50,000 validation; 50,000 test
- Selected model: logistic regression
- Test PR-AUC: 0.7432
- Test precision / recall / F1: 0.8958 / 0.6515 / 0.7544
- Command: `python training_model/train.py`
