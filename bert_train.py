import os
import numpy as np
import pandas as pd

X = pd.read_csv('train-balanced-sarcasm.csv')

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, BertTokenizer
from datasets import Dataset
import accelerate
import torch
import re
import evaluate
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split


X.dropna(subset=['comment'], inplace=True)
X['label'].value_counts()
def preprocessing(s):
    s = str(s).lower().strip()
    s = re.sub('\n', '', s)
    s = re.sub(r"([?!,\":;\(\)])", r" \1 ", s)
    s = re.sub('[ ]{2,}', ' ', s).strip()
    return s

SEED = 1
X_train, X_valid = train_test_split(X,random_state=SEED)

tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny')

def tokenize_function(data):
    return tokenizer(data["comment"], padding="max_length", truncation=True, max_length=100)

X_train_ds = Dataset.from_pandas(X_train[['comment','label']])
X_valid_ds = Dataset.from_pandas(X_valid[['comment','label']])

tokenized_train = X_train_ds.map(tokenize_function, batched=True)
tokenized_valid = X_valid_ds.map(tokenize_function, batched=True)

small_train_dataset = tokenized_train.shuffle(seed=SEED).select(range(100000))
small_eval_dataset = tokenized_valid.shuffle(seed=SEED).select(range(100000))

small_train_dataset.set_format("torch")
small_eval_dataset.set_format("torch")

model = AutoModelForSequenceClassification.from_pretrained('prajjwal1/bert-tiny', num_labels=2)

metric = evaluate.load("accuracy")

def compute_metrics(p):
  pred, labels = p
  pred = np.argmax(pred, axis=1)

  accuracy = accuracy_score(y_true=labels, y_pred=pred)
  recall = recall_score(y_true=labels, y_pred=pred)
  precision = precision_score(y_true=labels, y_pred=pred)
  f1 = f1_score(y_true=labels, y_pred=pred)

  return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

args = TrainingArguments(
    output_dir="./mymodel3",
    evaluation_strategy="steps",
    eval_steps=50,
    logging_strategy="steps",
    logging_steps=100,
    save_strategy="steps",
    save_steps=200,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="tensorboard"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.evaluate()

tokenizer.save_pretrained("./tokenizer")
trainer.save_model("./model")

