import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from torch import cuda, device
import torch
import os 
from transformers import AutoTokenizer,AutoModelForSequenceClassification
import numpy as np


# choose model size and task

#LOAD THE RIGHT DATA AND GET THE MODEL NAME, OUTPUT PATH

dataset = load_dataset("csv", data_files={"train":"emotions_ohe_train.csv", "test":"emotions_ohe_test.csv"})

output_path = "emo_class"

model_name = "astrosbd/french_emotion_camembert" 
tokenizer = AutoTokenizer.from_pretrained(model_name) 
model = AutoModelForSequenceClassification.from_pretrained(model_name)
config = model.config

import os
if not os.path.exists(output_path):
    os.makedirs(output_path)
    os.makedirs(output_path + '/results')


os.environ["CUDA_VISIBLE_DEVICES"] = "4"


labels = [label for label in dataset['train'].features.keys() if label not in ['input']]
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}

print(id2label)


def preprocess_data(examples):
  # take a batch of texts
  text = examples["input"]
  # encode them
  encoding = tokenizer(text, padding="max_length", truncation=True, max_length=128)
  # add labels
  labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
  # create numpy array of shape (batch_size, num_labels)
  labels_matrix = np.zeros((len(text), len(labels)))
  # fill numpy array
  for idx, label in enumerate(labels):
    labels_matrix[:, idx] = labels_batch[label]

  encoding["labels"] = labels_matrix.tolist()
  
  return encoding


#PREPROCESS THE DATASETS
encoded_dataset = dataset.map(preprocess_data, batched=True, remove_columns=dataset['train'].column_names)

example = encoded_dataset['train'][0]
print(example.keys())

tokenizer.decode(example['input_ids'])

encoded_dataset.set_format("torch")


#LOAD THE MODEL
#Setting `problem_type` to be "multi_label_classification" makes sure we use the appropriate loss function, BCEWithLogitsLoss
#The output layer has `len(labels)` output neurons, and we set the id2label and label2id mappings.

from transformers import AutoModelForSequenceClassification


model = AutoModelForSequenceClassification.from_pretrained(model_name, 
                                                        problem_type="multi_label_classification", 
                                                        num_labels=len(labels),
                                                        id2label=id2label,
                                                        label2id=label2id)

#check the device
device = torch.device(0)
model = model.to(device)
print(model.device)


#SET THE ARGUMENTS AND HYPER-PARAMETERS

batch_size = 16
metric_name = "f1"

from transformers import TrainingArguments, Trainer

args = TrainingArguments(
    output_path,
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=10,
    logging_steps=20,
    weight_decay=0.01,
    gradient_accumulation_steps = 2,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    save_total_limit = 2,
    #push_to_hub=True,
)


#MULTILABEL METRICS
# while training, we need to define a `compute_metrics` function, that returns a dictionary with the desired metric values


from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import EvalPrediction
import torch
    
def multi_label_metrics(predictions, labels, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1': f1_micro_average,
            'roc_auc': roc_auc,
            'accuracy': accuracy}
    return metrics

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, 
            tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds, 
        labels=p.label_ids)
    return result


#verification

print(encoded_dataset['train'][0]['labels'].type())

print(encoded_dataset['train']['input_ids'][0])

#outputs = model(input_ids=encoded_dataset['train']['input_ids'][0].unsqueeze(0).to(device), labels=encoded_dataset['train'][0]['labels'].unsqueeze(0).to(device))

outputs = model(encoded_dataset['train']['input_ids'][0].unsqueeze(0).to(device))


#Initialise trainer module

trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset = encoded_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

#TRAIN!
trainer.train()

#EVALUATION ON VALIDATION SET

trainer.evaluate()

model = trainer.model


device = torch.device(0)
model = model.to(device)