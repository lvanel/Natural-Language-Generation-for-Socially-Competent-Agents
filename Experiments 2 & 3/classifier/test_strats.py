from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import pandas as pd
from tqdm import tqdm

#model_name = "/home/lvanel/condgen/classifier/camembert-large/checkpoint-770"
#model_name = "/home/lvanel/condgen/classifier/flaubert_large/checkpoint-1510"
model_name = '/home/lvanel/condgen/classifier/camembert-base/checkpoint-770'
tokenizer = AutoTokenizer.from_pretrained(model_name) 
model = AutoModelForSequenceClassification.from_pretrained(model_name)

test = pd.read_csv('/home/lvanel/condgen/data/labels_test.csv', encoding='UTF-8')
texts = list(test['input'])

labels = [label for label in test.columns if label not in ['input', 'gt']]
id2label = {idx:label for idx, label in enumerate(labels)}

annots =  []

for i, row in test.iterrows():
    label = []
    for col in labels:
        label.append(row[col])
    
    annots.append(label.index(1))

preds = []

for text in tqdm(texts):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512) 
    outputs = model(**inputs)

    prediction = torch.nn.functional.softmax(outputs.logits, dim=-1) 
    predicted_emotion = prediction.argmax().item()
    preds.append(predicted_emotion)

targs = set(list(set([id2label[x] for x in annots])) + list(set([id2label[x] for x in preds])))

from sklearn.metrics import classification_report
print(classification_report(annots, preds, target_names = targs))