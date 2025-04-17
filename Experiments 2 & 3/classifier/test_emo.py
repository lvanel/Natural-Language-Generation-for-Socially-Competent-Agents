from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import pandas as pd

#model_name = "astrosbd/french_emotion_camembert" 
model_name = "/home/lvanel/condgen/classifier/emo_class/checkpoint-40"
tokenizer = AutoTokenizer.from_pretrained(model_name) 
model = AutoModelForSequenceClassification.from_pretrained(model_name)
config = model.config

test = pd.read_csv('emotions_ohe_test.csv', encoding='UTF-8')
texts = list(test['input'])

labels = [label for label in test.columns if label not in ['input']]
id2label = {idx:label for idx, label in enumerate(labels)}

annots =  []

for i, row in test.iterrows():
    label = []
    for col in labels:
        label.append(row[col])
    
    annots.append(label.index(1))

preds = []

for text in texts:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512) 
    outputs = model(**inputs)

    prediction = torch.nn.functional.softmax(outputs.logits, dim=-1) 
    predicted_emotion = prediction.argmax().item()
    #preds.append(config.id2label[predicted_emotion])
    preds.append(predicted_emotion)

targs = list(set([config.id2label[x] for x in annots]))
#targs = config.id2label.values()

from sklearn.metrics import classification_report
print(classification_report(annots, preds, target_names = targs))