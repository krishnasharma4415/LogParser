import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
from transformers import RobertaTokenizer, RobertaModel
import numpy as np
import torch.nn as nn
import re
from collections import OrderedDict

df = pd.read_csv("dataset.csv")
df = df.head(22000)

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_df(df):
    features = ['log']
    target_labels = ['timestamp', 'LineId', 'Component', 'Content', 'EventId', 'EventTemplate']
    label_encoders = {}
    label_columns = []

    df['log'] = df['log'].apply(preprocess_text)

    for col in target_labels:
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col])
        label_encoders[col] = le
        label_columns.append(col + '_encoded')

    return df, label_columns, label_encoders

df, label_columns, label_encoders = preprocess_df(df)
        
class MultiTaskRobertaModel(nn.Module):
    def __init__(self, model_name, num_labels):
        super(MultiTaskRobertaModel, self).__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifiers = nn.ModuleList([nn.Linear(self.roberta.config.hidden_size, num_label) for num_label in num_labels])
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        pooled_output = self.dropout(sequence_output[:, 0])

        logits = [classifier(pooled_output) for classifier in self.classifiers]

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            losses = [loss_fct(logit, label) for logit, label in zip(logits, labels.T)]
            loss = sum(losses)
            return loss, logits
        return logits

device = torch.device("cpu")
model = MultiTaskRobertaModel('roberta-base', [len(le.classes_) for le in label_encoders.values()])
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()
tokenizer = RobertaTokenizer.from_pretrained('./tokenizer')

def predict_log_details(log_text, model, tokenizer, label_encoders, device):
    model.eval()
    inputs = tokenizer(preprocess_text(log_text), return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs)
    
    predictions = [logit.cpu().numpy().argmax(axis=1)[0] for logit in logits]
    confidence_scores = [torch.softmax(logit, dim=1).max().item() for logit in logits]
    
    interpreted_predictions = OrderedDict([
        ('LineId', label_encoders['LineId'].inverse_transform([predictions[1]])[0]),
        ('timestamp', label_encoders['timestamp'].inverse_transform([predictions[0]])[0]),
        ('Component', label_encoders['Component'].inverse_transform([predictions[2]])[0]),
        ('Content', label_encoders['Content'].inverse_transform([predictions[3]])[0]),
        ('EventId', label_encoders['EventId'].inverse_transform([predictions[4]])[0]),
        ('EventTemplate', label_encoders['EventTemplate'].inverse_transform([predictions[5]])[0])
    ])
    
    result = OrderedDict()
    result['input_log'] = log_text
    result['predictions'] = OrderedDict()
    
    for i, (key, value) in enumerate(interpreted_predictions.items()):
        result['predictions'][key] = {
            'value': value,
            'confidence': f"{confidence_scores[i]:.4f}"
        }
    
    return result

new_log = "32,2016-09-28,04:30:31,Info,CBS,Warning: Unrecognized packageExtended attribute.,E50,Warning: Unrecognized packageExtended attribute."
predicted_details = predict_log_details(new_log, model, tokenizer, label_encoders, device)

def print_predictions(predictions):
    print("Input Log:")
    print(f"  {predictions['input_log']}")
    print("\nPredictions:")
    for key, value in predictions['predictions'].items():
        print(f"  {key}:")
        print(f"    Value: {value['value']}")
        print(f"    Confidence: {value['confidence']}")
        print()

print_predictions(predicted_details)