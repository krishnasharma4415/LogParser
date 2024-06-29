import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaModel, AdamW, get_linear_schedule_with_warmup
from datetime import datetime
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import logging
import torch.nn as nn
from tqdm import tqdm, trange
import re
from sklearn.model_selection import KFold
import seaborn as sns
import matplotlib.pyplot as plt
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

df = pd.read_csv("dataset.csv")
df.info()
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

        logger.info(f"Column: {col}")
        logger.info(f"Unique values: {len(np.unique(df[col + '_encoded']))}")
        logger.info(f"Min: {df[col + '_encoded'].min()}, Max: {df[col + '_encoded'].max()}")
        logger.info("---")

    return df, label_columns, label_encoders

df, label_columns, label_encoders = preprocess_df(df)
class LogDataset(Dataset):
    def __init__(self, logs, labels, tokenizer, max_len):
        self.logs = logs
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.num_labels = [len(np.unique(labels[:, i])) for i in range(labels.shape[1])]

    def __len__(self):
        return len(self.logs)

    def __getitem__(self, item):
        log = str(self.logs[item])
        labels = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            log,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'log_text': log,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.long)
        }
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

print("Unique label values:", np.unique(df[label_columns].values))
print("Max label value:", np.max(df[label_columns].values))
print("Min label value:", np.min(df[label_columns].values))
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
num_labels = [len(np.unique(df[col])) for col in label_columns]
print("Number of labels for each task:", num_labels)
model = MultiTaskRobertaModel('roberta-base', num_labels)

logs = df['log'].values.tolist()
labels = df[label_columns].values

dataset = LogDataset(logs, labels, tokenizer, max_len=128)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

num_epochs = 10
batch_size = 16
learning_rate = 2e-5
num_warmup_steps = 0
num_folds = 5

kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
best_model = None
best_val_loss = float('inf')

for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    print(f"Fold {fold + 1}/{num_folds}")
    
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
    val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
    
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_subsampler)
    
    model = MultiTaskRobertaModel('roberta-base', num_labels).to(device)
    
    optimizer = AdamW(model.parameters(), lr=learning_rate, correct_bias=False)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps)

    for epoch in trange(num_epochs, desc="Epochs"):
        model.train()
        total_loss = 0
        all_preds = [[] for _ in range(len(label_columns))]
        all_labels = [[] for _ in range(len(label_columns))]
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for batch in progress_bar:
            optimizer.zero_grad()
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
            }
            labels = batch['labels'].to(device)
            loss, logits = model(**inputs, labels=labels)
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            for i, logit in enumerate(logits):
                preds = logit.argmax(dim=-1).detach().cpu().numpy()
                all_preds[i].extend(preds)
                all_labels[i].extend(labels[:, i].cpu().numpy())
            
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(train_loader)
        
        model.eval()
        val_loss = 0
        val_preds = [[] for _ in range(len(label_columns))]
        val_labels = [[] for _ in range(len(label_columns))]
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = {
                    'input_ids': batch['input_ids'].to(device),
                    'attention_mask': batch['attention_mask'].to(device),
                }
                labels = batch['labels'].to(device)
                loss, logits = model(**inputs, labels=labels)
                val_loss += loss.item()
                
                for i, logit in enumerate(logits):
                    preds = logit.argmax(dim=-1).detach().cpu().numpy()
                    val_preds[i].extend(preds)
                    val_labels[i].extend(labels[:, i].cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        
        logger.info(f"Fold {fold + 1}, Epoch {epoch + 1}/{num_epochs}")
        logger.info(f"Training Loss: {avg_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
        
        for i, (preds, labels) in enumerate(zip(val_preds, val_labels)):
            precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
            accuracy = accuracy_score(labels, preds)
            
            logger.info(f"Task {i+1} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
            
            cm = confusion_matrix(labels, preds)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d')
            plt.title(f'Confusion Matrix for Task {i+1}')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.savefig(f'confusion_matrix_fold{fold+1}_task{i+1}_epoch{epoch+1}.png')
            plt.close()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model.state_dict()
            torch.save(best_model, './best_model.pth')
            print(f"New best model saved with validation loss: {best_val_loss}")


print(f"Best model had a validation loss of {best_val_loss}")

model.load_state_dict(torch.load('./best_model.pth'))

tokenizer.save_pretrained('./tokenizer')
def predict_log_details(log_text, model, tokenizer, label_encoders, device):
    model.eval()
    inputs = tokenizer(preprocess_text(log_text), return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs)
    
    predictions = [logit.cpu().numpy().argmax(axis=1)[0] for logit in logits]
    confidence_scores = [torch.softmax(logit, dim=1).max().item() for logit in logits]
    
    interpreted_predictions = {
        'LineId': label_encoders['LineId'].inverse_transform([predictions[1]])[0],
        'timestamp': label_encoders['timestamp'].inverse_transform([predictions[0]])[0],
        'Component': label_encoders['Component'].inverse_transform([predictions[2]])[0],
        'Content': label_encoders['Content'].inverse_transform([predictions[3]])[0],
        'EventId': label_encoders['EventId'].inverse_transform([predictions[4]])[0],
        'EventTemplate': label_encoders['EventTemplate'].inverse_transform([predictions[5]])[0]
    }
    
    for key, value in interpreted_predictions.items():
        interpreted_predictions[key] = {
            'prediction': value,
            'confidence': confidence_scores[list(interpreted_predictions.keys()).index(key)]
        }

    return interpreted_predictions

model.load_state_dict(torch.load('./best_model.pth'))

new_log = "32,2016-09-28,04:30:31,Info,CBS,Warning: Unrecognized packageExtended attribute.,E50,Warning: Unrecognized packageExtended attribute."
predicted_details = predict_log_details(new_log, model, tokenizer, label_encoders, device)
print(predicted_details)
def batch_predict_log_details(log_texts, model, tokenizer, label_encoders, device, batch_size=32):
    model.eval()
    all_predictions = []
    
    for i in range(0, len(log_texts), batch_size):
        batch_texts = log_texts[i:i+batch_size]
        batch_texts = [preprocess_text(text) for text in batch_texts]
        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            logits = model(**inputs)
        
        batch_predictions = [logit.cpu().numpy().argmax(axis=1) for logit in logits]
        confidence_scores = [torch.softmax(logit, dim=1).max(dim=1)[0].cpu().numpy() for logit in logits]
        
        for j in range(len(batch_texts)):
            predictions = [pred[j] for pred in batch_predictions]
            scores = [score[j] for score in confidence_scores]
            
            interpreted_predictions = {
                'LineId': label_encoders['LineId'].inverse_transform([predictions[1]])[0],
                'timestamp': label_encoders['timestamp'].inverse_transform([predictions[0]])[0],
                'Component': label_encoders['Component'].inverse_transform([predictions[2]])[0],
                'Content': label_encoders['Content'].inverse_transform([predictions[3]])[0],
                'EventId': label_encoders['EventId'].inverse_transform([predictions[4]])[0],
                'EventTemplate': label_encoders['EventTemplate'].inverse_transform([predictions[5]])[0]
            }
            
            for key, value in interpreted_predictions.items():
                interpreted_predictions[key] = {
                    'prediction': value,
                    'confidence': scores[list(interpreted_predictions.keys()).index(key)]
                }
            
            all_predictions.append(interpreted_predictions)
    
    return all_predictions

batch_logs = [new_log] * 5 
batch_predictions = batch_predict_log_details(batch_logs, model, tokenizer, label_encoders, device)
print("Batch predictions:", batch_predictions)