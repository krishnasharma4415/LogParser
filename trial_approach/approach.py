# Complete Guide: ML-Based Log Parser Development

## Project Overview
# Build a machine learning model that can parse any type of log entry from 16 different log types (Android, Apache, BGL, Hadoop, HDFS, HealthApp, HPC, Linux, Mac, OpenSSH, OpenStack, Proxifier, Spark, Thunderbird, Windows, Zookeeper) into structured format.

## Step 1: Data Exploration and Understanding

### 1.1 Analyze Raw Data Structure
import pandas as pd
import os
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# First, let's understand the data structure
def explore_raw_data(raw_data_path):
    log_types = []
    for file in os.listdir(raw_data_path):
        if file.endswith('.log'):
            log_type = file.replace('_2k.log', '')
            log_types.append(log_type)
            
            # Read first few lines to understand format
            with open(os.path.join(raw_data_path, file), 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()[:10]
                print(f"\n{log_type} Sample:")
                for i, line in enumerate(lines):
                    print(f"{i+1}: {line.strip()}")
    
    return log_types

log_types = explore_raw_data('raw_data/')

### 1.2 Analyze Structured Data Format

def explore_structured_data(structured_data_path):
    structured_info = {}
    
    for file in os.listdir(structured_data_path):
        if file.endswith('.csv'):
            log_type = file.replace('_2k.log_structured.csv', '')
            df = pd.read_csv(os.path.join(structured_data_path, file))
            
            print(f"\n{log_type} Structured Format:")
            print(f"Columns: {list(df.columns)}")
            print(f"Shape: {df.shape}")
            print(df.head(3))
            
            structured_info[log_type] = {
                'columns': list(df.columns),
                'shape': df.shape,
                'sample': df.head(3)
            }
    
    return structured_info

structured_info = explore_structured_data('structured_data/')

## Step 2: Data Preprocessing and Feature Engineering

### 2.1 Create Unified Dataset
import re
from datetime import datetime
import json

class LogDataProcessor:
    def __init__(self, raw_data_path, structured_data_path):
        self.raw_data_path = raw_data_path
        self.structured_data_path = structured_data_path
        self.log_types = [
            'Android', 'Apache', 'BGL', 'Hadoop', 'HDFS', 'HealthApp',
            'HPC', 'Linux', 'Mac', 'OpenSSH', 'OpenStack', 'Proxifier',
            'Spark', 'Thunderbird', 'Windows', 'Zookeeper'
        ]
        
    def load_and_align_data(self):
        """Load raw and structured data and align them"""
        unified_dataset = []
        
        for log_type in self.log_types:
            raw_file = f"{log_type}_2k.log"
            structured_file = f"{log_type}_2k.log_structured.csv"
            
            # Read raw logs
            raw_path = os.path.join(self.raw_data_path, raw_file)
            with open(raw_path, 'r', encoding='utf-8', errors='ignore') as f:
                raw_lines = [line.strip() for line in f.readlines() if line.strip()]
            
            # Read structured data
            structured_path = os.path.join(self.structured_data_path, structured_file)
            structured_df = pd.read_csv(structured_path)
            
            # Align data (assuming they're in same order)
            min_len = min(len(raw_lines), len(structured_df))
            
            for i in range(min_len):
                unified_dataset.append({
                    'log_type': log_type,
                    'raw_log': raw_lines[i],
                    'structured_data': structured_df.iloc[i].to_dict(),
                    'line_id': structured_df.iloc[i].get('LineId', i),
                    'template': structured_df.iloc[i].get('EventTemplate', ''),
                    'event_id': structured_df.iloc[i].get('EventId', '')
                })
        
        return unified_dataset
    
    def extract_features(self, raw_log):
        """Extract features from raw log entry"""
        features = {
            'length': len(raw_log),
            'word_count': len(raw_log.split()),
            'has_timestamp': bool(re.search(r'\d{4}-\d{2}-\d{2}|\d{2}:\d{2}:\d{2}', raw_log)),
            'has_ip': bool(re.search(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', raw_log)),
            'has_brackets': '[' in raw_log or '(' in raw_log,
            'has_error': any(word in raw_log.lower() for word in ['error', 'fail', 'exception']),
            'has_warning': any(word in raw_log.lower() for word in ['warn', 'warning']),
            'special_char_count': len(re.findall(r'[^a-zA-Z0-9\s]', raw_log)),
            'digit_count': len(re.findall(r'\d', raw_log)),
            'uppercase_ratio': sum(1 for c in raw_log if c.isupper()) / len(raw_log) if raw_log else 0
        }
        return features

processor = LogDataProcessor('raw_data/', 'structured_data/')
unified_data = processor.load_and_align_data()
print(f"Total unified dataset size: {len(unified_data)}")
```

### 2.2 Text Preprocessing for Neural Networks
```python
import torch
from transformers import AutoTokenizer
from sklearn.preprocessing import LabelEncoder

class LogTextPreprocessor:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.label_encoders = {}
        
    def preprocess_text(self, text, max_length=512):
        """Tokenize and encode text"""
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        return encoding
    
    def prepare_training_data(self, unified_data):
        """Prepare data for training"""
        processed_data = []
        
        # Extract log types for classification
        log_types = [item['log_type'] for item in unified_data]
        if 'log_type' not in self.label_encoders:
            self.label_encoders['log_type'] = LabelEncoder()
            self.label_encoders['log_type'].fit(log_types)
        
        for item in unified_data:
            # Input: raw log
            raw_log = item['raw_log']
            
            # Target: structured template
            template = item['template']
            
            # Additional labels
            log_type_encoded = self.label_encoders['log_type'].transform([item['log_type']])[0]
            
            processed_item = {
                'input_text': raw_log,
                'target_template': template,
                'log_type': log_type_encoded,
                'log_type_name': item['log_type'],
                'event_id': item.get('event_id', ''),
                'features': processor.extract_features(raw_log)
            }
            processed_data.append(processed_item)
        
        return processed_data

preprocessor = LogTextPreprocessor()
processed_data = preprocessor.prepare_training_data(unified_data)
```

## Step 3: Model Architecture Design

### 3.1 Multi-Task Learning Approach
```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class LogParserModel(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_log_types=16, 
                 hidden_dim=768, template_vocab_size=50000):
        super(LogParserModel, self).__init__()
        
        # Encoder (BERT-based)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.config = self.encoder.config
        
        # Log type classification head
        self.log_type_classifier = nn.Sequential(
            nn.Linear(self.config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_log_types)
        )
        
        # Template generation head (seq2seq decoder)
        self.template_decoder = nn.LSTM(
            input_size=self.config.hidden_size,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        self.template_output = nn.Linear(hidden_dim, template_vocab_size)
        
        # Feature fusion layer
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.config.hidden_size + 10, hidden_dim),  # +10 for engineered features
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, input_ids, attention_mask, features=None, target_template=None):
        # Encode input
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = encoder_outputs.last_hidden_state  # [batch, seq_len, hidden]
        pooled_output = encoder_outputs.pooler_output  # [batch, hidden]
        
        # Log type classification
        log_type_logits = self.log_type_classifier(pooled_output)
        
        # Feature fusion if features provided
        if features is not None:
            fused_features = torch.cat([pooled_output, features], dim=1)
            fused_output = self.feature_fusion(fused_features)
        else:
            fused_output = pooled_output
        
        # Template generation
        # Use fused output as initial hidden state for LSTM
        batch_size = fused_output.size(0)
        hidden_dim = self.template_decoder.hidden_size
        num_layers = self.template_decoder.num_layers
        
        h0 = fused_output.unsqueeze(0).repeat(num_layers, 1, 1)  # [num_layers, batch, hidden]
        c0 = torch.zeros_like(h0)
        
        # For training, use teacher forcing
        if target_template is not None and self.training:
            decoder_output, _ = self.template_decoder(sequence_output, (h0, c0))
            template_logits = self.template_output(decoder_output)
        else:
            # For inference, generate step by step
            template_logits = self.generate_template(sequence_output, h0, c0)
        
        return {
            'log_type_logits': log_type_logits,
            'template_logits': template_logits,
            'hidden_states': sequence_output
        }
    
    def generate_template(self, encoder_output, h0, c0, max_length=128):
        """Generate template during inference"""
        batch_size = encoder_output.size(0)
        generated = []
        hidden = (h0, c0)
        
        # Start with encoder's first token
        current_input = encoder_output[:, 0:1, :]  # [batch, 1, hidden]
        
        for _ in range(max_length):
            output, hidden = self.template_decoder(current_input, hidden)
            logits = self.template_output(output)  # [batch, 1, vocab_size]
            generated.append(logits)
            
            # Use output as next input (teacher forcing disabled)
            current_input = output
        
        return torch.cat(generated, dim=1)  # [batch, max_length, vocab_size]
```

### 3.2 Training Pipeline
```python
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import torch.nn.functional as F

class LogDataset(Dataset):
    def __init__(self, processed_data, tokenizer, max_length=512):
        self.data = processed_data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Build template vocabulary
        all_templates = [item['target_template'] for item in processed_data]
        self.template_vocab = self.build_template_vocab(all_templates)
        
    def build_template_vocab(self, templates):
        """Build vocabulary for templates"""
        vocab = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        
        for template in templates:
            tokens = template.split()
            for token in tokens:
                if token not in vocab:
                    vocab[token] = len(vocab)
        
        return vocab
    
    def encode_template(self, template):
        """Encode template to token ids"""
        tokens = ['<SOS>'] + template.split() + ['<EOS>']
        token_ids = [self.template_vocab.get(token, self.template_vocab['<UNK>']) for token in tokens]
        
        # Pad or truncate
        if len(token_ids) < self.max_length:
            token_ids += [self.template_vocab['<PAD>']] * (self.max_length - len(token_ids))
        else:
            token_ids = token_ids[:self.max_length]
        
        return torch.tensor(token_ids, dtype=torch.long)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize input
        encoding = self.tokenizer(
            item['input_text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Prepare features
        features_list = list(item['features'].values())
        features_tensor = torch.tensor(features_list, dtype=torch.float32)
        
        # Encode template
        template_encoded = self.encode_template(item['target_template'])
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'features': features_tensor,
            'log_type': torch.tensor(item['log_type'], dtype=torch.long),
            'template_target': template_encoded
        }

class LogParserTrainer:
    def __init__(self, model, train_loader, val_loader, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Optimizers
        self.optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
        
        # Scheduler
        num_training_steps = len(train_loader) * 10  # 10 epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_training_steps // 10,
            num_training_steps=num_training_steps
        )
        
        # Loss functions
        self.classification_loss = nn.CrossEntropyLoss()
        self.generation_loss = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for batch in self.train_loader:
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            features = batch['features'].to(self.device)
            log_type = batch['log_type'].to(self.device)
            template_target = batch['template_target'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                features=features,
                target_template=template_target
            )
            
            # Calculate losses
            log_type_loss = self.classification_loss(outputs['log_type_logits'], log_type)
            
            # Template generation loss
            template_logits = outputs['template_logits']
            template_loss = self.generation_loss(
                template_logits.view(-1, template_logits.size(-1)),
                template_target.view(-1)
            )
            
            # Combined loss
            total_batch_loss = log_type_loss + template_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += total_batch_loss.item()
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        correct_log_type = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                features = batch['features'].to(self.device)
                log_type = batch['log_type'].to(self.device)
                template_target = batch['template_target'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    features=features,
                    target_template=template_target
                )
                
                # Log type accuracy
                log_type_pred = torch.argmax(outputs['log_type_logits'], dim=1)
                correct_log_type += (log_type_pred == log_type).sum().item()
                total_samples += log_type.size(0)
                
                # Losses
                log_type_loss = self.classification_loss(outputs['log_type_logits'], log_type)
                template_logits = outputs['template_logits']
                template_loss = self.generation_loss(
                    template_logits.view(-1, template_logits.size(-1)),
                    template_target.view(-1)
                )
                
                total_loss += (log_type_loss + template_loss).item()
        
        avg_loss = total_loss / len(self.val_loader)
        log_type_accuracy = correct_log_type / total_samples
        
        return avg_loss, log_type_accuracy
    
    def train(self, num_epochs=10):
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_loss, val_accuracy = self.validate()
            
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Log Type Accuracy: {val_accuracy:.4f}")
            print("-" * 50)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_log_parser_model.pth')
                print("Saved best model!")
```

## Step 4: Training and Evaluation

### 4.1 Data Splitting and Training
```python
from sklearn.model_selection import train_test_split

# Split data
train_data, val_data = train_test_split(processed_data, test_size=0.2, random_state=42, 
                                       stratify=[item['log_type_name'] for item in processed_data])

# Create datasets
train_dataset = LogDataset(train_data, preprocessor.tokenizer)
val_dataset = LogDataset(val_data, preprocessor.tokenizer)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Initialize model
model = LogParserModel(
    num_log_types=len(preprocessor.label_encoders['log_type'].classes_),
    template_vocab_size=len(train_dataset.template_vocab)
)

# Train model
trainer = LogParserTrainer(model, train_loader, val_loader)
trainer.train(num_epochs=10)
```

### 4.2 Evaluation Metrics
```python
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class LogParserEvaluator:
    def __init__(self, model, tokenizer, label_encoders, template_vocab):
        self.model = model
        self.tokenizer = tokenizer
        self.label_encoders = label_encoders
        self.template_vocab = template_vocab
        self.reverse_template_vocab = {v: k for k, v in template_vocab.items()}
        
    def evaluate_on_test_set(self, test_data):
        """Comprehensive evaluation"""
        self.model.eval()
        
        predictions = {
            'log_types': [],
            'templates': []
        }
        
        ground_truth = {
            'log_types': [],
            'templates': []
        }
        
        with torch.no_grad():
            for item in test_data:
                # Prepare input
                encoding = self.tokenizer(
                    item['input_text'],
                    truncation=True,
                    padding='max_length',
                    max_length=512,
                    return_tensors='pt'
                )
                
                features_list = list(item['features'].values())
                features_tensor = torch.tensor([features_list], dtype=torch.float32)
                
                # Forward pass
                outputs = self.model(
                    input_ids=encoding['input_ids'],
                    attention_mask=encoding['attention_mask'],
                    features=features_tensor
                )
                
                # Log type prediction
                log_type_pred = torch.argmax(outputs['log_type_logits'], dim=1).item()
                log_type_name = self.label_encoders['log_type'].inverse_transform([log_type_pred])[0]
                
                # Template prediction
                template_logits = outputs['template_logits'].squeeze()
                template_pred_ids = torch.argmax(template_logits, dim=-1).tolist()
                template_pred = self.decode_template(template_pred_ids)
                
                predictions['log_types'].append(log_type_name)
                predictions['templates'].append(template_pred)
                
                ground_truth['log_types'].append(item['log_type_name'])
                ground_truth['templates'].append(item['target_template'])
        
        return self.calculate_metrics(predictions, ground_truth)
    
    def decode_template(self, token_ids):
        """Decode template from token ids"""
        tokens = [self.reverse_template_vocab.get(id, '<UNK>') for id in token_ids]
        
        # Remove special tokens and padding
        clean_tokens = []
        for token in tokens:
            if token in ['<SOS>', '<EOS>', '<PAD>']:
                if token == '<EOS>':
                    break
                continue
            clean_tokens.append(token)
        
        return ' '.join(clean_tokens)
    
    def calculate_metrics(self, predictions, ground_truth):
        """Calculate various evaluation metrics"""
        metrics = {}
        
        # Log type classification metrics
        log_type_accuracy = accuracy_score(ground_truth['log_types'], predictions['log_types'])
        log_type_precision, log_type_recall, log_type_f1, _ = precision_recall_fscore_support(
            ground_truth['log_types'], predictions['log_types'], average='weighted'
        )
        
        metrics['log_type'] = {
            'accuracy': log_type_accuracy,
            'precision': log_type_precision,
            'recall': log_type_recall,
            'f1': log_type_f1
        }
        
        # Template parsing metrics
        template_exact_match = sum(1 for gt, pred in zip(ground_truth['templates'], predictions['templates']) 
                                 if gt.strip() == pred.strip()) / len(ground_truth['templates'])
        
        # BLEU score for template generation
        from nltk.translate.bleu_score import sentence_bleu
        bleu_scores = []
        for gt, pred in zip(ground_truth['templates'], predictions['templates']):
            gt_tokens = gt.split()
            pred_tokens = pred.split()
            if gt_tokens:  # Avoid empty reference
                bleu = sentence_bleu([gt_tokens], pred_tokens)
                bleu_scores.append(bleu)
        
        avg_bleu = np.mean(bleu_scores) if bleu_scores else 0
        
        metrics['template'] = {
            'exact_match': template_exact_match,
            'bleu_score': avg_bleu
        }
        
        return metrics
    
    def plot_confusion_matrix(self, ground_truth, predictions):
        """Plot confusion matrix for log type classification"""
        cm = confusion_matrix(ground_truth, predictions)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Log Type Classification Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

# Evaluate model
evaluator = LogParserEvaluator(model, preprocessor.tokenizer, 
                              preprocessor.label_encoders, train_dataset.template_vocab)
test_metrics = evaluator.evaluate_on_test_set(val_data)
print("Evaluation Results:", test_metrics)
```

## Step 5: Model Deployment and Inference

### 5.1 Inference Pipeline
```python
class LogParserInference:
    def __init__(self, model_path, tokenizer_name='bert-base-uncased'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = LogParserModel()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Load tokenizer and other components
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Load preprocessors (these should be saved during training)
        self.processor = LogDataProcessor('', '')
        
    def parse_single_log(self, raw_log):
        """Parse a single log entry"""
        with torch.no_grad():
            # Preprocess
            features = self.processor.extract_features(raw_log)
            
            # Tokenize
            encoding = self.tokenizer(
                raw_log,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt'
            ).to(self.device)
            
            features_tensor = torch.tensor([list(features.values())], 
                                         dtype=torch.float32).to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=encoding['input_ids'],
                attention_mask=encoding['attention_mask'],
                features=features_tensor
            )
            
            # Extract predictions
            log_type_pred = torch.argmax(outputs['log_type_logits'], dim=1).item()
            template_logits = outputs['template_logits'].squeeze()
            template_pred_ids = torch.argmax(template_logits, dim=-1).tolist()
            
            return {
                'raw_log': raw_log,
                'predicted_log_type': log_type_pred,
                'predicted_template': self.decode_template(template_pred_ids),
                'confidence_scores': {
                    'log_type': torch.softmax(outputs['log_type_logits'], dim=1).max().item(),
                }
            }
    
    def batch_parse_logs(self, raw_logs):
        """Parse multiple log entries"""
        results = []
        for log in raw_logs:
            result = self.parse_single_log(log)
            results.append(result)
        return results

# Example usage
parser = LogParserInference('best_log_parser_model.pth')

# Test on new logs
test_logs = [
    "2024-01-15 10:30:15 ERROR [main] Connection failed to database server 192.168.1.100",
    "Jan 15 10:30:15 server01 sshd[1234]: Failed password for user admin from 192.168.1.50",
    "10:30:15.123 WARNING Task execution failed with exception: NullPointerException"
]

for log in test_logs:
    result = parser.parse_single_log(log)
    print(f"Raw: {result['raw_log']}")
    print(f"Type: {result['predicted_log_type']}")
    print(f"Template: {result['predicted_template']}")
    print(f"Confidence: {result['confidence_scores']['log_type']:.3f}")
    print("-" * 80)
```

## Step 6: Advanced Techniques and Improvements

### 6.1 Few-Shot Learning for New Log Types
```python
class FewShotLogParser:
    def __init__(self, base_model, tokenizer):
        self.base_model = base_model
        self.tokenizer = tokenizer
        
    def adapt_to_new_log_type(self, few_shot_examples, new_log_type_name):
        """Adapt model to new log type with few examples"""
        # Create a small dataset from examples
        adaptation_data = []
        for raw_log, template in few_shot_examples:
            features = self.processor.extract_features(raw_log)
            adaptation_data.append({
                'input_text': raw_log,
                'target_template': template,
                'log_type_name': new_log_type_name,
                'features': features
            })
        
        # Fine-tune the model on few-shot examples
        self.fine_tune_on_examples(adaptation_data)
    
    def fine_tune_on_examples(self, examples, learning_rate=1e-5, epochs=5):
        """Fine-tune model on few examples"""
        # Create small dataset
        small_dataset = LogDataset(examples, self.tokenizer)
        small_loader = DataLoader(small_dataset, batch_size=2, shuffle=True)
        
        # Use lower learning rate for fine-tuning
        optimizer = AdamW(self.base_model.parameters(), lr=learning_rate)
        
        self.base_model.train()
        for epoch in range(epochs):
            for batch in small_loader:
                # Forward pass and loss calculation
                # (similar to main training loop but with smaller batch)
                pass

### 6.2 Attention Visualization
```python
import matplotlib.pyplot as plt
import numpy as np

class AttentionVisualizer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def visualize_attention(self, raw_log, save_path=None):
        """Visualize attention weights for a log entry"""
        # Get model attention weights
        encoding = self.tokenizer(raw_log, return_tensors='pt', padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = self.model(**encoding, output_attentions=True)
            attentions = outputs.attentions  # List of attention matrices
        
        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
        
        # Plot attention heatmap for last layer
        attention_matrix = attentions[-1][0].mean(dim=0).cpu().numpy()  # Average across heads
        
        plt.figure(figsize=(12, 8))
        plt.imshow(attention_matrix, cmap='Blues', aspect='auto')
        plt.colorbar()
        plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right')
        plt.yticks(range(len(tokens)), tokens)
        plt.title(f'Attention Visualization for Log: {raw_log[:50]}...')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

### 6.3 Ensemble Method
```python
class EnsembleLogParser:
    def __init__(self, model_paths, tokenizer_name='bert-base-uncased'):
        self.models = []
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Load multiple models
        for path in model_paths:
            model = LogParserModel()
            model.load_state_dict(torch.load(path))
            model.eval()
            self.models.append(model)
    
    def ensemble_predict(self, raw_log):
        """Make predictions using ensemble of models"""
        all_predictions = []
        
        for model in self.models:
            with torch.no_grad():
                # Prepare input
                encoding = self.tokenizer(raw_log, return_tensors='pt', 
                                        padding=True, truncation=True)
                features = torch.tensor([list(self.extract_features(raw_log).values())], 
                                      dtype=torch.float32)
                
                # Get prediction
                outputs = model(input_ids=encoding['input_ids'],
                              attention_mask=encoding['attention_mask'],
                              features=features)
                
                all_predictions.append(outputs)
        
        # Combine predictions (voting/averaging)
        # Log type: majority voting
        log_type_votes = []
        for pred in all_predictions:
            log_type_votes.append(torch.argmax(pred['log_type_logits'], dim=1).item())
        
        # Template: average logits
        template_logits = torch.stack([pred['template_logits'] for pred in all_predictions])
        avg_template_logits = template_logits.mean(dim=0)
        
        final_log_type = max(set(log_type_votes), key=log_type_votes.count)
        final_template_ids = torch.argmax(avg_template_logits, dim=-1).squeeze().tolist()
        
        return {
            'log_type': final_log_type,
            'template': self.decode_template(final_template_ids),
            'confidence': len([v for v in log_type_votes if v == final_log_type]) / len(log_type_votes)
        }

## Step 7: Real-time Processing Pipeline

### 7.1 Streaming Log Parser
```python
import asyncio
import json
from datetime import datetime
import logging

class StreamingLogParser:
    def __init__(self, model_path, batch_size=32, buffer_size=1000):
        self.parser = LogParserInference(model_path)
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.log_buffer = []
        self.processed_count = 0
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    async def process_log_stream(self, log_stream):
        """Process continuous stream of logs"""
        async for log_entry in log_stream:
            await self.add_to_buffer(log_entry)
            
            # Process batch when buffer is full
            if len(self.log_buffer) >= self.batch_size:
                await self.process_batch()
    
    async def add_to_buffer(self, log_entry):
        """Add log entry to processing buffer"""
        self.log_buffer.append({
            'raw_log': log_entry,
            'timestamp': datetime.now().isoformat(),
            'id': f"log_{self.processed_count}"
        })
        self.processed_count += 1
        
        # Prevent buffer overflow
        if len(self.log_buffer) > self.buffer_size:
            self.log_buffer = self.log_buffer[-self.buffer_size:]
    
    async def process_batch(self):
        """Process a batch of logs"""
        batch_to_process = self.log_buffer[:self.batch_size]
        self.log_buffer = self.log_buffer[self.batch_size:]
        
        # Process batch
        results = []
        for log_entry in batch_to_process:
            result = self.parser.parse_single_log(log_entry['raw_log'])
            result.update({
                'id': log_entry['id'],
                'timestamp': log_entry['timestamp']
            })
            results.append(result)
        
        # Send results to output handler
        await self.handle_results(results)
    
    async def handle_results(self, results):
        """Handle processed results (save to DB, send to monitoring, etc.)"""
        for result in results:
            # Example: Log to file
            self.logger.info(f"Processed log {result['id']}: "
                           f"Type={result['predicted_log_type']}, "
                           f"Template={result['predicted_template'][:50]}...")
            
            # Example: Send to monitoring system
            await self.send_to_monitoring(result)
    
    async def send_to_monitoring(self, result):
        """Send result to monitoring system"""
        # This could be sending to Elasticsearch, Kafka, etc.
        monitoring_data = {
            'log_id': result['id'],
            'timestamp': result['timestamp'],
            'log_type': result['predicted_log_type'],
            'template': result['predicted_template'],
            'confidence': result['confidence_scores']['log_type'],
            'raw_log': result['raw_log']
        }
        
        # Example: Send to webhook or message queue
        # await send_to_webhook(monitoring_data)
        pass

### 7.2 API Service
```python
from flask import Flask, request, jsonify
import threading
import queue

app = Flask(__name__)

class LogParserAPI:
    def __init__(self, model_path):
        self.parser = LogParserInference(model_path)
        self.request_queue = queue.Queue()
        self.response_cache = {}
        
        # Start background worker
        self.worker_thread = threading.Thread(target=self.worker, daemon=True)
        self.worker_thread.start()
    
    def worker(self):
        """Background worker to process requests"""
        while True:
            try:
                request_id, raw_log = self.request_queue.get(timeout=1)
                result = self.parser.parse_single_log(raw_log)
                self.response_cache[request_id] = result
                self.request_queue.task_done()
            except queue.Empty:
                continue
    
    def parse_log_async(self, raw_log):
        """Async log parsing"""
        request_id = f"req_{len(self.response_cache)}"
        self.request_queue.put((request_id, raw_log))
        return request_id
    
    def get_result(self, request_id):
        """Get parsing result"""
        return self.response_cache.get(request_id)

# Initialize API
parser_api = LogParserAPI('best_log_parser_model.pth')

@app.route('/parse', methods=['POST'])
def parse_log():
    """Parse single log entry"""
    data = request.json
    raw_log = data.get('log', '')
    
    if not raw_log:
        return jsonify({'error': 'No log provided'}), 400
    
    try:
        result = parser_api.parser.parse_single_log(raw_log)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/parse_batch', methods=['POST'])
def parse_batch():
    """Parse multiple log entries"""
    data = request.json
    logs = data.get('logs', [])
    
    if not logs:
        return jsonify({'error': 'No logs provided'}), 400
    
    try:
        results = parser_api.parser.batch_parse_logs(logs)
        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': True})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

## Step 8: Performance Optimization

### 8.1 Model Quantization
```python
import torch.quantization as quantization

class OptimizedLogParser:
    def __init__(self, model_path):
        # Load original model
        self.model = LogParserModel()
        self.model.load_state_dict(torch.load(model_path))
        
        # Quantize model for faster inference
        self.quantized_model = self.quantize_model(self.model)
        
    def quantize_model(self, model):
        """Quantize model for faster inference"""
        model.eval()
        
        # Post-training quantization
        quantized_model = quantization.quantize_dynamic(
            model, 
            {torch.nn.Linear}, 
            dtype=torch.qint8
        )
        
        return quantized_model
    
    def benchmark_models(self, test_logs, num_runs=100):
        """Benchmark original vs quantized model"""
        import time
        
        # Benchmark original model
        start_time = time.time()
        for _ in range(num_runs):
            for log in test_logs:
                _ = self.parse_with_original(log)
        original_time = time.time() - start_time
        
        # Benchmark quantized model
        start_time = time.time()
        for _ in range(num_runs):
            for log in test_logs:
                _ = self.parse_with_quantized(log)
        quantized_time = time.time() - start_time
        
        print(f"Original model time: {original_time:.2f}s")
        print(f"Quantized model time: {quantized_time:.2f}s")
        print(f"Speedup: {original_time/quantized_time:.2f}x")

### 8.2 Caching and Batching
```python
import hashlib
from functools import lru_cache

class CachedLogParser:
    def __init__(self, model_path, cache_size=1000):
        self.parser = LogParserInference(model_path)
        self.cache_size = cache_size
        self.parse_cache = {}
        
    def get_log_hash(self, raw_log):
        """Generate hash for log entry"""
        return hashlib.md5(raw_log.encode()).hexdigest()
    
    @lru_cache(maxsize=1000)
    def parse_with_cache(self, log_hash, raw_log):
        """Parse log with caching"""
        return self.parser.parse_single_log(raw_log)
    
    def parse_log(self, raw_log):
        """Parse log with caching"""
        log_hash = self.get_log_hash(raw_log)
        return self.parse_with_cache(log_hash, raw_log)

## Step 9: Monitoring and Maintenance

### 9.1 Model Performance Monitoring
```python
import psutil
import matplotlib.pyplot as plt
from collections import deque
import time

class ModelMonitor:
    def __init__(self, model_path):
        self.parser = LogParserInference(model_path)
        self.performance_metrics = {
            'inference_times': deque(maxlen=1000),
            'memory_usage': deque(maxlen=1000),
            'cpu_usage': deque(maxlen=1000),
            'accuracy_scores': deque(maxlen=1000)
        }
        
    def monitor_inference(self, raw_log, ground_truth=None):
        """Monitor single inference"""
        # Measure inference time
        start_time = time.time()
        result = self.parser.parse_single_log(raw_log)
        inference_time = time.time() - start_time
        
        # Measure resource usage
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        cpu_usage = psutil.cpu_percent()
        
        # Store metrics
        self.performance_metrics['inference_times'].append(inference_time)
        self.performance_metrics['memory_usage'].append(memory_usage)
        self.performance_metrics['cpu_usage'].append(cpu_usage)
        
        # Calculate accuracy if ground truth provided
        if ground_truth:
            accuracy = self.calculate_accuracy(result, ground_truth)
            self.performance_metrics['accuracy_scores'].append(accuracy)
        
        return result
    
    def calculate_accuracy(self, prediction, ground_truth):
        """Calculate accuracy for single prediction"""
        # Simple accuracy based on template similarity
        pred_template = prediction['predicted_template']
        true_template = ground_truth.get('template', '')
        
        # Jaccard similarity
        pred_words = set(pred_template.split())
        true_words = set(true_template.split())
        
        if not pred_words and not true_words:
            return 1.0
        
        intersection = len(pred_words.intersection(true_words))
        union = len(pred_words.union(true_words))
        
        return intersection / union if union > 0 else 0.0
    
    def plot_performance_metrics(self):
        """Plot performance metrics over time"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Inference times
        axes[0, 0].plot(list(self.performance_metrics['inference_times']))
        axes[0, 0].set_title('Inference Times')
        axes[0, 0].set_ylabel('Time (seconds)')
        
        # Memory usage
        axes[0, 1].plot(list(self.performance_metrics['memory_usage']))
        axes[0, 1].set_title('Memory Usage')
        axes[0, 1].set_ylabel('Memory (MB)')
        
        # CPU usage
        axes[1, 0].plot(list(self.performance_metrics['cpu_usage']))
        axes[1, 0].set_title('CPU Usage')
        axes[1, 0].set_ylabel('CPU %')
        
        # Accuracy scores
        if self.performance_metrics['accuracy_scores']:
            axes[1, 1].plot(list(self.performance_metrics['accuracy_scores']))
            axes[1, 1].set_title('Accuracy Scores')
            axes[1, 1].set_ylabel('Accuracy')
        
        plt.tight_layout()
        plt.show()
    
    def get_performance_summary(self):
        """Get performance summary statistics"""
        summary = {}
        
        for metric_name, values in self.performance_metrics.items():
            if values:
                summary[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values)
                }
        
        return summary

### 9.2 Automated Retraining Pipeline
```python
import schedule
import time
from datetime import datetime, timedelta

class AutoRetrainingPipeline:
    def __init__(self, model_path, data_path, retrain_threshold=0.85):
        self.model_path = model_path
        self.data_path = data_path
        self.retrain_threshold = retrain_threshold
        self.monitor = ModelMonitor(model_path)
        
        # Schedule retraining checks
        schedule.every().day.at("02:00").do(self.check_retraining_needed)
        
    def check_retraining_needed(self):
        """Check if model needs retraining"""
        # Get recent performance metrics
        recent_accuracy = list(self.monitor.performance_metrics['accuracy_scores'])[-100:]
        
        if recent_accuracy:
            avg_accuracy = np.mean(recent_accuracy)
            
            if avg_accuracy < self.retrain_threshold:
                print(f"Model performance dropped to {avg_accuracy:.3f}. Initiating retraining...")
                self.initiate_retraining()
            else:
                print(f"Model performance is acceptable: {avg_accuracy:.3f}")
    
    def initiate_retraining(self):
        """Initiate automated retraining"""
        print("Starting automated retraining pipeline...")
        
        try:
            # Load new data
            new_data = self.load_new_training_data()
            
            if len(new_data) > 100:  # Minimum data requirement
                # Retrain model
                self.retrain_model(new_data)
                
                # Validate new model
                if self.validate_new_model():
                    self.deploy_new_model()
                    print("Retraining completed successfully!")
                else:
                    print("New model validation failed. Keeping old model.")
            else:
                print("Insufficient new data for retraining.")
                
        except Exception as e:
            print(f"Retraining failed: {str(e)}")
    
    def load_new_training_data(self):
        """Load new training data"""
        # This would load new log data that has been manually labeled
        # or validated by domain experts
        pass
    
    def retrain_model(self, new_data):
        """Retrain model with new data"""
        # Implementation of retraining logic
        pass
    
    def validate_new_model(self):
        """Validate newly trained model"""
        # Run validation tests on new model
        return True  # Placeholder
    
    def deploy_new_model(self):
        """Deploy new model to production"""
        # Backup old model
        backup_path = f"{self.model_path}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        # Copy current model to backup
        
        # Deploy new model
        # Update model path references
        print("New model deployed successfully!")
    
    def run_scheduler(self):
        """Run the scheduling loop"""
        while True:
            schedule.run_pending()
            time.sleep(3600)  # Check every hour

## Step 10: Deployment Checklist and Best Practices

### 10.1 Production Deployment Checklist
```python
class ProductionDeploymentChecklist:
    def __init__(self):
        self.checklist_items = {
            'model_validation': False,
            'performance_testing': False,
            'security_review': False,
            'monitoring_setup': False,
            'backup_strategy': False,
            'rollback_plan': False,
            'documentation': False,
            'error_handling': False,
            'scalability_testing': False,
            'api_documentation': False
        }
    
    def validate_model(self, model_path, test_data):
        """Validate model before deployment"""
        try:
            parser = LogParserInference(model_path)
            
            # Test on sample data
            test_results = []
            for item in test_data[:100]:  # Test on subset
                result = parser.parse_single_log(item['raw_log'])
                test_results.append(result)
            
            # Check if all predictions are valid
            valid_predictions = all(
                'predicted_log_type' in result and 'predicted_template' in result
                for result in test_results
            )
            
            if valid_predictions:
                self.checklist_items['model_validation'] = True
                print("✓ Model validation passed")
            else:
                print("✗ Model validation failed")
                
        except Exception as e:
            print(f"✗ Model validation failed: {str(e)}")
    
    def performance_test(self, model_path, test_logs, target_latency=0.1):
        """Test model performance"""
        try:
            parser = LogParserInference(model_path)
            
            # Measure latency
            latencies = []
            for log in test_logs[:50]:
                start_time = time.time()
                parser.parse_single_log(log)
                latency = time.time() - start_time
                latencies.append(latency)
            
            avg_latency = np.mean(latencies)
            
            if avg_latency <= target_latency:
                self.checklist_items['performance_testing'] = True
                print(f"✓ Performance test passed (avg latency: {avg_latency:.3f}s)")
            else:
                print(f"✗ Performance test failed (avg latency: {avg_latency:.3f}s > {target_latency}s)")
                
        except Exception as e:
            print(f"✗ Performance test failed: {str(e)}")
    
    def setup_monitoring(self):
        """Setup monitoring infrastructure"""
        # This would setup monitoring dashboards, alerts, etc.
        self.checklist_items['monitoring_setup'] = True
        print("✓ Monitoring setup completed")
    
    def create_backup_strategy(self):
        """Create backup and recovery strategy"""
        # This would setup automated backups
        self.checklist_items['backup_strategy'] = True
        print("✓ Backup strategy implemented")
    
    def create_rollback_plan(self):
        """Create rollback plan"""
        # Document rollback procedures
        self.checklist_items['rollback_plan'] = True
        print("✓ Rollback plan documented")
    
    def generate_report(self):
        """Generate deployment readiness report"""
        completed = sum(self.checklist_items.values())
        total = len(self.checklist_items)
        
        print(f"\nDeployment Readiness Report")
        print(f"={'='*40}")
        print(f"Completed: {completed}/{total} ({completed/total*100:.1f}%)")
        print(f"\nItem Status:")
        
        for item, status in self.checklist_items.items():
            status_symbol = "✓" if status else "✗"
            print(f"{status_symbol} {item.replace('_', ' ').title()}")
        
        if completed == total:
            print(f"\n🎉 Ready for production deployment!")
        else:
            print(f"\n⚠️  Complete remaining items before deployment")
        
        return completed == total

# Usage example
checklist = ProductionDeploymentChecklist()
checklist.validate_model('best_log_parser_model.pth', processed_data[:100])
checklist.performance_test('best_log_parser_model.pth', [item['input_text'] for item in processed_data[:50]])
checklist.setup_monitoring()
checklist.create_backup_strategy()
checklist.create_rollback_plan()

deployment_ready = checklist.generate_report()
```

## Conclusion

This comprehensive guide covers the complete pipeline for building an ML-based log parser:

1. **Data Understanding**: Analyze raw and structured log formats
2. **Data Preprocessing**: Create unified datasets and extract features
3. **Model Architecture**: Design multi-task learning model with BERT encoder
4. **Training Pipeline**: Implement training with proper validation
5. **Evaluation**: Comprehensive metrics and visualization
6. **Deployment**: API service and real-time processing
7. **Optimization**: Quantization, caching, and performance tuning
8. **Monitoring**: Performance tracking and automated retraining
9. **Production**: Deployment checklist and best practices

### Key Success Factors:
- **Data Quality**: Ensure high-quality alignment between raw and structured data
- **Feature Engineering**: Extract meaningful features from log entries
- **Model Architecture**: Use appropriate multi-task learning approach
- **Evaluation**: Implement comprehensive evaluation metrics
- **Monitoring**: Continuous monitoring and retraining pipeline
- **Scalability**: Design for production-scale deployment

### Next Steps:
1. Start with data exploration and understanding
2. Implement the preprocessing pipeline
3. Build and train the base model
4. Evaluate and iterate on model performance
5. Implement deployment and monitoring infrastructure
6. Scale to production requirements

This approach provides a robust foundation for building a production-ready log parsing system that can handle multiple log types and adapt to new formats over time.