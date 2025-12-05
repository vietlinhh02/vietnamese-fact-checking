"""
Script để fine-tune PhoBERT cho bài toán Claim Detection trên Kaggle
Task 6.2: Binary classification (claim vs non-claim)
Requirements: 1.1, 1.2, 1.5
"""

import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Cấu hình
class Config:
    MODEL_NAME = "vinai/phobert-base"
    MAX_LENGTH = 128  # Claim detection chỉ cần câu đơn, không cần context dài
    BATCH_SIZE = 32   # Tăng batch size vì input ngắn hơn
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 5
    WARMUP_STEPS = 500
    WEIGHT_DECAY = 0.01
    
    # Đường dẫn dataset - Cần dataset có label claim/non-claim
    TRAIN_PATH = "/kaggle/input/your-dataset/claim_detection_train.jsonl"
    VAL_PATH = "/kaggle/input/your-dataset/claim_detection_val.jsonl"
    TEST_PATH = "/kaggle/input/your-dataset/claim_detection_test.jsonl"
    
    # Label mapping cho binary classification
    LABEL2ID = {"non-claim": 0, "claim": 1}
    ID2LABEL = {0: "non-claim", 1: "claim"}
    NUM_LABELS = 2

# Load và xử lý dữ liệu
def load_data(file_path):
    """
    Load dữ liệu từ file JSONL
    Format: {"text": "câu văn bản", "label": "claim" hoặc "non-claim"}
    """
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            records.append({
                'text': data['text'],
                'label': data['label']
            })
    
    return pd.DataFrame(records)

def prepare_input_text(row):
    """
    Chuẩn bị text input cho model
    Với claim detection, chỉ cần câu văn bản đơn
    """
    return row['text']

# Custom Dataset
class FactCheckDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length, label2id):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = label2id
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Chuẩn bị text
        text = prepare_input_text(row)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Label
        label = self.label2id[row['label']]
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Metrics
def compute_metrics(pred):
    """
    Compute metrics cho binary classification
    Requirement 1.1, 1.2: Accuracy, Precision, Recall, F1
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='binary', pos_label=1)
    recall = recall_score(labels, preds, average='binary', pos_label=1)
    f1 = f1_score(labels, preds, average='binary', pos_label=1)
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def main():
    print("=" * 50)
    print("FINE-TUNING PHOBERT CHO CLAIM DETECTION")
    print("Task 6.2: Binary Classification (claim vs non-claim)")
    print("=" * 50)
    
    # Kiểm tra GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Device: {device}")
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    
    # Load dữ liệu
    print("\n[1] Đang load dữ liệu...")
    train_df = load_data(Config.TRAIN_PATH)
    val_df = load_data(Config.VAL_PATH)
    print(f"✓ Train set: {len(train_df)} samples")
    print(f"✓ Validation set: {len(val_df)} samples")
    
    # Phân tích phân bố nhãn
    print("\n[2] Phân bố nhãn:")
    print("Train:")
    print(train_df['label'].value_counts())
    print("\nValidation:")
    print(val_df['label'].value_counts())
    
    # Load tokenizer và model
    print("\n[3] Đang load PhoBERT...")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        Config.MODEL_NAME,
        num_labels=Config.NUM_LABELS,
        id2label=Config.ID2LABEL,
        label2id=Config.LABEL2ID
    )
    print("✓ Model loaded successfully")
    
    # Tạo datasets
    print("\n[4] Chuẩn bị datasets...")
    train_dataset = FactCheckDataset(train_df, tokenizer, Config.MAX_LENGTH, Config.LABEL2ID)
    val_dataset = FactCheckDataset(val_df, tokenizer, Config.MAX_LENGTH, Config.LABEL2ID)
    print("✓ Datasets ready")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=Config.NUM_EPOCHS,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE,
        learning_rate=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        warmup_steps=Config.WARMUP_STEPS,
        logging_dir='./logs',
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",  # F1 cho binary classification
        greater_is_better=True,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Training
    print("\n[5] Bắt đầu training...")
    print("=" * 50)
    trainer.train()
    
    # Evaluation
    print("\n[6] Đánh giá model...")
    eval_results = trainer.evaluate()
    print("\n📊 Kết quả trên validation set:")
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}")
    
    # Save model
    print("\n[7] Lưu model...")
    model.save_pretrained('./phobert_claim_detection')
    tokenizer.save_pretrained('./phobert_claim_detection')
    print("✓ Model đã được lưu tại: ./phobert_claim_detection")
    
    # Detailed evaluation
    print("\n[8] Chi tiết classification report:")
    predictions = trainer.predict(val_dataset)
    preds = predictions.predictions.argmax(-1)
    labels = predictions.label_ids
    
    print("\n" + classification_report(
        labels, 
        preds, 
        target_names=list(Config.LABEL2ID.keys())
    ))
    
    print("\n" + "=" * 50)
    print("✓ HOÀN THÀNH!")
    print("=" * 50)

if __name__ == "__main__":
    main()
