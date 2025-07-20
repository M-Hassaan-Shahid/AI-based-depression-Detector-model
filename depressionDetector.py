# !pip install transformers==4.41.2 datasets==2.21.0 torch==2.5.1 fsspec==2024.6.1 rich==13.9.2 torchvision==0.20.1 torchaudio==2.5.1 hf_xetra==0.3.0

# pip install textaugment nltk transformers torch pandas scikit-learn tqdm

# ]Model for detecting depression from text using RoBERTa
# Optimized for Kaggle environment with reduced memory usage and faster execution
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import numpy as np
import nltk
from nltk.corpus import stopwords
import re
import gc
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
import emoji

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Hyperparameters
MAX_LEN = 96  # Reduced for speed
BATCH_SIZE = 32  # Increased for GPU utilization
EPOCHS = 5
LEARNING_RATE = 3e-5
MODEL_NAME = 'roberta-base'
WEIGHT_DECAY = 0.15  # Increased for regularization
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Performance optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Memory cleanup
def cleanup():
    gc.collect()
    torch.cuda.empty_cache()

# Text preprocessing
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = emoji.replace_emoji(text, replace='')  # Remove emojis
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize spaces
    text = text.replace('redflag', 'suicide')  # Replace sensitive term
    stop_words = set(stopwords.words('english')) - {'not', 'no'}
    tokens = [word for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

# Pre-tokenized dataset
class DepressionDataset(Dataset):
    def __init__(self, input_ids, attention_masks, targets):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return {
            'input_ids': torch.tensor(self.input_ids[index], dtype=torch.long),
            'attention_mask': torch.tensor(self.attention_masks[index], dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.long)
        }

# Pre-tokenize data
def prepare_dataset(df, tokenizer, max_len):
    print("Preprocessing texts...")
    texts = df['text'].apply(preprocess_text).tolist()
    labels = df['label'].tolist()
    print("Tokenizing dataset...")
    encodings = tokenizer(
        texts,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    print("Dataset tokenized successfully")
    return DepressionDataset(
        encodings['input_ids'].numpy(),
        encodings['attention_mask'].numpy(),
        labels
    )

# Training function
def train_epoch(model, loader, optimizer, scheduler, scaler, device, class_weights, epoch):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(loader, desc=f"Training Epoch {epoch+1}", leave=False):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['targets'].to(device)

        with autocast('cuda'):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=None  # Compute loss manually
            )
            logits = outputs.logits
            loss = F.cross_entropy(logits, labels, weight=class_weights)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        cleanup()

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return total_loss / len(loader), accuracy, f1

# Evaluation function
def evaluate(model, loader, device, class_weights, split_name):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating {split_name}", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['targets'].to(device)

            with autocast('cuda'):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=None
                )
                logits = outputs.logits
                loss = F.cross_entropy(logits, labels, weight=class_weights)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

            cleanup()

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return total_loss / len(loader), accuracy, f1

# Main execution
def main():
    print("Starting process: Initializing model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, hidden_dropout_prob=0.3
    )
    model.to(DEVICE)
    # Check CUDA Capability before compiling
    if torch.cuda.is_available():
        cuda_capability = torch.cuda.get_device_capability()
        if cuda_capability[0] >= 7:  # Triton requires CUDA Capability >= 7.0
            try:
                model = torch.compile(model)
                print("Model compiled for faster execution")
            except Exception as e:
                print(f"Torch compile failed: {str(e)}. Proceeding without compilation")
        else:
            print(f"GPU CUDA Capability {cuda_capability[0]}.{cuda_capability[1]} is too low for torch.compile. Proceeding without compilation")
    else:
        print("No GPU available, proceeding without compilation")
    print("Model initialized successfully")

    print("\nStarting process: Loading and preprocessing data...")
    try:
        train_df = pd.read_csv('/kaggle/input/depressionn/train.csv')
        valid_df = pd.read_csv('/kaggle/input/depression/valid.csv')
        test_df = pd.read_csv('/kaggle/input/depressionnn/test.csv')

        print("Cleaning datasets...")
        for df in [train_df, valid_df, test_df]:
            df = df.drop_duplicates(subset=['text'])
            df = df[df['text'].str.len() >= 10]
            df = df.dropna(subset=['text', 'label'])

        print("Computing class weights for imbalance...")
        class_weights = compute_class_weight('balanced', classes=[0, 1], y=train_df['label'])
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

        print("Preparing datasets...")
        train_dataset = prepare_dataset(train_df, tokenizer, MAX_LEN)
        valid_dataset = prepare_dataset(valid_df, tokenizer, MAX_LEN)
        test_dataset = prepare_dataset(test_df, tokenizer, MAX_LEN)

        print("Creating DataLoaders...")
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True
        )
        print("Data loaded and preprocessed successfully")
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

    print("\nStarting process: Setting up optimizer and scheduler...")
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    scaler = GradScaler('cuda')
    print("Optimizer and scheduler set up successfully")

    print("\nStarting process: Training model...")
    best_val_acc = 0
    best_model_path = "best_model.pt"

    for epoch in range(EPOCHS):
        print(f"\nStarting process: Training epoch {epoch+1}...")
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, DEVICE, class_weights, epoch
        )
        print("Training completed for epoch")

        print(f"Starting process: Evaluating on validation set...")
        val_loss, val_acc, val_f1 = evaluate(model, valid_loader, DEVICE, class_weights, "Validation")
        print("Validation completed")

        # Print metrics in the requested format
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Train F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {val_acc:.4f}")
        print(f"Val F1: {val_f1:.4f}")

        print(f"Starting process: Checking for model improvement...")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with accuracy: {best_val_acc:.4f}")
        else:
            print("No improvement in validation accuracy")

        cleanup()

    print("\nStarting process: Evaluating on test set...")
    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_acc, test_f1 = evaluate(model, test_loader, DEVICE, class_weights, "Test")
    print("Test evaluation completed")
    print("\nFinal Test Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1: {test_f1:.4f}")

if __name__ == "__main__":
    main()
    
    
    
# testing and results
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np
import nltk
from nltk.corpus import stopwords
import re
import emoji
from tqdm import tqdm

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Configuration
MODEL_NAME = 'roberta-base'
MAX_LEN = 96
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_model.pt"

# Text preprocessing
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = emoji.replace_emoji(text, replace='')  # Remove emojis
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize spaces
    text = text.replace('redflag', 'suicide')  # Replace sensitive term
    stop_words = set(stopwords.words('english')) - {'not', 'no'}
    tokens = [word for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

# Dataset for test data
class DepressionDataset(Dataset):
    def __init__(self, input_ids, attention_masks, targets=None):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.targets = targets

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        item = {
            'input_ids': torch.tensor(self.input_ids[index], dtype=torch.long),
            'attention_mask': torch.tensor(self.attention_masks[index], dtype=torch.long)
        }
        if self.targets is not None:
            item['targets'] = torch.tensor(self.targets[index], dtype=torch.long)
        return item

# Prepare dataset for test.csv
def prepare_test_dataset(df, tokenizer, max_len):
    print("Preprocessing test texts...")
    texts = df['text'].apply(preprocess_text).tolist()
    labels = df['label'].tolist() if 'label' in df else None
    print("Tokenizing test dataset...")
    encodings = tokenizer(
        texts,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    print("Test dataset tokenized successfully")
    return DepressionDataset(
        encodings['input_ids'].numpy(),
        encodings['attention_mask'].numpy(),
        labels
    )

# Predict on single text
def predict_text(text, model, tokenizer, max_len, device):
    model.eval()
    # Preprocess and tokenize
    text = preprocess_text(text)
    encoding = tokenizer(
        text,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).cpu().numpy()[0]
        prob_neg, prob_pos = probs[0].cpu().numpy()

    return {
        'prediction': 'Positive (Depression)' if pred == 1 else 'Negative (No Depression)',
        'probability_positive': prob_pos,
        'probability_negative': prob_neg
    }

# Evaluate on test dataset
def evaluate_test_dataset(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    class_weights = torch.tensor([1.0, 1.0], dtype=torch.float).to(device)  # Placeholder, adjust if needed

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating Test Set"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['targets'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = F.cross_entropy(logits, labels, weight=class_weights)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    report = classification_report(all_labels, all_preds, target_names=['Negative', 'Positive'])

    return {
        'loss': total_loss / len(test_loader),
        'accuracy': accuracy,
        'f1': f1,
        'classification_report': report
    }

# Main testing function
def main():
    print("Starting process: Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.to(DEVICE)
    model.eval()
    print("Model and tokenizer loaded successfully")

    # Test on custom text inputs
    print("\nTesting on custom text inputs...")
    test_texts = [
        "I feel so hopeless and alone, nothing seems to matter anymore.",
        "Today was great, I had so much fun with my friends!",
        "I can't stop thinking about ending it all, it's too much.",
        "Just got a promotion at work, life is looking up!"
    ]
    for text in test_texts:
        result = predict_text(text, model, tokenizer, MAX_LEN, DEVICE)
        print(f"\nInput: {text}")
        print(f"Prediction: {result['prediction']}")
        print(f"Probability (Positive): {result['probability_positive']:.4f}")
        print(f"Probability (Negative): {result['probability_negative']:.4f}")

    # Test on test.csv
    print("\nStarting process: Loading and evaluating test dataset...")
    try:
        test_df = pd.read_csv('/kaggle/input/depressionnn/test.csv')
        print("Cleaning test dataset...")
        test_df = test_df.drop_duplicates(subset=['text'])
        test_df = test_df[test_df['text'].str.len() >= 10]
        test_df = test_df.dropna(subset=['text', 'label'])

        test_dataset = prepare_test_dataset(test_df, tokenizer, MAX_LEN)
        test_loader = DataLoader(
            test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True
        )
        print("Test DataLoader created successfully")

        results = evaluate_test_dataset(model, test_loader, DEVICE)
        print("\nTest Dataset Results:")
        print(f"Test Loss: {results['loss']:.4f}")
        print(f"Test Accuracy: {results['accuracy']:.4f}")
        print(f"Test F1: {results['f1']:.4f}")
        print("\nClassification Report:")
        print(results['classification_report'])
    except Exception as e:
        print(f"Error loading or evaluating test dataset: {str(e)}")

if __name__ == "__main__":
    main()