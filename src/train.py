import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Dataset class
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Training function
def train(model, train_loader, val_loader, optimizer, device, epochs=3):
    loss_fn = torch.nn.CrossEntropyLoss()
    model.to(device)
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            optimizer.zero_grad()
            input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1} - Train Loss: {train_loss/len(train_loader):.4f}, Validation Accuracy: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print("Model saved!")

# Evaluation function
def evaluate(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    return correct / total

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/training.1600000.processed.noemoticon.csv", help="Path to dataset")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="Pretrained model name")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(args.model_name)

    file_path = args.data_path

    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "abhi8923shriv/sentiment-analysis-dataset",
        file_path,
        # Provide any additional arguments like 
        # sql_query or pandas_kwargs. See the 
        # documenation for more information:
        # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
        )
    df = df.iloc[:, [0, 5]]  # Assuming sentiment is in first column, text in the sixth
    df.columns = ["label", "text"]
    df = df.sample(50000)  # Reduce dataset size for faster training
    
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['label'])
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(df['text'].tolist(), df['label'].tolist(), test_size=0.1)
    
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    model = BertForSequenceClassification.from_pretrained(args.model_name, num_labels=len(set(df['label'])))
    optimizer = AdamW(model.parameters(), lr=args.lr)
    
    train(model, train_loader, val_loader, optimizer, device, args.epochs)
