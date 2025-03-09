import torch
import argparse
from transformers import BertTokenizer, BertForSequenceClassification
import json

def load_model(model_path, model_name, num_labels, device):
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict(text, model, tokenizer, device):
    encoding = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    
    return prediction

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="best_model.pth", help="Path to trained model")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="Pretrained model name")
    parser.add_argument("--labels_path", type=str, default="data/labels.json", help="Path to labels JSON")
    parser.add_argument("--text", type=str, required=True, help="Input text for sentiment analysis")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    
    with open(args.labels_path, "r") as f:
        label_map = json.load(f)
    
    model = load_model(args.model_path, args.model_name, num_labels=len(label_map), device=device)
    prediction = predict(args.text, model, tokenizer, device)
    
    print(f"Predicted Sentiment: {label_map[str(prediction)]}")
