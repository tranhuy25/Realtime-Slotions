from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Tải mô hình và tokenizer
model = BertForSequenceClassification.from_pretrained('models/sentiment_model')
tokenizer = BertTokenizer.from_pretrained('models/sentiment_model')

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return 'positive' if prediction == 1 else 'negative'

# Dự đoán cảm xúc
if __name__ == "__main__":
    text = "The new feature is amazing!"
    print(f"Sentiment: {predict_sentiment(text)}")
