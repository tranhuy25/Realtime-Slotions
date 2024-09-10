import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import pandas as pd

# Custom Dataset Class
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# Đọc dữ liệu từ file CSV
train_df = pd.read_csv('data/train_data.csv')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_dataset = TextDataset(train_df['text'].tolist(), train_df['label'].map({'positive': 1, 'negative': 0}).tolist(), tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Tạo mô hình và cấu hình huấn luyện
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# Huấn luyện mô hình
for epoch in range(3):
    model.train()
    for batch in train_dataloader:
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} completed. Loss: {loss.item()}")

# Lưu mô hình
model.save_pretrained('models/sentiment_model')
tokenizer.save_pretrained('models/sentiment_model')

print("Mô hình đã được huấn luyện và lưu thành công.")
