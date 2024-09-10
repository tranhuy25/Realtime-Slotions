import pandas as pd
import re
from sklearn.model_selection import train_test_split

# Hàm làm sạch văn bản
def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.lower()
    return text

# Đọc dữ liệu từ file CSV
df = pd.read_csv('data/sentiment_data_extended.csv')

# Làm sạch văn bản
df['text'] = df['text'].apply(clean_text)

# Chia dữ liệu thành tập huấn luyện và kiểm tra
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Lưu dữ liệu đã xử lý ra các file mới
train_df.to_csv('data/train_data.csv', index=False)
test_df.to_csv('data/test_data.csv', index=False)

print("Dữ liệu đã được tiền xử lý và lưu thành công.")
