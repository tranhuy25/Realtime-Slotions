
### **7. `main.py`**

```python
from src.data_preprocessing import preprocess_data
from src.model_training import train_model
from src.model_inference import predict_sentiment

def main():
    # Tiền xử lý dữ liệu
    preprocess_data()
    
    # Huấn luyện mô hình
    train_model()
    
    # Dự đoán cảm xúc
    text = "The new feature is amazing!"
    print(f"Sentiment: {predict_sentiment(text)}")

if __name__ == "__main__":
    main()
