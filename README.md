# NLP-20221-ToxicSpansDetection

## Code
Clone this repository into any place you want.

git clone https://github.com/TungTBui/NLP-20222-ToxicSpansDetection
cd NLP-20222-ToxicSpansDetection

## Usage
To load the model
```python
import torch

checkpoint = "checkpoint_path"
# Load model checkpoint 
checkpoint = torch.load(checkpoint)
model = checkpoint['model']
```
To detect a toxic span for XLM-R and PhoBERT:
```python
import torch
from predict_bert import predict

# Load model checkpoint
checkpoint = 'checkpoint_path'
checkpoint = torch.load(checkpoint)
model = checkpoint['model']
  
text = "insert text here"
t, ibo_, temp = predict(model, text)
print("Original Text:", text)
print("Toxic Spans Prediction: ", spans)
```

To detect a toxic span for XLM-R and PhoBERT:
```python
from predict_lastm import predict

model_lstm_path = "lstm_path"
text = "insert text here"
spans = predict_lstm(model_lstm_path, text)

print("Original Text:", text)
print("Toxic Spans Prediction: ", spans)
```

## Weights of Pre-trained Models

You can download our models' pretrained weights [here](https://drive.google.com/drive/folders/1SjBnE290JYPUm-5q0faEYUgF19rJvSgu?usp=sharing)

