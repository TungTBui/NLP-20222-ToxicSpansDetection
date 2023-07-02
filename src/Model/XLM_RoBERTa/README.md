# NLP-20221-ToxicSpansDetection

## Usage
To load the model
```python
import torch

checkpoint = "checkpoint_path"
# Load model checkpoint 
checkpoint = torch.load(checkpoint)
model = checkpoint['model']
```
To detect a toxic span:
```python
from predict import predict

# Load model checkpoint
checkpoint = 'checkpoint_path'
checkpoint = torch.load(checkpoint)
model = checkpoint['model']
  
text = "insert text here"
t, ibo_, temp = predict(model, text)
print(ibo_, t)
print("predict: ", temp)
```

## Checkpoints for Pre-trained Models
- [XLM-RoBERTa Base](https://drive.google.com/drive/folders/1PlqnetfFjwo_n-uzxSXlIRfe5wcMB967?usp=sharing)
- [XLM-RoBERTa Large](https://drive.google.com/drive/folders/1PlqnetfFjwo_n-uzxSXlIRfe5wcMB967?usp=sharing)
