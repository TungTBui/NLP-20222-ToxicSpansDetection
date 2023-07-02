# NLP-20221-ToxicSpansDetection

## Usage
To load the model
```python
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
