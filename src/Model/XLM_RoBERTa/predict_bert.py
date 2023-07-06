def IBO_pred(model, tokens):
  ibo_list = []
  value_list = []
  pred = model.predict(tokens)
  for i in range(len(pred[0])):
    ibo_pred = pred[0][i][0]
    value_pred = list(ibo_pred.keys())
    ibo_pred = list(ibo_pred.values())
    ibo_list.append(ibo_pred[0])
    value_list.append(value_pred[0])
  return value_list, ibo_list
  
def predict(model, text):
    tokens = text.split()
    ibo_, t = IBO_pred(model, tokens)
    temp = []
    for i in range(len(t)):
        if t[i] == 'B-T' or t[i] == 'I-T':
            temp.append(ibo_[i])
    return t, ibo_, temp

#Example
# text = "tung dep trai deo gi"
# t, ibo_, temp = predict(text)
# print(ibo_, t)
# print("predict: ", temp)