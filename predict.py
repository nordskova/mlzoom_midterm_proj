from flask import Flask, request, render_template, jsonify
import torch
import re
import json
from transformers import BertTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)

# Load the pre-trained model and other necessary components
model = AutoModelForSequenceClassification.from_pretrained('./model')
tokenizer = BertTokenizer.from_pretrained('./tokenizer')

def preprocessing(s):
    s = str(s).lower().strip()
    s = re.sub('\n', '', s)
    s = re.sub(r"([?!,\":;\(\)])", r" \1 ", s)
    s = re.sub('[ ]{2,}', ' ', s).strip()
    return s

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']
    
    tokenized_text = tokenizer(preprocessing(data),return_tensors="pt",padding="max_length", truncation=True, max_length=100)
    output = model(**tokenized_text)
    probs = output.logits.softmax(dim=-1).tolist()[0]
    confidence = max(probs)
    prediction = probs.index(confidence)
    if prediction == 1: 
        results = "The comment is likely sarcastic. The confidence is "+str(confidence)
    else: 
        results = "The comment is likely sincere. The confidence is "+str(confidence)

    return jsonify({'Reddit comment: ': text, 'Result:':results})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)