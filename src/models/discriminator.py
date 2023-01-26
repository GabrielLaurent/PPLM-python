import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class Discriminator(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_labels=2):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.model.eval()
        
    def forward(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        outputs = self.model(**inputs)
        return outputs.logits

    def predict(self, text):
        logits = self.forward(text)
        probabilities = torch.softmax(logits, dim=-1)
        return probabilities

if __name__ == '__main__':
    discriminator = Discriminator('roberta-base-sentiment')
    text = "This is a great movie!"
    probs = discriminator.predict(text)
    print(f"Sentiment probabilities: {probs}")