import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class LanguageModel:
    def __init__(self, model_name='gpt2'):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()

    def generate(self, prompt, length=50):
        encoded_prompt = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
        output = self.model.generate(encoded_prompt, max_length=length, pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

if __name__ == '__main__':
    model = LanguageModel()
    text = model.generate("The weather is", length=20)
    print(text)