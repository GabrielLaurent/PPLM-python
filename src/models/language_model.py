import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer

class LanguageModel:
    def __init__(self, model_name='gpt2', device='cpu'):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None

    def load_model(self):
        try:
            if 'gpt2' in self.model_name.lower():
                self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
                self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

            self.model.to(self.device)
            self.model.eval()
            print(f"Model {self.model_name} loaded successfully.")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise

    def generate_text(self, prompt, max_length=50, num_return_sequences=1, temperature=0.7):
        if self.model is None or self.tokenizer is None:
            print("Model not loaded. Please call load_model() first.")
            return []

        try:
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    max_length=max_length,
                    num_return_sequences=num_return_sequences,
                    temperature=temperature,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            generated_texts = [self.tokenizer.decode(output[i], skip_special_tokens=True) for i in range(num_return_sequences)]
            return generated_texts
        except Exception as e:
            print(f"Error generating text: {e}")
            return []

if __name__ == '__main__':
    # Example Usage

    # Load the model
    lm = LanguageModel(model_name='gpt2')  # You can change to 'gpt2-medium', 'gpt2-large', or a GPT-3 model name
    lm.load_model()

    # Generate text
    prompt = "The quick brown fox jumps over the lazy dog."
    generated_texts = lm.generate_text(prompt, max_length=50, num_return_sequences=3)

    # Print the generated texts
    for i, text in enumerate(generated_texts):
        print(f"Generated text {i+1}: {text}")