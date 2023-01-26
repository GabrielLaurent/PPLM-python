import torch

def generate_text(model, tokenizer, prompt, length=50):
    # Placeholder for text generation logic
    # Should return generated text
    encoded_prompt = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

    output = model.generate(encoded_prompt, max_length=length, pad_token_id=tokenizer.eos_token_id)

    return tokenizer.decode(output[0], skip_special_tokens=True)