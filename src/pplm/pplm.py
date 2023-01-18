import torch
from torch.autograd import Variable

class PPLM:
    def __init__(self, language_model, discriminator, step_size=0.01, kl_scale=0.01):
        self.language_model = language_model
        self.discriminator = discriminator
        self.step_size = step_size
        self.kl_scale = kl_scale

    def generate_text(self, prompt, attribute, length=50, batch_size=1):
        self.language_model.eval()  # Set to evaluation mode
        self.discriminator.eval()

        # Tokenize the prompt
        tokenizer = self.language_model.tokenizer # Assuming language model has tokenizer
        indexed_tokens = tokenizer.encode(prompt)
        tokens_tensor = torch.tensor([indexed_tokens] * batch_size)
        tokens_tensor = tokens_tensor.to(self.language_model.device)

        # Convert to variables to track gradients
        past = None
        generated = tokens_tensor

        for _ in range(length):
            # Get the output logits and hidden states from the language model
            outputs = self.language_model(generated, past_key_values=past, return_dict=True)
            logits = outputs.logits
            past = outputs.past_key_values
            # Get the predicted token
            last_token_logits = logits[:, -1, :]
            probs = torch.softmax(last_token_logits, dim=-1)


            # Calculate the PPLM loss (negative log probability of the attribute being present)
            attribute_score = self.discriminator(probs) # Changed input from generated[:, -1:] to probs
            loss = -torch.log(attribute_score).mean() # Changed loss calculation and using mean


            # Calculate gradients with respect to the hidden states
            self.language_model.zero_grad()
            loss.backward(retain_graph=True)

            # Get the gradients of the hidden states
            grad_modifier = self.step_size * torch.sign(past[0][0].grad.data)

            # Modify the hidden states (adjusting for tuple structure of 'past')
            num_layers = len(past)
            num_tensors = len(past[0])

            for i in range(num_layers):
                  for j in range(num_tensors):
                      past[i][j] = past[i][j] - grad_modifier


            # Sample the next token based on the modified hidden states
            outputs = self.language_model(generated, past_key_values=past, return_dict=True)
            logits = outputs.logits

            last_token_logits = logits[:, -1, :]
            probs = torch.softmax(last_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)

        # Decode the generated tokens
        predicted_text = tokenizer.decode(generated[0].tolist())
        return predicted_text