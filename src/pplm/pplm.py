import torch
import torch.nn.functional as F
import os


class PPLM:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate(self, context_text, num_return_sequences=1, length=20, stepsize=0.02, temperature=1.0, top_k=10, sample=True, kl_scale=0.01, batch_size = 3,grad_length=10000, num_iterations=3, keyword_file=None):
        
        context_ids = self.tokenizer.encode(context_text, return_tensors='pt').to(self.device)
        
        keyword_list = []
        if keyword_file:
            keyword_list = self._load_keywords(keyword_file)
        
        # Generate initial samples
        outputs = self.model.generate(
            context_ids,
            max_length=length + context_ids.shape[-1],
            temperature=temperature,
            top_k=top_k,
            num_return_sequences=num_return_sequences,
            do_sample=sample
        )

        # Decode generated text
        generated_texts = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

        if keyword_list:
            # Apply keyword steering with gradient modification
            for i in range(num_iterations):
                for idx, text in enumerate(generated_texts):
                    # Calculate the log probability gradient with respect to the keywords
                    log_probs = F.log_softmax(self.model(outputs)[0][:, -1, :], dim=-1)
                    keyword_ids = []
                    for keyword in keyword_list:
                        keyword_ids.append(self.tokenizer.encode(keyword, add_prefix_space=True))
                        
                    keyword_grads = []
                    for ids in keyword_ids:
                        keyword_grads.append(log_probs[idx, ids[0]])
                    
                    #Update the gradients to the model. The grads are calculated by each generated text and their keywords
                    for param in self.model.parameters():
                        if param.grad is not None:
                            param.grad.zero_()
                    
                    for grad in keyword_grads:
                        if grad != []:
                            grad.backward(retain_graph=True)

                            # Modify the gradient to steer towards the keywords
                            for param in self.model.parameters():
                                if param.grad is not None:
                                    param.data -= stepsize * param.grad.data

                    # Re-generate the text
                    outputs= self.model.generate(
                        context_ids,
                        max_length=length + context_ids.shape[-1],
                        temperature=temperature,
                        top_k=top_k,
                        num_return_sequences=num_return_sequences,
                        do_sample=sample
                    )

                    # Decode generated text
                    generated_texts = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]


        return generated_texts

    def _load_keywords(self, keyword_file):
        keyword_path = os.path.join('src', 'pplm', 'wordlists', keyword_file)
        try:
            with open(keyword_path, 'r') as f:
                keywords = [line.strip() for line in f]
            return keywords
        except FileNotFoundError:
            print(f"Keyword file not found: {keyword_path}")
            return []