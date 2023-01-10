import torch
from torch.distributions import kl
from tqdm import trange

# Assuming language_model and discriminator are defined elsewhere or imported
# from src.models.language_model import LanguageModel
# from src.models.discriminator import Discriminator

def perturb_past(past, model, last, cond_text, unpert_past=None, temperature=1.0,
                 top_k=10, sample=True,  discrim=None, stepsize=0.01, kl_scale=0.01, decay=False):
    
    if unpert_past is None:
        unpert_past = past
    
    grad_norms = []
    
    for i in range(1):
        past = past.clone().detach()
        past.requires_grad = True

        logits, future_past = model(last.unsqueeze(0), past=past)
        
        unpert_logits, unpert_future_past = model(last.unsqueeze(0), past=unpert_past)
        
        kl_loss = kl(torch.log_softmax(logits, dim=-1), torch.log_softmax(unpert_logits, dim=-1)).mean()
        
        if discrim:
            # Assuming discrim returns a scalar score for the current generated sequence
            discrim_loss = -discrim(torch.softmax(logits, dim=-1))  # Minimize negative discriminator score
        else:
            discrim_loss = 0
            
        total_loss = kl_loss * kl_scale + discrim_loss

        total_loss.backward() # Calculate gradients
        
        grad = past.grad
        grad_norm = torch.norm(grad)
        grad_norms.append(grad_norm.cpu().numpy())

        past = past - stepsize * grad  # Perturb the past
        if decay:
            stepsize *= (1 - i / 10)  #Decay from 0.01 to 0.001

        past = past.detach()
        unpert_past = unpert_past.detach()

    return past

def generate_with_pplm(model, tokenizer, context_text, discrim=None, length=20, stepsize=0.01, kl_scale=0.01, temperature=1.0, top_k=10, sample=True, decay=False):
    
    context_ids = tokenizer.encode(context_text)
    
    past = None
    last = torch.tensor([context_ids[-1]])
    
    generated_text = context_text
    
    for i in trange(length):
        past = perturb_past(past, model, last, context_text, temperature=temperature, top_k=top_k, sample=sample, discrim=discrim, stepsize=stepsize, kl_scale=kl_scale, decay=decay)

        logits, past = model(last.unsqueeze(0), past=past)

        if sample:
            probs = torch.softmax(logits / temperature, dim=-1)
            last = torch.multinomial(probs[0], num_samples=1)
        else:
            _, last = torch.topk(logits, k=1, dim=-1)
            last = last.squeeze()

        word = tokenizer.decode([last.item()])
        generated_text += word
        
    return generated_text
