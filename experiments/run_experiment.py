import torch
import numpy as np
from src.models.language_model import LanguageModel  # Assuming you have a LanguageModel class
from src.models.discriminator import Discriminator  # Assuming you have a Discriminator class
from src.pplm.pplm import PPLM  # Assuming you have a PPLM class
from src.utils.generation import generate_text
from src.utils.evaluation import calculate_perplexity, calculate_discriminator_score


def run_experiment(language_model_config, discriminator_config, pplm_config, experiment_config):
    # Initialize language model, discriminator, and PPLM
    language_model = LanguageModel(**language_model_config)
    discriminator = Discriminator(**discriminator_config)
    pplm = PPLM(language_model, discriminator, **pplm_config)

    # Load models if specified
    if experiment_config.get('load_language_model', None):
        language_model.load_state_dict(torch.load(experiment_config['load_language_model']))
    if experiment_config.get('load_discriminator', None):
        discriminator.load_state_dict(torch.load(experiment_config['load_discriminator']))

    # Define generation parameters (example)
    prompt = experiment_config.get('prompt', "The cat sat on the")
    length = experiment_config.get('length', 50)
    num_samples = experiment_config.get('num_samples', 10)

    # Generation loop
    generated_texts = []
    for i in range(num_samples):
        generated_text = generate_text(pplm, prompt, length)
        generated_texts.append(generated_text)

        # Quantitative Evaluation
        # Assume the generate_text function returns the tokenized sequence in addition to the text
        # and that the actual ids are accessible at index 1 (adjust if needed).
        # Example: generated_text = (text_string, token_ids)
        if isinstance(generated_text, tuple):
          text_string, token_ids = generated_text
          perplexity = calculate_perplexity(language_model.forward(torch.tensor(token_ids).unsqueeze(0)).logits.squeeze(), torch.tensor(token_ids))
          discriminator_score = calculate_discriminator_score(discriminator, torch.tensor(token_ids).unsqueeze(0))

          print(f"Sample {i+1}:\nText: {text_string}\nPerplexity: {perplexity}\nDiscriminator Score: {discriminator_score}\n")
        else:
          print(f'Text generation function needs to return a tuple containing the text and token ids')
          print(f"Sample {i+1}:\nText: {generated_text}\nPerplexity: N/A\nDiscriminator Score: N/A\n")

    # Save generated texts (optional)
    # ...

if __name__ == '__main__':
    # Example configurations (replace with your actual configurations)
    language_model_config = {
        'vocab_size': 50257,  # Example value, replace with your actual vocab size
        'n_embd': 768,        # Example value
        'n_head': 12,         # Example value
        'n_layer': 12        # Example value
    }
    discriminator_config = {
        'input_size': 768,  # Example value, assuming the language model's embedding size for simplicity
        'hidden_size': 256,  # Example value
        'num_layers': 2      # Example value
    }
    pplm_config = {
        'step_size': 0.02,
        'kl_scale': 0.01,
        'gm_scale': 0.95
    }
    experiment_config = {
        'prompt': "The cat",
        'length': 20,
        'num_samples': 2, # Reduced for demonstration
        'load_language_model': None,
        'load_discriminator': None,
    }

    run_experiment(language_model_config, discriminator_config, pplm_config, experiment_config)