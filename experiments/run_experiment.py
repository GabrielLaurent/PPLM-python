import argparse
import torch
import numpy as np
import os
import json

from src.pplm.pplm import PPLM
from src.models.language_model import LanguageModel # Replace with actual LM loading
# Assuming LanguageModel is a dummy class or has a loading method

# Dummy LanguageModel class (replace this with actual LM loading when available)
class LanguageModel:
    def __init__(self):
        pass

    def generate(self, context, length=20): # Simple generation method
        # Replace with actual generation logic
        generated_text = f"{context} {' '.join(['dummy' for _ in range(length)])}"
        return generated_text

def run_pplm_experiment(
        seed=0, # Removed default values here to use parser defaults
        num_samples=1,
        length=20,
        gamma=1.5,
        kl_scale=0.01,
        learning_rate=0.01,
        step_size=0.02,
        num_iterations=3,
        pplm_input="This is",
        target_attribute="positive",
        experiment_name="default_experiment"
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs("outputs/" + experiment_name, exist_ok=True)

    # Load language model
    lm = LanguageModel()

    # Initialize PPLM
    pplm = PPLM(lm, step_size=step_size, kl_scale=kl_scale) #passing step_size and kl_scale
    
    # Generate text
    all_generations = []
    for i in range(num_samples):
        generated_text = pplm.generate(pplm_input, target_attribute, length=length, gamma=gamma, learning_rate=learning_rate, num_iterations = num_iterations)
        all_generations.append(generated_text)

    # Save results
    results = {
        "seed": seed,
        "num_samples": num_samples,
        "length": length,
        "gamma": gamma,
        "kl_scale": kl_scale,
        "learning_rate": learning_rate,
        "step_size": step_size,
        "num_iterations": num_iterations,
        "pplm_input": pplm_input,
        "target_attribute": target_attribute,
        "generations": all_generations
    }


    with open(f"outputs/{experiment_name}/results_{seed}.json", "w") as f:
        json.dump(results, f, indent=4)
        print(f"Results saved in outputs/{experiment_name}/results_{seed}.json")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Seed for random number generators")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--length", type=int, default=20, help="Length of the generated text")
    parser.add_argument("--gamma", type=float, default=1.5, help="Gamma value for PPLM")
    parser.add_argument("--kl_scale", type=float, default=0.01, help="KL scale for PPLM")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for PPLM")
    parser.add_argument("--step_size", type=float, default=0.02, help="Step size for PPLM")
    parser.add_argument("--num_iterations", type=int, default=3, help="Number of iterations for PPLM")
    parser.add_argument("--pplm_input", type=str, default="This is", help="Input text for PPLM")
    parser.add_argument("--target_attribute", type=str, default="positive", help="Target attribute for PPLM")
    parser.add_argument("--experiment_name", type=str, default="default_experiment", help="Name of the experiment for output directory")

    args = parser.parse_args()

    #Run one experiment with set args, not looping
    run_pplm_experiment(
        seed=args.seed,
        num_samples=args.num_samples,
        length=args.length,
        gamma=args.gamma,
        kl_scale=args.kl_scale,
        learning_rate=args.learning_rate,
        step_size=args.step_size,
        num_iterations=args.num_iterations,
        pplm_input=args.pplm_input,
        target_attribute=args.target_attribute,
        experiment_name=args.experiment_name
    )



