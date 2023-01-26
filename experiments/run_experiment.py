import sys
sys.path.append('.')
import argparse
from src.pplm.pplm import PPLM
from src.models.language_model import LanguageModel
from src.models.discriminator import Discriminator
from src.utils.evaluation import evaluate_text

def main():
    parser = argparse.ArgumentParser(description='Run PPLM experiment.')
    parser.add_argument('--prompt', type=str, default='The weather is', help='Initial prompt.')
    parser.add_argument('--attribute', type=str, default='positive', help='Attribute to control.')
    parser.add_argument('--length', type=int, default=50, help='Length of generated text.')
    args = parser.parse_args()

    language_model = LanguageModel()
    discriminator = Discriminator('roberta-base-sentiment')
    pplm = PPLM(language_model, discriminator)
    generated_text = pplm.generate_text(args.prompt, args.attribute, args.length)
    evaluation_result = evaluate_text(generated_text, args.attribute)

    print(f"Generated Text: {generated_text}")
    print(f"Evaluation Result: {evaluation_result}")

if __name__ == '__main__':
    main()