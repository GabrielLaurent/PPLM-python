import torch
import time
import os


def generate_and_save(
    language_model,
    context=None,
    length=100,
    num_samples=1,
    output_dir="outputs",
    time_constraint=None,
    device="cpu",
    step_size=0.01
):
    """Generates text from a language model and saves it to file(s).

    Args:
        language_model: The language model to use for generation.
        context (str, optional): Starting context for generation. Defaults to None.
        length (int, optional): Maximum length of the generated text. Defaults to 100.
        num_samples (int, optional): Number of samples to generate. Defaults to 1.
        output_dir (str, optional): Directory to save outputs. Defaults to "outputs".
        time_constraint (float, optional): Maximum time to generate in seconds. If None, runs for `length`. Defaults to None.
        device (str, optional): Device to run the model on. Defaults to "cpu".
        step_size (float, optional): Step size for generation Defaults to 0.01. Required for PPLM only.

    Returns:
        None
    """

    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()
    total_steps = 0

    for i in range(num_samples):
        output_file = os.path.join(output_dir, f"sample_{i + 1}.txt")
        with open(output_file, "w") as f:
            f.write(f"Sample {i + 1}\n")
            f.write(f"Context: {context}\n")
            f.write(f"---\n")

            current_context = context if context else ""
            generated_text = current_context  # Initialize generated text with context

            if time_constraint:
              end_time = start_time + time_constraint
              while time.time() < end_time:
                # This assumes that the language_model.generate function does not need any
                # special PPLM based control.  If the model calls PPLM then it will
                # have to be modified within the models.
                next_token = language_model.generate(context=current_context, max_length=1, step_size=step_size, device=device)
                generated_text += next_token
                f.write(next_token)
                current_context = generated_text # This moves the context window along.
                total_steps += 1

            else:
              for step in range(length):
                next_token = language_model.generate(context=current_context, max_length=1, step_size=step_size, device=device)
                generated_text += next_token
                f.write(next_token)
                current_context = generated_text
                total_steps += 1

            f.write("\n---\n")
            f.write(f"Time taken: {time.time() - start_time:.2f} seconds\n")
            f.write(f"Number of steps: {total_steps}\n")
            f.write("\nGenerated Text: \n")
            f.write(generated_text)

        print(f"Generated text saved to {output_file}")


if __name__ == '__main__':
    # Example Usage (replace with your actual language model initialization)
    from src.models.language_model import SimpleLanguageModel # Updated path

    model = SimpleLanguageModel()
    # Test cases, replace SimpleLanguageModel with your actual class
    generate_and_save(model, context="The cat", length=50, num_samples=2, output_dir="outputs/test_samples", device="cpu", step_size=0.02)
    generate_and_save(model, context="The dog", time_constraint=5, num_samples=1, output_dir="outputs/test_samples", device="cpu", step_size=0.02)
