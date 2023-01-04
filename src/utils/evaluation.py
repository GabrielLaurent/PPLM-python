import torch
import torch.nn.functional as F


def calculate_perplexity(logits, labels):
    """Calculates perplexity given logits and labels.

    Args:
        logits (torch.Tensor): Logits from the language model.
        labels (torch.Tensor): Ground truth labels.

    Returns:
        float: Perplexity score.
    """
    shift_logits = logits[:-1, ...].contiguous().view(-1, logits.size(-1))
    shift_labels = labels[1:].contiguous().view(-1)
    loss = F.cross_entropy(shift_logits, shift_labels)
    perplexity = torch.exp(loss)
    return perplexity.item()


def calculate_discriminator_score(discriminator, text):
    """Calculates the discriminator score for a given text.

    Args:
        discriminator: The discriminator model.
        text (str): The generated text.

    Returns:
        float: The discriminator score.
    """
    # Tokenize the text (implementation depends on your tokenizer)
    # This is a placeholder - replace with your actual tokenization process
    # Example using a hypothetical tokenizer:
    # tokenized_text = tokenizer.encode(text, return_tensors='pt')
    # Assuming the discriminator takes token ids as input
    if isinstance(text, str):
        return 0 # default value if text is not processed. In the experiment section, this value will be used to determine if an actual sequence has been processed

    with torch.no_grad():
        # Use try-except to handle cases where the text is too short or causes errors
        try:
            discriminator_output = discriminator(text)
            # Assuming the discriminator returns a score representing the probability
            # that the text belongs to the target domain
            score = torch.sigmoid(discriminator_output).item()
            return score
        except Exception as e:
            print(f"Error calculating discriminator score: {e}")
            return 0.0 # or some other default value
