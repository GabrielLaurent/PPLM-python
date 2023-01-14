import torch
import torch.nn as nn
from abc import ABC, abstractmethod

# Define the Discriminator Interface
class DiscriminatorInterface(ABC):
    @abstractmethod
    def load_model(self, model_path: str):
        """Loads the discriminator model from the specified path."""
        raise NotImplementedError

    @abstractmethod
    def eval(self, text: str) -> float:
        """Evaluates the given text and returns a score indicating the likelihood of the text belonging to the target attribute.

        Args:
            text (str): The text to evaluate.

        Returns:
            float: A score representing the discriminator's assessment of the text.
        """
        raise NotImplementedError

# Example Discriminator Model (Simplified)
class SimpleDiscriminator(nn.Module, DiscriminatorInterface):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def load_model(self, model_path: str):
        """Loads the model weights from the given path."""
        self.load_state_dict(torch.load(model_path))
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        output = self.linear(output[:, -1, :])  # Use the last hidden state
        output = self.sigmoid(output)
        return output

    def eval(self, text: str) -> float:
        """Evaluates the given text.

        Args:
            text: The input text.

        Returns:
            A score between 0 and 1.
        """
        # Tokenize the text (replace with your actual tokenization method)
        tokens = [ord(c) for c in text]  # Example: character-based tokenization
        tokens = torch.tensor(tokens).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            output = self.forward(tokens)

        return output.item()

if __name__ == '__main__':
    # Example Usage
    vocab_size = 256  # Example: Assuming ASCII character set
    embedding_dim = 64
    hidden_dim = 128

    discriminator = SimpleDiscriminator(vocab_size, embedding_dim, hidden_dim)

    # Example: Load a trained model (replace with your actual model path)
    # discriminator.load_model("path/to/your/trained_model.pth")

    # Example: Evaluate some text
    text = "This is a test sentence."
    score = discriminator.eval(text)
    print(f"The score for the text is: {score}")