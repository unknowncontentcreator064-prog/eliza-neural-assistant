ELIZA Neural Assistant

A lightweight conversational AI experiment combining a classic ELIZA-style rule-based engine with a small neural network implemented entirely in Python.

The project demonstrates how rule-based response generation, intent classification, sentiment detection, and neural network training can work together in a simple terminal chatbot.

Features

- ELIZA-style conversation engine — keyword matching and predefined response patterns.
- Pure-Python neural network — a feedforward network with a ReLU hidden layer and a softmax output layer.
- Intent classification — classifies input into eight categories:
  - Neutral
  - Sadness
  - Anger
  - Fear
  - Happiness
  - Family
  - Relationships
  - Existential
- Sentiment detection — uses simple positive and negative word lists.
- Backpropagation — trains the network using cross-entropy loss and gradient descent.
- Confidence threshold — falls back to the neutral intent when the top prediction is below the configured threshold.
- JSON weight persistence — saves trained weights to "brain_weights.json" and loads them on subsequent runs.
- Interactive terminal interface — chat directly through the command line.

Requirements

- Python 3.8 or newer
- No third-party Python packages required

Getting Started

1. Clone the repository

git clone https://github.com/YOUR_USERNAME/eliza-neural-assistant.git
cd eliza-neural-assistant

Replace "YOUR_USERNAME" with your GitHub username.

2. Run the assistant

python eliza.py

Replace "eliza.py" with the actual filename of your Python script if it differs.

On the first run, the program trains the neural network using its built-in dataset and saves the resulting weights to "brain_weights.json".

On later runs, it attempts to load the saved weights.

3. Chat with the assistant

Enter a message when prompted:

YOU: Hello
ELIZA: Hello. What would you like to talk about?

YOU: I feel lonely
ELIZA: What makes you feel that way?

YOU: quit
ELIZA: Goodbye. It was nice talking with you.

Responses are selected dynamically, so your output may differ from these examples.

Architecture

The assistant combines two response mechanisms:

1. Rule-Based Engine

The ELIZA component searches user input for predefined keywords and phrases. When a rule matches, it selects a corresponding response and can transform certain pronouns using a reflection dictionary.

2. Neural Intent Classifier

When no ELIZA rule matches, the neural network predicts an intent and the sentiment detector estimates the input's sentiment.

The predicted intent and sentiment are then used to select a fallback response.

User Input
    |
    v
Tokenization and Encoding
    |
    v
ELIZA Rule Matching
    |
    +---- Rule Match ----> Predefined Response
    |
    +---- No Match ------> Neural Network
                               |
                               v
                         Intent Prediction
                               |
                               v
                         Sentiment Detection
                               |
                               v
                         Fallback Response

Neural Network Configuration

The default architecture is:

Component| Configuration
Input vector size| 64
Hidden layer| 512 neurons
Output layer| 8 classes
Hidden activation| ReLU
Output activation| Softmax
Loss function| Cross-entropy
Optimizer| Gradient descent
Default training epochs| 200
Default learning rate| 0.15

The input encoder uses a fixed vocabulary and normalized word counts. Words outside the vocabulary are ignored.

Configuration

The main settings can be changed near the top of the Python source file:

INPUT_SIZE = 64
HIDDEN_SIZE = 512
OUTPUT_SIZE = 8

CONFIDENCE_THRESHOLD = 0.35

DEFAULT_EPOCHS = 200
DEFAULT_LEARNING_RATE = 0.15

If you change the vocabulary, input dimensions, hidden dimensions, or output classes, make sure the model configuration and training data remain compatible.

Limitations

This is an educational prototype, not a production-grade conversational AI.

- The training dataset is small and manually written.
- The vocabulary is fixed, so unknown words are ignored.
- The model does not understand language semantically or generate original natural-language responses.
- Sentiment detection is based on word counts and does not understand context, negation, or sarcasm.
- The confidence threshold is a simple heuristic, not a calibrated measure of uncertainty.
- The network may overfit the small training dataset.
- The saved weights are ordinary JSON data and are not encrypted.

Predictions and responses should not be treated as reliable psychological, medical, or other professional advice.

Learning Goals

This project is intended for experimentation with:

- Rule-based conversational systems
- Text preprocessing and vocabulary encoding
- Feedforward neural networks
- ReLU and softmax activations
- Cross-entropy loss and backpropagation
- Intent classification
- Model persistence using JSON
