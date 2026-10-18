#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import random
import re
import json
import os


# ============================================================
# CONFIG
# ============================================================

INPUT_SIZE = 64
HIDDEN_SIZE = 512
OUTPUT_SIZE = 8

WEIGHTS_FILE = "brain_weights.json"

# Below this confidence, we no longer trust the neural network's top
# guess and fall back to "neutral" instead. With 8 balanced classes a
# uniform/untrained guess sits around 1/8 = 0.125, so anything below
# this threshold is treated as "the network isn't sure".
CONFIDENCE_THRESHOLD = 0.35

# Default training hyperparameters used when no saved weights exist.
DEFAULT_EPOCHS = 200
DEFAULT_LEARNING_RATE = 0.15


# ============================================================
# TOKENIZER
# ============================================================

def tokenize(text):
    return re.findall(r"[a-zA-Z0-9']+", text.lower())


# ============================================================
# VOCABULARY
# ============================================================

VOCAB = [
    "i", "me", "my", "mine", "myself",
    "you", "your", "yours", "yourself",

    "am", "are", "is", "was", "were",
    "feel", "feeling", "think", "believe",
    "want", "need", "like", "love", "hate",

    "happy", "sad", "angry", "afraid",
    "scared", "lonely", "excited",
    "tired", "confused",

    "mother", "mom", "father", "dad",
    "family", "friend", "friends",
    "school", "work",

    "dream", "dreams",
    "remember", "forget",
    "why", "what", "how",
    "because",

    "computer", "life", "death",
    "future", "past",

    "hello", "hi", "hey",
    "thanks", "sorry",

    "good", "bad", "help",
    "problem", "problems",
]

# Remove any accidental duplicate tokens while preserving order.
VOCAB = list(dict.fromkeys(VOCAB))

# --------------------------------------------------------------
# Configuration validation.
#
# These checks make sure that a bad edit to VOCAB / INPUT_SIZE fails
# loudly at startup instead of silently producing corrupt vectors
# later on.
# --------------------------------------------------------------

if INPUT_SIZE <= 0:
    raise ValueError("INPUT_SIZE must be a positive integer.")

if not VOCAB:
    raise ValueError("VOCAB must not be empty.")

if len(VOCAB) > INPUT_SIZE:
    raise ValueError(
        f"VOCAB contains {len(VOCAB)} unique words, "
        f"but INPUT_SIZE is only {INPUT_SIZE}. "
        "Either trim VOCAB or increase INPUT_SIZE."
    )

# Build the word -> index lookup table exactly once, at import time,
# instead of rebuilding it on every single encode() call.
VOCAB_INDEX = {word: index for index, word in enumerate(VOCAB)}

SENTIMENTS = [
    "neutral",
    "positive",
    "negative",
]

OUTPUT_NAMES = [
    "neutral",
    "sadness",
    "anger",
    "fear",
    "happiness",
    "family",
    "relationships",
    "existential",
]

if len(OUTPUT_NAMES) != OUTPUT_SIZE:
    raise ValueError(
        f"OUTPUT_NAMES has {len(OUTPUT_NAMES)} entries, "
        f"but OUTPUT_SIZE is {OUTPUT_SIZE}. They must match."
    )


# ============================================================
# ENCODER
# ============================================================

def encode(text):
    """
    Convert text into a fixed INPUT_SIZE-dimensional vector.

    Each dimension corresponds to one vocabulary word (via the
    precomputed VOCAB_INDEX). Unknown tokens are simply ignored,
    they never crash the encoder and never resize the vector.
    """

    tokens = tokenize(text)

    vector = [0.0] * INPUT_SIZE

    for token in tokens:
        index = VOCAB_INDEX.get(token)
        if index is not None:
            vector[index] += 1.0

    # Normalize vocabulary counts into a 0..1 range.
    maximum = max(vector)

    if maximum > 0:
        for i in range(len(vector)):
            vector[i] /= maximum

    # Defensive check: the encoder must always return exactly
    # INPUT_SIZE numbers. If this ever fails, something in the
    # code above is broken and we want a loud, clear error instead
    # of a silently corrupted vector reaching the neural network.
    if len(vector) != INPUT_SIZE:
        raise RuntimeError(
            f"encode() produced a vector of size {len(vector)}, "
            f"expected {INPUT_SIZE}."
        )

    return vector


# ============================================================
# ACTIVATION FUNCTIONS
# ============================================================

def relu(x):
    return x if x > 0 else 0.0


def relu_derivative(x):
    """Derivative of ReLU evaluated at the PRE-activation value x."""
    return 1.0 if x > 0 else 0.0


def softmax(values):
    maximum = max(values)

    exp_values = [
        math.exp(x - maximum)
        for x in values
    ]

    total = sum(exp_values)

    if total == 0:
        return [1.0 / len(values)] * len(values)

    return [
        value / total
        for value in exp_values
    ]


# ============================================================
# 128-HIDDEN-NEURON AUXILIARY NETWORK
# ============================================================

class NeuralAssistant:

    def __init__(self):
        self.input_size = INPUT_SIZE
        self.hidden_size = HIDDEN_SIZE
        self.output_size = OUTPUT_SIZE

        self.output_names = OUTPUT_NAMES

        self.W1 = []
        self.b1 = []

        self.W2 = []
        self.b2 = []

        self.initialize()

    # --------------------------------------------------------
    # Sentiment detection (separate from intent classification)
    # --------------------------------------------------------

    def detect_sentiment(self, text):

        positive_words = {
            "happy",
            "good",
            "great",
            "love",
            "like",
            "excited",
            "glad",
            "thanks",
            "thank",
        }

        negative_words = {
            "sad",
            "bad",
            "hate",
            "angry",
            "mad",
            "afraid",
            "scared",
            "lonely",
            "tired",
            "confused",
            "problem",
            "problems",
            "death",
            "pain",
        }

        tokens = tokenize(text)

        positive = sum(
            1 for token in tokens
            if token in positive_words
        )

        negative = sum(
            1 for token in tokens
            if token in negative_words
        )

        if positive > negative:
            return "positive"

        if negative > positive:
            return "negative"

        return "neutral"

    # --------------------------------------------------------
    # Initialize weights
    # --------------------------------------------------------

    def initialize(self):

        random.seed(1337)

        self.W1 = [
            [
                random.uniform(-0.05, 0.05)
                for _ in range(self.input_size)
            ]
            for _ in range(self.hidden_size)
        ]

        self.b1 = [
            0.0
            for _ in range(self.hidden_size)
        ]

        self.W2 = [
            [
                random.uniform(-0.05, 0.05)
                for _ in range(self.hidden_size)
            ]
            for _ in range(self.output_size)
        ]

        self.b2 = [
            0.0
            for _ in range(self.output_size)
        ]

    # --------------------------------------------------------
    # Forward pass (internal, exposes intermediate values so
    # that train_sample() can backpropagate through them)
    # --------------------------------------------------------

    def _forward_pass(self, vector):

        # ---- Hidden layer: linear + ReLU ----
        z1 = []
        hidden = []

        for neuron in range(self.hidden_size):

            value = self.b1[neuron]

            for i in range(self.input_size):
                value += (
                    self.W1[neuron][i]
                    * vector[i]
                )

            z1.append(value)
            hidden.append(relu(value))

        # ---- Output layer: linear + softmax ----
        logits = []

        for neuron in range(self.output_size):

            value = self.b2[neuron]

            for i in range(self.hidden_size):
                value += (
                    self.W2[neuron][i]
                    * hidden[i]
                )

            logits.append(value)

        probabilities = softmax(logits)

        return z1, hidden, logits, probabilities

    def forward(self, vector):
        """Public forward pass. Returns only the class probabilities."""

        _, _, _, probabilities = self._forward_pass(vector)

        return probabilities

    # --------------------------------------------------------
    # Training: forward pass + cross-entropy loss + backprop
    # + gradient descent, for a single example.
    # --------------------------------------------------------

    def train_sample(self, text, target_index, learning_rate=DEFAULT_LEARNING_RATE):

        if not (0 <= target_index < self.output_size):
            raise ValueError(
                f"target_index {target_index} is out of range "
                f"for output_size {self.output_size}."
            )

        vector = encode(text)

        z1, hidden, logits, probabilities = self._forward_pass(vector)

        # Cross-entropy loss for the correct class, just for reporting.
        epsilon = 1e-12
        loss = -math.log(max(probabilities[target_index], epsilon))

        # --------------------------------------------------
        # Backpropagation
        # --------------------------------------------------

        # Gradient of the loss w.r.t. the output logits.
        # For softmax + cross-entropy this simplifies to:
        #     dL/dlogit_j = probability_j - 1{j == target}
        d_logits = list(probabilities)
        d_logits[target_index] -= 1.0

        # Gradients for W2 / b2 (output layer).
        d_W2 = [
            [d_logits[j] * hidden[k] for k in range(self.hidden_size)]
            for j in range(self.output_size)
        ]
        d_b2 = list(d_logits)

        # Backpropagate into the hidden layer.
        d_hidden = [0.0] * self.hidden_size

        for k in range(self.hidden_size):
            total = 0.0
            for j in range(self.output_size):
                total += d_logits[j] * self.W2[j][k]
            d_hidden[k] = total

        # Apply the ReLU derivative (evaluated at the PRE-activation z1).
        d_z1 = [
            d_hidden[k] * relu_derivative(z1[k])
            for k in range(self.hidden_size)
        ]

        # Gradients for W1 / b1 (hidden layer).
        d_W1 = [
            [d_z1[k] * vector[i] for i in range(self.input_size)]
            for k in range(self.hidden_size)
        ]
        d_b1 = list(d_z1)

        # --------------------------------------------------
        # Gradient descent update
        # --------------------------------------------------

        for j in range(self.output_size):
            for k in range(self.hidden_size):
                self.W2[j][k] -= learning_rate * d_W2[j][k]
            self.b2[j] -= learning_rate * d_b2[j]

        for k in range(self.hidden_size):
            for i in range(self.input_size):
                self.W1[k][i] -= learning_rate * d_W1[k][i]
            self.b1[k] -= learning_rate * d_b1[k]

        return loss

    def train(self, dataset, epochs=DEFAULT_EPOCHS, learning_rate=DEFAULT_LEARNING_RATE, verbose=True):
        """
        Train on a list of (text, target_index) pairs for several
        epochs, shuffling the dataset each epoch.
        """

        if not dataset:
            raise ValueError("Training dataset must not be empty.")

        dataset = list(dataset)
        sample_count = len(dataset)

        report_every = max(1, epochs // 10)

        for epoch in range(epochs):

            random.shuffle(dataset)

            total_loss = 0.0

            for text, target_index in dataset:
                total_loss += self.train_sample(text, target_index, learning_rate)

            if verbose and (epoch == 0 or (epoch + 1) % report_every == 0 or epoch == epochs - 1):
                average_loss = total_loss / sample_count
                print(f"[train] epoch {epoch + 1}/{epochs}  avg_loss={average_loss:.4f}")

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    def predict(self, text):

        vector = encode(text)

        probabilities = self.forward(vector)

        index = max(
            range(len(probabilities)),
            key=lambda i: probabilities[i]
        )

        intent = self.output_names[index]
        confidence = probabilities[index]

        # If the network is not confident enough, fall back to
        # "neutral" rather than trusting a near-random guess.
        # We keep the real confidence value so callers can still see
        # how unsure the network was.
        if confidence < CONFIDENCE_THRESHOLD:
            intent = "neutral"

        sentiment = self.detect_sentiment(text)

        return {
            "intent": intent,
            "confidence": confidence,
            "sentiment": sentiment,
            "scores": {
                self.output_names[i]: probabilities[i]
                for i in range(len(probabilities))
            }
        }

    # --------------------------------------------------------
    # Save weights
    # --------------------------------------------------------

    def save(self, filename=WEIGHTS_FILE):

        data = {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "vocab": VOCAB,
            "output_names": self.output_names,
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2,
        }

        with open(
            filename,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                data,
                file,
                separators=(",", ":")
            )

    # --------------------------------------------------------
    # Load weights
    # --------------------------------------------------------

    def load(self, filename=WEIGHTS_FILE):
        """
        Load previously trained weights. Returns True only if the
        file exists, is valid JSON, and its dimensions (and, if
        present, its vocabulary) match this network's configuration.
        Any mismatch is treated as "no usable weights" rather than
        raising, so the caller can simply retrain from scratch.
        """

        if not os.path.exists(filename):
            return False

        try:
            with open(
                filename,
                "r",
                encoding="utf-8"
            ) as file:
                data = json.load(file)

            required_keys = {"input_size", "hidden_size", "output_size", "W1", "b1", "W2", "b2"}

            if not required_keys.issubset(data.keys()):
                return False

            if data["input_size"] != self.input_size:
                return False

            if data["hidden_size"] != self.hidden_size:
                return False

            if data["output_size"] != self.output_size:
                return False

            W1 = data["W1"]
            b1 = data["b1"]
            W2 = data["W2"]
            b2 = data["b2"]

            # Structural validation: shapes must match exactly.
            if len(W1) != self.hidden_size:
                return False
            if any(len(row) != self.input_size for row in W1):
                return False
            if len(b1) != self.hidden_size:
                return False

            if len(W2) != self.output_size:
                return False
            if any(len(row) != self.hidden_size for row in W2):
                return False
            if len(b2) != self.output_size:
                return False

            # If the saved file recorded the vocabulary it was trained
            # with, make sure it still matches. If VOCAB changed since
            # training, the saved weights no longer mean anything for
            # the current encode() function, so we reject them.
            if "vocab" in data and data["vocab"] != VOCAB:
                return False

            self.W1 = W1
            self.b1 = b1
            self.W2 = W2
            self.b2 = b2

            return True

        except Exception:
            return False


# ============================================================
# ELIZA REFLECTION SYSTEM
# ============================================================

REFLECTIONS = {
    "am": "are",
    "was": "were",

    "i": "you",
    "i'm": "you are",
    "i'd": "you would",
    "i'll": "you will",
    "i've": "you have",

    "me": "you",
    "my": "your",
    "mine": "yours",
    "myself": "yourself",

    "you": "I",
    "you're": "I am",
    "you'd": "I would",
    "you'll": "I will",
    "you've": "I have",

    "your": "my",
    "yours": "mine",
    "yourself": "myself",
}


def reflect(text):

    words = text.split()
    result = []

    for word in words:

        match = re.match(
            r"^([^\w']*)([\w']+)([^\w']*)$",
            word
        )

        if not match:
            result.append(word)
            continue

        prefix, core, suffix = match.groups()

        replacement = REFLECTIONS.get(
            core.lower(),
            core
        )

        result.append(
            prefix + replacement + suffix
        )

    return " ".join(result)


# ============================================================
# ELIZA RULES
# ============================================================

RULES = [

    (
        ["hello", "hi", "hey"],
        [
            "Hello. How are you feeling today?",
            "Hello. What would you like to talk about?",
            "Hi. Tell me what is on your mind.",
        ]
    ),

    (
        ["sad", "unhappy", "miserable"],
        [
            "What makes you feel that way?",
            "How long have you felt this way?",
            "Tell me more about your sadness.",
        ]
    ),

    (
        ["angry", "mad", "furious"],
        [
            "What makes you angry?",
            "Tell me more about your anger.",
            "Why do you think you feel this way?",
        ]
    ),

    (
        ["afraid", "scared", "fear", "frightened"],
        [
            "What are you afraid of?",
            "Why do you feel afraid?",
            "Tell me more about your fear.",
        ]
    ),

    (
        ["happy", "glad", "excited"],
        [
            "What makes you feel that way?",
            "Tell me more about your happiness.",
            "How long have you felt this way?",
        ]
    ),

    (
        ["mother", "mom", "mommy", "mama"],
        [
            "Tell me more about your mother.",
            "How do you feel about your mother?",
            "Does your mother influence you strongly?",
        ]
    ),

    (
        ["father", "dad", "daddy", "papa"],
        [
            "Tell me more about your father.",
            "How do you feel about your father?",
            "Does your father influence you strongly?",
        ]
    ),

    (
        ["family"],
        [
            "Tell me more about your family.",
            "How do you feel about your family?",
            "What is your relationship with your family like?",
        ]
    ),

    (
        ["friend", "friends"],
        [
            "Tell me more about your friends.",
            "Do you feel understood by your friends?",
            "What do you value in friendship?",
        ]
    ),

    (
        ["love", "loved", "loving"],
        [
            "What does love mean to you?",
            "Tell me more about your feelings of love.",
            "How does love make you feel?",
        ]
    ),

    (
        ["hate"],
        [
            "Why do you feel hatred?",
            "Tell me more about what you hate.",
            "What makes you feel that way?",
        ]
    ),

    (
        ["remember"],
        [
            "What do you remember?",
            "Does that memory still affect you?",
            "Why do you remember that now?",
        ]
    ),

    (
        ["forget"],
        [
            "Why do you want to forget?",
            "What makes forgetting important to you?",
            "Do you often try to forget things?",
        ]
    ),

    (
        ["dream", "dreams"],
        [
            "What do your dreams mean to you?",
            "Tell me more about your dreams.",
            "Do you often dream about this?",
        ]
    ),

    (
        ["why"],
        [
            "Why do you think?",
            "What do you think the reason is?",
            "What makes you ask why?",
        ]
    ),

    (
        ["because"],
        [
            "Is that the real reason?",
            "What other reasons might there be?",
            "Could there be another reason?",
        ]
    ),

    (
        ["i am", "i'm"],
        [
            "How long have you been {input}?",
            "Why do you tell me you are {input}?",
            "How does being {input} make you feel?",
        ]
    ),

    (
        ["i feel"],
        [
            "Tell me more about feeling {input}.",
            "Why do you feel {input}?",
            "Do you often feel {input}?",
        ]
    ),

    (
        ["i want"],
        [
            "Why do you want {input}?",
            "What would it mean if you got {input}?",
            "What makes that important to you?",
        ]
    ),

    (
        ["i think"],
        [
            "Why do you think {input}?",
            "Do you really think {input}?",
            "What makes you think {input}?",
        ]
    ),

    (
        ["you are", "you're"],
        [
            "Why do you think I am {input}?",
            "What makes you say I am {input}?",
            "Perhaps you are really talking about yourself.",
        ]
    ),

    (
        ["computer"],
        [
            "Do computers worry you?",
            "What do you think about machines?",
            "Do you think computers can understand people?",
        ]
    ),
]


# ============================================================
# RULE MATCHING
# ============================================================

def find_rule(text):

    lowered = text.lower()

    for keywords, responses in RULES:

        for keyword in keywords:

            pattern = (
                r"\b"
                + re.escape(keyword)
                + r"\b"
            )

            match = re.search(
                pattern,
                lowered
            )

            if match:
                return keywords, responses, keyword

    return None


# ============================================================
# NEURAL FALLBACK RESPONSES (sentiment-aware)
# ============================================================
#
# Each intent maps to a small dict of response pools keyed by
# sentiment. "default" is always available; "positive" / "negative"
# pools are blended in on top of "default" when they match the
# detected sentiment, so sentiment genuinely changes which responses
# can be picked, without removing any of the original wording.

AUXILIARY_RESPONSES = {

    "sadness": {
        "default": [
            "Can you tell me more about what is troubling you?",
            "How long have you been feeling this way?",
        ],
        "negative": [
            "That sounds heavy to carry. What has been the hardest part?",
            "I am sorry you are feeling this way. What do you think brought it on?",
        ],
    },

    "anger": {
        "default": [
            "What do you think is behind your anger?",
            "Tell me more about what is upsetting you.",
        ],
        "negative": [
            "That sounds really frustrating. What happened?",
        ],
    },

    "fear": {
        "default": [
            "What is making you feel afraid?",
            "Tell me more about this fear.",
        ],
        "negative": [
            "That sounds unsettling. What would help you feel safer?",
        ],
    },

    "happiness": {
        "default": [
            "What is making you feel happy?",
            "Tell me more about that feeling.",
        ],
        "positive": [
            "That is wonderful to hear. What made this moment stand out?",
        ],
    },

    "family": {
        "default": [
            "Tell me more about your family.",
            "How does your family make you feel?",
        ],
        "positive": [
            "It sounds like your family means a lot to you.",
        ],
        "negative": [
            "Family matters can be complicated. What is going on?",
        ],
    },

    "relationships": {
        "default": [
            "Tell me more about your relationships.",
            "How do your relationships affect you?",
        ],
        "positive": [
            "That sounds like a meaningful connection. Tell me more.",
        ],
        "negative": [
            "Relationships can be difficult sometimes. What is on your mind?",
        ],
    },

    "existential": {
        "default": [
            "What does that mean to you?",
            "Tell me more about how you see life.",
        ],
        "negative": [
            "Those thoughts can feel heavy. What has been on your mind lately?",
        ],
    },

    "neutral": {
        "default": [
            "Please tell me more.",
            "Can you elaborate on that?",
            "What does that mean to you?",
        ],
    },
}


def select_fallback_response(intent, sentiment):

    intent_pools = AUXILIARY_RESPONSES.get(intent, AUXILIARY_RESPONSES["neutral"])

    responses = list(intent_pools.get("default", []))

    if sentiment in intent_pools:
        responses += intent_pools[sentiment]

    if not responses:
        responses = AUXILIARY_RESPONSES["neutral"]["default"]

    return random.choice(responses)


# ============================================================
# ELIZA RESPONSE
# ============================================================

def eliza_response(text, neural):

    result = find_rule(text)

    if result:

        keywords, responses, keyword = result

        pattern = (
            r"\b"
            + re.escape(keyword)
            + r"\b"
        )

        remainder = re.sub(
            pattern,
            "",
            text,
            count=1,
            flags=re.IGNORECASE
        ).strip()

        remainder = reflect(remainder)

        response = random.choice(responses)

        if "{input}" in response:

            if remainder:
                response = response.replace(
                    "{input}",
                    remainder
                )
            else:
                response = response.replace(
                    "{input}",
                    "that"
                )

        return response

    # --------------------------------------------------------
    # No ELIZA rule matched.
    # The trained neural network drives the fallback selection,
    # using both intent and sentiment.
    # --------------------------------------------------------

    prediction = neural.predict(text)

    return select_fallback_response(
        prediction["intent"],
        prediction["sentiment"],
    )


# ============================================================
# TRAINING DATA
# ============================================================
#
# Small, hand-written dataset covering all 8 intents with several
# example sentences each. This is only meant to demonstrate that the
# network can actually learn something from real gradient descent,
# not to build a production-quality classifier.

TRAINING_DATA = [

    # ---- sadness ----
    ("i am sad", "sadness"),
    ("i feel lonely", "sadness"),
    ("i feel miserable", "sadness"),
    ("i feel tired and sad", "sadness"),
    ("i am so lonely today", "sadness"),
    ("i feel confused and sad", "sadness"),

    # ---- anger ----
    ("i am angry", "anger"),
    ("i am mad", "anger"),
    ("this makes me furious", "anger"),
    ("i hate this", "anger"),
    ("i feel so angry right now", "anger"),
    ("you make me angry", "anger"),

    # ---- fear ----
    ("i am afraid", "fear"),
    ("i feel scared", "fear"),
    ("this scares me", "fear"),
    ("i am afraid of the future", "fear"),
    ("i feel very afraid", "fear"),
    ("i am scared and confused", "fear"),

    # ---- happiness ----
    ("i am happy", "happiness"),
    ("i feel great", "happiness"),
    ("i am excited", "happiness"),
    ("i feel very happy today", "happiness"),
    ("i love this", "happiness"),
    ("i feel good and excited", "happiness"),

    # ---- family ----
    ("my mother", "family"),
    ("my father", "family"),
    ("my family", "family"),
    ("i love my mother", "family"),
    ("my family is important to me", "family"),
    ("my father and my mother", "family"),

    # ---- relationships ----
    ("my friend", "relationships"),
    ("my relationship", "relationships"),
    ("i love someone", "relationships"),
    ("my friends are important to me", "relationships"),
    ("i need my friend", "relationships"),
    ("my relationship with my friend", "relationships"),

    # ---- existential ----
    ("what is life", "existential"),
    ("why are we alive", "existential"),
    ("i think about death", "existential"),
    ("what is the meaning of life", "existential"),
    ("i think about the future and the past", "existential"),
    ("why do we exist", "existential"),

    # ---- neutral ----
    ("hello", "neutral"),
    ("how are you", "neutral"),
    ("what is a computer", "neutral"),
    ("hi there", "neutral"),
    ("hey how are you", "neutral"),
    ("thanks", "neutral"),
    ("sorry", "neutral"),
]


def build_training_dataset():
    """Convert (text, intent_name) pairs into (text, target_index) pairs."""

    dataset = []

    for text, intent_name in TRAINING_DATA:

        if intent_name not in OUTPUT_NAMES:
            raise ValueError(
                f"TRAINING_DATA references unknown intent '{intent_name}'."
            )

        target_index = OUTPUT_NAMES.index(intent_name)
        dataset.append((text, target_index))

    return dataset


# ============================================================
# TERMINAL
# ============================================================

def banner():

    print()
    print("=" * 58)
    print("                         ELIZA")
    print("              Classic rule-based engine")
    print("                + 128-neuron assistant")
    print("=" * 58)
    print()
    print("Type 'quit' or 'exit' to leave.")
    print()


def prepare_neural_assistant():
    """
    Load a previously trained network if a valid one is saved on
    disk. Otherwise, initialize a fresh network, train it on
    TRAINING_DATA, and save the result so future runs can skip
    training.
    """

    neural = NeuralAssistant()

    if neural.load():
        print(f"Loaded existing trained weights from '{WEIGHTS_FILE}'.")
        return neural

    print("No valid saved weights found. Training a new model...")
    print("(This is a small pure-Python neural network trained on a")
    print(" tiny hand-written dataset -- it is not a language model")
    print(" and does not understand language, it just recognizes")
    print(" simple word patterns for the 8 intents above.)")
    print()

    dataset = build_training_dataset()
    neural.train(dataset, epochs=DEFAULT_EPOCHS, learning_rate=DEFAULT_LEARNING_RATE)

    neural.save()
    print(f"\nTraining complete. Weights saved to '{WEIGHTS_FILE}'.")

    return neural


def main():

    neural = prepare_neural_assistant()

    banner()

    while True:

        try:

            user_input = input("YOU: ").strip()

        except KeyboardInterrupt:

            print("\nELIZA: Goodbye.")
            break

        except EOFError:

            print("\nELIZA: Goodbye.")
            break

        if not user_input:

            print("ELIZA: Please say something.")
            continue

        if user_input.lower() in {
            "quit",
            "exit",
            "bye",
            "goodbye",
            "/quit",
            "/exit",
        }:

            print(
                "ELIZA: Goodbye. "
                "It was nice talking with you."
            )

            break

        response = eliza_response(
            user_input,
            neural
        )

        print("ELIZA:", response)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
