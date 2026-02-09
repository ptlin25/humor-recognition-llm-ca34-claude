"""
Data preparation for humor rank analysis.
Loads humor (Short Jokes) and non-humor text, creates balanced dataset.
"""
import json
import random
import os
import numpy as np
from datasets import load_from_disk
from pathlib import Path

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

PROJECT_ROOT = Path(__file__).parent.parent


def load_short_jokes(n_samples=2000):
    """Load and sample from Short Jokes dataset."""
    ds = load_from_disk(str(PROJECT_ROOT / "datasets" / "short_jokes" / "train"))
    # Filter to reasonable length jokes (10-200 chars) to avoid trivial length cues
    texts = [row["Joke"] for row in ds if 10 < len(row["Joke"]) < 200]
    random.shuffle(texts)
    return texts[:n_samples]


# Factual / non-humorous sentences that match joke-like length and style
# These are encyclopedic, instructional, or news-style sentences
NON_HUMOR_TEMPLATES = [
    "The capital of {country} is {capital}.",
    "Water boils at 100 degrees Celsius at sea level.",
    "The human body contains approximately 206 bones.",
    "Photosynthesis converts carbon dioxide into oxygen.",
    "The speed of light is approximately 299,792 kilometers per second.",
    "DNA stands for deoxyribonucleic acid.",
    "The Earth orbits the Sun once every 365.25 days.",
    "Gravity keeps the planets in orbit around the Sun.",
    "The Pacific Ocean is the largest ocean on Earth.",
    "Cells are the basic building blocks of all living organisms.",
    "Iron is the most abundant element in Earth's core.",
    "The Great Wall of China stretches over 13,000 miles.",
    "Oxygen makes up about 21 percent of Earth's atmosphere.",
    "The Nile River is often considered the longest river in the world.",
    "Diamonds are formed under extreme heat and pressure.",
    "The moon has no atmosphere and no weather.",
    "Coral reefs support about 25 percent of all marine species.",
    "The Amazon rainforest produces a significant portion of the world's oxygen.",
    "Sound travels faster through water than through air.",
    "Earthquakes are caused by the movement of tectonic plates.",
]

COUNTRY_CAPITAL_PAIRS = [
    ("France", "Paris"), ("Germany", "Berlin"), ("Japan", "Tokyo"),
    ("Brazil", "Brasilia"), ("Australia", "Canberra"), ("Canada", "Ottawa"),
    ("Egypt", "Cairo"), ("India", "New Delhi"), ("Mexico", "Mexico City"),
    ("Russia", "Moscow"), ("Spain", "Madrid"), ("Italy", "Rome"),
    ("Argentina", "Buenos Aires"), ("Sweden", "Stockholm"), ("Norway", "Oslo"),
    ("Poland", "Warsaw"), ("Greece", "Athens"), ("Turkey", "Ankara"),
    ("Thailand", "Bangkok"), ("Kenya", "Nairobi"),
]

FACTUAL_SENTENCES = [
    "The average adult human has about 5 liters of blood.",
    "Mercury is the smallest planet in our solar system.",
    "The chemical symbol for gold is Au.",
    "An octopus has three hearts.",
    "The Sahara Desert is the largest hot desert in the world.",
    "Honey never spoils if stored properly.",
    "Venus is the hottest planet in our solar system.",
    "A group of flamingos is called a flamboyance.",
    "The longest bone in the human body is the femur.",
    "Bamboo is the fastest growing plant in the world.",
    "A light year measures distance, not time.",
    "The first computer programmer was Ada Lovelace.",
    "Mount Everest is the tallest mountain above sea level.",
    "A marathon is exactly 26.2 miles long.",
    "The human brain weighs about 3 pounds.",
    "Steel is an alloy of iron and carbon.",
    "Penguins are found naturally only in the Southern Hemisphere.",
    "The periodic table has 118 confirmed elements.",
    "Lightning strikes the Earth about 8 million times per day.",
    "The human eye can distinguish about 10 million different colors.",
    "Salt is composed of sodium and chlorine atoms.",
    "The Eiffel Tower was built for the 1889 World Fair.",
    "Pluto was reclassified as a dwarf planet in 2006.",
    "The speed of sound in air is about 343 meters per second.",
    "Glass is made primarily from sand.",
    "The deepest part of the ocean is the Mariana Trench.",
    "A day on Venus is longer than a year on Venus.",
    "The first email was sent in 1971.",
    "Cats have five toes on their front paws and four on their back.",
    "The smallest country in the world is Vatican City.",
    "Coffee beans are actually seeds from a cherry-like fruit.",
    "The Great Barrier Reef is the largest living structure on Earth.",
    "Human teeth are as strong as shark teeth.",
    "The Statue of Liberty was a gift from France.",
    "Bananas are naturally slightly radioactive.",
    "The average cloud weighs about 1.1 million pounds.",
    "Spiders are not insects; they are arachnids.",
    "The human nose can detect over 1 trillion different scents.",
    "Mars has two moons named Phobos and Deimos.",
    "The first antibiotic discovered was penicillin.",
    "An adult human body contains about 60 percent water.",
    "The longest river in Asia is the Yangtze.",
    "Copper is an excellent conductor of electricity.",
    "The largest organ in the human body is the skin.",
    "Helium is the second most abundant element in the universe.",
    "The Titanic sank on its maiden voyage in 1912.",
    "A blue whale's heart is the size of a small car.",
    "Shakespeare wrote 37 plays during his lifetime.",
    "The Earth's inner core is about as hot as the surface of the Sun.",
    "Butterflies taste with their feet.",
    "The first photograph was taken in 1826.",
    "Polar bears have black skin under their white fur.",
    "The longest mountain range on Earth is the Mid-Atlantic Ridge.",
    "Rubber bands last longer when refrigerated.",
    "A hummingbird's heart beats about 1,200 times per minute.",
    "The coldest temperature ever recorded was minus 89.2 degrees Celsius.",
    "Paper was invented in China around 105 AD.",
    "The average person walks about 100,000 miles in a lifetime.",
    "Saturn's rings are made mostly of ice and rock.",
    "The first commercial airline flight was in 1914.",
]


def generate_non_humor_texts(n_samples=2000):
    """Generate non-humorous factual sentences."""
    texts = []

    # Use templates with country/capital pairs
    for country, capital in COUNTRY_CAPITAL_PAIRS:
        texts.append(f"The capital of {country} is {capital}.")

    # Add all factual sentences
    texts.extend(FACTUAL_SENTENCES)

    # Add template-based sentences (without the country one)
    texts.extend(NON_HUMOR_TEMPLATES[1:])

    # Repeat and shuffle to get enough samples
    if len(texts) < n_samples:
        # Create variations by combining facts
        base = list(texts)
        while len(texts) < n_samples:
            random.shuffle(base)
            texts.extend(base)

    random.shuffle(texts)
    return texts[:n_samples]


def prepare_dataset(n_humor=2000, n_non_humor=2000, val_frac=0.2, test_frac=0.2):
    """Prepare balanced humor/non-humor dataset with splits."""
    humor_texts = load_short_jokes(n_humor)
    non_humor_texts = generate_non_humor_texts(n_non_humor)

    # Ensure balanced
    n = min(len(humor_texts), len(non_humor_texts))
    humor_texts = humor_texts[:n]
    non_humor_texts = non_humor_texts[:n]

    texts = humor_texts + non_humor_texts
    labels = [1] * n + [0] * n  # 1=humor, 0=non-humor

    # Shuffle together
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined)
    texts, labels = list(texts), list(labels)

    # Split
    total = len(texts)
    n_test = int(total * test_frac)
    n_val = int(total * val_frac)
    n_train = total - n_test - n_val

    train_texts, train_labels = texts[:n_train], labels[:n_train]
    val_texts, val_labels = texts[n_train:n_train+n_val], labels[n_train:n_train+n_val]
    test_texts, test_labels = texts[n_train+n_val:], labels[n_train+n_val:]

    print(f"Dataset prepared: train={len(train_texts)}, val={len(val_texts)}, test={len(test_texts)}")
    print(f"Train humor ratio: {sum(train_labels)/len(train_labels):.2f}")
    print(f"Val humor ratio: {sum(val_labels)/len(val_labels):.2f}")
    print(f"Test humor ratio: {sum(test_labels)/len(test_labels):.2f}")

    return {
        "train": {"texts": train_texts, "labels": train_labels},
        "val": {"texts": val_texts, "labels": val_labels},
        "test": {"texts": test_texts, "labels": test_labels},
    }


if __name__ == "__main__":
    data = prepare_dataset()
    # Save for later use
    output_path = PROJECT_ROOT / "results" / "prepared_data.json"
    with open(output_path, "w") as f:
        json.dump(data, f)
    print(f"Saved to {output_path}")

    # Show examples
    print("\n=== Humor examples ===")
    for t, l in zip(data["train"]["texts"][:3], data["train"]["labels"][:3]):
        if l == 1:
            print(f"  [{l}] {t[:100]}")
    print("\n=== Non-humor examples ===")
    for t, l in zip(data["train"]["texts"][:10], data["train"]["labels"][:10]):
        if l == 0:
            print(f"  [{l}] {t[:100]}")
