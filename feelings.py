"""Feelings: a handful of numbers from 0 to 1 that decide how a creature sounds."""

NAMES = ["energy", "positivity", "curiosity", "tension", "warmth", "chattiness", "confidence"]

NEUTRAL = {name: 0.5 for name in NAMES}

PRESETS = {
    "happy": dict(energy=0.65, positivity=0.85, warmth=0.75),
    "excited": dict(energy=1.0, positivity=0.9, chattiness=0.8, confidence=0.8),
    "proud": dict(energy=0.7, positivity=0.9, confidence=1.0),
    "curious": dict(curiosity=1.0, energy=0.55, warmth=0.6),
    "confused": dict(curiosity=0.8, confidence=0.0, tension=0.5),
    "worried": dict(tension=0.9, positivity=0.3, confidence=0.3),
    "sad": dict(positivity=0.05, energy=0.25, confidence=0.3),
    "grumpy": dict(positivity=0.2, warmth=0.05, tension=0.6, chattiness=0.2),
    "sleepy": dict(energy=0.0, chattiness=0.1, tension=0.0),
    "affectionate": dict(warmth=1.0, positivity=0.8, energy=0.4, tension=0.0),
}


def random_feelings(rng):
    return {name: rng.random() for name in NAMES}


def preset(mood, rng):
    return blur({**NEUTRAL, **PRESETS[mood]}, rng, 0.1)


def blur(feelings, rng, amount):
    return {name: jitter(value, rng, amount) for name, value in feelings.items()}


def jitter(value, rng, amount=0.07):
    return min(max(value + rng.gauss(0, amount), 0.0), 1.0)
