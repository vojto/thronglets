"""Reading feelings from text with the Jev judge model on TypeSafe."""

import json
import os
import urllib.request
from pathlib import Path

from feelings import jitter

# One Jev score question per feeling: instructions plus a scale from low to high.
QUESTIONS = {
    "energy": ("How much energy does this message carry?",
               ["Drowsy and flat", "Calm", "Steady", "Lively", "Bursting with excitement"]),
    "positivity": ("How good is the news or the feeling in this message?",
                   ["Bad news, upset or apologetic", "Somewhat disappointing", "Neutral", "Good", "Delighted or triumphant"]),
    "curiosity": ("How much is the writer asking or wondering, versus stating?",
                  ["Plain statement", "Mostly statements", "Some open questions", "Asks the reader something", "Mainly a question"]),
    "tension": ("How tense, worried or alarmed is the message?",
                ["Completely relaxed", "At ease", "Some concern", "Worried", "Alarmed"]),
    "warmth": ("How warm and friendly is the tone?",
               ["Dry and technical", "Businesslike", "Friendly", "Warm", "Affectionate and playful"]),
    "chattiness": ("How long and elaborate is the message?",
                   ["A few words", "A sentence or two", "A short paragraph", "Several paragraphs", "Long and detailed"]),
    "confidence": ("How sure of itself does the writer sound?",
                   ["Lost or confused", "Hesitant", "Fairly sure", "Confident", "Proud of the result"]),
}


def read_feelings(message, rng):
    """Let Jev rate the message on every feeling, then turn the ratings into feeling values."""
    return {name: feeling_from(answer, rng) for name, answer in ask(message).items()}


def ask(message):
    questions = {
        name: {"type": "score", "instructions": instructions, "criteria": scale}
        for name, (instructions, scale) in QUESTIONS.items()
    }
    request = urllib.request.Request(
        "https://api.typesafe.ai/v1/systemone",
        data=json.dumps({"model": "jev-latest", "state": {"message": message[:4000]}, "questions": questions}).encode(),
        headers={"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.load(response)["answers"]


def feeling_from(answer, rng):
    """Half Jev's average rating, half a level drawn from Jev's own probabilities.

    When Jev is unsure between two readings, the creature sometimes sounds one
    way and sometimes the other.
    """
    levels = [int(level) for level in answer["probabilities"]]
    top = max(levels)
    drawn = rng.choices(levels, weights=list(answer["probabilities"].values()))[0]
    return jitter(0.5 * answer["score"] / top + 0.5 * drawn / top, rng)


def api_key():
    if key := os.environ.get("TYPESAFE_API_KEY"):
        return key
    for line in (Path(__file__).parent / ".env").read_text().splitlines():
        if line.startswith("TYPESAFE_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("TYPESAFE_API_KEY is missing from the environment and .env")
