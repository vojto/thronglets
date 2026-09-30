#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy"]
# ///
"""Procedural babble for tiny virtual creatures, in the spirit of Black Mirror's Thronglets.

Every phrase is synthesized from scratch: a buzzy voice source is pushed through
vowel formants (shifted up, as if from a very small throat), cut into syllables
with pitch glides and wobble, and strung together into a melody.

What the creature sounds like is decided by a handful of feelings, each a number
from 0 to 1 (energy, positivity, curiosity, ...). For real text, the Jev judge
model on TypeSafe reads the message and rates each feeling; there are no mood labels.

  ./thronglets.py say                          # random feelings
  ./thronglets.py say --mood worried           # a preset, see PRESETS
  ./thronglets.py say --text "It works!"       # let Jev hear a message and react to it
  ./thronglets.py chatter                      # endless conversation between a few creatures
  ./thronglets.py say --out hello.wav          # save instead of playing
  ./thronglets.py hook                         # Claude Code Stop hook, reads the hook JSON on stdin
"""

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
import time
import urllib.request
import wave
from pathlib import Path

import numpy as np
from scipy.signal import butter, lfilter

SAMPLE_RATE = 44100

# (F1, F2) formants of human vowels, in Hz. They get scaled up per creature.
VOWELS = {
    "a": (800, 1200),
    "e": (500, 1900),
    "i": (300, 2300),
    "o": (500, 900),
    "u": (330, 800),
}

# Each feeling is one Jev score question: instructions plus a scale from 0 to 1.
FEELINGS = {
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

NEUTRAL = {name: 0.5 for name in FEELINGS}

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


def judge_feelings(message):
    """Ask Jev to rate the message on every feeling. Returns Jev's answers, one per feeling."""
    questions = {
        name: {"type": "score", "instructions": instructions, "criteria": scale}
        for name, (instructions, scale) in FEELINGS.items()
    }
    request = urllib.request.Request(
        "https://api.typesafe.ai/v1/systemone",
        data=json.dumps({"model": "jev-latest", "state": {"message": message[:4000]}, "questions": questions}).encode(),
        headers={"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.load(response)["answers"]


def feelings_from(answers, rng):
    """Turn Jev's answers into feeling values, with some randomness.

    Half of each value is Jev's average rating; the other half is a level drawn
    from Jev's own probabilities. When Jev is unsure between two readings, the
    creature sometimes sounds one way and sometimes the other.
    """
    feelings = {}
    for name, answer in answers.items():
        levels = [int(level) for level in answer["probabilities"]]
        top = max(levels)
        drawn = rng.choices(levels, weights=list(answer["probabilities"].values()))[0]
        feelings[name] = jitter(0.5 * answer["score"] / top + 0.5 * drawn / top, rng)
    return feelings


def jitter(value, rng, amount=0.07):
    return min(max(value + rng.gauss(0, amount), 0.0), 1.0)


def api_key():
    if key := os.environ.get("TYPESAFE_API_KEY"):
        return key
    for line in (Path(__file__).parent / ".env").read_text().splitlines():
        if line.startswith("TYPESAFE_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("TYPESAFE_API_KEY is missing from the environment and .env")


class Creature:
    def __init__(self, rng, pan=0.0):
        self.rng = rng
        self.base_pitch = rng.uniform(420, 750)
        self.formant_scale = rng.uniform(1.5, 1.9)
        self.brightness = rng.uniform(0.6, 1.0)
        self.pan = pan
        # Each creature favours a couple of vowels, which gives it a recognizable "accent".
        self.favourite_vowels = rng.sample(list(VOWELS), 2)

    def phrase(self, feelings):
        f = {**NEUTRAL, **feelings}
        energy, positivity, curiosity = f["energy"], f["positivity"], f["curiosity"]
        tension, warmth, confidence = f["tension"], f["warmth"], f["confidence"]

        pitch = self.base_pitch * (0.7 + 0.3 * energy + 0.2 * positivity + 0.1 * tension)
        length = 0.22 - 0.15 * energy + 0.04 * warmth
        count = max(1, round(1 + 2 * f["chattiness"] + 3 * energy + self.rng.uniform(-1, 1)))
        gap = 0.08 - 0.07 * energy + 0.04 * (1 - confidence)
        wobble = 0.015 + 0.07 * tension + 0.03 * (1 - confidence)
        glide = 0.1 + 0.3 * energy
        chirp_chance = 0.03 + 0.25 * energy * positivity
        bounces = self.rng.uniform(1.5, 3.5)
        vowel_weights = {
            vowel: (2 if vowel in self.favourite_vowels else 1) * weight
            for vowel, weight in {
                "a": 1 + positivity,
                "e": 1 + tension,
                "i": 0.5 + tension + energy,
                "o": 0.5 + 2 * warmth,
                "u": 0.3 + 2 * warmth * (1 - energy),
            }.items()
        }
        brightness = min(max(self.brightness + 0.3 * tension - 0.3 * warmth, 0.3), 1.2)

        pieces = []
        for index in range(count):
            progress = index / max(count - 1, 1)
            melody = (
                1
                + 0.6 * curiosity * progress**2  # questions rise at the end
                - 0.35 * (1 - positivity) * progress  # bad news sinks
                + 0.2 * energy * positivity * np.sin(progress * np.pi * bounces)  # joy bounces
            )
            syllable_pitch = pitch * melody * self.rng.uniform(0.92, 1.1)
            if self.rng.random() < chirp_chance:
                pieces.append(self.chirp(syllable_pitch))
            else:
                duration = length * self.rng.uniform(0.7, 1.3)
                vowel = self.rng.choices(list(vowel_weights), weights=list(vowel_weights.values()))[0]
                pieces.append(self.syllable(syllable_pitch, duration, vowel, wobble, glide, brightness))
            pieces.append(np.zeros(int(gap * self.rng.uniform(0.3, 1.5) * SAMPLE_RATE)))

        # A proud, happy creature finishes with a little upward squeak.
        if confidence > 0.75 and positivity > 0.6:
            pieces.append(self.chirp(pitch * 1.2, rising=True))

        return self.stereo(np.concatenate(pieces))

    def syllable(self, pitch, duration, vowel, wobble, glide, brightness):
        samples = int(duration * SAMPLE_RATE)
        t = np.arange(samples) / SAMPLE_RATE

        # Pitch glide across the syllable plus a little vibrato.
        glide_curve = np.linspace(1, self.rng.uniform(1 - glide, 1 + glide), samples)
        vibrato = 1 + wobble * np.sin(2 * np.pi * self.rng.uniform(7, 14) * t)
        frequency = pitch * glide_curve * vibrato
        phase = 2 * np.pi * np.cumsum(frequency) / SAMPLE_RATE

        # Band-limited buzzy source: harmonics falling off like a voice.
        source = np.zeros(samples)
        harmonic_count = int(9000 / frequency.max())
        for harmonic in range(1, harmonic_count + 1):
            source += np.sin(harmonic * phase) / harmonic ** (2 - brightness)

        voice = self.formants(source, VOWELS[vowel])
        if self.rng.random() < 0.4:
            consonant = self.consonant()
            voice[: len(consonant)] += consonant

        return voice * envelope(samples, attack=0.008, release=min(0.04, duration / 2))

    def formants(self, source, vowel):
        out = 0.3 * source
        for frequency, gain in zip((f * self.formant_scale for f in vowel), (1.0, 0.7)):
            low, high = frequency * 0.8, min(frequency * 1.2, SAMPLE_RATE / 2 - 100)
            b, a = butter(2, [low, high], btype="band", fs=SAMPLE_RATE)
            out += gain * lfilter(b, a, source)
        return out

    def consonant(self):
        """A tiny noise burst at the syllable start, like a soft t, k or s."""
        samples = int(self.rng.uniform(0.008, 0.025) * SAMPLE_RATE)
        b, a = butter(2, self.rng.uniform(3000, 7000), btype="high", fs=SAMPLE_RATE)
        noise = lfilter(b, a, np.random.default_rng(self.rng.getrandbits(32)).standard_normal(samples))
        return 0.3 * noise * np.linspace(1, 0, samples)

    def chirp(self, pitch, rising=None):
        """A pure whistle sweep, the little squeak between words."""
        if rising is None:
            rising = self.rng.random() < 0.5
        samples = int(self.rng.uniform(0.05, 0.12) * SAMPLE_RATE)
        sweep = np.linspace(pitch * 1.6, pitch * (2.6 if rising else 1.0), samples)
        phase = 2 * np.pi * np.cumsum(sweep) / SAMPLE_RATE
        return 0.6 * np.sin(phase) * envelope(samples, attack=0.005, release=0.03)

    def stereo(self, mono):
        left = np.sqrt((1 - self.pan) / 2)
        right = np.sqrt((1 + self.pan) / 2)
        return np.stack([mono * left, mono * right], axis=1)


def envelope(samples, attack, release):
    env = np.ones(samples)
    attack_samples = min(int(attack * SAMPLE_RATE), samples)
    release_samples = min(int(release * SAMPLE_RATE), samples - attack_samples)
    env[:attack_samples] = np.linspace(0, 1, attack_samples)
    if release_samples:
        env[-release_samples:] = np.linspace(1, 0, release_samples)
    return env


def save(stereo, path):
    stereo = stereo / (np.abs(stereo).max() + 1e-9) * 0.8
    with wave.open(path, "wb") as file:
        file.setnchannels(2)
        file.setsampwidth(2)
        file.setframerate(SAMPLE_RATE)
        file.writeframes((stereo * 32767).astype(np.int16).tobytes())


def play(stereo):
    with tempfile.NamedTemporaryFile(suffix=".wav") as file:
        save(stereo, file.name)
        subprocess.run(["afplay", file.name], check=True)


def preset(mood, rng):
    return {name: jitter(value, rng, 0.1) for name, value in {**NEUTRAL, **PRESETS[mood]}.items()}


def say(args, rng):
    if args.text:
        feelings = feelings_from(judge_feelings(args.text), rng)
    elif args.mood:
        feelings = preset(args.mood, rng)
    else:
        feelings = {name: rng.random() for name in FEELINGS}
    print("  ".join(f"{name} {value:.2f}" for name, value in feelings.items()))

    audio = Creature(rng).phrase(feelings)
    if args.out:
        save(audio, args.out)
        print(f"Saved {args.out}")
    else:
        play(audio)


def chatter(args, rng):
    creatures = [Creature(rng, pan=pan) for pan in np.linspace(-0.7, 0.7, args.creatures)]
    print(f"{len(creatures)} creatures are talking. Press Ctrl+C to stop.")
    try:
        while True:
            play(rng.choice(creatures).phrase(preset(args.mood or rng.choice(list(PRESETS)), rng)))
            time.sleep(rng.uniform(0.1, 1.2))
    except KeyboardInterrupt:
        pass


def hook(args, rng):
    """Claude Code Stop hook: react to the assistant's last reply, then return immediately."""
    event = json.load(sys.stdin)
    message = event.get("last_assistant_message") or last_message_from_transcript(event.get("transcript_path"))
    try:
        feelings = feelings_from(judge_feelings(message), rng)
    except Exception:
        feelings = {name: jitter(value, rng, 0.2) for name, value in NEUTRAL.items()}

    # Same session, same creature voice.
    creature = Creature(random.Random(event.get("session_id")))
    creature.rng = rng
    path = tempfile.mktemp(suffix=".wav")
    save(creature.phrase(feelings), path)
    subprocess.Popen(["sh", "-c", f"afplay '{path}'; rm '{path}'"], start_new_session=True)


def last_message_from_transcript(path):
    if not path:
        return ""
    with open(path) as file:
        for line in reversed(file.readlines()):
            entry = json.loads(line)
            if entry.get("type") == "assistant":
                content = entry["message"]["content"]
                text = " ".join(part.get("text", "") for part in content if part.get("type") == "text")
                if text:
                    return text
    return ""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["say", "chatter", "hook"])
    parser.add_argument("--mood", choices=list(PRESETS))
    parser.add_argument("--text", help="let Jev read this message and react to it")
    parser.add_argument("--creatures", type=int, default=3, help="how many creatures talk in chatter mode")
    parser.add_argument("--out", help="save the phrase to a WAV file instead of playing it")
    parser.add_argument("--seed", type=int, help="repeat an exact phrase")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    {"say": say, "chatter": chatter, "hook": hook}[args.command](args, rng)


if __name__ == "__main__":
    main()
