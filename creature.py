"""A tiny creature with its own voice that babbles according to how it feels."""

import random

import numpy as np
from scipy.signal import butter, lfilter

from audio import SAMPLE_RATE, envelope, pan, seconds, silence
from feelings import NEUTRAL

# (F1, F2) formants of human vowels, in Hz. They get scaled up per creature.
VOWELS = {
    "a": (800, 1200),
    "e": (500, 1900),
    "i": (300, 2300),
    "o": (500, 900),
    "u": (330, 800),
}


class Creature:
    # MARK: Lifecycle

    def __init__(self, rng, pan=0.0):
        self.rng = rng
        self.base_pitch = rng.uniform(420, 750)
        self.formant_scale = rng.uniform(1.5, 1.9)
        self.brightness = rng.uniform(0.6, 1.0)
        self.pan = pan
        # Each creature favours a couple of vowels, which gives it a recognizable "accent".
        self.favourite_vowels = rng.sample(list(VOWELS), 2)

    @classmethod
    def for_session(cls, session_id, rng):
        """Same session, same voice; what it says still comes from rng."""
        creature = cls(random.Random(session_id))
        creature.rng = rng
        return creature

    # MARK: Speaking

    def phrase(self, feelings):
        """A sentence of babble whose melody, pace and vowels follow the feelings."""
        f = {**NEUTRAL, **feelings}
        pitch = self.base_pitch * (0.7 + 0.3 * f["energy"] + 0.2 * f["positivity"] + 0.1 * f["tension"])
        count = max(1, round(1 + 2 * f["chattiness"] + 3 * f["energy"] + self.rng.uniform(-1, 1)))
        length = 0.22 - 0.15 * f["energy"] + 0.04 * f["warmth"]
        gap = 0.08 - 0.07 * f["energy"] + 0.04 * (1 - f["confidence"])
        voice = dict(
            wobble=0.015 + 0.07 * f["tension"] + 0.03 * (1 - f["confidence"]),
            glide=0.1 + 0.3 * f["energy"],
            brightness=min(max(self.brightness + 0.3 * f["tension"] - 0.3 * f["warmth"], 0.3), 1.2),
        )
        chirp_chance = 0.03 + 0.25 * f["energy"] * f["positivity"]
        vowel_weights = self._vowel_weights(f)
        melody = self._melody(f, count)

        pieces = []
        for step in melody:
            vowel = self.rng.choices(list(vowel_weights), weights=list(vowel_weights.values()))[0]
            duration = length * self.rng.uniform(0.7, 1.3)
            pieces.append(self._word(pitch * step, duration, vowel, chirp_chance, **voice))
            pieces.append(silence(gap * self.rng.uniform(0.3, 1.5)))

        # A proud, happy creature finishes with a little upward squeak.
        if f["confidence"] > 0.75 and f["positivity"] > 0.6:
            pieces.append(self._chirp(pitch * 1.2, is_rising=True))

        return pan(np.concatenate(pieces), self.pan)

    def poke(self):
        """A startled reaction to being poked: one to three quick, high syllables or squeaks."""
        pitch = self.base_pitch * self.rng.uniform(1.0, 1.4)
        pieces = []
        for _ in range(self.rng.choice([1, 1, 2, 2, 3])):
            vowel = self.rng.choice(["a", "i", "e", "o", *self.favourite_vowels])
            duration = self.rng.uniform(0.05, 0.13)
            voice = dict(wobble=self.rng.uniform(0.02, 0.08), glide=self.rng.uniform(0.15, 0.5), brightness=self.brightness)
            pieces.append(self._word(pitch, duration, vowel, chirp_chance=0.35, **voice))
            pieces.append(silence(self.rng.uniform(0.01, 0.06)))
            pitch *= self.rng.uniform(0.85, 1.25)
        return pan(np.concatenate(pieces), self.pan)

    # MARK: Phrasing

    def _vowel_weights(self, f):
        weights = {
            "a": 1 + f["positivity"],
            "e": 1 + f["tension"],
            "i": 0.5 + f["tension"] + f["energy"],
            "o": 0.5 + 2 * f["warmth"],
            "u": 0.3 + 2 * f["warmth"] * (1 - f["energy"]),
        }
        return {vowel: (2 if vowel in self.favourite_vowels else 1) * weight for vowel, weight in weights.items()}

    def _melody(self, f, count):
        """Pitch multiplier for each syllable of the phrase."""
        bounces = self.rng.uniform(1.5, 3.5)
        steps = []
        for index in range(count):
            progress = index / max(count - 1, 1)
            step = (
                1
                + 0.6 * f["curiosity"] * progress**2  # questions rise at the end
                - 0.35 * (1 - f["positivity"]) * progress  # bad news sinks
                + 0.2 * f["energy"] * f["positivity"] * np.sin(progress * np.pi * bounces)  # joy bounces
            )
            steps.append(step * self.rng.uniform(0.92, 1.1))
        return steps

    def _word(self, pitch, duration, vowel, chirp_chance, **voice):
        if self.rng.random() < chirp_chance:
            return self._chirp(pitch)
        return self._syllable(pitch, duration, vowel, **voice)

    # MARK: Sounds

    def _syllable(self, pitch, duration, vowel, wobble, glide, brightness):
        samples = seconds(duration)
        t = np.arange(samples) / SAMPLE_RATE

        # Pitch glide across the syllable plus a little vibrato.
        glide_curve = np.linspace(1, self.rng.uniform(1 - glide, 1 + glide), samples)
        vibrato = 1 + wobble * np.sin(2 * np.pi * self.rng.uniform(7, 14) * t)
        frequency = pitch * glide_curve * vibrato

        voice = self._formants(buzz(frequency, brightness), VOWELS[vowel])
        if self.rng.random() < 0.4:
            consonant = self._consonant()
            voice[: len(consonant)] += consonant

        return voice * envelope(samples, attack=0.008, release=min(0.04, duration / 2))

    def _formants(self, source, vowel):
        out = 0.3 * source
        for frequency, gain in zip((f * self.formant_scale for f in vowel), (1.0, 0.7)):
            low, high = frequency * 0.8, min(frequency * 1.2, SAMPLE_RATE / 2 - 100)
            b, a = butter(2, [low, high], btype="band", fs=SAMPLE_RATE)
            out += gain * lfilter(b, a, source)
        return out

    def _consonant(self):
        """A tiny noise burst at the syllable start, like a soft t, k or s."""
        samples = seconds(self.rng.uniform(0.008, 0.025))
        b, a = butter(2, self.rng.uniform(3000, 7000), btype="high", fs=SAMPLE_RATE)
        noise = lfilter(b, a, np.random.default_rng(self.rng.getrandbits(32)).standard_normal(samples))
        return 0.3 * noise * np.linspace(1, 0, samples)

    def _chirp(self, pitch, is_rising=None):
        """A pure whistle sweep, the little squeak between words."""
        if is_rising is None:
            is_rising = self.rng.random() < 0.5
        samples = seconds(self.rng.uniform(0.05, 0.12))
        sweep = np.linspace(pitch * 1.6, pitch * (2.6 if is_rising else 1.0), samples)
        return 0.6 * np.sin(phase(sweep)) * envelope(samples, attack=0.005, release=0.03)


def buzz(frequency, brightness):
    """Band-limited buzzy source following a frequency curve: harmonics falling off like a voice."""
    fundamental = phase(frequency)
    source = np.zeros(len(frequency))
    for harmonic in range(1, int(9000 / frequency.max()) + 1):
        source += np.sin(harmonic * fundamental) / harmonic ** (2 - brightness)
    return source


def phase(frequency):
    return 2 * np.pi * np.cumsum(frequency) / SAMPLE_RATE
