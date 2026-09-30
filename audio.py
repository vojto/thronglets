"""Generic audio plumbing: envelopes, panning, saving and playing sound."""

import subprocess
import tempfile
import wave

import numpy as np

SAMPLE_RATE = 44100
VOLUME = 0.35


def seconds(duration):
    return int(duration * SAMPLE_RATE)


def silence(duration):
    return np.zeros(seconds(duration))


def envelope(samples, attack, release):
    env = np.ones(samples)
    attack_samples = min(seconds(attack), samples)
    release_samples = min(seconds(release), samples - attack_samples)
    env[:attack_samples] = np.linspace(0, 1, attack_samples)
    if release_samples:
        env[-release_samples:] = np.linspace(1, 0, release_samples)
    return env


def pan(mono, position):
    """Place a mono signal between left (-1) and right (1)."""
    left = np.sqrt((1 - position) / 2)
    right = np.sqrt((1 + position) / 2)
    return np.stack([mono * left, mono * right], axis=1)


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
        subprocess.run(["afplay", "-v", str(VOLUME), file.name], check=True)


def play_in_background(stereo):
    """Start playing and return immediately; the file is removed once playback ends."""
    path = tempfile.mktemp(suffix=".wav")
    save(stereo, path)
    subprocess.Popen(["sh", "-c", f"afplay -v {VOLUME} '{path}'; rm '{path}'"], start_new_session=True)
