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
  ./thronglets.py say --mood worried           # a preset, see feelings.PRESETS
  ./thronglets.py say --text "It works!"       # let Jev hear a message and react to it
  ./thronglets.py poke                         # a short squeak, as if poked
  ./thronglets.py chatter                      # endless conversation between a few creatures
  ./thronglets.py say --out hello.wav          # save instead of playing
  ./thronglets.py hook                         # Claude Code Stop hook, reads the hook JSON on stdin
  ./thronglets.py prompt-hook                  # Claude Code UserPromptSubmit hook, squeaks as if poked
"""

import argparse
import random
import time

import numpy as np

import claude_code
from audio import play, save
from creature import Creature
from feelings import PRESETS, preset, random_feelings
from jev import read_feelings


def say(args, rng):
    if args.text:
        feelings = read_feelings(args.text, rng)
    elif args.mood:
        feelings = preset(args.mood, rng)
    else:
        feelings = random_feelings(rng)
    print("  ".join(f"{name} {value:.2f}" for name, value in feelings.items()))
    output(Creature(rng).phrase(feelings), args.out)


def poke(args, rng):
    output(Creature(rng).poke(), args.out)


def chatter(args, rng):
    creatures = [Creature(rng, pan=position) for position in np.linspace(-0.7, 0.7, args.creatures)]
    print(f"{len(creatures)} creatures are talking. Press Ctrl+C to stop.")
    try:
        while True:
            play(rng.choice(creatures).phrase(preset(args.mood or rng.choice(list(PRESETS)), rng)))
            time.sleep(rng.uniform(0.1, 1.2))
    except KeyboardInterrupt:
        pass


def output(audio, path):
    if path:
        save(audio, path)
        print(f"Saved {path}")
    else:
        play(audio)


COMMANDS = {
    "say": say,
    "poke": poke,
    "chatter": chatter,
    "hook": lambda args, rng: claude_code.on_reply(rng),
    "prompt-hook": lambda args, rng: claude_code.on_prompt(rng),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=list(COMMANDS))
    parser.add_argument("--mood", choices=list(PRESETS))
    parser.add_argument("--text", help="let Jev read this message and react to it")
    parser.add_argument("--creatures", type=int, default=3, help="how many creatures talk in chatter mode")
    parser.add_argument("--out", help="save the sound to a WAV file instead of playing it")
    parser.add_argument("--seed", type=int, help="repeat an exact sound")
    args = parser.parse_args()

    COMMANDS[args.command](args, random.Random(args.seed))


if __name__ == "__main__":
    main()
