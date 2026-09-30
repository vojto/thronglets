"""Claude Code hooks: the creature reacts to the conversation."""

import json
import sys

from audio import play_in_background
from creature import Creature
from feelings import NEUTRAL, blur
from jev import read_feelings


def on_reply(rng):
    """Stop hook: react to the assistant's last reply, then return immediately."""
    event = read_event()
    message = event.get("last_assistant_message") or last_message_from_transcript(event.get("transcript_path"))
    try:
        feelings = read_feelings(message, rng)
    except Exception:
        feelings = blur(NEUTRAL, rng, 0.2)
    play_in_background(Creature.for_session(event.get("session_id"), rng).phrase(feelings))


def on_prompt(rng):
    """UserPromptSubmit hook: squeak as if poked, then return immediately."""
    event = read_event()
    play_in_background(Creature.for_session(event.get("session_id"), rng).poke())


def read_event():
    return {} if sys.stdin.isatty() else json.load(sys.stdin)


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
