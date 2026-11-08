from __future__ import annotations

import argparse
import difflib
import logging
import random
import re
import string
import sys
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

__version__ = "1.0.0"

BOT_NAME = "DecodeBot"
MAX_INPUT_LENGTH = 500
FUZZY_CUTOFF = 0.85
logger = logging.getLogger("decodebot")

class Intent(str, Enum):
    GREETING = "greeting"
    HOW_ARE_YOU = "how_are_you"
    NAME = "name"
    CREATOR = "creator"
    CAPABILITIES = "capabilities"
    RULE_BASED = "rule_based"
    THANKS = "thanks"
    HELP = "help"

class MatchType(str, Enum):
    EXACT = "exact"
    PHRASE = "phrase"
    FUZZY = "fuzzy"
    FALLBACK = "fallback"
    EMPTY = "empty"
    EXIT = "exit"

@dataclass(frozen=True)
class Reply:
    """Result of processing one user message."""
    text: str
    match_type: MatchType
    intent: Optional[Intent] = None
    should_exit: bool = False

RESPONSES: Dict[Intent, Sequence[str]] = {
    Intent.GREETING: (
        "Hello! How can I help you today?",
        "Hi there! What can I do for you?",
        "Hey! Nice to see you.",
    ),
    Intent.HOW_ARE_YOU: (
        "I'm just a bunch of if-else logic, but I'm running perfectly!",
        "All systems operational. How about you?",
    ),
    Intent.NAME: (
        f"I'm {BOT_NAME}, a rule-based chatbot built at DecodeLabs.",
    ),
    Intent.CREATOR: (
        "I was built as Project 1 of the DecodeLabs AI Industrial Training Kit.",
    ),
    Intent.CAPABILITIES: (
        "I can greet you, answer a few predefined questions, and exit when you "
        "say 'bye'. Type 'help' to see example inputs.",
    ),
    Intent.RULE_BASED: (
        "A rule-based chatbot follows explicit instructions written by a "
        "developer. It does not learn, which makes it predictable, traceable "
        "and free of hallucinations.",
    ),
    Intent.THANKS: (
        "You're welcome!",
        "Happy to help!",
    ),
    Intent.HELP: (
        "Try: hello, how are you, what is your name, who made you, "
        "what can you do, what is a rule based chatbot, thanks, or bye.",
    ),
}

PHRASES: Dict[Intent, Tuple[str, ...]] = {
    Intent.GREETING: ("hello", "hi", "hey", "good morning", "good afternoon",
                      "good evening"),
    Intent.HOW_ARE_YOU: ("how are you", "how are you doing", "whats up"),
    Intent.NAME: ("what is your name", "whats your name", "who are you"),
    Intent.CREATOR: ("who made you", "who created you", "who built you"),
    Intent.CAPABILITIES: ("what can you do", "what do you do"),
    Intent.RULE_BASED: ("what is a rule based chatbot", "rule based chatbot"),
    Intent.THANKS: ("thanks", "thank you"),
    Intent.HELP: ("help",),
}

EXIT_COMMANDS: FrozenSet[str] = frozenset(
    {"exit", "quit", "bye", "goodbye", "see you"}
)

WELCOME_MESSAGE = (
    f"Hi! I'm {BOT_NAME}, a rule-based chatbot. "
    "Type 'help' for ideas or 'exit' to quit."
)
GOODBYE_MESSAGE = "Goodbye! Have a great day."
INTERRUPT_MESSAGE = "Session interrupted. Goodbye!"
FALLBACK_MESSAGE = (
    "Sorry, I do not understand that. Type 'help' to see what I can answer."
)
EMPTY_INPUT_MESSAGE = "Please type something so I can respond."
TOO_LONG_MESSAGE = (
    f"That message is too long (max {MAX_INPUT_LENGTH} characters). "
    "Please shorten it."
)

_PUNCTUATION_TABLE = {ord(c): None for c in string.punctuation}
_PUNCTUATION_TABLE[ord("-")] = " "
_WHITESPACE_RE = re.compile(r"\s+")

def sanitize(raw_text: str) -> str:
    """Normalize raw input into a canonical form.

    Steps: Unicode normalization, lowercase, trim, punctuation removal and
    whitespace collapsing. Example: '  HELLO!!  ' -> 'hello'.
    """
    text = unicodedata.normalize("NFKC", raw_text).lower().strip()
    text = text.translate(_PUNCTUATION_TABLE)
    return _WHITESPACE_RE.sub(" ", text).strip()

class ChatBot:
    """Deterministic, rule-based response engine."""

    def __init__(
        self,
        enable_fuzzy: bool = True,
        rng: Optional[random.Random] = None,
    ) -> None:
        self._enable_fuzzy = enable_fuzzy
        self._rng = rng or random.Random()

        self._intent_map: Dict[str, Intent] = {
            phrase: intent
            for intent, phrases in PHRASES.items()
            for phrase in phrases
        }
        self._validate_knowledge_base()

        ordered = sorted(self._intent_map, key=len, reverse=True)
        self._phrase_pattern = re.compile(
            r"\b(" + "|".join(re.escape(p) for p in ordered) + r")\b"
        )

    def _validate_knowledge_base(self) -> None:
        """Fail fast at start-up if the data is inconsistent."""
        missing = [i.value for i in PHRASES if i not in RESPONSES]
        if missing:
            raise ValueError(f"Intents without responses: {missing}")
        if len(RESPONSES) < 5:
            raise ValueError("Knowledge base must contain at least 5 intents.")

    def _match(self, clean_input: str) -> Tuple[Optional[Intent], MatchType]:

        intent = self._intent_map.get(clean_input)
        if intent is not None:
            return intent, MatchType.EXACT

        found = self._phrase_pattern.search(clean_input)
        if found:
            return self._intent_map[found.group(1)], MatchType.PHRASE

        if self._enable_fuzzy:
            close: List[str] = difflib.get_close_matches(
                clean_input, self._intent_map.keys(), n=1, cutoff=FUZZY_CUTOFF
            )
            if close:
                return self._intent_map[close[0]], MatchType.FUZZY

        return None, MatchType.FALLBACK

    def respond(self, raw_text: str) -> Reply:
        """Process one raw user message and return a Reply."""
        if len(raw_text) > MAX_INPUT_LENGTH:
            return Reply(TOO_LONG_MESSAGE, MatchType.FALLBACK)

        clean_input = sanitize(raw_text)

        if not clean_input:
            return Reply(EMPTY_INPUT_MESSAGE, MatchType.EMPTY)

        if clean_input in EXIT_COMMANDS:
            return Reply(GOODBYE_MESSAGE, MatchType.EXIT, should_exit=True)

        intent, match_type = self._match(clean_input)
        if intent is None:
            return Reply(FALLBACK_MESSAGE, MatchType.FALLBACK)

        return Reply(
            text=self._rng.choice(RESPONSES[intent]),
            match_type=match_type,
            intent=intent,
        )

def run_chat_loop(bot: ChatBot, debug: bool = False) -> None:
    """Run the interactive loop until an exit command or interruption."""
    print(f"{BOT_NAME}: {WELCOME_MESSAGE}")

    while True:
        try:
            raw_text = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print(f"\n{BOT_NAME}: {INTERRUPT_MESSAGE}")
            logger.info("Session interrupted by user.")
            break

        reply = bot.respond(raw_text)
        logger.info(
            "user=%r intent=%s match=%s",
            raw_text,
            reply.intent.value if reply.intent else None,
            reply.match_type.value,
        )

        print(f"{BOT_NAME}: {reply.text}")
        if debug:
            intent_name = reply.intent.value if reply.intent else "none"
            print(f"  [debug] intent={intent_name} match={reply.match_type.value}")

        if reply.should_exit:
            break  

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DecodeLabs rule-based AI chatbot (Project 1)."
    )
    parser.add_argument("--debug", action="store_true",
                        help="show matched intent and match type")
    parser.add_argument("--no-fuzzy", action="store_true",
                        help="disable typo-tolerant matching")
    parser.add_argument("--log", metavar="FILE",
                        help="write a session log to FILE")
    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {__version__}")
    return parser.parse_args(argv)

def configure_logging(log_file: Optional[str]) -> None:
    if log_file:
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
        )
    else:
        logger.addHandler(logging.NullHandler())

def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log)
    bot = ChatBot(enable_fuzzy=not args.no_fuzzy)
    run_chat_loop(bot, debug=args.debug)
    return 0

if __name__ == "__main__":
    sys.exit(main())