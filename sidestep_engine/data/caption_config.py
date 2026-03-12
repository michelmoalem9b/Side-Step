"""
Shared caption generation configuration.

Provides the system prompt (user-overridable from file), default
generation parameters, and prompt building utilities used by all
caption providers.  Custom prompts are loaded from
``~/.config/sidestep/caption_prompt.txt`` (or platform equivalent).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Default generation parameters ────────────────────────────────────
DEFAULT_TEMPERATURE: float = 0.45
DEFAULT_MAX_TOKENS: int = 4096
DEFAULT_TOP_P: float = 0.9
DEFAULT_PRESENCE_PENALTY: float = 0.0
DEFAULT_FREQUENCY_PENALTY: float = 0.0
DEFAULT_REPETITION_PENALTY: float = 1.05
DEFAULT_GEMINI_MODEL: str = "gemini-2.5-flash"
DEFAULT_OPENAI_MODEL: str = "gpt-4o"
DEFAULT_LOCAL_MODEL: str = "Qwen/Qwen2.5-Omni-7B"

GEMINI_MODEL_SUGGESTIONS: list[str] = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-pro",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite-preview",
    "gemini-3.1-pro-preview",
]

OPENAI_MODEL_SUGGESTIONS: list[str] = [
    "gpt-4o",
    "gpt-4o-audio-preview",
    "gpt-4o-mini",
    "gpt-4o-mini-audio-preview",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-audio",
    "gpt-audio-1.5",
    "gpt-audio-mini",
]

DEFAULT_LOCAL_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating "
    "text and speech."
)

_DEFAULT_PROMPT_INSTRUCTIONS = (
    "Write high-quality structured music dataset metadata grounded in the song's "
    "audible content. If audio is attached, analyze the actual audio first and use "
    "title, artist, and lyrics only as weak secondary context.\n\n"
    "Return EXACTLY 5 lines in plain text and nothing else. Each field must start at "
    "the beginning of its own new line. Never place two fields on the same line.\n\n"
    "Use this exact output template:\n"
    "caption: <single-line paragraph with EXACTLY 9 complete sentences>\n"
    "genre: <comma-separated genre/style tags, e.g. 'bass house, electro house'>\n"
    "bpm: <estimated BPM as integer, e.g. 120>\n"
    "key: <musical key, e.g. 'C minor' or 'F# major'>\n"
    "signature: <time signature, e.g. '4/4'>\n\n"
    "Caption rules:\n"
    "- The caption must be one line with exactly 9 complete sentences.\n"
    "- Start `caption:` on line 1, `genre:` on line 2, `bpm:` on line 3, `key:` on line 4, and `signature:` on line 5.\n"
    "- Do not merge fields together. For example, do not output `genre: ... bpm: ... key: ...` on one line.\n"
    "- Do not use markdown, bullets, numbering, code fences, labels before the template, or commentary after the template.\n"
    "- Write for machine-learning training metadata, not for reviews, marketing copy, or listener-facing blurbs.\n"
    "- Prioritize musically useful descriptors: groove, drum pattern, bass design, harmonic density, melodic motifs, instrumentation, synthesis, timbre, texture, dynamics, stereo space, effects, and arrangement.\n"
    "- Use concrete audio evidence such as syncopated hats, sidechained bass, plucked synth lead, saturated kick, washed reverb tails, filtered risers, wide stereo pads, chopped vocal textures, dry upfront drums, or distorted reese bass when applicable.\n"
    "- Avoid vague or low-value phrases such as 'nice vibe', 'good energy', 'keeps you moving', 'hard to resist', 'captivating journey', 'atmospheric progression', 'listeners feel', or 'emotionally resonant' unless backed by specific sonic detail.\n"
    "- Avoid generic openings like 'This track is' when more specific wording can be used immediately.\n"
    "- Keep explicit metadata in the dedicated fields, not in the caption: do not state exact BPM numbers, key names, or time signatures in the caption unless absolutely necessary for a rare musically specific point.\n"
    "- Mention stereo width, panning, depth, or imaging only when those traits are clearly audible with high confidence; if uncertain, prefer safer mix descriptors such as dry, wet, dense, open, compressed, bright, dark, upfront, distant, or saturated.\n"
    "- If you mention a buildup, drop, break, climax, or outro, specify what changes musically: which layers enter, which layers drop out, which filters open, how percussion changes, how the bass changes, or how the energy is reshaped.\n"
    "- Sentence 1: identify the core genre/subgenre, tempo feel, groove character, and overall intensity without restating exact metadata fields.\n"
    "- Sentence 2: describe drum design and groove, including kick, snare/clap, hats, percussion, swing, syncopation, and pulse.\n"
    "- Sentence 3: describe the bass design and low-end behavior, including weight, tone, movement, rhythm, sustain, and kick interaction.\n"
    "- Sentence 4: describe harmony and melody, including chords, tonal center, motifs, riffs, leads, pads, stabs, arps, or vocal hooks.\n"
    "- Sentence 5: describe sound design and timbre, including synth character, source type, texture, brightness, distortion, saturation, envelopes, layering, and spectral character.\n"
    "- Sentence 6: describe mix treatment and space, including reverb, delay, compression, transient shape, filtering, automation, density, and only clearly audible spatial placement if confidence is high.\n"
    "- Sentence 7: describe the opening section and buildup in detail, including which elements are introduced first and how tension is created.\n"
    "- Sentence 8: describe the drop and any break section in detail, including what the drop contains, what hits hardest, and which elements are removed or exposed during the break.\n"
    "- Sentence 9: describe the late-song payoff, climax, or outro in detail, including how the track escalates, peaks, resolves, strips back, or closes.\n"
    "- Do not mention the artist name or song title in the caption.\n"
    "- If audio is not attached or a field cannot be determined from available evidence, write N/A for that field instead of guessing."
)


def _prompt_override_path() -> Path:
    """Path to the user's custom system prompt file."""
    from sidestep_engine.settings import settings_dir
    return settings_dir() / "caption_prompt.txt"


def _generation_override_path() -> Path:
    """Path to the user's custom generation-settings override file."""
    from sidestep_engine.settings import settings_dir
    return settings_dir() / "caption_generation.json"


def get_system_prompt(provider: str = "api") -> Optional[str]:
    """Return the system prompt for the given provider.

    For ``"local"`` providers, returns the built-in local system prompt.
    For API providers, returns ``None`` — they use
    :func:`get_prompt_instructions` instead (injected as a user message).

    Returns:
        System prompt string, or ``None`` for API providers.
    """
    if provider == "local":
        return DEFAULT_LOCAL_SYSTEM_PROMPT
    return None


def get_prompt_instructions() -> str:
    """Return the prompt instructions, with optional user file override.

    Checks ``~/.config/sidestep/caption_prompt.txt`` for a user-supplied
    prompt.  Falls back to the built-in default.
    """
    p = _prompt_override_path()
    if p.is_file():
        try:
            text = p.read_text(encoding="utf-8").strip()
            if text:
                logger.debug("Loaded custom caption prompt from %s", p)
                return text
        except OSError as exc:
            logger.warning("Failed to read custom prompt: %s", exc)
    return _DEFAULT_PROMPT_INSTRUCTIONS


def get_generation_settings() -> dict[str, float | int]:
    """Load generation parameters, merging user overrides from file.

    Reads ``~/.config/sidestep/caption_generation.json`` if present.
    Unknown keys are ignored; invalid values fall back to defaults.
    """
    settings: dict[str, float | int] = {
        "temperature": DEFAULT_TEMPERATURE,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "top_p": DEFAULT_TOP_P,
        "presence_penalty": DEFAULT_PRESENCE_PENALTY,
        "frequency_penalty": DEFAULT_FREQUENCY_PENALTY,
        "repetition_penalty": DEFAULT_REPETITION_PENALTY,
    }
    p = _generation_override_path()
    if not p.is_file():
        return settings
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read caption generation settings: %s", exc)
        return settings
    if not isinstance(raw, dict):
        return settings
    for key in settings:
        value = raw.get(key)
        if isinstance(settings[key], int):
            if isinstance(value, (int, float)) and int(value) > 0:
                settings[key] = int(value)
        else:
            if isinstance(value, (int, float)):
                settings[key] = float(value)
    return settings


def resolve_generation_settings(**overrides: object) -> dict[str, float | int]:
    """Merge caller overrides on top of file-based + default settings."""
    settings = get_generation_settings()
    for key, value in overrides.items():
        if value is None or key not in settings:
            continue
        if isinstance(settings[key], int):
            if isinstance(value, (int, float)) and int(value) > 0:
                settings[key] = int(value)
        else:
            if isinstance(value, (int, float)):
                settings[key] = float(value)
    return settings


def build_user_prompt(
    title: str,
    artist: str,
    lyrics_excerpt: str = "",
    *,
    audio_attached: bool = False,
    google_search: bool = False,
) -> str:
    """Build the user prompt from song metadata.

    Args:
        title: Song title.
        artist: Artist name.
        lyrics_excerpt: Optional lyrics text (truncated to 500 chars).
        audio_attached: Whether an audio file is attached to this request.
        google_search: Whether Google Search grounding is enabled.

    Returns:
        Formatted user prompt string.
    """
    prompt = (
        f"{get_prompt_instructions()}\n\n"
        "Song metadata:\n"
        f"Audio attached to this request: {'yes' if audio_attached else 'no'}\n"
        f"Title: {title}\n"
        f"Artist: {artist}"
    )
    if lyrics_excerpt:
        prompt += f"\nLyrics excerpt:\n{lyrics_excerpt[:500]}"
    prompt += (
        "\n\nReminder: audible evidence comes first; metadata and lyrics are secondary."
    )
    if google_search:
        prompt += (
            "\n\nYou have access to Google Search. You may search for the track "
            "by title and artist to find genre classifications, release context, "
            "or production details that can enrich your description. However, "
            "everything you hear in the audio takes absolute priority over "
            "anything found online — never contradict or override what you "
            "actually hear. Use search results only to supplement and add "
            "context that the audio alone cannot provide."
        )
    return prompt


# Keys we expect from the structured AI response
_STRUCTURED_KEYS = frozenset({"caption", "genre", "bpm", "key", "signature"})
_STRUCTURED_FIELD_RE = re.compile(
    r"(?is)(?<!\w)(caption|genre|bpm|key|signature)\s*:",
)


def _normalize_structured_value(value: str) -> str:
    """Collapse whitespace and strip a structured field value."""
    return re.sub(r"\s+", " ", value).strip()


def _split_structured_tail(key: str, value: str) -> tuple[str, str]:
    """Split a structured value into (clean_value, leftover_tail) by key type."""
    normalized = _normalize_structured_value(value)
    if key == "genre":
        return normalized.rstrip(" .;,"), ""
    if key == "bpm":
        match = re.match(r"^(\d{1,3})(.*)$", normalized)
        if match:
            return match.group(1), match.group(2).lstrip(" .:-")
        return normalized, ""
    if key == "signature":
        match = re.match(r"^(\d{1,2}(?:\s*/\s*\d{1,2})?)(.*)$", normalized)
        if match:
            return match.group(1).replace(" ", ""), match.group(2).lstrip(" .:-")
        return normalized, ""
    if key == "key":
        match = re.match(
            r"^([A-G](?:#|b)?(?:\s+(?:major|minor))?)(?=$|[\s,.;:/-])(.*)$",
            normalized,
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(1).strip(), match.group(2).lstrip(" .:-")
        return normalized, ""
    return normalized, ""


def parse_structured_response(text: str) -> dict[str, str]:
    """Parse a structured ``key: value`` AI response into a dict.

    Falls back to treating the entire text as a caption if no
    recognised keys are found (backward compat with plain-text
    prompts or user-overridden system prompts).

    Args:
        text: Raw response text from the caption provider.

    Returns:
        Dict with sidecar-compatible keys.  Values that are
        ``"N/A"`` or empty are omitted.
    """
    if not text:
        return {}

    matches = list(_STRUCTURED_FIELD_RE.finditer(text))
    if not matches:
        return {"caption": text.strip()}

    result: dict[str, str] = {}
    caption_fragments: list[str] = []

    prefix = text[:matches[0].start()].strip(" \t\r\n:-")
    if prefix:
        caption_fragments.append(_normalize_structured_value(prefix))

    for idx, match in enumerate(matches):
        key = match.group(1).lower()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        raw_value = text[start:end].strip()
        if not raw_value:
            continue
        value, tail = _split_structured_tail(key, raw_value)
        if value and value.lower() != "n/a":
            if key == "caption" and key in result:
                result[key] = f"{result[key]} {value}".strip()
            else:
                result[key] = value
        if tail and tail.lower() != "n/a":
            caption_fragments.append(tail)

    if caption_fragments:
        recovered_caption = _normalize_structured_value(" ".join(caption_fragments))
        if recovered_caption:
            if result.get("caption"):
                result["caption"] = (
                    f"{result['caption']} {recovered_caption}"
                ).strip()
            else:
                result["caption"] = recovered_caption

    # Fallback: if no structured keys found, treat whole text as caption
    if not result:
        result["caption"] = text.strip()

    return result
