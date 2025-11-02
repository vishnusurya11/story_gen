"""Context management tools for shared state across agents"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

CONTEXT_FILE = "session_cache/context.json"


def initialize_context() -> Dict[str, Any]:
    """Initialize a new context file with empty structure."""
    context = {
        "hero_arc": {},
        "source_story": {},
        "setting": {},
        "beats": [],
        "characters": [],
        "chapters": [],
        "current_step": "hero_arc",
        "current_beat_index": 0,
        "retry_count": 0,
        "errors": []
    }

    # Ensure directory exists
    os.makedirs("session_cache", exist_ok=True)

    with open(CONTEXT_FILE, "w") as f:
        json.dump(context, f, indent=2)

    return context


def load_context() -> Dict[str, Any]:
    """Load context from file, initialize if doesn't exist."""
    if not os.path.exists(CONTEXT_FILE):
        return initialize_context()

    with open(CONTEXT_FILE, "r") as f:
        return json.load(f)


def save_context(context: Dict[str, Any]) -> None:
    """Save context to file."""
    os.makedirs("session_cache", exist_ok=True)
    with open(CONTEXT_FILE, "w") as f:
        json.dump(context, f, indent=2)


def update_context_field(field_name: str, value: Any) -> None:
    """Update a specific field in the context."""
    context = load_context()
    context[field_name] = value
    save_context(context)


def get_context_field(field_name: str) -> Any:
    """Get a specific field from the context."""
    context = load_context()
    return context.get(field_name)


def get_previous_chapters(count: int = 3) -> list:
    """Get the last N chapters from context."""
    chapters = get_context_field("chapters") or []
    return chapters[-count:] if chapters else []


def add_chapter(chapter_data: Dict[str, Any]) -> None:
    """Add a chapter to the context."""
    context = load_context()
    context["chapters"].append(chapter_data)
    save_context(context)


def increment_retry_count() -> int:
    """Increment and return retry count."""
    context = load_context()
    context["retry_count"] += 1
    save_context(context)
    return context["retry_count"]


def reset_retry_count() -> None:
    """Reset retry count to 0."""
    update_context_field("retry_count", 0)


def add_error(error_message: str) -> None:
    """Add an error to the context."""
    context = load_context()
    context["errors"].append(error_message)
    save_context(context)
