"""Validation utilities for content quality checks"""

import re
from typing import List, Tuple


def count_words(text: str) -> int:
    """Count words in text."""
    # Remove extra whitespace and count
    words = text.split()
    return len(words)


def extract_character_names(text: str) -> List[str]:
    """Extract potential character names (capitalized words)."""
    # Simple heuristic: words that start with capital letters
    # This is not perfect but works for basic validation
    words = text.split()
    potential_names = [w.strip('.,!?";:') for w in words if w and w[0].isupper()]
    # Remove common words
    common_words = {"The", "A", "An", "In", "On", "At", "To", "For", "Of", "And", "But", "Or", "I", "He", "She", "It", "They"}
    names = [name for name in potential_names if name not in common_words]
    return list(set(names))  # Unique names


def check_contains_keywords(text: str, keywords: List[str], case_sensitive: bool = False) -> Tuple[bool, List[str]]:
    """
    Check if text contains all keywords.

    Returns:
        Tuple of (all_found: bool, missing_keywords: List[str])
    """
    if not case_sensitive:
        text = text.lower()
        keywords = [k.lower() for k in keywords]

    missing = []
    for keyword in keywords:
        if keyword not in text:
            missing.append(keyword)

    return (len(missing) == 0, missing)


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate simple word overlap similarity between two texts.

    Returns:
        Float between 0 and 1 representing similarity
    """
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    if not words1 or not words2:
        return 0.0

    intersection = words1 & words2
    union = words1 | words2

    return len(intersection) / len(union) if union else 0.0


def validate_json_structure(data: dict, required_keys: List[str]) -> Tuple[bool, List[str]]:
    """
    Validate that a dictionary contains all required keys.

    Returns:
        Tuple of (valid: bool, missing_keys: List[str])
    """
    missing = [key for key in required_keys if key not in data]
    return (len(missing) == 0, missing)


def check_sentence_count(text: str) -> int:
    """Count approximate number of sentences in text."""
    # Simple heuristic: count sentence-ending punctuation
    return len(re.findall(r'[.!?]+', text))


def check_paragraph_count(text: str) -> int:
    """Count approximate number of paragraphs in text."""
    # Split by double newlines
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    return len(paragraphs)


def check_dialogue_present(text: str) -> bool:
    """Check if text contains dialogue (quotation marks)."""
    return '"' in text or '"' in text or '"' in text or "'" in text


def assess_narrative_quality(text: str) -> dict:
    """
    Perform basic quality assessment of narrative text.

    Returns dict with metrics:
        - word_count
        - sentence_count
        - paragraph_count
        - has_dialogue
        - avg_sentence_length
    """
    word_count = count_words(text)
    sentence_count = check_sentence_count(text)
    paragraph_count = check_paragraph_count(text)
    has_dialogue = check_dialogue_present(text)

    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "paragraph_count": paragraph_count,
        "has_dialogue": has_dialogue,
        "avg_sentence_length": round(avg_sentence_length, 1),
        "quality_score": calculate_quality_score(word_count, sentence_count, paragraph_count, has_dialogue)
    }


def calculate_quality_score(word_count: int, sentence_count: int, paragraph_count: int, has_dialogue: bool) -> float:
    """
    Calculate a simple quality score (0-10) based on basic metrics.
    """
    score = 0.0

    # Word count score (up to 3 points)
    if word_count >= 1000:
        score += 3.0
    elif word_count >= 800:
        score += 2.0
    elif word_count >= 600:
        score += 1.0

    # Paragraph variety (up to 2 points)
    if paragraph_count >= 5:
        score += 2.0
    elif paragraph_count >= 3:
        score += 1.0

    # Sentence variety (up to 2 points)
    if sentence_count >= 30:
        score += 2.0
    elif sentence_count >= 20:
        score += 1.0

    # Dialogue presence (up to 2 points)
    if has_dialogue:
        score += 2.0

    # Average sentence length check (up to 1 point)
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
    if 15 <= avg_sentence_length <= 25:
        score += 1.0

    return round(score, 1)
