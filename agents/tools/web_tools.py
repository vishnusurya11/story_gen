"""Web search and fetch tools for agents"""

import os
from typing import Optional, List, Dict, Any
from langchain_core.tools import tool


@tool
def web_search_tool(query: str) -> str:
    """
    Search the web for information.

    Args:
        query: The search query string

    Returns:
        Search results as formatted text
    """
    # Note: This is a placeholder. In actual implementation, this would use
    # the WebSearch tool available in the Claude Code environment.
    # For now, return a note to use the system WebSearch
    return f"WEB_SEARCH_NEEDED: {query}"


@tool
def web_fetch_tool(url: str, extraction_prompt: str) -> str:
    """
    Fetch and extract information from a URL.

    Args:
        url: The URL to fetch
        extraction_prompt: What information to extract

    Returns:
        Extracted information as text
    """
    # Note: This is a placeholder. In actual implementation, this would use
    # the WebFetch tool available in the Claude Code environment.
    return f"WEB_FETCH_NEEDED: {url} - {extraction_prompt}"


def search_and_summarize(query: str, llm: Any) -> str:
    """
    Perform a web search and use LLM to summarize results.

    Args:
        query: Search query
        llm: Language model instance

    Returns:
        Summarized search results
    """
    # Placeholder for web search integration
    # In practice, this would call the actual WebSearch tool
    from langchain_core.messages import HumanMessage

    prompt = f"""I need to search for: {query}

Please provide a comprehensive answer based on general knowledge about this topic.
If this is about a specific story, novel, or movie, include:
- Plot summary
- Main themes
- Key characters
- Story structure/beats
- Cultural significance

If this is about a story structure or writing technique, include:
- Definition and explanation
- How it works
- Examples
- When to use it
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


def search_story_information(story_name: str, llm: Any) -> Dict[str, Any]:
    """
    Search for information about a story and structure it.

    Args:
        story_name: Name of the story/novel/movie
        llm: Language model instance

    Returns:
        Dictionary with story information
    """
    from langchain_core.messages import HumanMessage
    import json

    prompt = f"""Provide detailed information about the story "{story_name}" in JSON format:

{{
  "title": "Full official title",
  "author_or_creator": "Name of author/director",
  "type": "novel/movie/short story/play/myth",
  "year": "Year created/published",
  "genre": "Primary genre",
  "plot_summary": "Brief plot summary (2-3 sentences)",
  "main_theme": "Central theme",
  "protagonist": "Main character name",
  "antagonist": "Opposing force or character",
  "key_beats": [
    "Opening situation",
    "Inciting incident",
    "Midpoint turning point",
    "Climax",
    "Resolution"
  ],
  "story_structure": "Description of how the story is structured"
}}

Provide only valid JSON, no additional text."""

    response = llm.invoke([HumanMessage(content=prompt)])

    try:
        # Try to parse JSON from response
        content = response.content.strip()
        # Remove markdown code blocks if present
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]

        return json.loads(content.strip())
    except json.JSONDecodeError:
        # Fallback if JSON parsing fails
        return {
            "title": story_name,
            "type": "unknown",
            "plot_summary": response.content[:200],
            "main_theme": "To be determined",
            "story_structure": response.content
        }


def search_setting_inspiration(setting_genre: str, llm: Any) -> str:
    """
    Search for world-building inspiration for a given setting.

    Args:
        setting_genre: Type of setting (medieval, sci-fi, etc.)
        llm: Language model instance

    Returns:
        Rich setting description and world-building details
    """
    from langchain_core.messages import HumanMessage

    prompt = f"""Create detailed world-building elements for a {setting_genre} setting.

Include:
1. Visual atmosphere and aesthetic
2. Technology/magic level
3. Social structures
4. Common architectural styles
5. Transportation methods
6. Daily life details
7. Cultural norms
8. Typical conflicts or tensions
9. Sensory details (what it looks/sounds/smells like)
10. Unique world-building elements that make it memorable

Be creative and specific. This is for a novel, so make it vivid and immersive."""

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


def search_character_names(setting_genre: str, culture: str, llm: Any) -> List[str]:
    """
    Generate appropriate character names for a given setting.

    Args:
        setting_genre: Type of setting
        culture: Cultural inspiration
        llm: Language model instance

    Returns:
        List of potential character names
    """
    from langchain_core.messages import HumanMessage

    prompt = f"""Generate 20 character names appropriate for a {setting_genre} setting with {culture} cultural influences.

Provide a mix of:
- 10 protagonist-type names (heroic, memorable)
- 5 antagonist-type names (powerful, imposing)
- 5 supporting character names (diverse, interesting)

List only the names, one per line, no descriptions."""

    response = llm.invoke([HumanMessage(content=prompt)])

    # Parse names from response
    names = [line.strip().strip('- ').strip('* ') for line in response.content.split('\n') if line.strip()]
    return [name for name in names if name and not name.startswith('#')]


def search_genre_writing_techniques(genre: str, llm: Any) -> str:
    """
    Search for writing techniques specific to a genre.

    Args:
        genre: Story genre
        llm: Language model instance

    Returns:
        Writing technique guidelines
    """
    from langchain_core.messages import HumanMessage

    prompt = f"""What are the key writing techniques and style conventions for {genre} fiction?

Include:
1. Pacing expectations
2. Narrative voice (first/third person preferences)
3. Description style
4. Dialogue conventions
5. Tone and atmosphere
6. Common tropes to embrace or avoid
7. Reader expectations
8. Sentence structure and rhythm

Be specific and actionable for a writer."""

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content
