"""
Pydantic schemas for structured LLM outputs.

All agents use these schemas with LangChain's .with_structured_output()
to ensure reliable, validated JSON responses from LLMs.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


# Agent 2: Story Source Agent
class StoryInfo(BaseModel):
    """Information about the source story to retell/adapt."""
    title: str = Field(description="The title of the source story")
    type: str = Field(description="Type of story (novel, film, myth, short story, etc.)")
    author_or_creator: str = Field(description="Original author or creator")
    main_theme: str = Field(description="The core theme of the story")
    story_structure: str = Field(description="Brief description of the story's structure")
    plot_summary: str = Field(description="2-3 paragraph plot summary")
    key_beats: List[str] = Field(description="List of 5-7 major story beats or plot points")


# Agent 3: Setting Agent
class SettingInfo(BaseModel):
    """Rich world-building details for the story setting."""
    genre: str = Field(description="The setting genre (medieval, sci-fi, space, etc.)")
    description: str = Field(description="Detailed 2-3 paragraph description of the world")
    key_locations: List[str] = Field(description="3-5 important locations in this world")
    atmosphere: str = Field(description="Overall mood and atmosphere of the setting")
    unique_elements: List[str] = Field(description="2-4 unique aspects that make this setting distinctive")


# Agent 4: Beat Sheet Agent
class BeatDescription(BaseModel):
    """Single beat in the Save the Cat story structure."""
    number: int = Field(description="Beat number (1-15)")
    name: str = Field(description="Name of the beat (e.g., 'Opening Image', 'Catalyst')")
    percentage: str = Field(description="Percentage point in story where beat occurs")
    description: str = Field(description="Detailed 2-4 paragraph description of what happens in this beat for THIS specific story")


# Agent 5: Character Agent
class Character(BaseModel):
    """Detailed character profile."""
    name: str = Field(description="Character's name")
    role: str = Field(description="Role in story (Hero, Antagonist, Mentor, Love Interest, etc.)")
    age: str = Field(description="Approximate age or age range")
    primary_want: str = Field(description="What the character consciously desires (external goal)")
    primary_need: str = Field(description="What the character actually needs to become whole (internal growth)")
    personality_traits: str = Field(description="2-3 sentences describing personality")
    backstory: str = Field(description="Brief backstory (2-3 sentences)")
    internal_flaw: str = Field(description="Their internal flaw or wound")
    arc_summary: str = Field(description="How they will change (or fail to change) through the story")
    key_conflicts: str = Field(description="Internal and external conflicts they face")


# Agent 5.5: Scene Breakdown Agent
class Scene(BaseModel):
    """Single scene within a chapter."""
    scene_number: int = Field(description="Scene number within this chapter (1, 2, 3...)")
    setting: str = Field(description="Specific location for this scene (e.g., 'Cargo bay of starship at dawn')")
    characters: List[str] = Field(description="Names of characters present in this scene")
    time_mood: str = Field(description="Time of day and emotional atmosphere (e.g., 'Early morning, tense and uncertain')")
    purpose: str = Field(description="What this scene accomplishes narratively")
    key_events: List[str] = Field(description="3-5 specific things that happen in this scene")
    emotional_beat: str = Field(description="How characters feel or change emotionally during this scene")
    word_target: int = Field(description="Target word count for this scene (200-300)", ge=200, le=300)


class SceneList(BaseModel):
    """Collection of scenes for a chapter."""
    scenes: List[Scene] = Field(description="List of 5-10 scenes for this chapter")


# Story Source - Web Search Output (if needed)
class StorySearchResult(BaseModel):
    """Result from web search about a story."""
    title: str = Field(description="Story title")
    summary: str = Field(description="Brief summary of what was found")
    relevant_info: str = Field(description="Key information about the story")


# Setting Inspiration - Web Search Output (if needed)
class SettingSearchResult(BaseModel):
    """Result from web search about setting inspiration."""
    genre: str = Field(description="The setting genre")
    inspiration: str = Field(description="Inspirational details about this type of setting")
    examples: List[str] = Field(description="2-3 examples of works in this setting")
