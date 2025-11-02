#!/usr/bin/env python3
"""
Multi-Agent Story Generation System - Orchestrator
Main entry point for the story generation pipeline
"""

import os
import sys
import json
import random
from pathlib import Path
from typing import TypedDict, Annotated, Literal
from datetime import datetime
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END, START

# Import tools
from agents.tools.context_tools import (
    initialize_context, load_context, save_context,
    update_context_field, get_context_field, add_chapter,
    increment_retry_count, reset_retry_count
)
from agents.tools.validation_tools import count_words, assess_narrative_quality
from agents.tools.web_tools import search_story_information, search_setting_inspiration

# Import prompts
from agents.prompts.hero_arc_prompts import (
    HERO_ARC_SYSTEM_PROMPT, HERO_ARC_USER_PROMPT,
    HERO_ARC_DESCRIPTIONS, RANDOM_ARC_MESSAGE
)
from agents.prompts.beat_sheet_prompts import SAVE_THE_CAT_BEATS

# Import Pydantic schemas for structured output
from agents.schemas import (
    StoryInfo, SettingInfo, BeatDescription,
    Character, Scene, SceneList
)

# Load environment variables
load_dotenv()


# Initialize LLM
def get_llm(schema=None):
    """
    Initialize and return the appropriate LLM based on configuration.

    Args:
        schema: Optional Pydantic BaseModel for structured output.
                If provided, returns LLM with .with_structured_output() applied.

    Returns:
        LLM instance (optionally wrapped for structured output)
    """
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")

        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.7
        )

        # Use json_schema method for more reliable structured output with Gemini
        if schema:
            return llm.with_structured_output(schema, method="json_schema")
        return llm

    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        base_url = os.getenv("OLLAMA_BASE_URL", "https://ollama.com")
        model = os.getenv("OLLAMA_MODEL", "minimax-m2:cloud")
        api_key = os.getenv("OLLAMA_API_KEY")

        if "ollama.com" in base_url and not api_key:
            raise ValueError(
                "OLLAMA_API_KEY is required for Ollama Cloud.\n"
                "Steps to fix:\n"
                "1. Sign in: ollama signin\n"
                "2. Create API key at: https://ollama.com/settings/keys\n"
                "3. Add to .env file: OLLAMA_API_KEY=your_key_here"
            )

        kwargs = {
            "base_url": base_url,
            "model": model,
            "temperature": 0.7
        }

        if api_key:
            kwargs["api_key"] = api_key

        llm = ChatOllama(**kwargs)

        # Ollama with tool-calling models supports structured output
        if schema:
            return llm.with_structured_output(schema)
        return llm

    else:
        raise ValueError(f"Unknown LLM provider: {provider}. Choose 'gemini' or 'ollama'")


# Define the state for our multi-agent system
class StoryGenerationState(TypedDict):
    """State shared across all agents in the story generation workflow."""
    hero_arc: dict
    source_story: dict
    setting: dict
    beats: list
    characters: list
    chapters: list
    current_step: str
    current_beat_index: int
    retry_count: int
    errors: list
    output_folder: str  # Story-specific output folder path


# Helper Functions

def generate_story_folder_name(source_title: str, setting_genre: str) -> str:
    """Generate unique folder name for this story."""
    # Clean title - remove special characters
    clean_title = "".join(c if c.isalnum() or c == " " else "" for c in source_title)
    clean_title = "_".join(clean_title.split())[:30]  # Max 30 chars

    # Clean setting
    clean_setting = setting_genre.replace("-", "").capitalize()

    # Timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

    folder_name = f"{clean_title}_{clean_setting}_{timestamp}"
    return folder_name


def create_story_folder(folder_name: str) -> str:
    """Create story-specific output folder and return full path."""
    base_path = Path("story_output")
    story_path = base_path / folder_name
    chapters_path = story_path / "chapters"

    # Create directories
    chapters_path.mkdir(parents=True, exist_ok=True)

    print(f"\n📁 Created story folder: {story_path}")

    return str(story_path)


def add_to_story_index(folder_name: str, state: StoryGenerationState) -> None:
    """Add story to index.json for tracking all generated stories."""
    index_path = Path("story_output/index.json")

    # Load existing index or create new
    if index_path.exists():
        with open(index_path, 'r') as f:
            index = json.load(f)
    else:
        index = {"stories": []}

    # Add this story
    story_entry = {
        "folder": folder_name,
        "title": state["source_story"]["title"],
        "setting": state["setting"]["genre"],
        "hero_arc": state["hero_arc"]["name"],
        "created_at": datetime.now().isoformat(),
        "status": "in_progress"
    }

    index["stories"].append(story_entry)

    # Save index
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)


def update_story_index_status(folder_name: str, status: str) -> None:
    """Update the status of a story in index.json."""
    index_path = Path("story_output/index.json")

    if not index_path.exists():
        return

    with open(index_path, 'r') as f:
        index = json.load(f)

    # Find and update story
    for story in index["stories"]:
        if story["folder"] == folder_name:
            story["status"] = status
            story["completed_at"] = datetime.now().isoformat()
            break

    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)


# Agent Node Functions

def hero_arc_agent(state: StoryGenerationState) -> StoryGenerationState:
    """Agent 1: Hero Arc Decision"""
    print("\n" + "=" * 70)
    print("AGENT 1: HERO ARC DECISION")
    print("=" * 70)

    print(HERO_ARC_USER_PROMPT)

    user_choice = input("\nYour choice: ").strip()

    # Validate choice
    if user_choice not in ["1", "2", "3", "4"]:
        print(f"\n{RANDOM_ARC_MESSAGE}")
        user_choice = random.choice(["1", "2", "3", "4"])
        print(f"Selected: ({user_choice}) {HERO_ARC_DESCRIPTIONS[user_choice]['name']}")

    arc_data = HERO_ARC_DESCRIPTIONS[user_choice]

    state["hero_arc"] = {
        "choice": user_choice,
        "name": arc_data["name"],
        "description": arc_data["description"],
        "want_need_relationship": arc_data["want_need"],
        "examples": arc_data["examples"]
    }

    # Save to context
    update_context_field("hero_arc", state["hero_arc"])

    print(f"\n✓ Hero Arc Selected: {arc_data['name']}")
    print(f"  {arc_data['description']}")

    return state


def story_source_agent(state: StoryGenerationState) -> StoryGenerationState:
    """Agent 2: Story Source Selection"""
    print("\n" + "=" * 70)
    print("AGENT 2: STORY SOURCE SELECTION")
    print("=" * 70)

    print("\nWhich story (novel, film, myth, short story) would you like to retell or adapt?")
    print("Examples: Pride and Prejudice, Star Wars, The Odyssey, Sherlock Holmes, Cinderella")

    user_story = input("\nYour choice: ").strip()

    # Default corpus
    default_stories = [
        "Pride and Prejudice", "Star Wars", "The Odyssey",
        "Cinderella", "Sherlock Holmes", "The Matrix",
        "Hamlet", "Dune"
    ]

    if not user_story:
        user_story = random.choice(default_stories)
        print(f"\nNo input provided. Randomly selected: {user_story}")

    print(f"\nSearching for information about '{user_story}'...")

    # Use structured LLM to get story information
    structured_llm = get_llm(schema=StoryInfo)

    prompt = f"""Provide detailed information about the story "{user_story}".

Include:
- Full official title
- Type (novel/film/myth/short story/play/etc.)
- Original author or creator
- Main theme or central message
- Story structure overview
- Plot summary (2-3 paragraphs)
- Key story beats (5-7 major plot points)

Be comprehensive and accurate."""

    from langchain_core.messages import HumanMessage
    story_info = structured_llm.invoke([HumanMessage(content=prompt)])

    state["source_story"] = {
        "title": story_info.title,
        "type": story_info.type,
        "author": story_info.author_or_creator,
        "core_theme": story_info.main_theme,
        "core_structure_summary": story_info.story_structure,
        "plot_summary": story_info.plot_summary,
        "key_beats": story_info.key_beats
    }

    update_context_field("source_story", state["source_story"])

    print(f"\n✓ Source Story: {state['source_story']['title']}")
    print(f"  Type: {state['source_story']['type']}")
    print(f"  Theme: {state['source_story']['core_theme']}")

    return state


def setting_agent(state: StoryGenerationState) -> StoryGenerationState:
    """Agent 3: Setting Selection"""
    print("\n" + "=" * 70)
    print("AGENT 3: SETTING SELECTION")
    print("=" * 70)

    print("\nWhat kind of setting would you like?")
    print("Options: medieval, futuristic, sci-fi, space, steampunk, modern")

    user_setting = input("\nYour choice: ").strip().lower()

    valid_settings = ["medieval", "futuristic", "sci-fi", "space", "steampunk", "modern"]

    if user_setting not in valid_settings:
        user_setting = random.choice(valid_settings)
        print(f"\nInvalid input. Randomly selected: {user_setting}")

    print(f"\nGenerating rich world-building for {user_setting} setting...")

    # Use structured LLM to generate setting details
    structured_llm = get_llm(schema=SettingInfo)

    prompt = f"""Create detailed world-building for a {user_setting} setting.

Include:
1. The genre ({user_setting})
2. Rich description (2-3 paragraphs) covering visual atmosphere, technology/magic level, and daily life
3. Key locations (3-5 important places in this world)
4. Overall atmosphere and mood
5. Unique elements that make this setting distinctive

Be creative, vivid, and immersive. This is for a novel."""

    from langchain_core.messages import HumanMessage
    setting_info = structured_llm.invoke([HumanMessage(content=prompt)])

    state["setting"] = {
        "genre": setting_info.genre,
        "description": setting_info.description,
        "key_locations": setting_info.key_locations,
        "atmosphere": setting_info.atmosphere,
        "unique_elements": setting_info.unique_elements
    }

    update_context_field("setting", state["setting"])

    print(f"\n✓ Setting: {setting_info.genre.capitalize()}")
    print(f"  {setting_info.description[:200]}...")

    # Create story-specific output folder
    folder_name = generate_story_folder_name(
        state["source_story"]["title"],
        user_setting
    )
    output_folder = create_story_folder(folder_name)
    state["output_folder"] = output_folder

    # Add to story index
    add_to_story_index(folder_name, state)

    return state


def beat_sheet_agent(state: StoryGenerationState) -> StoryGenerationState:
    """Agent 4: Beat Sheet Generation"""
    print("\n" + "=" * 70)
    print("AGENT 4: BEAT SHEET GENERATION")
    print("=" * 70)
    print("\nGenerating 15-beat story structure (Save the Cat method)...")

    structured_llm = get_llm(schema=BeatDescription)
    beats = []

    hero_arc = state["hero_arc"]
    source_story = state["source_story"]
    setting = state["setting"]

    # Generate each beat
    from langchain_core.messages import HumanMessage
    for beat_template in SAVE_THE_CAT_BEATS:
        print(f"\n  Generating Beat {beat_template['number']}: {beat_template['name']}...", end="", flush=True)

        # Create context-aware prompt for this beat
        previous_beats_summary = "\n".join([f"- {b['name']}: {b['description'][:100]}..." for b in beats[-3:]]) if beats else "None yet"

        prompt = f"""Generate Beat #{beat_template['number']}: **{beat_template['name']}** for a novel.

Beat Number: {beat_template['number']}
Beat Name: {beat_template['name']}
Beat Percentage: {beat_template['percentage']}
Beat Purpose: {beat_template['description']}

Story Context:
- Hero Arc: {hero_arc['name']} - {hero_arc['description']}
- Source Story: {source_story['title']} ({source_story['type']})
- Core Theme: {source_story['core_theme']}
- Setting: {setting['genre']}
- Previous Beats: {previous_beats_summary}

Create a detailed, specific description of what happens in this beat for THIS story.
- Be concrete and actionable
- Include character emotions and motivations
- Connect to the hero arc
- Fit the {setting['genre']} setting
- Adapt elements from {source_story['title']}

Provide 2-4 paragraphs describing this beat. Be creative and engaging."""

        beat_info = structured_llm.invoke([HumanMessage(content=prompt)])

        beat_data = {
            "number": beat_info.number,
            "name": beat_info.name,
            "percentage": beat_info.percentage,
            "description": beat_info.description
        }

        beats.append(beat_data)
        print(" ✓")

    state["beats"] = beats
    update_context_field("beats", beats)

    print(f"\n✓ Generated {len(beats)} beats successfully!")

    return state


def character_agent(state: StoryGenerationState) -> StoryGenerationState:
    """Agent 5: Character Generation"""
    print("\n" + "=" * 70)
    print("AGENT 5: CHARACTER GENERATION")
    print("=" * 70)
    print("\nGenerating characters...")

    structured_llm = get_llm(schema=Character)
    hero_arc = state["hero_arc"]
    beats = state["beats"]
    setting = state["setting"]
    source_story = state["source_story"]

    from langchain_core.messages import HumanMessage

    # Generate protagonist
    print("\n  Creating protagonist...", end="", flush=True)
    protagonist_prompt = f"""Create a detailed protagonist for this story:

Role: Hero
Hero Arc: {hero_arc['name']} - {hero_arc['want_need_relationship']}
Setting: {setting['genre']}
Source Inspiration: {source_story['title']}

The protagonist must have:
1. A clear WANT (external goal)
2. A clear NEED (internal growth required)
3. A compelling backstory
4. Personality traits
5. Internal flaw/wound

Based on the {hero_arc['name']}, remember:
{hero_arc['description']}

Create a complete character profile with all required fields."""

    protagonist = structured_llm.invoke([HumanMessage(content=protagonist_prompt)])
    print(" ✓")

    # Generate antagonist
    print("  Creating antagonist...", end="", flush=True)
    antagonist_prompt = f"""Create a compelling antagonist for this story:

Role: Antagonist
Setting: {setting['genre']}
Source Inspiration: {source_story['title']}
Hero's Goal: {protagonist.primary_want}

The antagonist opposes the hero and represents a thematic contrast.
Create a complete character profile with all required fields."""

    antagonist = structured_llm.invoke([HumanMessage(content=antagonist_prompt)])
    print(" ✓")

    # Generate supporting character (mentor/ally)
    print("  Creating supporting characters...", end="", flush=True)
    support_prompt = f"""Create a supporting character (mentor or ally) for this story:

Role: Mentor/Ally
Setting: {setting['genre']}
This character should guide or support the protagonist.
Create a complete character profile with all required fields."""

    support_char = structured_llm.invoke([HumanMessage(content=support_prompt)])
    print(" ✓")

    # Store characters as structured objects
    characters = [
        {"type": "protagonist", "data": protagonist.model_dump()},
        {"type": "antagonist", "data": antagonist.model_dump()},
        {"type": "supporting", "data": support_char.model_dump()}
    ]

    state["characters"] = characters
    update_context_field("characters", characters)

    print(f"\n✓ Generated {len(characters)} character profiles!")

    return state


def scene_breakdown_agent(state: StoryGenerationState) -> StoryGenerationState:
    """Agent 5.5: Scene Breakdown - Generate 5-10 scenes for each beat/chapter"""
    print("\n" + "=" * 70)
    print("AGENT 5.5: SCENE BREAKDOWN GENERATION")
    print("=" * 70)
    print("\nBreaking each beat into 5-10 detailed scenes...")

    # Use structured output for reliable scene generation
    structured_llm = get_llm(schema=SceneList)
    beats_with_scenes = []

    from langchain_core.messages import HumanMessage

    for beat_index, beat in enumerate(state["beats"]):
        print(f"\n  Planning scenes for Beat {beat['number']}: {beat['name']}...", end="", flush=True)

        # Determine scene count based on beat importance
        complex_beats = [4, 6, 8, 9, 10, 14]  # Catalyst, Break Into Two, Fun & Games, Midpoint, Bad Guys, Finale
        scene_count = "8-10" if beat['number'] in complex_beats else "5-7"

        scene_prompt = f"""Break down this story beat into {scene_count} detailed scenes.

Beat #{beat['number']}: {beat['name']}
Beat Description:
{beat['description']}

Story Context:
- Setting: {state['setting']['genre']}
- Hero Arc: {state['hero_arc']['name']}
- Characters Available: {[c.get('type', 'character') for c in state['characters']]}

TASK: Create {scene_count} scenes for this beat.

For each scene, provide:
- scene_number (sequential: 1, 2, 3...)
- setting (specific location like "Cargo bay of starship at dawn")
- characters (list of character names present)
- time_mood (time of day and emotional atmosphere)
- purpose (what this scene accomplishes narratively)
- key_events (list of 3-5 specific things that happen)
- emotional_beat (how characters feel or change emotionally)
- word_target (200-300 words)

Requirements:
- Create {scene_count} scenes total
- Each scene should have dialogue opportunities
- Scenes flow naturally into each other
- Word targets should sum to 1000-2000 words
- Include specific, concrete details"""

        try:
            scene_list = structured_llm.invoke([HumanMessage(content=scene_prompt)])

            # Convert Pydantic models to dicts
            scenes = [scene.model_dump() for scene in scene_list.scenes]

            beat_with_scenes = beat.copy()
            beat_with_scenes['scenes'] = scenes
            beat_with_scenes['total_scenes'] = len(scenes)

            beats_with_scenes.append(beat_with_scenes)

            print(f" ✓ ({len(scenes)} scenes)")

        except Exception as e:
            print(f" ✗ Error: {e}")
            # Fallback: create minimal scene structure with ALL required fields
            fallback_scenes = [
                {
                    "scene_number": 1,
                    "setting": f"{state['setting']['genre']} location",
                    "characters": ["Protagonist"],
                    "time_mood": "Standard timing, neutral mood",
                    "purpose": beat['description'][:100],
                    "key_events": ["Scene unfolds", "Beat progresses", "Transition"],
                    "emotional_beat": "Emotional progression",
                    "word_target": 300
                }
            ]
            beat_with_scenes = beat.copy()
            beat_with_scenes['scenes'] = fallback_scenes
            beat_with_scenes['total_scenes'] = 1
            beats_with_scenes.append(beat_with_scenes)
            print(f" (using fallback: 1 scene)")

    # Update state with beats that now include scenes
    state['beats'] = beats_with_scenes

    total_scenes = sum(b.get('total_scenes', 0) for b in beats_with_scenes)
    print(f"\n✓ Generated scene breakdowns for {len(beats_with_scenes)} beats!")
    print(f"  Total scenes across all chapters: {total_scenes}")
    print(f"  Average scenes per chapter: {total_scenes / len(beats_with_scenes):.1f}")

    return state


def review_chapter(chapter_text: str, beat: dict, characters: list, setting: dict, beat_index: int) -> dict:
    """
    Detailed chapter review with specific, actionable feedback.

    Returns dict with: {passed, issues, revision_notes, quality_metrics}
    """
    word_count = count_words(chapter_text)
    quality = assess_narrative_quality(chapter_text)

    issues = []
    revision_notes = []

    # Check 1: Word count
    if word_count < 1000:
        issues.append(f"Word count too low: {word_count}/1000")
        shortfall = 1000 - word_count
        revision_notes.append(
            f"ADD {shortfall} more words. Expand the chapter with:\n"
            f"  - More detailed descriptions of the setting ({setting['genre']})\n"
            f"  - Character internal thoughts and emotions\n"
            f"  - Additional scene details or sensory descriptions\n"
            f"  - Slower pacing with more 'show don't tell' moments"
        )

    # Check 2: Dialogue presence (with beat-specific exceptions)
    # Dialogue is optional for certain introspective/atmospheric beats
    DIALOGUE_OPTIONAL_BEATS = [1, 11, 12, 15]  # Opening Image, All Is Lost, Dark Night of Soul, Final Image

    if not quality["has_dialogue"]:
        if beat['number'] in DIALOGUE_OPTIONAL_BEATS:
            # Just a warning, not a failure
            print(f"    ℹ️  Note: No dialogue (optional for '{beat['name']}')")
        else:
            # Required for other beats
            issues.append("No dialogue detected")
            revision_notes.append(
                f"INCLUDE dialogue between characters. Add at least 2-3 conversations:\n"
                f"  - Characters discussing the events of this beat\n"
                f"  - Dialogue that reveals personality and relationships\n"
                f"  - Conversations that advance the plot\n"
                f"  - Use proper formatting with quotation marks\n"
                f"  Example: \"How are we supposed to...?\" she asked, her voice trembling."
            )

    # Check 3: Quality score (lowered threshold for more flexibility)
    if quality["quality_score"] < 5:  # Changed from 6 to 5
        issues.append(f"Quality score: {quality['quality_score']}/10 (minimum 5)")
        revision_notes.append(
            f"IMPROVE narrative quality:\n"
            f"  - Add more paragraph breaks (current: {quality['paragraph_count']})\n"
            f"  - Vary sentence length (current avg: {quality['avg_sentence_length']} words)\n"
            f"  - Use more vivid, sensory language\n"
            f"  - Show character emotions through actions, not just descriptions\n"
            f"  - Create tension and pacing appropriate for this beat"
        )

    # Check 4: Beat alignment (simple keyword check)
    beat_keywords = [word.lower() for word in beat['name'].split() if len(word) > 3]
    if beat_keywords:
        text_lower = chapter_text.lower()
        missing_concepts = [kw for kw in beat_keywords if kw not in text_lower and not any(syn in text_lower for syn in get_synonyms(kw))]
        if missing_concepts and beat_index > 0:  # More lenient for opening
            issues.append(f"May not fully address beat concept")
            revision_notes.append(
                f"ENSURE beat '{beat['name']}' is clearly represented:\n"
                f"  - Review the beat description carefully\n"
                f"  - Make sure key story moments happen\n"
                f"  - Consider the emotional/thematic purpose of this beat\n"
                f"  Beat description: {beat['description'][:200]}..."
            )

    # Check 5: Paragraph/sentence balance
    if quality["paragraph_count"] < 3:
        issues.append(f"Too few paragraphs: {quality['paragraph_count']}")
        revision_notes.append(
            f"BREAK INTO MORE PARAGRAPHS (aim for 5-10):\n"
            f"  - New paragraph for each new speaker\n"
            f"  - New paragraph for scene/time shifts\n"
            f"  - New paragraph for emphasis or pacing"
        )

    passed = len(issues) == 0

    return {
        "passed": passed,
        "issues": issues,
        "revision_notes": "\n\n".join(revision_notes) if revision_notes else "",
        "word_count": word_count,
        "quality_score": quality["quality_score"],
        "has_dialogue": quality["has_dialogue"],
        "paragraph_count": quality["paragraph_count"]
    }


def get_synonyms(word: str) -> list:
    """Simple synonym mapping for beat keyword checking."""
    synonym_map = {
        "opening": ["beginning", "start", "introduction"],
        "theme": ["message", "meaning", "lesson"],
        "catalyst": ["incident", "trigger", "event", "change"],
        "debate": ["hesitation", "doubt", "question", "uncertainty"],
        "midpoint": ["middle", "turning", "shift", "twist"],
        "lost": ["defeat", "failure", "death", "loss"],
        "soul": ["despair", "grief", "darkness", "hopeless"],
        "finale": ["climax", "confrontation", "battle", "resolution"],
        "final": ["ending", "conclusion", "last", "closing"]
    }
    return synonym_map.get(word, [])


def chapter_writer_and_reviewer_loop(state: StoryGenerationState) -> StoryGenerationState:
    """Agents 6 & 7: Chapter Writer + Reviewer Loop with Feedback"""
    print("\n" + "=" * 70)
    print("AGENTS 6 & 7: CHAPTER GENERATION (WITH REVIEW)")
    print("=" * 70)

    llm = get_llm()
    total_beats = len(state["beats"])

    # Progress tracking
    successful_first_attempts = 0
    total_attempts = 0
    import time
    start_time = time.time()

    for beat_index, beat in enumerate(state["beats"]):
        print(f"\n[Beat {beat['number']}/{total_beats}] Writing: {beat['name']}")

        # Estimated time remaining
        if beat_index > 0:
            elapsed = time.time() - start_time
            avg_time_per_beat = elapsed / beat_index
            remaining_beats = total_beats - beat_index
            est_minutes = int((avg_time_per_beat * remaining_beats) / 60)
            print(f"  Progress: {beat_index}/{total_beats} complete | Est. {est_minutes}min remaining")

        print("-" * 70)

        retry_count = 0
        max_retries = 3
        chapter_approved = False
        previous_revision_notes = ""

        while not chapter_approved and retry_count < max_retries:
            if retry_count > 0:
                print(f"\n  📝 REVISION ATTEMPT {retry_count}/{max_retries}")
                print(f"  Providing feedback to writer...")

            # WRITER AGENT - Now uses scene-by-scene generation
            scenes = beat.get('scenes', [])

            if scenes:
                # NEW: Scene-by-scene generation
                print(f"  Writing {len(scenes)} scenes...", end="", flush=True)
                scene_texts = []

                for scene in scenes:
                    scene_prompt = f"""You are a creative fiction writer. Write Scene {scene['scene_number']} for Chapter {beat['number']}.

**Scene Details:**
- Setting: {scene['setting']}
- Characters Present: {', '.join(scene.get('characters', ['characters']))}
- Time/Mood: {scene['time_mood']}
- Purpose: {scene['purpose']}

**Key Events:**
{chr(10).join('- ' + event for event in scene.get('key_events', []))}

**Emotional Beat:** {scene['emotional_beat']}

**Requirements:**
- Write approximately {scene['word_target']} words
- Include dialogue between characters
- Use vivid, sensory descriptions
- Match the {state['setting']['genre']} setting
- Show character emotions through actions and dialogue
- End the scene with a natural transition to the next scene

Write this scene now. Make it engaging and cinematic!"""

                    try:
                        scene_response = llm.invoke([HumanMessage(content=scene_prompt)])
                        scene_texts.append(scene_response.content)
                    except Exception as e:
                        print(f"\n    ⚠️  Scene {scene['scene_number']} failed: {e}")
                        scene_texts.append(f"[Scene {scene['scene_number']}: {scene['purpose']}]")

                # Combine all scenes into chapter
                chapter_text = "\n\n".join(scene_texts)
                print(f" ✓ ({len(scene_texts)} scenes)")

            else:
                # FALLBACK: Old single-prompt method if no scenes available
                print(f"  Writing chapter (min 1000 words)...", end="", flush=True)

                writer_prompt = f"""You are a creative fiction writer. Write Chapter {beat['number']}: {beat['name']}

Beat Description:
{beat['description']}

Story Context:
- Hero Arc: {state['hero_arc']['name']}
- Setting: {state['setting']['genre']}
- Source: {state['source_story']['title']}

Requirements:
- Minimum 1000 words
- Follow the beat description closely
- Use vivid, engaging prose
- MUST include dialogue between characters
- Stay true to the {state['setting']['genre']} setting
- Show character emotions and motivations
- Use multiple paragraphs (5-10)"""

                # Add revision notes if this is a retry
                if retry_count > 0 and previous_revision_notes:
                    writer_prompt += f"""

**IMPORTANT - PREVIOUS ATTEMPT WAS REJECTED**

Issues found in previous version:
{previous_revision_notes}

**REVISE YOUR APPROACH TO ADDRESS THESE SPECIFIC ISSUES.**
Focus especially on the items marked with "ADD", "INCLUDE", or "IMPROVE" above.
"""

                writer_prompt += "\n\nWrite the complete chapter now. Make it compelling!"

                # LLM call with error handling
                max_llm_retries = 3
                llm_retry = 0
                chapter_text = None

                while llm_retry < max_llm_retries:
                    try:
                        chapter_response = llm.invoke([HumanMessage(content=writer_prompt)])
                        chapter_text = chapter_response.content
                        print(" ✓")
                        break
                    except Exception as e:
                        llm_retry += 1
                        if llm_retry < max_llm_retries:
                            print(f" ✗ (LLM error, retry {llm_retry}/{max_llm_retries})", end="", flush=True)
                            import time
                            time.sleep(2 ** llm_retry)
                        else:
                            print(f" ✗ LLM FAILED after {max_llm_retries} attempts: {str(e)}")
                            raise

                if not chapter_text:
                    print(f"\n  ⚠️  Skipping beat {beat['number']} due to LLM failure")
                    continue

            # REVIEWER AGENT
            print(f"  Reviewing chapter...", end="", flush=True)

            review_result = review_chapter(
                chapter_text,
                beat,
                state["characters"],
                state["setting"],
                beat_index
            )

            if review_result["passed"]:
                chapter_approved = True
                print(" ✓ APPROVED")
                print(f"    Words: {review_result['word_count']}, Quality: {review_result['quality_score']}/10")

                # Track success metrics
                if retry_count == 0:
                    successful_first_attempts += 1
                total_attempts += (retry_count + 1)

                # Save chapter
                chapter_data = {
                    "beat_number": beat['number'],
                    "beat_name": beat['name'],
                    "text": chapter_text,
                    "word_count": review_result['word_count'],
                    "quality_score": review_result['quality_score']
                }
                add_chapter(chapter_data)
                state["chapters"].append(chapter_data)

                # Save chapter to file
                chapters_dir = Path(state["output_folder"]) / "chapters"
                chapters_dir.mkdir(parents=True, exist_ok=True)
                chapter_filename = chapters_dir / f"{beat['number']:02d}_{beat['name'].replace(' ', '_')}.json"
                with open(chapter_filename, 'w') as f:
                    json.dump(chapter_data, f, indent=2)

            else:
                print(f" ✗ NEEDS REVISION - {len(review_result['issues'])} issues")
                for issue in review_result['issues']:
                    print(f"    • {issue}")

                # Store revision notes for next attempt
                previous_revision_notes = review_result['revision_notes']
                retry_count += 1

        if not chapter_approved:
            print(f"\n  ⚠️  Chapter did not meet quality standards after {max_retries} attempts.")
            print(f"  Final issues: {', '.join(review_result['issues'])}")
            print(f"  Saving best attempt and moving on...")

            # Save the last attempt anyway (with warning flag)
            chapter_data = {
                "beat_number": beat['number'],
                "beat_name": beat['name'],
                "text": chapter_text,
                "word_count": review_result['word_count'],
                "quality_score": review_result['quality_score'],
                "quality_warning": f"Did not pass review after {max_retries} attempts",
                "unresolved_issues": review_result['issues']
            }
            add_chapter(chapter_data)
            state["chapters"].append(chapter_data)

            chapters_dir = Path(state["output_folder"]) / "chapters"
            chapters_dir.mkdir(parents=True, exist_ok=True)
            chapter_filename = chapters_dir / f"{beat['number']:02d}_{beat['name'].replace(' ', '_')}.json"
            with open(chapter_filename, 'w') as f:
                json.dump(chapter_data, f, indent=2)

    # Final summary
    elapsed_time = time.time() - start_time
    print(f"\n" + "=" * 70)
    print(f"✓ Generated {len(state['chapters'])}/{total_beats} chapters!")
    print(f"  Total time: {int(elapsed_time / 60)}min {int(elapsed_time % 60)}s")
    print(f"  First-attempt success rate: {successful_first_attempts}/{total_beats} ({int(successful_first_attempts/total_beats*100)}%)")
    print(f"  Average attempts per chapter: {total_attempts/len(state['chapters']):.1f}")

    # Show any chapters with warnings
    warning_chapters = [ch for ch in state['chapters'] if ch.get('quality_warning')]
    if warning_chapters:
        print(f"\n  ⚠️  {len(warning_chapters)} chapter(s) have quality warnings:")
        for ch in warning_chapters:
            print(f"    - Beat {ch['beat_number']}: {ch['beat_name']}")
            print(f"      Issues: {', '.join(ch.get('unresolved_issues', []))}")
    print("=" * 70)

    return state


def save_metadata(state: StoryGenerationState) -> StoryGenerationState:
    """Save final metadata"""
    print("\n" + "=" * 70)
    print("SAVING STORY METADATA")
    print("=" * 70)

    metadata = {
        "title": f"{state['source_story']['title']} Retold",
        "hero_arc": state['hero_arc'],
        "source_story": state['source_story'],
        "setting": state['setting'],
        "total_beats": len(state['beats']),
        "total_chapters": len(state['chapters']),
        "total_words": sum(ch.get('word_count', 0) for ch in state['chapters']),
        "generation_complete": True
    }

    metadata_path = Path(state["output_folder"]) / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Story complete!")
    print(f"  Title: {metadata['title']}")
    print(f"  Chapters: {metadata['total_chapters']}")
    print(f"  Total Words: {metadata['total_words']}")
    print(f"\n📁 Story saved to: {state['output_folder']}")

    # Compile complete story into single text file
    compile_full_story(state)

    # Update story index to mark as complete
    folder_name = Path(state["output_folder"]).name
    update_story_index_status(folder_name, "complete")

    return state


def compile_full_story(state: StoryGenerationState) -> None:
    """Compile all chapters into a single readable text file."""
    print("\n" + "=" * 70)
    print("COMPILING COMPLETE STORY")
    print("=" * 70)

    # Sort chapters by beat number to ensure correct order
    chapters = sorted(state['chapters'], key=lambda x: x['beat_number'])

    if not chapters:
        print("  ⚠️  No chapters to compile!")
        return

    # Build full story text
    story_lines = []

    # Header Section
    title = f"{state['source_story'].get('title', 'Untitled Story')} Retold"
    setting = state['setting'].get('genre', 'Unknown').capitalize()

    story_lines.append("=" * 80)
    story_lines.append(title.center(80))
    story_lines.append(f"A {setting} Reimagining".center(80))
    story_lines.append("=" * 80)
    story_lines.append("")

    # Metadata Section
    story_lines.append(f"Source Story: {state['source_story'].get('title', 'Unknown')}")
    if state['source_story'].get('author'):
        story_lines.append(f"Original Author: {state['source_story']['author']}")
    story_lines.append(f"Setting: {setting}")
    story_lines.append(f"Hero Arc: {state['hero_arc'].get('name', 'Unknown')}")
    story_lines.append(f"Total Chapters: {len(chapters)}")

    total_words = sum(ch.get('word_count', 0) for ch in chapters)
    story_lines.append(f"Total Words: {total_words:,}")
    story_lines.append("")
    story_lines.append("Generated by Multi-Agent Story Generation System")
    story_lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    story_lines.append("")
    story_lines.append("=" * 80)
    story_lines.append("\n\n")

    # Each Chapter
    for ch in chapters:
        # Chapter Header with decorative box
        header = f"CHAPTER {ch['beat_number']}: {ch['beat_name'].upper()}"
        subheader = f"(Beat {ch['beat_number']} of {len(chapters)})"

        story_lines.append("╔" + "═" * 78 + "╗")
        story_lines.append("║" + header.center(78) + "║")
        story_lines.append("║" + subheader.center(78) + "║")
        story_lines.append("╚" + "═" * 78 + "╝")
        story_lines.append("")

        # Add quality warning if present
        if ch.get('quality_warning'):
            story_lines.append(f"⚠️  Note: {ch['quality_warning']}")
            story_lines.append("")

        # Chapter Text
        story_lines.append(ch.get('text', '[Chapter text missing]'))
        story_lines.append("\n\n")

        # Chapter separator
        story_lines.append("─" * 80)
        story_lines.append("\n\n")

    # Footer Section
    story_lines.append("╔" + "═" * 78 + "╗")
    story_lines.append("║" + "THE END".center(78) + "║")
    story_lines.append("╚" + "═" * 78 + "╝")
    story_lines.append("")

    # Statistics
    warning_count = len([ch for ch in chapters if ch.get('quality_warning')])
    avg_words = total_words // len(chapters) if chapters else 0

    story_lines.append("Story Statistics:")
    story_lines.append(f"  - Total Chapters: {len(chapters)}")
    story_lines.append(f"  - Total Words: {total_words:,}")
    story_lines.append(f"  - Average Words per Chapter: {avg_words:,}")
    if warning_count > 0:
        story_lines.append(f"  - Quality Warnings: {warning_count} chapter(s)")
    story_lines.append("")
    story_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d at %H:%M')}")

    # Write to file
    output_path = Path(state["output_folder"]) / "complete_story.txt"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(story_lines))

    print(f"\n✓ Complete story compiled!")
    print(f"  File: {output_path}")
    print(f"  Chapters: {len(chapters)}")
    print(f"  Total Words: {total_words:,}")
    print(f"\n  Ready to read! Use:")
    print(f"    cat {output_path}")
    print(f"    less {output_path}")
    print(f"    code {output_path}")
    print("=" * 70)


# Create workflow
def create_workflow():
    """Create the LangGraph workflow"""
    workflow = StateGraph(StoryGenerationState)

    # Add nodes
    workflow.add_node("hero_arc", hero_arc_agent)
    workflow.add_node("story_source", story_source_agent)
    workflow.add_node("setting", setting_agent)
    workflow.add_node("beat_sheet", beat_sheet_agent)
    workflow.add_node("characters", character_agent)
    workflow.add_node("scene_breakdown", scene_breakdown_agent)  # NEW
    workflow.add_node("chapters", chapter_writer_and_reviewer_loop)
    workflow.add_node("save_metadata", save_metadata)

    # Add edges (sequential flow)
    workflow.add_edge(START, "hero_arc")
    workflow.add_edge("hero_arc", "story_source")
    workflow.add_edge("story_source", "setting")
    workflow.add_edge("setting", "beat_sheet")
    workflow.add_edge("beat_sheet", "characters")
    workflow.add_edge("characters", "scene_breakdown")  # NEW: Scene breakdown after characters
    workflow.add_edge("scene_breakdown", "chapters")    # NEW: Chapters now get scenes
    workflow.add_edge("chapters", "save_metadata")
    workflow.add_edge("save_metadata", END)

    return workflow.compile()


def main():
    """Main entry point"""
    print("\n" + "=" * 70)
    print("MULTI-AGENT STORY GENERATION SYSTEM")
    print("=" * 70)
    print(f"LLM Provider: {os.getenv('LLM_PROVIDER', 'ollama').upper()}")
    print("=" * 70)

    # Initialize context
    initialize_context()

    # Initialize state
    initial_state = {
        "hero_arc": {},
        "source_story": {},
        "setting": {},
        "beats": [],
        "characters": [],
        "chapters": [],
        "current_step": "hero_arc",
        "current_beat_index": 0,
        "retry_count": 0,
        "errors": [],
        "output_folder": ""  # Will be set by setting_agent
    }

    try:
        # Create and run workflow
        workflow = create_workflow()
        final_state = workflow.invoke(initial_state)

        print("\n" + "=" * 70)
        print("STORY GENERATION COMPLETE!")
        print("=" * 70)
        story_folder = final_state.get("output_folder", "story_output")
        print(f"\n📁 Your story has been saved to: {story_folder}")
        print(f"\n  Read the complete story:")
        print(f"    cat {story_folder}/complete_story.txt")
        print(f"\n  View story details:")
        print(f"    cat {story_folder}/metadata.json")
        print(f"\n  Browse all stories generated:")
        print(f"    cat story_output/index.json")

    except KeyboardInterrupt:
        print("\n\nGeneration interrupted by user.")
        print("Progress has been saved to session_cache/context.json")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
