# Structured Output Implementation

## Overview

All agents now use LangChain's `.with_structured_output()` with Pydantic models for reliable, validated JSON responses. This eliminates crashes from missing keys and unpredictable JSON formats.

## Problem Solved

**Before:** Scene Breakdown Agent returned unpredictable JSON → crashed on missing `time_mood` key
```
KeyError: 'time_mood'
```

**After:** Pydantic schema guarantees all required fields are present

## Implementation

### 1. Pydantic Schemas ([agents/schemas.py](agents/schemas.py))

All structured outputs now have Pydantic models:

- **`StoryInfo`** - Story Source Agent output
- **`SettingInfo`** - Setting Agent output
- **`BeatDescription`** - Single beat from Beat Sheet Agent
- **`Character`** - Character profile
- **`Scene`** - Single scene with all required fields
- **`SceneList`** - List of scenes (wrapper for Scene Breakdown Agent)

### 2. Updated `get_llm()` Function

```python
def get_llm(schema=None):
    """
    Returns LLM with optional structured output.

    Args:
        schema: Pydantic BaseModel for structured output

    Returns:
        LLM or LLM with .with_structured_output() applied
    """
    # Create base LLM
    if provider == "gemini":
        llm = ChatGoogleGenerativeAI(...)
        if schema:
            return llm.with_structured_output(schema, method="json_schema")
        return llm

    elif provider == "ollama":
        llm = ChatOllama(...)
        if schema:
            return llm.with_structured_output(schema)
        return llm
```

**Key differences:**
- **Gemini**: Uses `method="json_schema"` for native responseSchema (more reliable)
- **Ollama**: Default method works with tool-calling models like minimax-m2:cloud

### 3. Updated Agents

#### Agent 2: Story Source
```python
structured_llm = get_llm(schema=StoryInfo)
story_info = structured_llm.invoke([HumanMessage(content=prompt)])

# Access as object attributes
state["source_story"] = {
    "title": story_info.title,
    "type": story_info.type,
    "author": story_info.author_or_creator,
    # ...
}
```

#### Agent 3: Setting
```python
structured_llm = get_llm(schema=SettingInfo)
setting_info = structured_llm.invoke([HumanMessage(content=prompt)])

# Structured fields guaranteed
state["setting"] = {
    "genre": setting_info.genre,
    "description": setting_info.description,
    "key_locations": setting_info.key_locations,
    "atmosphere": setting_info.atmosphere,
    "unique_elements": setting_info.unique_elements
}
```

#### Agent 4: Beat Sheet
```python
structured_llm = get_llm(schema=BeatDescription)

# Generate 15 beats with guaranteed structure
for beat_template in SAVE_THE_CAT_BEATS:
    beat_info = structured_llm.invoke([HumanMessage(content=prompt)])

    beat_data = {
        "number": beat_info.number,
        "name": beat_info.name,
        "percentage": beat_info.percentage,
        "description": beat_info.description
    }
```

#### Agent 5: Characters
```python
structured_llm = get_llm(schema=Character)

# Generate each character as validated Pydantic object
protagonist = structured_llm.invoke([HumanMessage(content=prompt)])
antagonist = structured_llm.invoke([HumanMessage(content=prompt)])
support_char = structured_llm.invoke([HumanMessage(content=prompt)])

# Store as dicts
characters = [
    {"type": "protagonist", "data": protagonist.model_dump()},
    {"type": "antagonist", "data": antagonist.model_dump()},
    {"type": "supporting", "data": support_char.model_dump()}
]
```

#### Agent 5.5: Scene Breakdown (CRITICAL FIX)
```python
structured_llm = get_llm(schema=SceneList)

# Generate scenes with ALL required fields guaranteed
scene_list = structured_llm.invoke([HumanMessage(content=prompt)])

# Convert to dicts - ALL fields present (scene_number, setting, characters,
# time_mood, purpose, key_events, emotional_beat, word_target)
scenes = [scene.model_dump() for scene in scene_list.scenes]

beat_with_scenes['scenes'] = scenes
```

**This fixes the crash:** `scene['time_mood']` is now guaranteed to exist.

#### Agent 6: Chapter Writer (Consumer)
```python
# Now safe - all fields guaranteed by Pydantic validation
for scene in scenes:
    scene_prompt = f"""
**Scene Details:**
- Setting: {scene['setting']}
- Characters: {', '.join(scene['characters'])}
- Time/Mood: {scene['time_mood']}  # ← NO MORE KeyError!
- Purpose: {scene['purpose']}

**Key Events:**
{chr(10).join('- ' + event for event in scene['key_events'])}

**Emotional Beat:** {scene['emotional_beat']}

Write approximately {scene['word_target']} words
"""
```

## Benefits

### 1. **No More Crashes**
- ✅ Guaranteed field presence
- ✅ Type validation (Pydantic)
- ✅ No more `KeyError` exceptions

### 2. **Reliable JSON Parsing**
- ✅ No manual cleanup of markdown code blocks
- ✅ No manual JSON parsing with try/except
- ✅ LLM forced to return valid schema

### 3. **Better Error Messages**
- Pydantic validation errors are descriptive
- Shows exactly which field is missing/invalid
- Easier debugging

### 4. **Type Safety**
- IDE autocomplete works
- Type hints available
- Catches errors at development time

### 5. **Provider Compatibility**
- Works with both Gemini and Ollama
- Gemini uses native responseSchema (very reliable)
- Ollama works with tool-calling models

## Pydantic Schema Example

```python
from pydantic import BaseModel, Field
from typing import List

class Scene(BaseModel):
    """Single scene within a chapter."""
    scene_number: int = Field(description="Scene number (1, 2, 3...)")
    setting: str = Field(description="Specific location")
    characters: List[str] = Field(description="Character names present")
    time_mood: str = Field(description="Time and emotional atmosphere")
    purpose: str = Field(description="What scene accomplishes")
    key_events: List[str] = Field(description="3-5 things that happen")
    emotional_beat: str = Field(description="Character emotions")
    word_target: int = Field(description="Target words (200-300)", ge=200, le=300)

class SceneList(BaseModel):
    """Collection of scenes for a chapter."""
    scenes: List[Scene] = Field(description="5-10 scenes")
```

**Validation:**
- `word_target` must be between 200-300 (`ge=200, le=300`)
- `characters` must be a list
- All required fields must be present
- Types must match (int, str, List[str], etc.)

## Fallback Handling

Even with structured output, the system has fallbacks:

```python
try:
    scene_list = structured_llm.invoke([HumanMessage(content=prompt)])
    scenes = [scene.model_dump() for scene in scene_list.scenes]
except Exception as e:
    print(f" ✗ Error: {e}")
    # Fallback with ALL required fields
    fallback_scenes = [
        {
            "scene_number": 1,
            "setting": f"{setting} location",
            "characters": ["Protagonist"],
            "time_mood": "Standard timing, neutral mood",
            "purpose": beat_description,
            "key_events": ["Scene unfolds", "Beat progresses"],
            "emotional_beat": "Emotional progression",
            "word_target": 300
        }
    ]
```

## Files Changed

1. **NEW**: `agents/schemas.py` (~100 lines)
   - All Pydantic schemas

2. **UPDATED**: `agents/orchestrator.py` (~150 lines modified)
   - `get_llm()` function with schema parameter
   - Story Source Agent (uses StoryInfo)
   - Setting Agent (uses SettingInfo)
   - Beat Sheet Agent (uses BeatDescription)
   - Character Agent (uses Character)
   - Scene Breakdown Agent (uses SceneList) ← **Critical fix**
   - Import statements

3. **NEW**: `STRUCTURED_OUTPUT.md` (this file)
   - Complete documentation

## Testing

Run the orchestrator - it should no longer crash:

```bash
uv run python agents/orchestrator.py
```

**Expected behavior:**
- All agents generate structured output
- No more KeyError exceptions
- Scene breakdown returns valid scenes with all fields
- Chapter writer safely accesses all scene fields

## Compatibility

- ✅ **Gemini 1.5+**: Uses `json_schema` method with native responseSchema
- ✅ **Ollama (MiniMax M2)**: Uses default method with tool-calling support
- ✅ **Pydantic v2**: All schemas use Pydantic v2 syntax

## Migration Notes

- Old code that manually parsed JSON is replaced with structured output
- Scene dictionaries now guaranteed to have all required keys
- No breaking changes to state structure - dicts still used in state
- Pydantic objects converted to dicts with `.model_dump()`

## Future Enhancements

Potential improvements:
1. Add more validation (e.g., scene_number must be sequential)
2. Add custom validators (e.g., validate character names exist)
3. Use structured output for chapter text (prose validation)
4. Add retry logic with exponential backoff for validation failures
