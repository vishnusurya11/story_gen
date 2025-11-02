#!/usr/bin/env python3
"""Quick test script to verify agents are working"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from agents.tools.context_tools import initialize_context, load_context, save_context
from agents.tools.validation_tools import count_words, assess_narrative_quality
from agents.prompts.hero_arc_prompts import HERO_ARC_DESCRIPTIONS
from agents.prompts.beat_sheet_prompts import SAVE_THE_CAT_BEATS

print("Testing Multi-Agent Story Generation System...")
print("=" * 60)

# Test 1: Context Tools
print("\n1. Testing Context Tools...")
try:
    context = initialize_context()
    print(f"   ✓ Context initialized with keys: {list(context.keys())}")

    # Test save/load
    context['test_field'] = "test_value"
    save_context(context)
    loaded = load_context()
    assert loaded['test_field'] == "test_value"
    print("   ✓ Context save/load working")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 2: Validation Tools
print("\n2. Testing Validation Tools...")
try:
    test_text = """This is a test chapter. It contains multiple sentences.

    This paragraph has dialogue. "Hello," said the character. "How are you?"

    The narrative continues with vivid descriptions of the setting."""

    word_count = count_words(test_text)
    quality = assess_narrative_quality(test_text)
    print(f"   ✓ Word count: {word_count}")
    print(f"   ✓ Quality score: {quality['quality_score']}/10")
    print(f"   ✓ Has dialogue: {quality['has_dialogue']}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 3: Prompt Libraries
print("\n3. Testing Prompt Libraries...")
try:
    assert len(HERO_ARC_DESCRIPTIONS) == 4
    print(f"   ✓ Hero Arc Descriptions loaded: {len(HERO_ARC_DESCRIPTIONS)} types")

    assert len(SAVE_THE_CAT_BEATS) == 15
    print(f"   ✓ Save the Cat Beats loaded: {len(SAVE_THE_CAT_BEATS)} beats")

    # Check first beat structure
    first_beat = SAVE_THE_CAT_BEATS[0]
    assert 'name' in first_beat
    assert 'description' in first_beat
    print(f"   ✓ First beat: {first_beat['name']}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 4: LLM Initialization
print("\n4. Testing LLM Initialization...")
try:
    from agents.orchestrator import get_llm
    llm = get_llm()
    print(f"   ✓ LLM initialized: {type(llm).__name__}")

    # Quick test
    from langchain_core.messages import HumanMessage
    response = llm.invoke([HumanMessage(content="Say 'test successful' and nothing else.")])
    print(f"   ✓ LLM responds: {response.content[:50]}...")
except Exception as e:
    print(f"   ✗ Error: {e}")
    print(f"      Make sure your .env file has OLLAMA_API_KEY set!")

# Test 5: File Structure
print("\n5. Testing File Structure...")
required_dirs = ["agents", "agents/tools", "agents/prompts", "session_cache", "story_output", "story_output/chapters"]
for dir_path in required_dirs:
    if os.path.exists(dir_path):
        print(f"   ✓ {dir_path} exists")
    else:
        print(f"   ✗ {dir_path} missing!")

print("\n" + "=" * 60)
print("Basic tests complete!")
print("\nTo run the full story generation:")
print("  uv run python agents/orchestrator.py")
print("=" * 60)
