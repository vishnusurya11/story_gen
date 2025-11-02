# Multi-Agent Story Generation System

A sophisticated story-writing application powered by LangChain's multi-agent system. This project uses **8 specialized AI agents** that collaborate to generate complete novels based on minimal user input.

## Features

- **8 Specialized Agents**: Hero Arc, Story Source, Setting, Beat Sheet, Characters, Scene Breakdown, Writer, and Reviewer
- **Save the Cat Beat Sheet**: Industry-standard 15-beat story structure methodology
- **Scene-Based Generation**: Each chapter broken into 5-10 detailed scenes for better structure
- **Character Arc Framework**: Wants vs Needs system for compelling character development
- **Story Library**: Each story saved to unique folder - build a collection without overwriting
- **Flexible LLM Support**: Use either Google Gemini API or Ollama Cloud (MiniMax M2)
- **Quality Control**: Automatic chapter review with specific feedback and retry mechanism
- **Web-Enhanced**: Agents use web search to research stories and world-building
- **Complete Story Export**: Full readable text file plus structured JSON files

## Architecture

The system uses a **sequential multi-agent pipeline** with 8 specialized agents:

1. **Hero Arc Agent**: Determines protagonist's wants vs needs journey (4 arc types)
2. **Story Source Agent**: Selects and analyzes source story to retell/adapt
3. **Setting Agent**: Creates rich world-building for chosen genre AND creates story-specific folder
4. **Beat Sheet Agent**: Generates 15-beat Save the Cat outline
5. **Character Agent**: Creates protagonist, antagonist, and supporting cast
6. **Scene Breakdown Agent**: Plans 5-10 detailed scenes per chapter (NEW!)
7. **Chapter Writer Agent**: Writes chapters scene-by-scene with specific prompts
8. **Chapter Reviewer Agent**: Validates quality with detailed feedback and requests targeted rewrites

## Prerequisites

- Python 3.11 or higher
- [UV package manager](https://github.com/astral-sh/uv)
- An [ollama.com](https://ollama.com) account (free - for MiniMax M2 Cloud access)
- *Optional*: Google Gemini API key (alternative to Ollama)

## Installation

### 1. Install UV

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Clone and Setup

```bash
cd explroe
uv sync
```

### 3. Configure Environment

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` and configure your LLM provider:

#### Option A: Using MiniMax M2 Cloud (FREE - Recommended)

**Step 1:** Sign in to Ollama

```bash
ollama signin
```

**Step 2:** Create an API key

Visit https://ollama.com/settings/keys and create a new API key.

**Step 3:** Update your `.env` file

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=https://ollama.com
OLLAMA_MODEL=minimax-m2:cloud
OLLAMA_API_KEY=your_actual_api_key_here
```

**That's it!** MiniMax M2 Cloud is free during the beta period and runs on Ollama's cloud infrastructure.

#### Option B: Using Google Gemini

```env
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your_actual_api_key_here
```

Get your API key from: https://makersuite.google.com/app/apikey

## Usage

### Quick Start

Run the multi-agent story generator:

```bash
uv run python agents/orchestrator.py
```

### User Interaction

You'll be asked **3 simple questions**:

1. **Hero Arc**: Choose from 4 character arc types (wants vs needs)
   ```
   (1) Gets WANT and NEED
   (2) Gets WANT but not NEED
   (3) Gets NEED but not WANT
   (4) Gets NEITHER
   ```

2. **Story Source**: Which story to retell/adapt?
   ```
   Examples: Pride and Prejudice, Star Wars, Sherlock Holmes, The Odyssey
   ```

3. **Setting**: Choose the genre/setting
   ```
   Options: medieval, futuristic, sci-fi, space, steampunk, modern
   ```

### What Happens Next

The agents will autonomously:
1. **Generate 15-beat outline** using Save the Cat methodology
2. **Create characters** with arcs, wants, needs, conflicts
3. **Break each chapter into 5-10 detailed scenes** for better structure
4. **Write 15 chapters** scene-by-scene (1000+ words each, one per beat)
5. **Review each chapter** with specific feedback and retry if needed
6. **Save everything** to a unique story folder (won't overwrite previous stories!)

### Progress Tracking

- **Session cache**: `session_cache/context.json` tracks real-time progress
- **Story index**: `story_output/index.json` lists all generated stories
- **Story folder**: `story_output/{Title}_{Setting}_{Timestamp}/`
  - `complete_story.txt` - Full readable story
  - `metadata.json` - Story configuration and stats
  - `chapters/*.json` - Individual chapter files with quality metrics

### Expected Output

A complete 15-chapter novel (~15,000 words minimum) in its own unique folder:

```
story_output/
├── index.json                                    # All stories index
└── Pride_and_Prejudice_Scifi_2025-11-02_14-30/  # Your story
    ├── complete_story.txt                        # Read this!
    ├── metadata.json                             # Story details
    └── chapters/                                 # Individual chapters
        ├── 01_Opening_Image.json
        ├── 02_Theme_Stated.json
        └── ... (15 total)
```

**Ready for:**
- Immediate reading (`cat complete_story.txt`)
- Export to ebook formats
- Further editing
- Publishing platforms
- Building a story library (each generation creates new folder!)

## Example Story Combinations

**Classic Stories to Retell:**
- Pride and Prejudice in space setting
- Sherlock Holmes in medieval fantasy
- The Odyssey in cyberpunk future
- Cinderella in steampunk world
- Star Wars in modern-day setting

## Project Structure

```
explroe/
├── agents/
│   ├── orchestrator.py              # Main entry point (8 agents)
│   ├── tools/
│   │   ├── context_tools.py         # Session management
│   │   ├── validation_tools.py      # Quality checks
│   │   └── web_tools.py             # Web search integration
│   └── prompts/
│       ├── hero_arc_prompts.py      # Hero arc definitions
│       └── beat_sheet_prompts.py    # Save the Cat beats
├── session_cache/
│   └── context.json                 # Runtime state (auto-generated)
├── story_output/                    # Story library
│   ├── index.json                   # All stories index
│   ├── Story1_Setting_2025-11-02_14-30/  # Unique per story
│   │   ├── complete_story.txt
│   │   ├── metadata.json
│   │   └── chapters/                # 15 chapter JSON files
│   └── Story2_Setting_2025-11-02_16-00/  # Another story
│       ├── complete_story.txt
│       ├── metadata.json
│       └── chapters/
├── story_agent.py                   # Legacy simple agent (reference)
├── prd.md                           # Product requirements document
├── FIXES.md                         # Critical fixes documentation
├── SCENE_SYSTEM.md                  # Scene breakdown system docs
├── STORY_FOLDERS.md                 # Story folder feature docs
├── pyproject.toml                   # UV/Python configuration
├── .env                             # Your API keys (not committed)
├── .env.example                     # Environment template
├── .gitignore                       # Git ignore rules
└── README.md                        # This file
```

## How It Works

The system uses **LangGraph StateGraph** to orchestrate a sequential pipeline:

1. **State Management**: Shared state flows through all agents via `StoryGenerationState`
2. **Sequential Execution**: Each agent runs in order, building on previous agents' outputs
3. **Context Persistence**: Progress saved to `session_cache/context.json` in real-time
4. **Quality Loop**: Writer/Reviewer agents loop until chapters meet standards (max 3 retries)
5. **Structured Output**: All content saved as JSON for easy processing

Each agent uses the configured LLM (Gemini or Ollama) to perform its specific role.

## Customization

### Adding New Agents

Edit [story_agent.py](story_agent.py) to add new specialized agents:

```python
@tool
def my_custom_agent(state_data: str) -> str:
    """Description of what this agent does."""
    llm = get_llm()
    prompt = "Your custom prompt..."
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content
```

### Changing Story Length

Modify the `write_story` function's prompt to adjust word count:

```python
# In write_story function
5. Is approximately 2000-3000 words  # Change this line
```

### Adjusting Creativity

Modify temperature in `get_llm()` function (0.0 = focused, 1.0 = creative):

```python
temperature=0.9  # More creative
```

## Troubleshooting

### Ollama Cloud connection errors
- Make sure you're signed in: `ollama signin`
- Check `OLLAMA_BASE_URL=https://ollama.com` in `.env`
- Verify model name is `minimax-m2:cloud`

### "GOOGLE_API_KEY not found" (if using Gemini)
- Ensure `.env` file exists and contains `GOOGLE_API_KEY=your_key`
- Check that you copied from `.env.example`

### Slow story generation
- This is normal - the agents make multiple LLM calls
- MiniMax M2 Cloud is generally fast and free during beta
- Gemini is another fast option if you have an API key

## License

MIT

## Contributing

Feel free to open issues or submit pull requests!


# 1. Install dependencies
uv sync

# 2. Setup environment
cp .env.example .env

# 3. Sign in to Ollama (FREE)
ollama signin

# 4. Run the story agent
uv run python story_agent.py