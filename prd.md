# Multi-Agent Story Generation System – PRD (v1)

## 1. Vision & Objective  
Create a modular, agent-driven storytelling pipeline where the user interacts with a sequence of specialized agents. The system progresses through:  
1. Hero Arc decision  
2. Story source selection  
3. Setting choice  
4. Beat sheet generation  
5. Character generation & arcs  
6. Chapter generation + review  
All user input is minimal — the agents handle structure, logic and quality control. The output is a fully-fleshed story saved as structured files (folders/JSON) ready for further use.

---

## 2. Agent Overview  

### Agent 1: Hero Arc Decision Agent  
**Purpose:** Determine the protagonist’s internal journey and narrative outcome (wants vs. needs).  
**User prompt:**  
> “Choose the hero arc:  
> (1) Gets what they *want* and *need*  
> (2) Gets what they *want* but *not what they *need*  
> (3) Gets what they *need* but *not what they *want*  
> (4) Gets *neither* what they want nor what they need”  
If the user’s response is invalid, this agent randomly chooses one of the four.  
**Reference:** StudioBinder article “What is a Character Arc — Definition & Types Explained”. :contentReference[oaicite:1]{index=1}  
**Output (JSON format):**  
```json
{
  "hero_arc_choice": "(one of the four)",
  "hero_arc_description": "…"
}
Agent 2: Story Source Agent
Purpose: Ask user to pick the story they want to retell or adapt (novel, movie, myth, short story) and extract the core skeletal structure (themes, major beats).
User prompt:

“Which story (novel, film, character’s journey, short story) would you like to retell or adapt?”
If user gives a valid known source, the agent extracts a basic structure; if not, picks a random source from a predefined corpus.
Output:

json
Copy code
{
  "source_story": "…",
  "source_type": "novel/movie/short story",
  "core_theme": "…",
  "core_structure_summary": "…"
}
Agent 3: Setting Selection Agent
Purpose: Determine setting/genre for the retelling (e.g., medieval, futuristic, sci-fi, space, steampunk, modern).
User prompt:

“What kind of setting would you like — medieval, futuristic, sci-fi, space, steampunk, modern?”
If invalid input, randomly assign one from the list.
Output:

json
Copy code
{
  "setting_genre": "…",
  "setting_description": "…"
}
Agent 4: Beat Sheet Agent
Purpose: Based on the hero arc choice, source story structure, and setting, generate a full beat sheet (e.g., the 15 beats of the Save the Cat! Writes a Novel method) to outline the story.
References:

Jessica Brody “How to Write Your Novel Using the Save the Cat Beat Sheet”. 
Jessica Brody

Save the Cat official beat sheets page. 
Save the Cat!®
+1

Input: hero_arc, source_story, setting
Output:

json
Copy code
{
  "beats": [
    {"beat_name": "Opening Image", "description": "…"},
    {"beat_name": "Theme Stated", "description": "…"},
    … (all beats) …
    {"beat_name": "Final Image", "description": "…"}
  ]
}
Agent 5: Character Generation Agent
Purpose: Generate characters (main hero, supporting casts, antagonist) with full arcs informed by hero arc, beats, and setting.
Input: hero_arc, beats
Output:

json
Copy code
{
  "characters": [
    {
      "name": "…",
      "role": "Hero/Antagonist/Support",
      "primary_want": "…",
      "primary_need": "…",
      "arc_summary": "…",
      "key_conflicts": "…"
    },
    … other characters …
  ]
}
Agent 6: Chapter Generation & Review Agents
Purpose:

Writer Agent: For each beat in beats list, generate a chapter text (minimum ~1000 words) consistent with setting, characters, hero arc, and beat description.

Reviewer Agent: Validate each chapter:

Word count ≥1000

Fidelity to beat description, character arcs, setting

Coherence and narrative flow
If fails any rule, flag for Writer Agent to rewrite.
Output Storage: Structured file system or JSON files. Example:

pgsql
Copy code
/story_output/
├─ metadata.json
└─ chapters/
   ├─ 01_Opening_Image.json
   ├─ 02_Theme_Stated.json
   └─ …
metadata.json sample:

json
Copy code
{
  "title": "…",
  "hero_arc": "…",
  "source_story": "…",
  "setting": "…",
  "word_target_per_chapter": 1000,
  "beats": true
}
3. Agent Coordination & Workflow
The orchestrator triggers the sequence: HeroArc → StorySource → Setting → BeatSheet → Characters → ChapterWriter + Reviewer.

Shared context store (e.g., /session_cache/context.json) holds cumulative user inputs and agent outputs.

Each agent reads from context, writes output back to context.

The Review Agent loops with the Writer Agent until chapters pass validation.

4. Data & Folder Structure
bash
Copy code
/agents/
   hero_arc_agent.py
   story_source_agent.py
   setting_agent.py
   beat_sheet_agent.py
   character_agent.py
   chapter_writer_agent.py
   chapter_reviewer_agent.py
   orchestrator.py

/session_cache/
   context.json

/story_output/
   metadata.json
   /chapters/
      01_Opening_Image.json
      02_Theme_Stated.json
      …
5. Validation Rules & Quality Gate
Step	Condition	If Failed
Hero Arc Choice	User gives invalid or missing choice	Random assignment
Setting Choice	Invalid/missing setting	Random assignment
Beat Sheet	Less than full set of beats	Regenerate beat sheet
Chapter Text	<1000 words OR off-beat/character/setting	Flag & rewrite
Consistency Check	Chapter deviates from beat, arc or setting	Flag & review

6. Future Expansion & Hooks
Add Image Generation Agent (WAN 2.2 / Qwen-Image) to produce visuals per chapter or scene.

Add Audio/Narration Agent to create voiceover or immersive audio layers.

Add Reader Feedback Agent to gather user feedback and iterate story improvement.

Add Analytics Agent to score pacing, emotional arc fulfillment, readability, and continuity.

7. Success Criteria
Minimal user input → full structured story output.

Agents modular and chainable.

All chapters meet word count, beat fidelity, character arc consistency.

Output format seamless for storage and further processing.

8. References
“What is a Character Arc — Definition & Types Explained” by StudioBinder. 
StudioBinder
+1

“How to Write Your Novel Using the Save the Cat Beat Sheet” by Jessica Brody. 
Jessica Brody

Save the Cat! Beat Sheets page. 
Save the Cat!®
+1

Prepared as reference spec for implementation in Claude-style agentic architecture.

yaml
Copy code

---

If you like this, I can **export** it as a `.md` file and send you a downloadable link.
::contentReference[oaicite:8]{index=8}