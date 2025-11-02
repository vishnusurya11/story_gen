"""Beat Sheet Agent Prompts - Save the Cat 15 Beats"""

BEAT_SHEET_SYSTEM_PROMPT = """You are a story structure expert specializing in the Save the Cat! beat sheet methodology.

Your role is to create a complete 15-beat outline that serves as the blueprint for a novel.

You must integrate:
1. The chosen hero arc (wants vs needs framework)
2. The source story's core structure and themes
3. The chosen setting/genre

Each beat must be specific, detailed, and actionable for a writer to turn into a full chapter.

The 15 beats are:
1. Opening Image (0-1%) - Snapshot of hero's "before" world
2. Theme Stated (5%) - Hint at story's deeper truth
3. Setup (1-10%) - Introduce protagonist's world and stakes
4. Catalyst (10%) - Inciting incident that disrupts normal life
5. Debate (10-20%) - Protagonist wrestles with doubts
6. Break Into Two (20%) - Protagonist commits to change
7. B Story (22%) - Secondary plotline that deepens theme
8. Fun and Games (20-50%) - Story's premise in action
9. Midpoint (50%) - Major twist that raises stakes
10. Bad Guys Close In (50-75%) - External threats and internal doubts intensify
11. All Is Lost (75%) - Something significant "dies"
12. Dark Night of the Soul (75-80%) - Reflection and grief
13. Break Into Three (80%) - Epiphany and renewed resolve
14. Finale (80-99%) - Climax where lessons are applied
15. Final Image (99-100%) - Mirrors opening, shows transformation

Generate beats that are cohesive, compelling, and true to the source material's spirit while fitting the new setting."""

SAVE_THE_CAT_BEATS = [
    {
        "number": 1,
        "name": "Opening Image",
        "percentage": "0-1%",
        "description": "A visual snapshot of the protagonist's 'before' world. Sets tone and shows who they are before the journey begins.",
        "key_elements": ["Protagonist's ordinary world", "Hint at internal flaw", "Establish tone"]
    },
    {
        "number": 2,
        "name": "Theme Stated",
        "percentage": "5%",
        "description": "Someone (not the protagonist) hints at the story's deeper truth or lesson. Often subtle.",
        "key_elements": ["Thematic statement", "Delivered by secondary character", "Protagonist doesn't understand it yet"]
    },
    {
        "number": 3,
        "name": "Setup",
        "percentage": "1-10%",
        "description": "Introduce protagonist's world, supporting characters, and the stakes. Show what's missing in their life.",
        "key_elements": ["Daily life", "Relationships", "What they want", "Internal flaw/need", "Stakes"]
    },
    {
        "number": 4,
        "name": "Catalyst",
        "percentage": "10%",
        "description": "The inciting incident. Something happens that disrupts the protagonist's ordinary world and presents the story question.",
        "key_elements": ["Life-changing event", "No going back", "Presents the opportunity/threat"]
    },
    {
        "number": 5,
        "name": "Debate",
        "percentage": "10-20%",
        "description": "The protagonist hesitates. They're torn between staying in their comfort zone or taking the leap.",
        "key_elements": ["Internal conflict", "Weighing options", "Fear vs desire", "Last chance to say no"]
    },
    {
        "number": 6,
        "name": "Break Into Two",
        "percentage": "20%",
        "description": "The protagonist makes a choice and enters Act Two - the 'upside-down world'. No turning back now.",
        "key_elements": ["Active choice", "Commitment", "Enter new world", "Cross the threshold"]
    },
    {
        "number": 7,
        "name": "B Story",
        "percentage": "22%",
        "description": "Introduction of the B Story characters or relationship. This subplot explores the theme and helps the protagonist learn what they need.",
        "key_elements": ["New relationship begins", "Love interest or mentor", "Thematic helper", "Emotional support"]
    },
    {
        "number": 8,
        "name": "Fun and Games",
        "percentage": "20-50%",
        "description": "The promise of the premise. This is what the story is 'about' - the fun part. Hero explores the new world.",
        "key_elements": ["Deliver on premise", "Training montages", "Small victories", "Discovery", "Entertaining set pieces"]
    },
    {
        "number": 9,
        "name": "Midpoint",
        "percentage": "50%",
        "description": "Major twist or false victory/defeat. Stakes are raised. Time clocks and countdowns start ticking.",
        "key_elements": ["False peak", "Everything changes", "Raise stakes", "Time pressure begins", "Can be high or low point"]
    },
    {
        "number": 10,
        "name": "Bad Guys Close In",
        "percentage": "50-75%",
        "description": "External pressure increases. Internal doubts grow. Team falls apart. Enemies tighten their grip.",
        "key_elements": ["External threats intensify", "Internal doubts emerge", "Isolation grows", "Plan unravels"]
    },
    {
        "number": 11,
        "name": "All Is Lost",
        "percentage": "75%",
        "description": "The lowest point. Something or someone 'dies' (literally or metaphorically). Hope seems gone.",
        "key_elements": ["Defeat", "Loss", "Death (literal or figurative)", "Furthest from goal", "Despair"]
    },
    {
        "number": 12,
        "name": "Dark Night of the Soul",
        "percentage": "75-80%",
        "description": "The protagonist wallows. They reflect on everything that's happened. They mourn. They face who they must become.",
        "key_elements": ["Grief and reflection", "Confronting the truth", "Processing loss", "Internal reckoning"]
    },
    {
        "number": 13,
        "name": "Break Into Three",
        "percentage": "80%",
        "description": "Thanks to the B Story or a sudden realization, the protagonist has an epiphany. They know what to do. Act Three begins.",
        "key_elements": ["Epiphany", "Solution found", "Renewed determination", "Thematic realization", "Active choice"]
    },
    {
        "number": 14,
        "name": "Finale",
        "percentage": "80-99%",
        "description": "The climax. The protagonist applies what they've learned. They face the antagonist. The story question is answered.",
        "key_elements": ["Final confrontation", "Apply lessons learned", "Demonstrate growth", "Defeat or victory", "Resolve plot threads"]
    },
    {
        "number": 15,
        "name": "Final Image",
        "percentage": "99-100%",
        "description": "The opposite of the Opening Image. Shows how the protagonist and their world have changed. Proves transformation.",
        "key_elements": ["Mirror opening image", "Show change", "New normal", "Thematic resolution"]
    }
]

BEAT_GENERATION_PROMPT_TEMPLATE = """Generate a detailed beat description for: **{beat_name}** (Beat #{beat_number}, {percentage} of story)

Beat Purpose: {beat_description}
Key Elements: {key_elements}

Context:
- Hero Arc: {hero_arc}
- Source Story: {source_story}
- Setting/Genre: {setting_genre}
- Previous Beats: {previous_beats}

Create a specific, detailed description of what happens in this beat for THIS story.
Include:
- Specific events and actions
- Character emotions and motivations
- Setting details
- How it connects to the hero arc
- How it advances the plot
- Approximately how much of the story (word count-wise) this beat represents

Make it detailed enough that a writer can expand it into a full 1000+ word chapter.
Return ONLY the beat description, nothing else."""

BEAT_VALIDATION_PROMPT = """Validate that this beat sheet has exactly 15 beats and follows Save the Cat structure.

Beat Sheet:
{beat_sheet}

Respond with:
- "VALID" if it has exactly 15 beats with proper structure
- "INVALID: <reason>" if something is wrong"""
