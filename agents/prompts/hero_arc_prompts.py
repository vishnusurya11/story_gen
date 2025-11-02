"""Hero Arc Agent Prompts"""

HERO_ARC_SYSTEM_PROMPT = """You are a character arc specialist and narrative expert.

Your role is to guide the user in selecting a hero's character arc based on the "wants vs needs" framework.

The four arc types are:

1. **Positive Arc (Gets WANT and NEED)**: The hero achieves their external goal (want) AND overcomes their internal flaw to get what they truly need. They grow and succeed. Example: Luke Skywalker in Star Wars.

2. **Corruption Arc (Gets WANT but not NEED)**: The hero achieves their external goal but fails to grow internally. They may win but lose their soul. Example: Walter White in Breaking Bad.

3. **Redemption Arc (Gets NEED but not WANT)**: The hero fails to achieve their external goal but grows internally and becomes a better person. Example: Ebenezer Scrooge in A Christmas Carol.

4. **Tragedy Arc (Gets NEITHER)**: The hero fails both externally and internally. They don't achieve their goal and don't grow. Example: Shakespeare's tragedies.

Guide the user clearly, explain each option if needed, and validate their choice."""

HERO_ARC_USER_PROMPT = """Please choose the type of hero's journey for your story:

(1) **Positive/Change Arc** - Hero gets what they WANT and what they NEED
    (Grows as a person AND achieves their goal)

(2) **Corruption Arc** - Hero gets what they WANT but NOT what they NEED
    (Achieves goal but loses themselves in the process)

(3) **Disillusionment Arc** - Hero gets what they NEED but NOT what they WANT
    (Grows as a person but doesn't achieve their external goal)

(4) **Tragedy Arc** - Hero gets NEITHER what they want NOR what they need
    (Fails to grow and fails to achieve goal)

Enter the number (1-4) of your choice:"""

HERO_ARC_DESCRIPTIONS = {
    "1": {
        "name": "Positive/Change Arc",
        "description": "The protagonist overcomes their internal flaw (need) and achieves their external goal (want). This is a story of growth and triumph.",
        "want_need": "Gets both WANT and NEED",
        "examples": "Luke Skywalker (Star Wars), Elizabeth Bennet (Pride & Prejudice), Simba (The Lion King)"
    },
    "2": {
        "name": "Corruption Arc",
        "description": "The protagonist achieves their external goal (want) but fails to overcome their internal flaw. They win but lose their soul.",
        "want_need": "Gets WANT but not NEED",
        "examples": "Walter White (Breaking Bad), Michael Corleone (The Godfather), Macbeth"
    },
    "3": {
        "name": "Disillusionment/Redemption Arc",
        "description": "The protagonist fails to achieve their external goal (want) but overcomes their internal flaw and becomes a better person (need).",
        "want_need": "Gets NEED but not WANT",
        "examples": "Ebenezer Scrooge (A Christmas Carol), Thor (Thor: Ragnarok), Javert (Les Misérables) - in reverse"
    },
    "4": {
        "name": "Tragedy Arc",
        "description": "The protagonist fails both externally and internally. They don't achieve their goal and don't grow or change for the better.",
        "want_need": "Gets neither WANT nor NEED",
        "examples": "Hamlet, Romeo and Juliet, The Great Gatsby (Jay Gatsby)"
    }
}

RANDOM_ARC_MESSAGE = """Invalid choice received. Randomly selecting a hero arc..."""

VALIDATION_PROMPT = """Validate that the user's input is one of: 1, 2, 3, or 4.

User input: {user_input}

Respond with either:
- "VALID: <number>" if valid (where <number> is 1, 2, 3, or 4)
- "INVALID" if not valid

Only output one of those two response formats, nothing else."""
