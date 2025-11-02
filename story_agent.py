#!/usr/bin/env python3
"""
Multi-agent story writing system using LangChain.
Uses a supervisor agent to coordinate specialized story-writing agents.
"""

import os
from typing import Annotated, TypedDict, Literal
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import create_react_agent

# Load environment variables
load_dotenv()

# Initialize LLM based on provider choice
def get_llm():
    """Initialize and return the appropriate LLM based on configuration."""
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.7
        )

    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        base_url = os.getenv("OLLAMA_BASE_URL", "https://ollama.com")
        model = os.getenv("OLLAMA_MODEL", "minimax-m2:cloud")
        api_key = os.getenv("OLLAMA_API_KEY")

        # Require API key for cloud endpoint
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

        # Add API key if provided
        if api_key:
            kwargs["api_key"] = api_key

        return ChatOllama(**kwargs)

    else:
        raise ValueError(f"Unknown LLM provider: {provider}. Choose 'gemini' or 'ollama'")


# Define the state for our multi-agent system
class StoryState(TypedDict):
    """State shared across all agents in the story writing workflow."""
    user_request: str
    plot: str
    characters: str
    story_draft: str
    final_story: str
    next_agent: str


# Define agent tools
@tool
def create_plot(request: str) -> str:
    """Create a story plot and outline based on user requirements."""
    llm = get_llm()

    prompt = f"""You are a plot designer for creative stories.

Based on this request: {request}

Create a detailed story outline including:
1. Setting (time and place)
2. Main conflict or challenge
3. Key plot points (beginning, middle, climax, resolution)
4. Themes and mood

Be creative and engaging. Return only the plot outline."""

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


@tool
def develop_characters(request: str, plot: str) -> str:
    """Develop characters based on the story request and plot."""
    llm = get_llm()

    prompt = f"""You are a character developer for creative stories.

Story request: {request}

Plot outline: {plot}

Create detailed character profiles for the main characters including:
1. Name and basic description
2. Personality traits and motivations
3. Background and relationships
4. Character arc within this story

Return only the character profiles."""

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


@tool
def write_story(request: str, plot: str, characters: str) -> str:
    """Write the complete story based on plot and characters."""
    llm = get_llm()

    prompt = f"""You are a creative story writer with excellent narrative skills.

Story request: {request}

Plot outline: {plot}

Characters: {characters}

Write a complete, engaging story that:
1. Follows the plot outline
2. Brings characters to life with dialogue and action
3. Uses vivid descriptions and engaging prose
4. Has a clear beginning, middle, and end
5. Is approximately 800-1200 words

Write the story now. Be creative and engaging!"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


@tool
def edit_story(story_draft: str) -> str:
    """Review and edit the story for quality, consistency, and polish."""
    llm = get_llm()

    prompt = f"""You are a professional story editor. Review and improve this story:

{story_draft}

Your tasks:
1. Fix any grammatical or spelling errors
2. Improve sentence flow and readability
3. Ensure consistency in character voices and plot
4. Enhance descriptions where needed
5. Polish the prose for maximum impact

Return the edited, final version of the story."""

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


# Create supervisor agent
def create_supervisor():
    """Create the supervisor agent that coordinates the workflow."""
    llm = get_llm()

    system_prompt = """You are a supervisor managing a team of story-writing specialists:
- plot_designer: Creates story outlines and structure
- character_developer: Develops character profiles
- writer: Writes the actual story
- editor: Edits and polishes the final story

Given a user's story request, coordinate these agents in the right order to create a great story.

Your workflow should be:
1. First, use plot_designer to create the plot
2. Then, use character_developer to develop characters
3. Then, use writer to write the story
4. Finally, use editor to polish it
5. End when you have the final story

Respond with the name of the next agent to call, or "FINISH" when done."""

    return system_prompt


# Define the workflow nodes
def supervisor_node(state: StoryState) -> StoryState:
    """Supervisor decides which agent to call next."""
    llm = get_llm()

    # Determine next step based on current state
    if not state.get("plot"):
        state["next_agent"] = "plot_designer"
    elif not state.get("characters"):
        state["next_agent"] = "character_developer"
    elif not state.get("story_draft"):
        state["next_agent"] = "writer"
    elif not state.get("final_story"):
        state["next_agent"] = "editor"
    else:
        state["next_agent"] = "FINISH"

    return state


def plot_designer_node(state: StoryState) -> StoryState:
    """Plot designer creates the story outline."""
    result = create_plot.invoke({"request": state["user_request"]})
    state["plot"] = result
    print("\n=== PLOT CREATED ===")
    print(result)
    print("=" * 50 + "\n")
    return state


def character_developer_node(state: StoryState) -> StoryState:
    """Character developer creates character profiles."""
    result = develop_characters.invoke({
        "request": state["user_request"],
        "plot": state["plot"]
    })
    state["characters"] = result
    print("\n=== CHARACTERS DEVELOPED ===")
    print(result)
    print("=" * 50 + "\n")
    return state


def writer_node(state: StoryState) -> StoryState:
    """Writer creates the story draft."""
    result = write_story.invoke({
        "request": state["user_request"],
        "plot": state["plot"],
        "characters": state["characters"]
    })
    state["story_draft"] = result
    print("\n=== STORY DRAFT WRITTEN ===")
    print(result[:500] + "..." if len(result) > 500 else result)
    print("=" * 50 + "\n")
    return state


def editor_node(state: StoryState) -> StoryState:
    """Editor polishes the final story."""
    result = edit_story.invoke({"story_draft": state["story_draft"]})
    state["final_story"] = result
    print("\n=== FINAL STORY EDITED ===")
    print("Story complete!")
    print("=" * 50 + "\n")
    return state


def route_agent(state: StoryState) -> str:
    """Route to the next agent based on supervisor's decision."""
    next_agent = state.get("next_agent", "")

    if next_agent == "FINISH":
        return "end"
    return next_agent


# Build the multi-agent workflow
def create_story_workflow():
    """Create and compile the story-writing workflow graph."""
    workflow = StateGraph(StoryState)

    # Add nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("plot_designer", plot_designer_node)
    workflow.add_node("character_developer", character_developer_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("editor", editor_node)

    # Add edges
    workflow.add_edge(START, "supervisor")

    # Conditional routing from supervisor
    workflow.add_conditional_edges(
        "supervisor",
        route_agent,
        {
            "plot_designer": "plot_designer",
            "character_developer": "character_developer",
            "writer": "writer",
            "editor": "editor",
            "end": END
        }
    )

    # After each agent completes, go back to supervisor
    workflow.add_edge("plot_designer", "supervisor")
    workflow.add_edge("character_developer", "supervisor")
    workflow.add_edge("writer", "supervisor")
    workflow.add_edge("editor", "supervisor")

    return workflow.compile()


def main():
    """Main function to run the story-writing agent system."""
    print("\n" + "=" * 60)
    print("Multi-Agent Story Writing System")
    print("=" * 60)
    print(f"Using LLM Provider: {os.getenv('LLM_PROVIDER', 'gemini').upper()}")
    print("=" * 60 + "\n")

    # Get user input
    print("What kind of story would you like me to write?")
    print("(e.g., 'A sci-fi story about an AI that becomes conscious')")
    print("(e.g., 'A mystery set in Victorian London')")
    print("(e.g., 'A heartwarming tale about friendship')\n")

    user_request = input("Your story request: ").strip()

    if not user_request:
        print("No request provided. Exiting.")
        return

    print(f"\nCreating your story: {user_request}")
    print("This may take a few minutes...\n")

    # Initialize state
    initial_state = {
        "user_request": user_request,
        "plot": "",
        "characters": "",
        "story_draft": "",
        "final_story": "",
        "next_agent": ""
    }

    # Create and run the workflow
    try:
        workflow = create_story_workflow()
        final_state = workflow.invoke(initial_state)

        # Display final story
        print("\n" + "=" * 60)
        print("YOUR FINAL STORY")
        print("=" * 60 + "\n")
        print(final_state["final_story"])
        print("\n" + "=" * 60)

        # Optionally save to file
        save = input("\nWould you like to save this story to a file? (y/n): ").strip().lower()
        if save == 'y':
            filename = input("Enter filename (e.g., my_story.txt): ").strip()
            if filename:
                with open(filename, 'w') as f:
                    f.write(f"Story Request: {user_request}\n\n")
                    f.write("=" * 60 + "\n\n")
                    f.write(final_state["final_story"])
                print(f"\nStory saved to {filename}")

    except Exception as e:
        print(f"\nError: {e}")
        print("Please check your configuration and try again.")


if __name__ == "__main__":
    main()
