import os
import asyncio
from dotenv import load_dotenv
import logging

from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent, BaseAgent
from google.adk.events import Event, EventActions
from google.adk.runners import Runner, print_event
from google.genai import types
from google.adk.tools.google_search_tool import GoogleSearchTool
from google.adk.utils.context_utils import Aclosing
from google.adk.sessions import InMemorySessionService
from google.adk.artifacts import InMemoryArtifactService


# 1. Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ResearchAgent")


# Load environment variables (e.g., GOOGLE_API_KEY) from .env file
load_dotenv()

MAX_WORKERS = 2


# --- STAGE 1: PLANNER ---
planner = LlmAgent(
    name="Planner",
    model="gemini-2.0-flash-lite",
    instruction="Break user request into 3 JSON strings. Example: ['A', 'B']",
    output_key="research_tasks"
)

# --- STAGE 2: DYNAMIC ORCHESTRATOR ---
class DynamicResearchOrchestrator(BaseAgent):
    async def _run_async_impl(self, ctx):
        tasks = ctx.session.state.get("research_tasks", [])
        logger.info(f"Orchestrating {len(tasks)} workers.")

        if len(tasks) > MAX_WORKERS:
            tasks = tasks[:MAX_WORKERS]

        workers = []
        for i, task in enumerate(tasks):
            worker = LlmAgent(
                name=f"Worker_{i}",
                model="gemini-2.0-flash-lite",
                instruction=f"Research this topic: {task}",
                tools=[GoogleSearchTool(model="gemini-2.0-flash-lite")],
                output_key=f"worker_report_{i}" 
            )
            workers.append(worker)

        fan_out = ParallelAgent(name="ResearchExecution", sub_agents=workers)

        # Use Aclosing to ensure the async generator is properly closed
        # even if an error occurs or the iteration is interrupted.
        async with Aclosing(fan_out.run_async(ctx)) as agen:
            async for event in agen:
                yield event

        # After fan_out is done, we aggregate the worker reports from the state
        reports = []
        for i in range(len(workers)):
            report = ctx.session.state.get(f"worker_report_{i}")
            if report:
                reports.append(f"--- Report {i} ---\n{report}")
        
        combined_output = "\n\n".join(reports)
        
        # In ADK, an agent's output is often captured from its final state.
        # We manually set it in the session state so the Validator can see it.
        ctx.session.state["Orchestrator_output"] = combined_output
        
        # Yield an event indicating this agent is done.
        yield Event(
            author=self.name,
            actions=EventActions(end_of_agent=True)
        )

# --- STAGE 3: VALIDATOR ---
validator = LlmAgent(
    name="Validator",
    model="gemini-2.0-flash-lite",
    # FIXED: Use Orchestrator_output instead of ResearchExecution_output
    instruction="""Review the findings in {Orchestrator_output}. 
    Filter out junk and consolidate the valid facts into 'cleaned_notes'.""",
    output_key="cleaned_research_notes"
)

# --- STAGE 4: SUMMARIZER ---
summarizer = LlmAgent(
    name="Summarizer",
    model="gemini-2.0-flash-lite",
    # FIXED: Again, use the {} syntax to pull the specific state key.
    instruction="""Create a Markdown report based on these notes: {cleaned_research_notes}.
    Ensure the tone is professional and the report is detailed.""",
    output_key="final_report"
)

# --- ASSEMBLY ---

root_agent = SequentialAgent(
    name="AutonomousResearcher",
    sub_agents=[
        planner,
        DynamicResearchOrchestrator(name="Orchestrator"),
        validator,
        summarizer
    ]
)

async def main():
    import sys
    
    # Get query from command line args or prompt user
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Enter your research query: ")
        
    if not query.strip():
        print("Error: No query provided.")
        return

    print(f"\nStarting research for: {query}\n")
    
    # Check for API Key
    if not os.getenv('GOOGLE_API_KEY'):
        print("WARNING: GOOGLE_API_KEY is not set. The agent will likely fail when calling the LLM.")
    
    runner = Runner(
        agent=root_agent, 
        app_name="ResearchApp",
        session_service=InMemorySessionService(),
        artifact_service=InMemoryArtifactService(),
        auto_create_session=True
    )

    # Wrap the query in a Content object as required by run_async
    new_message = types.Content(
        role="user",
        parts=[types.Part(text=query)]
    )
    
    async for event in runner.run_async(
        user_id="user_123", 
        session_id="session_456", 
        new_message=new_message
    ):
        print_event(event)

if __name__ == "__main__":
    asyncio.run(main())






