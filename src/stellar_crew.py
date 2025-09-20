# src/stellar_crew.py
from crewai import Agent, Task, Crew, Process
from llama_index.core import Settings
from src.config import init_settings
from src.agent_tools import vector_tool, kg_tool, extraction_tool

# Ensure settings are initialized
init_settings()
# Use the globally configured LLM for the agents
AGENT_LLM = Settings.llm

# --- Define Agents (Section 3.2) ---

# 1. Knowledge Retrieval Agent (The Research Specialist)
retrieval_agent = Agent(
    role='Knowledge Retrieval Specialist',
    goal='Retrieve comprehensive context from the dual-memory system (Vector and Graph).',
    backstory="You are an expert in hybrid knowledge systems. You know precisely when to use semantic search for nuanced context and when to use graph search for structured facts and relationships.",
    tools=[vector_tool, kg_tool],
    llm=AGENT_LLM,
    verbose=True,
    allow_delegation=False
)

# 2. Data Extraction Agent (The Data Analyst)
data_extraction_agent = Agent(
    role='Data Analyst',
    goal='Populate structured records (JSON) from unstructured text provided by the Retrieval agent.',
    backstory="You specialize in transforming conversations into structured data. You meticulously analyze context to identify key metrics, outcomes, and details, formatting them using the provided extraction tool.",
    tools=[extraction_tool],
    llm=AGENT_LLM,
    verbose=True,
    allow_delegation=False
)

# 3. Content Generation Agent (The Communications Expert)
content_generation_agent = Agent(
    role='Communications Expert',
    goal='Draft polished, human-readable content (summaries, emails) based on the retrieved context.',
    backstory="You are a master communicator. You synthesize information provided by the Retrieval agent into concise, professional, and actionable messages, adopting the appropriate tone for the task.",
    tools=[],
    llm=AGENT_LLM,
    verbose=True,
    allow_delegation=False
)

# Note: The SalesQueryRouterAgent from the blueprint is handled by the task definitions below.

# --- Define Tasks (Section IV) ---
# We define sequences of tasks for specific goals.

# Task Type 1: General Q&A / Summary
def create_general_query_tasks(user_query: str):
    retrieve_task = Task(
        description=f"Analyze the query: '{user_query}'. Use the Semantic and/or Structured search tools strategically to retrieve all necessary context from the knowledge base.",
        expected_output="A comprehensive context document containing all relevant information.",
        agent=retrieval_agent
    )
    generate_task = Task(
        description=f"Using the provided context, answer the user's original query: '{user_query}'. Ensure the answer is accurate, professional, and grounded only in the provided context.",
        expected_output="A polished, final answer to the user's query.",
        agent=content_generation_agent,
        context=[retrieve_task] # This task uses the output of the retrieve_task
    )
    return [retrieve_task, generate_task]

# Task Type 2: Structured Sales Record Generation (Section 4.1)
def create_structured_record_tasks(client_name: str):
    retrieve_task = Task(
        description=f"Retrieve the full context of the most recent meeting(s) with '{client_name}'. Focus on details required for a sales record: dates, participants, discussion points, outcomes, values, and action items.",
        expected_output=f"The complete context of recent interactions with {client_name}.",
        agent=retrieval_agent
    )
    extract_task = Task(
        description=f"Analyze the provided meeting context for '{client_name}'. Use the 'Structured Sales Record Extractor' tool to populate the SalesRecord JSON schema.",
        expected_output="A JSON object conforming to the SalesRecord schema.",
        agent=data_extraction_agent,
        context=[retrieve_task]
    )
    return [retrieve_task, extract_task]

# Task Type 3: Email Recap Generation (Section 4.2)
def create_email_recap_tasks(client_name: str):
    retrieve_task = Task(
        description=f"Retrieve the context of the most recent meeting with '{client_name}', focusing specifically on key discussion points, agreements, and action items.",
        expected_output=f"A summary of the key points and action items from the recent meeting with {client_name}.",
        agent=retrieval_agent
    )
    generate_task = Task(
        description=f"Draft a professional, concise, and friendly client-facing email recap for '{client_name}'. Include: 1. A brief thank you. 2. Bulleted list of key discussion points. 3. Clear summary of action items and next steps. The final output must be only the email text, ready to be sent.",
        expected_output="A fully drafted email (subject, greeting, body, closing).",
        agent=content_generation_agent,
        context=[retrieve_task]
    )
    return [retrieve_task, generate_task]

# --- Crew Execution ---

def run_crew(tasks: list):
    print(f"\n--- Starting Stellar Connect Crew Execution ({len(tasks)} tasks) ---")

    # Assemble the crew
    crew = Crew(
        agents=[retrieval_agent, data_extraction_agent, content_generation_agent],
        tasks=tasks,
        process=Process.sequential, # Tasks must be executed in order
        verbose=2, # Log agent thought processes
    )

    result = crew.kickoff()
    print("\n--- Crew Execution Finished ---")
    return result