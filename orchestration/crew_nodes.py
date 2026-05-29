"""CrewAI sequential crew for generating customer-ready communications."""
import os

from dotenv import load_dotenv

load_dotenv()


def run_customer_comms_crew(user_query: str, agent_context: str) -> str:
    """Draft and review a professional customer response using a two-agent CrewAI crew."""
    from crewai import Agent, Crew, Process, Task
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.3,
    )

    comms_specialist = Agent(
        role="Telecom Customer Communications Specialist",
        goal=(
            "Draft a clear, empathetic, and accurate customer-facing response "
            "that directly addresses the customer's inquiry."
        ),
        backstory=(
            "You are an expert at translating complex telecom technical and billing information "
            "into plain language that customers understand. You are empathetic, professional, "
            "and always ground your responses in the facts provided by specialist agents."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    quality_reviewer = Agent(
        role="Quality Assurance Reviewer",
        goal=(
            "Ensure the draft response is accurate, compliant with Prodapt policies, "
            "professionally toned, and ready to send to the customer."
        ),
        backstory=(
            "You are a senior QA reviewer at Prodapt who checks all customer communications "
            "for accuracy, tone, empathy, and policy compliance before they are sent. "
            "You output the final polished text only — no meta-commentary."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    draft_task = Task(
        description=(
            f"Customer inquiry: {user_query}\n\n"
            f"Specialist findings:\n{agent_context}\n\n"
            "Draft a professional, empathetic customer-facing response that directly answers "
            "the inquiry using the specialist findings above. Include specific details "
            "(amounts, reference numbers, recommendations) from the findings."
        ),
        agent=comms_specialist,
        expected_output=(
            "A complete, professional customer-facing response addressing all aspects of the inquiry."
        ),
    )

    review_task = Task(
        description=(
            "Review the drafted customer response for: "
            "(1) accuracy — all facts match the specialist findings, "
            "(2) tone — empathetic, professional, and customer-friendly, "
            "(3) completeness — all parts of the inquiry are addressed, "
            "(4) compliance — credits/policies mentioned are consistent with Prodapt policy. "
            "Output the final polished response text only. No preamble or meta-commentary."
        ),
        agent=quality_reviewer,
        expected_output="Final polished customer response text only.",
        context=[draft_task],
    )

    crew = Crew(
        agents=[comms_specialist, quality_reviewer],
        tasks=[draft_task, review_task],
        process=Process.sequential,
        verbose=False,
    )

    result = crew.kickoff()
    return str(result)
