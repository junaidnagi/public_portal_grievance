"""Routing agent: one role and one small factory."""
from crewai import Agent
ROLE = 'Routing'
GOAL = 'Explain submission preparation and source-supported escalation, distinguishing tentative mappings from verified procedure. Nothing has been submitted. Do not invent submission URLs, offices or deadlines.'

def create_agent(llm, rules: str) -> Agent:
    return Agent(role=ROLE, goal=GOAL, backstory="You assist Pakistani citizens cautiously. " + rules,
                 llm=llm, allow_delegation=False, verbose=False, max_iter=2,
                 max_retry_limit=0, max_execution_time=120)
