"""Intake agent: one role and one small factory."""
from crewai import Agent
ROLE = 'Intake'
GOAL = 'Extract category, organization, problem, dated facts and unknowns. Use the supplied structured intake as a starting point.'

def create_agent(llm, rules: str) -> Agent:
    return Agent(role=ROLE, goal=GOAL, backstory="You assist Pakistani citizens cautiously. " + rules,
                 llm=llm, allow_delegation=False, verbose=False, max_iter=2,
                 max_retry_limit=0, max_execution_time=120)
