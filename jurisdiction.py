"""Jurisdiction agent: one role and one small factory."""
from crewai import Agent
ROLE = 'Jurisdiction'
GOAL = 'Use retrieved evidence for initial authority and possible escalation. Clearly label demo guidance and mapping as unverified suggestions. Never treat demonstration text as law.'

def create_agent(llm, rules: str) -> Agent:
    return Agent(role=ROLE, goal=GOAL, backstory="You assist Pakistani citizens cautiously. " + rules,
                 llm=llm, allow_delegation=False, verbose=False, max_iter=2,
                 max_retry_limit=0, max_execution_time=120)
