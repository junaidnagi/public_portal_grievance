"""Readiness agent: one role and one small factory."""
from crewai import Agent
ROLE = 'Readiness'
GOAL = 'Explain the supplied score only when assessed. It is a self-reported generic demo checklist, not legally mandatory document validation. Otherwise say Not assessed. Never invent document availability.'

def create_agent(llm, rules: str) -> Agent:
    return Agent(role=ROLE, goal=GOAL, backstory="You assist Pakistani citizens cautiously. " + rules,
                 llm=llm, allow_delegation=False, verbose=False, max_iter=2,
                 max_retry_limit=0, max_execution_time=120)
