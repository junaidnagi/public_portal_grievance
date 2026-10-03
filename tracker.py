"""Tracking agent: one role and one small factory."""
from crewai import Agent
ROLE = 'Tracking'
GOAL = 'Suggest reference-number and follow-up steps. User dates are personal reminders, not legal deadlines. Explain manual status updates and that nothing has been filed automatically.'

def create_agent(llm, rules: str) -> Agent:
    return Agent(role=ROLE, goal=GOAL, backstory="You assist Pakistani citizens cautiously. " + rules,
                 llm=llm, allow_delegation=False, verbose=False, max_iter=2,
                 max_retry_limit=0, max_execution_time=120)
