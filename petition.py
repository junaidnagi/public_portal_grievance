"""Petition agent: one role and one small factory."""
from crewai import Agent
ROLE = 'Petition'
GOAL = 'Write a formal English complaint with addressee, subject, facts, requested relief, confirmed available attachments, date and signature placeholder. Use placeholders for missing facts. Omit unverified laws and identity numbers.'

def create_agent(llm, rules: str) -> Agent:
    return Agent(role=ROLE, goal=GOAL, backstory="You assist Pakistani citizens cautiously. " + rules,
                 llm=llm, allow_delegation=False, verbose=False, max_iter=2,
                 max_retry_limit=0, max_execution_time=120)
