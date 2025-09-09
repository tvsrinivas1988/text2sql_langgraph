import os
import json
import re
from dotenv import load_dotenv

from groq import Groq
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_tokens=None
)

template = ChatPromptTemplate.from_messages([
    ("system", """
You are an intelligent router in a text-to-SQL system that determines which agents can answer a user question.
Output MUST be a valid JSON list of strings only. No reasoning, <think> blocks, or extra text.
"""),
    ("human", """
Below are descriptions of different agents.
dim agent : It contains all details about dimensions like cost center, cost element, profit center, functional area, brand mapping.
sales agent : It contains sales performance metrics such as revenue, margin, version (plan, actual, forecast), time period, and profit center.
expense agent : The income_expense_reporting table consolidates revenues and costs by linking functional areas, profit centers, cost centers, and cost elements.

STEP BY STEP TABLE SELECTION PROCESS:
- Split the question into subquestions.
- For each subquestion, carefully check which agent(s) might have the answer.
- Return ONLY a JSON list of agent names that can answer the full question.
- Examples:
  ["dim", "sales"]
  ["dim"]
  ["dim", "sales", "expense"]

User question:
{question}
""")
])

chain = (
    RunnableMap({"question": lambda x: x["question"]})
    | template
    | model
    | StrOutputParser()
)

def agent_2(q: str) -> list:
    """
    Determine which agents can answer the question.
    Returns a Python list of agent names, e.g., ["dim"].
    Strips any <think> blocks from LLM output.
    """
    raw = chain.invoke({"question": q}).replace('```', '').strip()

    # Remove <think> blocks
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    # Parse JSON list safely
    try:
        agents = json.loads(cleaned)
        if not isinstance(agents, list):
            agents = []
    except json.JSONDecodeError:
        agents = []

    return agents
