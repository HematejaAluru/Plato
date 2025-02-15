# agents.py
import functools
from tools import arxiv_tool, python_repl
from utils import getllm, create_agent, agent_node, unwrapTools

# Research agent and node
def getResearchAgent(llm):
    research_agent = create_agent(
        llm,
        unwrapTools(llm, ["ArxivTool"]),
        system_message="You should provide accurate data for the chart_generator to use.",
    )
    return research_agent

# chart_generator
def getChartAgent(llm):
    chart_agent = create_agent(
        llm,
        unwrapTools(llm, ["PythonREPL"]),
        system_message="Any charts you display will be visible by the user.",
    )
    return chart_agent