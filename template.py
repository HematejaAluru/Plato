# flow.py
from utils import *
from agents import *
from langgraph.graph import END, StateGraph, START

# Create llm
llm = getllm("openai-gpt4o-mini")

# Create Required Nodes
research_node = functools.partial(agent_node, agent=getResearchAgent(llm), name="Researcher")
chart_node = functools.partial(agent_node, agent=getChartAgent(llm), name="chart_generator")
tool_node = ToolNode(unwrapTools(llm,[],True))

workflow = StateGraph(AgentState)

workflow.add_node("Researcher", research_node)
workflow.add_node("chart_generator", chart_node)
workflow.add_node("call_tool", tool_node)

workflow.add_conditional_edges(
    "Researcher",
    router,
    {"continue": "chart_generator", "call_tool": "call_tool", "__end__": END}
)
workflow.add_conditional_edges(
    "chart_generator",
    router,
    {"continue": "Researcher", "call_tool": "call_tool", "__end__": END}
)

workflow.add_conditional_edges(
    "call_tool",
    lambda x: x["sender"],
    {
        "Researcher": "Researcher",
        "chart_generator": "chart_generator",
    }
)
workflow.add_edge(START, "Researcher")
graph = workflow.compile()

# Lets draw the graph and look whats the flow
draw_graph(graph, True)

events = graph.stream(
    {
        "messages": [
            HumanMessage(
                content="Fetch an arxiv paper about deep learning networks"
                " then show any bar graph representing some metric"
                " Once you code it up, finish."
            )
        ],
    },
    # Maximum number of steps to take in the graph
    {"recursion_limit": 15},
)
for s in events:
    print(s)
    print("----")