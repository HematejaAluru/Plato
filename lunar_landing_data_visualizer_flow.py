from utils import *
from agents import *
import functools
from langgraph.graph import END, StateGraph, START

# lunar_data_visualizer_flow.py


def create_lunar_data_visualizer_flow():
    """
    Creates a flow for visualizing regolith/lunar soil data using research and chart generation agents.
    """
    # Create LLM
    llm = getllm("openai-gpt4o-mini")
    
    # Create required nodes
    research_node = functools.partial(agent_node, agent=getResearchAgent(llm), name="Researcher")
    chart_node = functools.partial(agent_node, agent=getChartAgent(llm), name="chart_generator")
    tool_node = ToolNode(unwrapTools(llm, [], True))
    
    # Create workflow
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("Researcher", research_node)
    workflow.add_node("chart_generator", chart_node)
    workflow.add_node("call_tool", tool_node)
    
    # Add conditional edges
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
    
    # Add tool edges
    workflow.add_conditional_edges(
        "call_tool",
        lambda x: x["sender"],
        {
            "Researcher": "Researcher",
            "chart_generator": "chart_generator",
        }
    )
    
    # Set starting node
    workflow.add_edge(START, "Researcher")
    
    # Compile graph
    return workflow.compile()

def run_lunar_data_visualizer(query=None):
    """
    Run the regolith data visualizer workflow with the given query.
    
    Args:
        query (str, optional): Query to pass to the workflow. Defaults to a standard regolith visualization request.
    """
    graph = create_lunar_data_visualizer_flow()
    
    # Draw the graph
    draw_graph(graph, True)
    
    if query is None:
        query = (
            "Find recent research papers about regolith or lunar soil properties "
            "and visualize the key data points with appropriate charts. "
            "Include composition percentages if available."
        )
    
    events = graph.stream(
        {
            "messages": [HumanMessage(content=query)],
        },
        {"recursion_limit": 15},
    )
    
    for s in events:
        print(s)
        print("----")

if __name__ == "__main__":
    run_lunar_data_visualizer()