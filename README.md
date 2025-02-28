# Plato

Plato is an agent-based framework for building complex AI workflows using multiple specialized agents that collaborate to solve tasks. Built on LangGraph and LangChain, Plato orchestrates LLM agents with different capabilities to work together.

## Features

- **Multi-Agent Workflows**: Create collaborative agents that can pass tasks between each other.
- **Specialized Agents**: Pre-built agents for research, data visualization, and more.
- **Tool Integration**: Easy integration of tools like ArXiv research and Python REPL.
- **Visualized Workflows**: Generate visual graphs of your agent workflows.
- **Extensible Architecture**: Build custom agents and tools for your specific needs.
- **Easily Extendable**: Add more tools, agents, and functionalities that LangGraph/LangChain support seamlessly.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/plato.git
cd plato

# Install dependencies
pip install -r requirements.txt

# Set up your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
```

## Quick Start

```python
from utils import *
from agents import *
from langgraph.graph import END, StateGraph, START
import functools

# Create LLM
llm = getllm("openai-gpt4o-mini")

# Create agents
research_node = functools.partial(agent_node, agent=getResearchAgent(llm), name="Researcher")
chart_node = functools.partial(agent_node, agent=getChartAgent(llm), name="chart_generator")
tool_node = ToolNode(unwrapTools(llm, [], True))

# Set up workflow graph
workflow = StateGraph(AgentState)

workflow.add_node("Researcher", research_node)
workflow.add_node("chart_generator", chart_node)
workflow.add_node("call_tool", tool_node)

# Configure workflow connections
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

# Visualize the workflow
draw_graph(graph)

# Run the workflow
events = graph.stream(
    {
        "messages": [
            HumanMessage(
                content="Fetch an arxiv paper about machine learning and visualize the key findings"
            )
        ],
    },
    {"recursion_limit": 15},
)

for s in events:
    print(s)
    print("----")
```

## Core Components

### Agents
- **Research Agent**: Searches ArXiv for scientific papers and extracts relevant information.
- **Chart Generator**: Creates visualizations based on data using matplotlib.

### Tools
- **ArXivTool**: Searches academic papers on arxiv.org.
- **PythonREPL**: Executes Python code for data processing and visualization.

## Example Flows

The repository includes example flows like `lunar_landing_data_visualizer_flow.py`, which creates visualizations of regolith/lunar soil data.

## Documentation

For more information on the available functions:

- `utils.py`: Core utilities and helper functions.
- `tools.py`: Available tools for agents.
- `agents.py`: Agent definitions and creation.
- `template.py`: Template for creating agent workflows.

## License

MIT

## Contributing

Contributions welcome! Please feel free to submit a Pull Request.

