# tools.py

import matplotlib
# Set the backend to 'Agg' before importing pyplot
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Annotated
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.utilities import ArxivAPIWrapper
from langchain_experimental.utilities import PythonREPL
from langchain.tools.base import BaseTool
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL

repl = PythonREPL()

@tool
def python_repl(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        # Add default plotting configuration
        setup_code = """
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
"""
        repl.run(setup_code)
        result = repl.run(code)
        # Save any generated plots
        if 'plt' in code:
            plt.savefig('output_plot.png')
            plt.close()
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    if 'plt' in code:
        result_str += "\nPlot has been saved as 'output_plot.png'"
    return (
        result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
    )

@tool
def arxiv_tool(
    query: Annotated[str, "Search query for arxiv.org"],
):
  """A wrapper around Arxiv.org Useful for when you need to answer questions about Physics, Mathematics,
  Computer Science, Quantitative Biology, Quantitative Finance, Statistics, Electrical Engineering, and Economics from scientific articles on arxiv.org. Input should be a search query."""
  try:
    arxiv = ArxivAPIWrapper()
    result = arxiv.run(query)
  except BaseException as e:
    return f"Failed to execute. Error: {repr(e)}"
  return (
      result + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
  )