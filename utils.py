# agentx.py

import os
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import operator
from tools import *
from PIL import Image
import matplotlib.pyplot as plt
from typing import Annotated, Sequence, TypedDict
from typing import Literal
from langgraph.prebuilt import ToolNode
from IPython.display import Image, display
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain.chains import RetrievalQA
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory, ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from IPython.display import Markdown as md
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings
)
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    ToolMessage,
    AIMessage,
    SystemMessage
)

os.environ["OPENAI_API_KEY"] = "[OPENAI_API_KEY]"

# Private Functions
def getllmMap():
  llmsMapSchema = {
      "openai-gpt4o-mini": ChatOpenAI(model="gpt-4o-mini"),
      "openai-gpt4": ChatOpenAI(model="gpt-4"),
      "openai-gpt35": ChatOpenAI(model="gpt-3.5-turbo")
  }
  return llmsMapSchema

def getToolsMap(llm):
  toolsMapSchema = {
    # "ReportCardGeneratorTool": ReportCardTool(llm=llm),
    "ArxivTool": arxiv_tool,
    "PythonREPL": python_repl
  }
  return toolsMapSchema

def getMemoryMap(llm=None, maxTokenLimit=200, memoryKey="chat_history", returnMessages=False):
  memoryMapSchema = {
    "ConversationBufferMemory": ConversationBufferMemory(memory_key=memoryKey, return_messages=returnMessages),
    "ConversationSummaryBufferMemory": ConversationSummaryBufferMemory(memory_key=memoryKey, llm=llm, max_token_limit=maxTokenLimit, return_messages= returnMessages)
  }
  return memoryMapSchema

# Public Functions
def getllm(llmType):
    llmMap = getllmMap()
    if(llmMap.get(llmType) == None):
        raise Exception("Invalid llm type")
    return llmMap[llmType]

def unwrapTools(llm, toolNames, all=False):
  toolsMap = getToolsMap(llm)

  for toolName in toolNames:
    if(toolsMap.get(toolName) == None):
      raise Exception(str("Invalid tool name - " + toolName))

  if(all):
    return list(toolsMap.values())

  return [toolsMap[toolName] for toolName in toolNames]

def getMemory(memoryType, llm, maxTokenLimit, memoryKey, returnMessages):
  memoryMap = getMemoryMap(llm, maxTokenLimit, memoryKey, returnMessages)
  if(memoryMap.get(memoryType) == None):
    raise Exception("Invalid memory type")
  return memoryMap[memoryType]

def create_agent(llm, tools, system_message: str):
    """Create an agent."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant, collaborating with other assistants."
                " Use the provided tools to progress towards answering the question."
                " If you are unable to fully answer, that's OK, another assistant with different tools "
                " will help where you left off. Execute what you can to make progress."
                " If you or any of the other assistants have the final answer or deliverable,"
                " prefix your response with FINAL ANSWER so the team knows to stop."
                " You have access to the following tools: {tool_names}.\n{system_message}",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    return prompt | llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str

def router(state) -> Literal["call_tool", "__end__", "continue"]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "call_tool"
    if "FINAL ANSWER" in last_message.content:
        return "__end__"
    return "continue"

def agent_node(state, agent, name):
  result = agent.invoke(state)
  # We convert the agent output into a format that is suitable to append to the global state
  if isinstance(result, ToolMessage):
      pass
  else:
      result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
  return {
      "messages": [result],
      # Since we have a strict workflow, we can
      # track the sender so we know who to pass to next.
      "sender": name,
  }

def draw_graph(graph, notJupyter = False):
  output_file="graph.png"
  if notJupyter:
    try:
        # Generate the image and save it to a file
        graph_image = graph.get_graph(xray=True).draw_mermaid_png()
        with open(output_file, "wb") as f:
            f.write(graph_image)

    except Exception as e:
        print("Error while creating the graph image: " + str(e))
        pass
  else:
    try:
        display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
    except Exception as e:
        print("Error while creating the graph image: " + str(e))
        pass