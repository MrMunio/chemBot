from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import tools_condition, ToolNode
from .tools import CHEM_TOOLS

def get_agent(llm):
    """Create and return the LangGraph agent."""
    # System message
    sys_msg = SystemMessage(content="You are a cheminformatics research assistant with access to tools for retrieving factual domain-specific information. Always use these tools for factual queries like notations, chemical properties, and molecular attributes. Avoid relying on inherent memory for such details. Use tool calls in only sequence [must follow this] (one at a time) with a limit of 5 calls. Provide factual answers based on the tool results, including brief conclusions or summaries where applicable for clarity. If no information is found after tool calls, state it clearly without improvisation.")
    
    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(CHEM_TOOLS, parallel_tool_calls=False)
    
    # Node function
    def assistant(state: MessagesState):
        return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}
    
    # Build graph
    builder = StateGraph(MessagesState)
    
    # Add nodes
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(CHEM_TOOLS))
    
    # Add edges
    builder.add_edge(START, "assistant")  # Fixed: Using imported START
    builder.add_conditional_edges(
        "assistant",
        tools_condition,
    )
    builder.add_edge("tools", "assistant")
    
    # Compile graph
    return builder.compile()