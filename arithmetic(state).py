from langchain_core.tools import tool
# from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AnyMessage,
    SystemMessage,
    ToolMessage,
    HumanMessage,
    AIMessage,
)
from typing_extensions import TypedDict, Annotated
import operator
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq

# ---------------- Model Setup ----------------
model = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key="YOUR_API_KEY"
    # (https://console.groq.com/keys)
)

# ---------------- Define Tools ----------------
@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b."""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Add a and b."""
    return a + b

@tool
def divide(a: int, b: int) -> float:
    """Divide a by b."""
    return a / b

@tool
def subtract(a: int, b: int) -> int:
    """Subtract b from a."""
    return a - b

tools = [add, multiply, divide, subtract]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)

# ---------------- Define State ----------------
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

# ---------------- Define LLM Node ----------------
def llm_call(state: MessagesState):
    """LLM decides whether to call a tool or respond directly."""
    messages = state["messages"]

    # Include conversation context + system prompt
    system_prompt = SystemMessage(
        content="""
You are a highly intelligent Math Expert that remembers the previous conversation naturally.
Always interpret references like 'it', 'that', or 'previous value' based on prior messages.

✅ Responsibilities:
- Understand and accurately solve any mathematical question.
- Use the conversation context to resolve ambiguous words like 'it'.
- Show steps only if the user explicitly requests them.
- Support arithmetic, algebra, geometry, trigonometry, calculus, statistics, and unit conversions.

🧠 Rules:
1. Never refuse a math question.
2. If unclear, politely ask for clarification.
3. Always verify your result before responding.
4. Use concise and clear responses.

⚙️ Output Format:
**Answer:** <numeric or algebraic result>
"""
    )

    response = model_with_tools.invoke([system_prompt] + messages)
    return {"messages": [response], "llm_calls": state.get("llm_calls", 0) + 1}

# ---------------- Tool Node ----------------
def tool_node(state: dict):
    """Performs the tool call and updates the conversation state."""
    result_msgs = []
    last_msg = state["messages"][-1]

    for tool_call in last_msg.tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result_msgs.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))

    return {"messages": result_msgs}

# ---------------- Conditional Logic ----------------
def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Route between tool execution and end response."""
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tool_node"
    return END

# ---------------- Build Graph ----------------
builder = StateGraph(MessagesState)
builder.add_node("llm_call", llm_call)
builder.add_node("tool_node", tool_node)
builder.add_edge(START, "llm_call")
builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
builder.add_edge("tool_node", "llm_call")
agent = builder.compile()

# ---------------- Run Interactive CLI ----------------
if __name__ == "__main__":
    print("\n🤖 Groq + LangGraph Memory-Based Math Agent")
    print("Type 'exit' or 'no' to quit.\n")

    state = {"messages": [], "llm_calls": 0}

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "no", "quit"]:
            print("Goodbye 👋")
            break

        # Keep conversation history
        state["messages"].append(HumanMessage(content=user_input))
        result_state = agent.invoke({"messages": state["messages"], "llm_calls": state["llm_calls"]})

        # Append model responses to message history
        for msg in result_state["messages"]:
            state["messages"].append(msg)
            if hasattr(msg, "content"):
                print("Bot:", msg.content)


        print("----------------\n")
