from langchain_core.tools import tool
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
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import os, time

# =========================================================
# 🔐 1️⃣  PINECONE SETUP
# =========================================================
PINECONE_API_KEY = "pcsk_7MmgmD_N1nwWacYopgdRA8BSRcFXQRURRBHCyutMJp72mLmSAxCwdbD98X8zj9T96KyfYV"
# https://docs.pinecone.io/reference/api/2025-04/admin/create_api_key?utm_source=chatgpt.com
INDEX_NAME = "groq-math-memory"
ENVIRONMENT = "us-east-1"

pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if not exists
if INDEX_NAME not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=ENVIRONMENT),
    )

index = pc.Index(INDEX_NAME)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# =========================================================
# 🤖 2️⃣  MODEL SETUP (Groq + LangGraph)
# =========================================================
model = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key="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",
) # (https://console.groq.com/keys)

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

# =========================================================
# 💾 3️⃣  STATE DEFINITION
# =========================================================
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

# =========================================================
# 🧠 4️⃣  HELPER FUNCTIONS FOR PINECONE MEMORY
# =========================================================
def save_memory_to_pinecone(user_text: str, bot_text: str):
    """Save both user input and bot reply into Pinecone."""
    try:
        if not user_text.strip() or not bot_text.strip():
            return
        vectors = []
        for prefix, text in [("user", user_text), ("bot", bot_text)]:
            embedding = embedding_model.encode([text])[0].tolist()
            vectors.append(
                (f"{prefix}-{abs(hash(text))}", embedding, {"role": prefix, "text": text})
            )
        index.upsert(vectors=vectors)
    except Exception as e:
        print(f"[Pinecone Save Error]: {e}")

def retrieve_relevant_context(user_query: str, top_k=3) -> str:
    """Retrieve semantically similar past memory."""
    if not user_query.strip():
        return ""
    try:
        query_embedding = embedding_model.encode([user_query])[0].tolist()
        results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
        context_texts = [match["metadata"]["text"] for match in results.get("matches", [])]
        return "\n".join(context_texts)
    except Exception as e:
        print(f"[Pinecone Retrieval Error]: {e}")
        return ""

# =========================================================
# 🧩 5️⃣  LLM NODE
# =========================================================
def llm_call(state: MessagesState):
    messages = state["messages"]
    user_query = messages[-1].content if isinstance(messages[-1], HumanMessage) else ""
    context_from_memory = retrieve_relevant_context(user_query)

    # Add system prompt
    system_prompt = SystemMessage(
        content=f"""
You are a highly intelligent Math Expert that remembers previous conversations.

🧠 Context from memory:
{context_from_memory}

✅ Responsibilities:
- Understand and accurately solve any mathematical question.
- Use conversation context to resolve ambiguous terms.
- Only show steps if user asks.
⚙️ Output Format:
**Answer:** <numeric or algebraic result>
"""
    )

    try:
        response = model_with_tools.invoke(
            [system_prompt] + messages,
            config={"timeout": 30},  # prevent infinite wait
        )
    except Exception as e:
        print(f"[Groq API Error]: {e}")
        return {
            "messages": [AIMessage(content="**Answer:** Sorry, Groq API did not respond.")],
            "llm_calls": state.get("llm_calls", 0) + 1,
        }

    return {"messages": [response], "llm_calls": state.get("llm_calls", 0) + 1}

# =========================================================
# 🛠️ 6️⃣  TOOL NODE
# =========================================================
def tool_node(state: dict):
    result_msgs = []
    last_msg = state["messages"][-1]
    for tool_call in getattr(last_msg, "tool_calls", []) or []:
        tool = tools_by_name.get(tool_call["name"])
        if tool:
            try:
                observation = tool.invoke(tool_call["args"])
            except Exception as e:
                observation = f"Tool error: {e}"
            result_msgs.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))
    return {"messages": result_msgs}

def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tool_node"
    return END

memory = MemorySaver()

# =========================================================
# 🔗 7️⃣  GRAPH SETUP
# =========================================================
builder = StateGraph(MessagesState)
builder.add_node("llm_call", llm_call)
builder.add_node("tool_node", tool_node)
builder.add_edge(START, "llm_call")
builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
builder.add_edge("tool_node", "llm_call")
agent = builder.compile()
react_graph_memory = builder.compile(checkpointer=memory)

# =========================================================
# 💬 8️⃣  INTERACTIVE CLI LOOP
# =========================================================
if __name__ == "__main__":
    print("\n🤖 Groq + LangGraph + Pinecone Memory Math Agent")
    print("Type 'exit' or 'quit' to stop.\n")

    state = {"messages": [], "llm_calls": 0}

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye 👋")
            break

        if not user_input:
            continue  # skip empty input safely
        if user_input.lower() in ["exit", "quit", "no"]:
            print("Goodbye 👋")
            break

        state["messages"].append(HumanMessage(content=user_input))
        config = {"configurable": {"thread_id": "1"}}

        try:
            result_state = agent.invoke(
                {"messages": state["messages"], "llm_calls": state["llm_calls"]},
                config,
            )
        except Exception as e:
            print(f"[Agent Invoke Error]: {e}")
            continue

        for msg in result_state["messages"]:
            state["messages"].append(msg)
            if hasattr(msg, "content") and msg.content.strip():
                print("Bot:", msg.content)
                try:
                    save_memory_to_pinecone(user_input, msg.content)
                except Exception as e:
                    print(f"[Memory Error]: {e}")

        print("----------------\n")
