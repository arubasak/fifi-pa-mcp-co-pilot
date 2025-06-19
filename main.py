import streamlit as st
import datetime
import asyncio
from functools import partial
import tiktoken

# --- Core Imports for the Agent Framework ---
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage, AIMessage

# --- Imports from your working code ---
from pinecone import Pinecone
from pinecone_plugins.assistant.models.chat import Message
from pinecone_plugins.assistant.control.core.client.exceptions import PineconeApiException

# --- Configuration & Secrets ---
st.set_page_config(page_title="FiFi Co-Pilot", layout="wide")
THREAD_ID = "fifi_production_v1"

try:
    # Adding all required secrets for the multi-tool agent
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
    PINECONE_ENVIRONMENT = st.secrets["PINECONE_REGION"] # Using the correct Pinecone SDK v3 term
    MCP_PIPEDREAM_URL = st.secrets.get("MCP_PIPEDREAM_URL")
    ASSISTANT_NAME = st.secrets.get("PINECONE_ASSISTANT_NAME", "fifi")
except KeyError as e:
    st.error(f"Missing critical secret: {e}. The app cannot continue.")
    st.stop()

# --- UPGRADE 1: Robust Memory Management ---
def count_tokens(messages: list, model_encoding: str = "cl100k_base") -> int:
    """Counts the number of tokens in a list of messages."""
    if not messages: return 0
    try:
        encoding = tiktoken.get_encoding(model_encoding)
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = 0
    for message in messages:
        num_tokens += 4
        for key, value in message.items():
            if value is not None:
                try:
                    num_tokens += len(encoding.encode(str(value)))
                except TypeError:
                    pass
    num_tokens += 2
    return num_tokens

def prune_history_if_needed(memory_instance: MemorySaver, thread_config: dict):
    """Checks token count and prunes history if it exceeds a limit."""
    MAX_HISTORY_TOKENS = 24000 # Keep it well within model limits
    MESSAGES_TO_KEEP = 12 # Keep the last 6 user/assistant pairs
    
    checkpoint = memory_instance.get(thread_config)
    if not checkpoint or "messages" not in checkpoint:
        return

    current_messages = checkpoint["messages"]
    token_count = count_tokens(current_messages)

    if token_count > MAX_HISTORY_TOKENS:
        print(f"History token count ({token_count}) > max ({MAX_HISTORY_TOKENS}). Pruning...")
        # Keep the last N interactions
        pruned_messages = current_messages[-MESSAGES_TO_KEEP:]
        memory_instance.put(thread_config, {"messages": pruned_messages})
        print("History pruned.")

# --- Tool Definitions ---
def query_pinecone_knowledge_base(query: str, assistant, memory_instance, thread_config: dict) -> str:
    """
    This function is now a "specialist tool" for the agent.
    It uses your working logic to query the Pinecone knowledge base.
    """
    # This logic is preserved from your original code, but now gets memory from the agent.
    checkpoint = memory_instance.get(thread_config)
    history_messages = checkpoint.get("messages", []) if checkpoint else []
    
    sdk_messages = [Message(role=msg.type, content=msg.content) for msg in history_messages]
    
    try:
        response_from_sdk = assistant.chat(messages=sdk_messages, model="gpt-4o")
        if hasattr(response_from_sdk, 'message') and hasattr(response_from_sdk.message, 'content'):
            return response_from_sdk.message.content or "(The assistant returned empty content.)"
        return "(Could not find content in the assistant's response.)"
    except Exception as e:
        return f"An error occurred while querying the knowledge base: {str(e)}"

# --- UPGRADE 2: Centralized Agent and Tool Initialization ---
@st.cache_resource(ttl=3600)
def initialize_agent_and_tools():
    """
    Creates the agent "Manager" and all its "Specialist" tools.
    """
    all_tools = []
    memory = MemorySaver()
    thread_config = {"configurable": {"thread_id": THREAD_ID}}

    # UPGRADE 3: Corrected Pinecone Initialization
    print("Initializing Pinecone Assistant with environment...")
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
        pinecone_assistant = pc.assistant.Assistant(assistant_name=ASSISTANT_NAME)
        
        pinecone_tool = Tool(
            name="get_product_and_knowledge_info",
            func=partial(query_pinecone_knowledge_base, assistant=pinecone_assistant, memory_instance=memory, thread_config=thread_config),
            description="Use for any questions about products, ingredients, recipes, or company knowledge."
        )
        all_tools.append(pinecone_tool)
        print("✅ Pinecone tool loaded.")
    except Exception as e:
        st.error(f"Fatal Error: Could not initialize Pinecone Assistant: {e}")
        st.stop()

    # UPGRADE 4: Robust WooCommerce Tool Integration
    print("Initializing WooCommerce tools...")
    if MCP_PIPEDREAM_URL:
        try:
            mcp_client = MultiServerMCPClient({"pipedream": {"url": MCP_PIPEDREAM_URL, "transport": "sse"}})
            woocommerce_tools = asyncio.run(mcp_client.get_tools())
            all_tools.extend(woocommerce_tools)
            print(f"✅ WooCommerce tools loaded ({len(woocommerce_tools)} found).")
        except Exception as e:
            st.warning(f"Could not load WooCommerce tools. E-commerce tasks will not work. Error: {e}")
    else:
        st.warning("WooCommerce URL not set. E-commerce tasks are disabled.")

    llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY, temperature=0)
    agent_executor = create_react_agent(llm, all_tools, checkpointer=memory)
    
    print("--- ✅ Agent is ready ---")
    return agent_executor, memory

# --- The Agent's Rulebook ---
SYSTEM_PROMPT = """You are FiFi, a specialized AI assistant for 1-2-Taste. You are a manager with access to specialist tools. Your job is to decide which tool to use based on the user's query.

1.  **Product & Knowledge Questions:** For any question about products, ingredients, recipes, applications, or company knowledge, you **MUST** use the `get_product_and_knowledge_info` tool.

2.  **E-commerce Tasks:** For tasks related to customer orders, shipping, or accounts, you **MUST** use the appropriate WooCommerce tool from your list.

Evaluate the user's latest query and route it to the correct specialist tool. Do not answer from your own memory.
"""

# --- Main App Logic with Agent Control Flow ---
def handle_submission(query):
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.query_to_process = query
    st.session_state.is_thinking = True

async def execute_agent(query, agent_executor, memory):
    prune_history_if_needed(memory, {"configurable": {"thread_id": THREAD_ID}})
    
    config = {"configurable": {"thread_id": THREAD_ID}}
    event = {"messages": [("system", SYSTEM_PROMPT), ("user", query)]}
    
    try:
        result = await agent_executor.ainvoke(event, config=config)
        reply = result["messages"][-1].content
    except Exception as e:
        reply = f"An error occurred: {e}"
    
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.session_state.is_thinking = False
    st.session_state.query_to_process = None

# --- Streamlit UI ---
st.title("1-2-Taste FiFi Co-Pilot")

try:
    agent_executor, memory = initialize_agent_and_tools()
    st.session_state.components_loaded = True
except Exception as e:
    st.error(f"Failed to initialize agent. Please refresh. Error: {e}")
    st.session_state.components_loaded = False
    st.stop()

if "messages" not in st.session_state: st.session_state.messages = []
if "is_thinking" not in st.session_state: st.session_state.is_thinking = False
if "query_to_process" not in st.session_state: st.session_state.query_to_process = None

# Render chat history and handle agent execution
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if st.session_state.is_thinking:
    with st.chat_message("assistant"):
        st.markdown("⌛ FiFi is thinking...")
    if st.session_state.query_to_process:
        asyncio.run(execute_agent(st.session_state.query_to_process, agent_executor, memory))
        st.rerun()

user_prompt = st.chat_input("Ask FiFi Co-Pilot...", disabled=st.session_state.is_thinking)
if user_prompt:
    handle_submission(user_prompt)
    st.rerun()

# Sidebar content remains the same...
st.sidebar.markdown("---")
# ... (download button, clear history, etc.)
