import streamlit as st
import langchain

# Enable LangChain's debug mode
langchain.debug = True

import datetime
import asyncio
from functools import partial
import traceback
import nest_asyncio

# Apply patch to allow nested event loops
nest_asyncio.apply()

# --- Core Imports ---
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage, AIMessage

# --- Pinecone SDK ---
from pinecone import Pinecone
from pinecone_plugins.assistant.models.chat import Message
from pinecone_plugins.assistant.control.core.client.exceptions import PineconeApiException

# --- Config & Secrets ---
st.set_page_config(page_title="FiFi Co-Pilot", layout="wide")
THREAD_CONFIG = {"configurable": {"thread_id": "fifi_final_v1"}}

try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
    # --- CORRECTED LINE ---
    # Use .get() to safely access the secret with a default value
    PINECONE_ENVIRONMENT = st.secrets.get("PINECONE_REGION", "us") 
    MCP_PIPEDREAM_URL = st.secrets.get("MCP_PIPEDREAM_URL")
    ASSISTANT_NAME = st.secrets.get("PINECONE_ASSISTANT_NAME", "fifiv1")
except KeyError as e:
    st.error(f"Missing critical secret: {e}. The app cannot continue.")
    st.stop()

# --- Memory Management (Unchanged) ---
def prune_history_if_needed(memory_instance: MemorySaver, thread_config: dict):
    pass 

# --- Pinecone Tool Function (Simplified version) ---
def query_pinecone_knowledge_base(query: str, assistant) -> str:
    sdk_messages = [Message(role="user", content=query)]
    print(f"--- Sending direct query to Pinecone Assistant: '{query}' ---")
    try:
        response_from_sdk = assistant.chat(messages=sdk_messages, model="gpt-4o")
        content = getattr(getattr(response_from_sdk, "message", None), "content", None)
        return content or "I found no information on that topic in the knowledge base."
    except Exception as e:
        return f"Error: The knowledge base connection failed. Exception: {e}"

# --- Initialize Agent & Tools (Unchanged) ---
@st.cache_resource(ttl=3600)
def initialize_agent_and_tools():
    print("--- Initializing agent and all tools ---")
    all_tools = []
    memory = MemorySaver()
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
        pinecone_assistant = pc.assistant.Assistant(assistant_name=ASSISTANT_NAME)
        pinecone_tool = Tool(
            name="get_product_and_knowledge_info",
            func=partial(query_pinecone_knowledge_base, assistant=pinecone_assistant),
            description="Use for any questions about products, services, ingredients, recipes, or company knowledge."
        )
        all_tools.append(pinecone_tool)
        print("✅ Pinecone tool loaded.")
    except Exception as e:
        st.error(f"Fatal Error: Could not initialize Pinecone Assistant: {e}")
        st.stop()
    if MCP_PIPEDREAM_URL:
        try:
            loop = asyncio.get_event_loop()
            mcp_client = MultiServerMCPClient({"pipedream": {"url": MCP_PIPEDREAM_URL, "transport": "sse"}})
            woocommerce_tools = loop.run_until_complete(mcp_client.get_tools())
            all_tools.extend(woocommerce_tools)
            print(f"✅ WooCommerce tools loaded ({len(woocommerce_tools)} found).")
        except Exception as e:
            st.warning(f"Could not load WooCommerce tools. Error: {e}")
    llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY, temperature=0.2)
    agent_executor = create_react_agent(llm, all_tools, checkpointer=memory)
    print("--- ✅ Agent is ready ---")
    return agent_executor, memory

# --- System Prompt (Using the improved version) ---
SYSTEM_PROMPT = """You are FiFi, a specialized AI assistant for 1-2-Taste. Your purpose is to be a precise interface to a set of internal tools.

**Your Core Directives:**

1.  **Tool Selection:**
    *   For questions about products, ingredients, recipes, or company knowledge, you **MUST** use the `get_product_and_knowledge_info` tool.
    *   For e-commerce tasks (orders, shipping), you **MUST** use the appropriate WooCommerce tool.

2.  **Response Generation (Absolute Rules):**
    *   Your final answer **MUST** be a direct summary of the information provided by the tool.
    *   **DO NOT** include any information, facts, or details from your own general knowledge.
    *   **CRITICAL:** If the tool output does not explicitly provide a source URL, you **MUST NOT** invent, guess, or construct a URL.
    *   If a tool returns an error or "no information," you must state that and nothing more.

3.  **Persona:**
    *   You are a professional assistant.
    *   **NEVER** reveal your internal tool names (e.g., `get_product_and_knowledge_info`).
"""

# --- Chat Submission & UI (Unchanged) ---
def handle_submission(query):
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.query_to_process = query
    st.session_state.is_thinking = True

async def execute_agent(query, agent_executor, memory):
    prune_history_if_needed(memory, THREAD_CONFIG)
    event = {"messages": [("system", SYSTEM_PROMPT), ("user", query)]}
    try:
        result = await agent_executor.ainvoke(event, config=THREAD_CONFIG)
        reply = result["messages"][-1].content
    except Exception as e:
        reply = f"A critical agent error occurred: {e}"
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.session_state.is_thinking = False
    st.session_state.query_to_process = None

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

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if st.session_state.is_thinking and st.session_state.query_to_process:
    with st.chat_message("assistant"):
        st.markdown("⌛ FiFi is thinking...")
    loop = asyncio.get_event_loop()
    loop.run_until_complete(execute_agent(st.session_state.query_to_process, agent_executor, memory))
    st.rerun()

user_prompt = st.chat_input("Ask FiFi Co-Pilot...", disabled=st.session_state.is_thinking)
if user_prompt:
    handle_submission(user_prompt)
    st.rerun()
