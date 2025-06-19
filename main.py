import streamlit as st
import datetime
import asyncio
from functools import partial
import tiktoken
import traceback
import nest_asyncio

# Apply patch to allow nested event loops in Streamlit
nest_asyncio.apply()

# --- Core Imports for the Agent Framework ---
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage, AIMessage

# --- Pinecone Assistant SDK ---
from pinecone import Pinecone
from pinecone_plugins.assistant.models.chat import Message
from pinecone_plugins.assistant.control.core.client.exceptions import PineconeApiException

# --- Configuration & Secrets ---
st.set_page_config(page_title="FiFi Co-Pilot (Debug Mode)", layout="wide")
THREAD_CONFIG = {"configurable": {"thread_id": "fifi_production_v1_debug"}}

try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
    PINECONE_ENVIRONMENT = st.secrets["PINECONE_REGION"]
    MCP_PIPEDREAM_URL = st.secrets.get("MCP_PIPEDREAM_URL")
    ASSISTANT_NAME = st.secrets.get("PINECONE_ASSISTANT_NAME", "fifi")
except KeyError as e:
    st.error(f"Missing critical secret: {e}. The app cannot continue.")
    st.stop()

# --- Memory Management ---
def count_tokens(messages: list, model_encoding: str = "cl100k_base") -> int:
    if not messages:
        return 0
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
    MAX_HISTORY_TOKENS = 24000
    MESSAGES_TO_KEEP = 12
    checkpoint = memory_instance.get(thread_config)
    if not checkpoint or "messages" not in checkpoint:
        return
    current_messages = checkpoint["messages"]
    token_count = count_tokens(current_messages)
    if token_count > MAX_HISTORY_TOKENS:
        print(f"History token count ({token_count}) > max ({MAX_HISTORY_TOKENS}). Pruning...")
        pruned_messages = current_messages[-MESSAGES_TO_KEEP:]
        memory_instance.put(thread_config, {"messages": pruned_messages})
        print("History pruned.")

# --- Tool Definitions ---
async def query_pinecone_knowledge_base(query: str, assistant, memory_instance, thread_config: dict) -> str:
    checkpoint = memory_instance.get(thread_config)
    history_messages = checkpoint.get("messages", []) if checkpoint else []

    sdk_messages = []
    for msg in history_messages:
        if isinstance(msg, (HumanMessage, AIMessage)):
            sdk_messages.append(Message(role=msg.type, content=msg.content))
        elif isinstance(msg, dict) and msg.get("type") != "system":
            sdk_messages.append(Message(role=msg.get("type"), content=msg.get("content")))

    try:
        print(f"\n--- DEBUG: Attempting to query Pinecone ---")
        print(f"Query: {query}")
        print(f"Messages sent: {[msg.dict() for msg in sdk_messages]}")

        response_from_sdk = await assistant.chat(messages=sdk_messages, model="gpt-4o")

        print(f"DEBUG: Response from SDK: {response_from_sdk}")

        content = getattr(getattr(response_from_sdk, "message", None), "content", None)
        return content or "(The assistant returned empty content.)"

    except Exception:
        error_details = traceback.format_exc()
        print(f"\n--- FATAL ERROR in Pinecone Tool ---\n{error_details}\n--- END OF ERROR ---\n")
        return f"An error occurred:\n\n```\n{error_details}\n```"

# --- Agent and Tool Initialization ---
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
            func=partial(query_pinecone_knowledge_base, assistant=pinecone_assistant, memory_instance=memory, thread_config=THREAD_CONFIG),
            description="Use for any questions about products, ingredients, recipes, or company knowledge."
        )
        all_tools.append(pinecone_tool)
        print("✅ Pinecone tool loaded.")
    except Exception as e:
        st.error(f"Fatal Error: Could not initialize Pinecone Assistant: {e}")
        st.stop()

    if MCP_PIPEDREAM_URL:
        try:
            mcp_client = MultiServerMCPClient({"pipedream": {"url": MCP_PIPEDREAM_URL, "transport": "sse"}})
            woocommerce_tools = asyncio.get_event_loop().run_until_complete(mcp_client.get_tools())
            all_tools.extend(woocommerce_tools)
            print(f"✅ WooCommerce tools loaded ({len(woocommerce_tools)} found).")
        except Exception as e:
            st.warning(f"Could not load WooCommerce tools. Error: {e}")
    else:
        st.warning("WooCommerce URL not set. E-commerce tasks are disabled.")

    llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY, temperature=0)
    agent_executor = create_react_agent(llm, all_tools, checkpointer=memory)

    print("--- ✅ Agent is ready ---")
    return agent_executor, memory

# --- Static System Prompt ---
SYSTEM_PROMPT = """You are FiFi, a specialized AI assistant for 1-2-Taste. Provide detailed yet clear answers about ingredients, products, sourcing, and food technology. Use the provided tools to get relevant data."""

# --- App Logic ---
def handle_submission(query):
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.query_to_process = query
    st.session_state.is_thinking = True

async def execute_agent(query, agent_executor, memory):
    prune_history_if_needed(memory, THREAD_CONFIG)
    config = THREAD_CONFIG
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
st.title("1-2-Taste FiFi Co-Pilot (Debug Mode)")

try:
    agent_executor, memory = initialize_agent_and_tools()
    st.session_state.components_loaded = True
except Exception as e:
    st.error(f"Failed to initialize agent. Please refresh. Error: {e}")
    st.session_state.components_loaded = False
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "is_thinking" not in st.session_state:
    st.session_state.is_thinking = False
if "query_to_process" not in st.session_state:
    st.session_state.query_to_process = None

# Chat History
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

# Optional sidebar utilities can go here
st.sidebar.markdown("---")
