import streamlit as st
import datetime
import asyncio
from functools import partial
import tiktoken
import traceback # Importing the traceback module for detailed error logging

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
st.set_page_config(page_title="FiFi Co-Pilot (Debug Mode)", layout="wide")
THREAD_ID = "fifi_production_v1_debug"

try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
    PINECONE_ENVIRONMENT = st.secrets["PINECONE_REGION"]
    MCP_PIPEDREAM_URL = st.secrets.get("MCP_PIPEDREAM_URL")
    ASSISTANT_NAME = st.secrets.get("PINECONE_ASSISTANT_NAME", "fifi")
except KeyError as e:
    st.error(f"Missing critical secret: {e}. The app cannot continue.")
    st.stop()

# --- Memory Management (Unchanged) ---
def count_tokens(messages: list, model_encoding: str = "cl100k_base") -> int:
    # ... (function is unchanged) ...
    if not messages: return 0
    try: encoding = tiktoken.get_encoding(model_encoding)
    except Exception: encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = 0
    for message in messages:
        num_tokens += 4
        for key, value in message.items():
            if value is not None:
                try: num_tokens += len(encoding.encode(str(value)))
                except TypeError: pass
    num_tokens += 2
    return num_tokens

def prune_history_if_needed(memory_instance: MemorySaver, thread_config: dict):
    # ... (function is unchanged) ...
    MAX_HISTORY_TOKENS = 24000
    MESSAGES_TO_KEEP = 12
    checkpoint = memory_instance.get(thread_config)
    if not checkpoint or "messages" not in checkpoint: return
    current_messages = checkpoint["messages"]
    token_count = count_tokens(current_messages)
    if token_count > MAX_HISTORY_TOKENS:
        print(f"History token count ({token_count}) > max ({MAX_HISTORY_TOKENS}). Pruning...")
        pruned_messages = current_messages[-MESSAGES_TO_KEEP:]
        memory_instance.put(thread_config, {"messages": pruned_messages})
        print("History pruned.")


# --- Tool Definitions ---
def query_pinecone_knowledge_base(query: str, assistant, memory_instance, thread_config: dict) -> str:
    """
    This is the tool we need to debug. It now returns full error details.
    """
    checkpoint = memory_instance.get(thread_config)
    history_messages = checkpoint.get("messages", []) if checkpoint else []
    
    # Filter out system messages and ensure we only have valid message types
    sdk_messages = [Message(role=msg.type, content=msg.content) for msg in history_messages if msg.type != "system"]
    
    # --- START OF DEBUGGING CHANGE ---
    try:
        # Print the exact messages being sent to the console for inspection
        print(f"\n--- DEBUG: Attempting to query Pinecone ---")
        print(f"Query: {query}")
        print(f"Messages being sent to SDK: {[msg.dict() for msg in sdk_messages]}")
        
        response_from_sdk = assistant.chat(messages=sdk_messages, model="gpt-4o")
        
        print(f"DEBUG: Response from SDK: {response_from_sdk}")
        
        if hasattr(response_from_sdk, 'message') and hasattr(response_from_sdk.message, 'content'):
            return response_from_sdk.message.content or "(The assistant returned empty content.)"
        return "(Could not find content in the assistant's response.)"

    except Exception as e:
        # Catch ANY exception, get the full traceback, and return it.
        # This ensures the error is not hidden.
        error_details = traceback.format_exc()
        
        # Print the full error to the console as well
        print(f"\n--- FATAL ERROR in Pinecone Tool ---")
        print(error_details)
        print(f"--- END OF ERROR ---\n")
        
        # Return the detailed error so it appears in the Streamlit UI
        return f"I encountered a critical error while trying to get information. Please show this to the developer:\n\n```\n{error_details}\n```"
    # --- END OF DEBUGGING CHANGE ---

# --- Agent and Tool Initialization ---
@st.cache_resource(ttl=3600)
def initialize_agent_and_tools():
    # ... (This function is unchanged from the last correct version) ...
    print("--- Initializing agent and all tools ---")
    all_tools = []
    memory = MemorySaver()
    thread_config = {"configurable": {"thread_id": THREAD_ID}}

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

# --- The Agent's Rulebook (Unchanged) ---
SYSTEM_PROMPT = """You are FiFi, a specialized AI assistant for 1-2-Taste...""" # Kept brief for clarity

# --- Main App Logic (Unchanged) ---
def handle_submission(query):
    # ... (function is unchanged) ...
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.query_to_process = query
    st.session_state.is_thinking = True

async def execute_agent(query, agent_executor, memory):
    # ... (function is unchanged) ...
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
st.title("1-2-Taste FiFi Co-Pilot (Debug Mode)")

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
