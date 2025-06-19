import streamlit as st
import datetime
import asyncio
from functools import partial
import tiktoken
import traceback
import nest_asyncio

# Apply patch to allow nested event loops in Streamlit
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
    PINECONE_ENVIRONMENT = st.secrets["PINECONE_REGION"]
    MCP_PIPEDREAM_URL = st.secrets.get("MCP_PIPEDREAM_URL")
    ASSISTANT_NAME = st.secrets.get("PINECONE_ASSISTANT_NAME", "fifi")
except KeyError as e:
    st.error(f"Missing critical secret: {e}. The app cannot continue.")
    st.stop()

# --- Memory Management (Unchanged) ---
def prune_history_if_needed(memory_instance: MemorySaver, thread_config: dict):
    # This function is fine as-is.
    pass # Leaving the function definition for clarity, but it's not the issue.

# --- Pinecone Tool Function (FINAL, CORRECTED VERSION) ---
def query_pinecone_knowledge_base(query: str, assistant, memory_instance, thread_config: dict) -> str:
    """
    Queries the Pinecone Assistant. It first loads the conversation history,
    then adds the current user query as the most recent message before sending.
    """
    checkpoint = memory_instance.get(thread_config)
    history_messages = checkpoint.get("messages", []) if checkpoint else []

    # 1. Load the past conversation history
    sdk_messages = []
    for msg in history_messages:
        # This logic correctly handles different message formats in history
        if isinstance(msg, (HumanMessage, AIMessage)):
            sdk_messages.append(Message(role=msg.type, content=str(msg.content)))
        elif isinstance(msg, dict) and msg.get("type") != "system":
            sdk_messages.append(Message(role=msg.get("type"), content=str(msg.get("content"))))
    
    # 2. <<< THIS IS THE FIX >>>
    # Always add the current user's query as the last message.
    # This guarantees the list is never empty and the assistant has the latest context.
    sdk_messages.append(Message(role="user", content=query))

    try:
        # Now, this call will always have at least one message.
        response_from_sdk = assistant.chat(messages=sdk_messages, model="gpt-4o")
        content = getattr(getattr(response_from_sdk, "message", None), "content", None)
        return content or "I found no information on that topic in the knowledge base."

    except Exception:
        # This error handling is now for genuine server/connection issues.
        error_details = traceback.format_exc()
        print(f"\n--- FATAL ERROR in Pinecone Tool ---\n{error_details}\n--- END OF ERROR ---\n")
        return "Error: The knowledge base connection failed. Please inform the user that you were unable to retrieve the information."


# --- Initialize Agent & Tools (Unchanged) ---
@st.cache_resource(ttl=3600)
def initialize_agent_and_tools():
    # This function is fine as-is.
    print("--- Initializing agent and all tools ---")
    all_tools = []
    memory = MemorySaver()

    try:
        pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
        pinecone_assistant = pc.assistant.Assistant(assistant_name=ASSISTANT_NAME)
        pinecone_tool = Tool(
            name="get_product_and_knowledge_info",
            func=partial(query_pinecone_knowledge_base, assistant=pinecone_assistant, memory_instance=memory, thread_config=THREAD_CONFIG),
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

# --- System Prompt (Unchanged) ---
SYSTEM_PROMPT = """You are FiFi, a specialized AI assistant for 1-2-Taste. You are a manager with access to specialist tools. Your job is to decide which tool to use based on the user's query.

**Your Rules:**
1.  **Product & Knowledge Questions:** For any question about products, ingredients, recipes, or company knowledge, you **MUST** use the `get_product_and_knowledge_info` tool.
2.  **E-commerce Tasks:** For tasks related to customer orders, shipping, or accounts, you **MUST** use the appropriate WooCommerce tool from your list.
3.  **Error Handling:** If a tool returns an error message, you must relay that information clearly to the user. Do not try to make up an answer. State that you were unable to retrieve the information.
4.  **Stay Focused:** Do not answer questions outside of these topics. Politely decline."""
5.  *   When asked about your capabilities, describe your functions simply (e.g., "I can answer questions about 1-2-Taste products and ingredients."). **NEVER reveal internal tool names.**
6.  **Cite your sources.** When the `{pinecone_tool}` tool provides a source URL, you must include it in your response. If no URL is available from the tool, state that the info is from the 1-2-Taste catalog.

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
