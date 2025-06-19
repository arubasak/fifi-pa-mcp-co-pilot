import streamlit as st
import datetime
import asyncio
from functools import partial

# --- Core Imports for the Agent Framework ---
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage, AIMessage

# --- Imports for Pinecone Assistant and Tools ---
from pinecone import Pinecone
from pinecone_plugins.assistant.models.chat import Message
from pinecone_plugins.assistant.control.core.client.exceptions import PineconeApiException

# --- Configuration & Secrets ---
st.set_page_config(page_title="FiFi Co-Pilot", layout="wide")
THREAD_ID = "fifi_fresh_start_v1"

try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
    MCP_PIPEDREAM_URL = st.secrets.get("MCP_PIPEDREAM_URL")
    ASSISTANT_NAME = st.secrets.get("PINECONE_ASSISTANT_NAME", "fifi")
except KeyError as e:
    st.error(f"Missing critical secret: {e}. The app cannot continue.")
    st.stop()


# --- Tool Definitions ---

# This function becomes the "Pinecone Knowledge Tool"
def query_pinecone_knowledge_base(query: str, assistant, memory_instance, thread_config: dict) -> str:
    """
    Use this tool for ANY question about 1-2-Taste products, services, ingredients,
    recipes, applications, or any other topic in the 1-2-Taste catalog.
    """
    if not assistant:
        return "Error: Pinecone Assistant client is not available."

    # Get history from the central agent's memory
    checkpoint = memory_instance.get(thread_config)
    history_messages = checkpoint.get("messages", []) if checkpoint else []

    # Convert LangChain messages to the format the Pinecone SDK expects
    sdk_messages = []
    for msg in history_messages:
        if isinstance(msg, HumanMessage):
            sdk_messages.append(Message(role="user", content=msg.content))
        elif isinstance(msg, AIMessage) and isinstance(msg.content, str) and msg.content:
            sdk_messages.append(Message(role="assistant", content=msg.content))

    try:
        # The Pinecone SDK gets the full context to do its job properly
        response_from_sdk = assistant.chat(messages=sdk_messages, model="gpt-4o")
        if hasattr(response_from_sdk, 'message') and hasattr(response_from_sdk.message, 'content'):
            return response_from_sdk.message.content or "(The assistant returned empty content.)"
        return "(Could not find content in the assistant's response.)"
    except Exception as e:
        return f"An error occurred while querying the knowledge base: {str(e)}"

# --- Agent and Tool Initialization (Cached for Performance) ---
@st.cache_resource(ttl=3600)
def initialize_agent_and_tools():
    """
    Creates the agent, its tools, and the memory system. This is the new core.
    """
    print("--- Initializing agent and all tools ---")
    all_tools = []

    # 1. Initialize the agent's memory FIRST.
    #    This is the crucial change. The memory must exist before any tools that use it.
    print("Initializing agent memory...")
    memory = MemorySaver()
    thread_config = {"configurable": {"thread_id": THREAD_ID}}

    # 2. Now that memory exists, create the Pinecone Tool.
    print("Initializing Pinecone Assistant...")
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        pinecone_assistant = pc.assistant.Assistant(assistant_name=ASSISTANT_NAME)
        
        # We pass the 'memory' variable we just created.
        pinecone_tool = Tool(
            name="get_product_and_knowledge_info",
            func=partial(
                query_pinecone_knowledge_base,
                assistant=pinecone_assistant,
                memory_instance=memory, # Use the local memory variable
                thread_config=thread_config
            ),
            description="Use for any questions about products, ingredients, recipes, or company knowledge."
        )
        all_tools.append(pinecone_tool)
        print("✅ Pinecone tool loaded successfully.")
    except Exception as e:
        st.error(f"Fatal Error: Could not initialize Pinecone Assistant: {e}")
        st.stop()

    # 3. Fetch the WooCommerce Tools (with robust error handling).
    print("Initializing WooCommerce tools...")
    if MCP_PIPEDREAM_URL:
        try:
            mcp_client = MultiServerMCPClient(
                {"pipedream": {"url": MCP_PIPEDREAM_URL, "transport": "sse"}}
            )
            woocommerce_tools = asyncio.run(mcp_client.get_tools())
            all_tools.extend(woocommerce_tools)
            print(f"✅ WooCommerce tools loaded successfully. ({len(woocommerce_tools)} tools found)")
        except Exception as e:
            st.warning(f"Could not load WooCommerce tools. The agent will work for product questions but not for e-commerce tasks. Error: {e}")
            print(f"⚠️ Could not load WooCommerce tools. Error: {e}")
    else:
        st.warning("WooCommerce URL is not configured. The agent will only answer product questions.")
        print("⚠️ WooCommerce URL not configured. Skipping.")

    # 4. Create the agent with all available tools and the memory checkpointer.
    print("Initializing the agent with available tools...")
    llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY, temperature=0)
    agent_executor = create_react_agent(llm, all_tools, checkpointer=memory)
    
    print("--- ✅ Agent is ready ---")
    return agent_executor

# --- System Prompt: The Agent's Core Instructions ---
SYSTEM_PROMPT = """You are FiFi, a specialized AI assistant for 1-2-Taste. You have two distinct functions. You must decide which one to use based on the user's query.

1.  **Product & Knowledge Expert:** For any question about products, ingredients, recipes, applications, or general company knowledge, you **MUST** use the `get_product_and_knowledge_info` tool. Do not try to answer these from memory.

2.  **E-commerce Assistant:** For any task related to customer orders, shipping, or accounts, you **MUST** use the appropriate WooCommerce tool from your list.

Evaluate the user's latest query and route it to the correct tool.
"""

# --- Main App Logic (with agent control flow) ---

def handle_submission(query):
    """Sets the app to 'thinking' mode and stores the query."""
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.query_to_process = query
    st.session_state.is_thinking = True

async def execute_agent(query, agent_executor):
    """Invokes the agent to get a response."""
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
    agent_executor = initialize_agent_and_tools()
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

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Show thinking indicator and execute agent if needed
if st.session_state.is_thinking:
    with st.chat_message("assistant"):
        st.markdown("⌛ FiFi is thinking...")
    if st.session_state.query_to_process:
        asyncio.run(execute_agent(st.session_state.query_to_process, agent_executor))
        st.rerun()

# The user input field
user_prompt = st.chat_input("Ask FiFi Co-Pilot...", disabled=st.session_state.is_thinking)
if user_prompt:
    handle_submission(user_prompt)
    st.rerun()

# Sidebar content remains the same...
st.sidebar.markdown("---")
# ... (download button, clear history, etc.)
