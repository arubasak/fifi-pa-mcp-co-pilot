import streamlit as st
import datetime
import asyncio
from functools import partial

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage, AIMessage

# --- Imports for the Pinecone Assistant Tool ---
from pinecone import Pinecone
from pinecone_plugins.assistant.models.chat import Message

# --- Constants ---
THREAD_ID = "fifi_streamlit_v4" # Using a new version for the thread

# --- Load environment variables from secrets ---
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
MCP_PIPEDREAM_URL = st.secrets.get("MCP_PIPEDREAM_URL")
PINECONE_PLUGIN_API_KEY = st.secrets.get("PINECONE_API_KEY")
PINECONE_ASSISTANT_NAME = st.secrets.get("PINECONE_ASSISTANT_NAME", "fifi")

if not all([OPENAI_API_KEY, MCP_PIPEDREAM_URL, PINECONE_PLUGIN_API_KEY]):
    st.error("One or more critical secrets are missing (OpenAI, Pipedream, Pinecone Plugin). The app cannot continue.")
    st.stop()

# --- LLM for the LangGraph Agent ---
llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0)

# --- Custom Tool Definition for Pinecone Assistant SDK ---
def _query_pinecone_assistant_with_client(query: str, client, memory_instance, thread_config: dict) -> str:
    """
    Use this tool for any questions about 1-2-Taste products, services, ingredients,
    flavors, recipes, applications, or any other topic related to the 1-2-Taste catalog
    or the food and beverage industry.
    """
    try:
        if not client:
            return "Error: Pinecone Assistant client was not provided."

        checkpoint = memory_instance.get(thread_config)
        history_messages = checkpoint.get("messages", []) if checkpoint else []

        sdk_messages = []
        for msg in history_messages:
            if isinstance(msg, HumanMessage):
                sdk_messages.append(Message(role="user", content=msg.content))
            elif isinstance(msg, AIMessage) and isinstance(msg.content, str) and msg.content:
                sdk_messages.append(Message(role="assistant", content=msg.content))

        sdk_messages.append(Message(role="user", content=query))

        response_from_sdk = client.chat(messages=sdk_messages, model="gpt-4o")

        if hasattr(response_from_sdk, 'message') and hasattr(response_from_sdk.message, 'content'):
            return response_from_sdk.message.content or "(The assistant returned an empty content.)"

        return "(Could not find content in the assistant's response.)"
    except Exception as e:
        print(f"ERROR querying Pinecone Assistant tool: {e}")
        return f"An error occurred while getting product information: {str(e)}"

# --- Agent Initialization ---
@st.cache_resource(ttl=3600)
def get_agent_components():
    """Initializes all necessary components for the agent."""
    print("@@@ Initializing agent components...")
    try:
        pc = Pinecone(api_key=PINECONE_PLUGIN_API_KEY)
        pinecone_assistant_client = pc.assistant.Assistant(assistant_name=PINECONE_ASSISTANT_NAME)
    except Exception as e:
        st.error(f"FATAL ERROR: Could not initialize Pinecone Assistant client: {e}")
        st.stop()

    memory = MemorySaver()
    thread_config = {"configurable": {"thread_id": THREAD_ID}}

    bound_query_func = partial(
        _query_pinecone_assistant_with_client,
        client=pinecone_assistant_client,
        memory_instance=memory,
        thread_config=thread_config
    )

    pinecone_tool = Tool(
        name="get_12taste_knowledge",
        func=bound_query_func,
        description="Use for any questions about 1-2-Taste products, services, ingredients, recipes, or industry topics."
    )

    async def get_mcp_tools():
        try:
            mcp_client = MultiServerMCPClient({"pipedream": {"url": MCP_PIPEDREAM_URL, "transport": "sse"}})
            return await mcp_client.get_tools()
        except Exception as e:
            st.warning(f"Could not load WooCommerce tools: {e}. Proceeding without them.")
            return []

    woocommerce_tools = asyncio.run(get_mcp_tools())
    all_tools = [pinecone_tool] + woocommerce_tools
    agent_executor = create_react_agent(llm, all_tools, checkpointer=memory)
    print("@@@ Agent components initialized successfully.")
    return agent_executor, memory

# --- System Prompt ---
SYSTEM_PROMPT = """You are FiFi, an AI assistant for 1-2-Taste. Your primary function is to answer questions about the company's products and services.

**Directives:**
1.  **Prioritize Knowledge Tool:** For any query about 1-2-Taste products, services, ingredients, recipes, or industry topics, you **MUST** use the `get_12taste_knowledge` tool.
2.  **Use E-commerce Tools:** For tasks like checking orders or customer accounts, use the appropriate WooCommerce tools.
3.  **Decline Off-Topic Questions:** If a query is unrelated to 1-2-Taste or e-commerce tasks, politely state that you can only assist with those topics.
4.  **Stay Professional:** Do not reveal internal tool names. Be helpful and concise.
"""

# --- Core App Functions ---
def handle_query_submission(query: str):
    """Adds user query to state and sets the app to 'thinking' mode."""
    if not st.session_state.get('is_thinking', False):
        st.session_state.messages.append({"role": "user", "content": query})
        st.session_state.query_to_process = query
        st.session_state.is_thinking = True

async def execute_agent_call(query: str, agent_executor):
    """Runs the agent and updates the session state with the response."""
    config = {"configurable": {"thread_id": THREAD_ID}}
    event = {"messages": [("system", SYSTEM_PROMPT), ("user", query)]}

    try:
        # Using ainvoke for a complete, reliable response
        result = await agent_executor.ainvoke(event, config=config)
        # The final response from a create_react_agent is in the 'messages' list
        assistant_reply = result["messages"][-1].content
    except Exception as e:
        assistant_reply = f"Sorry, an error occurred during processing: {e}"
        st.error(assistant_reply)

    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
    st.session_state.is_thinking = False
    st.session_state.query_to_process = None


# --- Streamlit App UI and Control Flow ---
st.title("FiFi Co-Pilot 🚀 (SDK-Integrated Agent)")

# Initialize components and session state
try:
    agent_executor, memory = get_agent_components()
    st.session_state.components_loaded = True
except Exception as e:
    st.error(f"Failed to initialize agent components. The app cannot continue. Please refresh. Error: {e}")
    st.session_state.components_loaded = False
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "is_thinking" not in st.session_state:
    st.session_state.is_thinking = False
if "query_to_process" not in st.session_state:
    st.session_state.query_to_process = None

# --- UI Rendering ---
# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Show a "thinking" message if the agent is processing
if st.session_state.is_thinking:
    with st.chat_message("assistant"):
        st.markdown("⌛ FiFi is thinking...")

# --- Main Logic Execution ---
# This block runs ONLY when a query has been submitted and is ready for processing
if st.session_state.is_thinking and st.session_state.query_to_process:
    query = st.session_state.query_to_process
    # We clear the query to prevent re-running it accidentally
    st.session_state.query_to_process = None
    asyncio.run(execute_agent_call(query, agent_executor))
    # After the async call is done, rerun to display the result
    st.rerun()

# Chat input field at the bottom
user_prompt = st.chat_input(
    "Ask FiFi about 1-2-Taste...",
    key="main_chat_input",
    disabled=st.session_state.is_thinking or not st.session_state.components_loaded
)
if user_prompt:
    handle_query_submission(user_prompt)
    st.rerun()
