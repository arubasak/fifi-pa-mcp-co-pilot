import streamlit as st
import datetime
import asyncio
import tiktoken
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
from pinecone_plugins.assistant.control.core.client.exceptions import PineconeApiException

# --- Constants ---
MAX_HISTORY_TOKENS = 90000
MESSAGES_TO_KEEP_AFTER_PRUNING = 10
TOKEN_MODEL_ENCODING = "cl100k_base"
THREAD_ID = "fifi_streamlit_v3"

# --- Load environment variables from secrets ---
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
MCP_PIPEDREAM_URL = st.secrets.get("MCP_PIPEDREAM_URL")
PINECONE_PLUGIN_API_KEY = st.secrets.get("PINECONE_API_KEY")
PINECONE_ASSISTANT_NAME = st.secrets.get("PINECONE_ASSISTANT_NAME", "fifi")

if not all([OPENAI_API_KEY, MCP_PIPEDREAM_URL, PINECONE_PLUGIN_API_KEY]):
    st.error("One or more critical secrets are missing (OpenAI, Pipedream, Pinecone Plugin).")
    st.stop()

# --- LLM for the LangGraph Agent ---
# This LLM's primary job is to decide which tool to use.
llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0)


# --- Custom Tool Definition for Pinecone Assistant SDK ---
def _query_pinecone_assistant_with_client(query: str, client, memory_instance, thread_config: dict) -> str:
    """
    Use this tool for any questions about 1-2-Taste products, services, ingredients,
    flavors, recipes, applications, or any other topic related to the 1-2-Taste catalog
    or the food and beverage industry. This is your primary source of knowledge.
    """
    try:
        if not client:
            return "Error: Pinecone Assistant client was not provided."

        # 1. Get the conversational history from the LangGraph checkpointer.
        checkpoint = memory_instance.get(thread_config)
        history_messages = checkpoint.get("messages", []) if checkpoint else []

        # 2. Convert LangChain messages to Pinecone SDK Message format.
        #    The Pinecone SDK will handle its own context and memory.
        sdk_messages = []
        for msg in history_messages:
            if isinstance(msg, HumanMessage):
                sdk_messages.append(Message(role="user", content=msg.content))
            elif isinstance(msg, AIMessage):
                # Ensure we don't pass tool call information, only the content.
                if isinstance(msg.content, str) and msg.content:
                    sdk_messages.append(Message(role="assistant", content=msg.content))
        
        # Add the current user query as the latest message
        sdk_messages.append(Message(role="user", content=query))
        
        # 3. Call the Pinecone Assistant with the full conversational history.
        #    We let the SDK manage the context for its semantic search.
        response_from_sdk = client.chat(messages=sdk_messages, model="gpt-4o")

        if hasattr(response_from_sdk, 'message') and hasattr(response_from_sdk.message, 'content'):
            return response_from_sdk.message.content or "(The assistant returned an empty content.)"

        return "(Could not find content in the assistant's response.)"
    except Exception as e:
        print(f"ERROR querying Pinecone Assistant tool: {e}")
        return f"An error occurred while trying to get product information: {str(e)}"

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

    # The memory instance will be passed to our custom tool
    memory = MemorySaver()

    # Bind the client and memory to the tool function
    bound_query_func = partial(
        _query_pinecone_assistant_with_client,
        client=pinecone_assistant_client,
        memory_instance=memory,
        thread_config={"configurable": {"thread_id": THREAD_ID}}
    )

    pinecone_tool = Tool(
        name="get_12taste_knowledge",
        func=bound_query_func,
        description=(
            "Use for questions about 1-2-Taste products, services, ingredients, recipes, or industry topics. "
            "Pass the user's original question directly to this tool."
        )
    )

    # Fetch other tools, e.g., WooCommerce
    async def get_mcp_tools():
        mcp_client = MultiServerMCPClient({"pipedream": {"url": MCP_PIPEDREAM_URL, "transport": "sse"}})
        return await mcp_client.get_tools()

    try:
        woocommerce_tools = asyncio.run(get_mcp_tools())
    except Exception as e:
        st.warning(f"Could not load WooCommerce tools: {e}. Proceeding without them.")
        woocommerce_tools = []


    all_tools = [pinecone_tool] + woocommerce_tools

    # The agent's main job is now primarily routing to the correct tool.
    agent_executor = create_react_agent(llm, all_tools, checkpointer=memory)
    print("@@@ Agent components initialized successfully.")
    return agent_executor, memory

# --- System Prompt ---
SYSTEM_PROMPT = """You are FiFi, an AI assistant for 1-2-Taste.

Your capabilities are divided into two areas:
1.  **Product & Knowledge Expert**: For any query about 1-2-Taste products, services, ingredients, recipes, or industry topics, you **MUST** use the `get_12taste_knowledge` tool. This is your primary function.
2.  **E-commerce Assistant**: For tasks related to customer orders, accounts, or shipping, use the appropriate WooCommerce tool based on its description.

**NEVER** answer product or knowledge questions from your own general knowledge. Always use the `get_12taste_knowledge` tool.
If a query is outside these two areas, politely state that you can only assist with 1-2-Taste topics.
Do not reveal your internal tool names to the user.
"""

# --- Main App Logic ---
st.title("FiFi Co-Pilot 🚀 (SDK-Integrated Agent)")

try:
    agent_executor, memory = get_agent_components()
    st.session_state.components_loaded = True
except Exception as e:
    st.error(f"Failed to initialize agent components: {e}")
    st.session_state.components_loaded = False
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

async def get_agent_response(user_query: str):
    """Invokes the agent and streams the response."""
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            config = {"configurable": {"thread_id": THREAD_ID}}
            
            # The agent will now manage history in its checkpointer.
            # We add the system prompt to every call to ensure it's respected.
            event = {
                "messages": [
                    ("system", SYSTEM_PROMPT),
                    ("user", user_query)
                ]
            }
            
            async for chunk in agent_executor.astream(event, config=config):
                # The final response is in the 'messages' of the last chunk
                if "messages" in chunk:
                    last_message = chunk["messages"][-1]
                    if last_message.content:
                        full_response = last_message.content
                        message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            error_message = f"Sorry, an error occurred: {e}"
            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})


if user_prompt := st.chat_input("Ask FiFi about 1-2-Taste products...", disabled=not st.session_state.get("components_loaded")):
    asyncio.run(get_agent_response(user_prompt))
