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

from pinecone import Pinecone
from pinecone_plugins.assistant.models.chat import Message
from pinecone_plugins.assistant.control.core.client.exceptions import PineconeApiException

st.write("🔍 App is starting...")

MAX_HISTORY_TOKENS = 90000
MESSAGES_TO_KEEP_AFTER_PRUNING = 6
TOKEN_MODEL_ENCODING = "cl100k_base"

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
MCP_PIPEDREAM_URL = st.secrets.get("MCP_PIPEDREAM_URL")
PINECONE_PLUGIN_API_KEY = st.secrets.get("PINECONE_API_KEY")
PINECONE_ASSISTANT_NAME = st.secrets.get("PINECONE_ASSISTANT_NAME", "fifi")

st.write(f"✅ Secrets loaded: OPENAI={bool(OPENAI_API_KEY)}, MCP={bool(MCP_PIPEDREAM_URL)}, PINECONE={bool(PINECONE_PLUGIN_API_KEY)}")

if not all([OPENAI_API_KEY, MCP_PIPEDREAM_URL, PINECONE_PLUGIN_API_KEY]):
    st.error("❌ One or more critical secrets are missing (OpenAI, Pipedream, Pinecone Plugin).")
    st.stop()

llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY, temperature=0.2)
THREAD_ID = "fifi_streamlit_session"

st.write("✅ LLM Initialized")

try:
    pc = Pinecone(api_key=PINECONE_PLUGIN_API_KEY)
    pinecone_assistant_client = pc.assistant.Assistant(assistant_name=PINECONE_ASSISTANT_NAME)
    st.write("✅ Pinecone Assistant initialized")
except Exception as e:
    st.error(f"❌ Failed to initialize Pinecone Assistant: {e}")
    st.stop()

def _query_pinecone_assistant_with_client(query: str, client) -> dict:
    try:
        if not client:
            return {"final_answer": "Error: Pinecone Assistant client not initialized."}

        sdk_message = Message(role="user", content=query)
        response_from_sdk = client.chat(messages=[sdk_message], model="gpt-4o")

        if hasattr(response_from_sdk, 'message') and hasattr(response_from_sdk.message, 'content'):
            content = response_from_sdk.message.content or "(The assistant returned an empty response.)"
            return {"final_answer": content}

        return {"final_answer": "(No content returned from Pinecone Assistant.)"}
    except Exception as e:
        print(f"ERROR querying Pinecone Assistant tool: {e}")
        return {"final_answer": f"Error getting product info: {str(e)}"}

def get_system_prompt(agent_components):
    pinecone_tool = agent_components['pinecone_tool_name']
    return f"""You are FiFi, a specialized AI assistant for 1-2-Taste.

**Primary Directives:**
1.  **Tool Prioritization:**
    *   For any query about 1-2-Taste products, services, ingredients, recipes, or industry topics, try using the `{pinecone_tool}` tool.
    *   For e-commerce tasks like orders or customer accounts, use the appropriate WooCommerce tool based on its description.
    *   If the product tool doesn't help, you may use your general knowledge to assist the user.

2.  **User-Facing Persona:**
    *   When asked about your capabilities, describe your functions simply (e.g., "I can answer questions about 1-2-Taste products and ingredients.").
    *   Cite your sources where available. If no link is available, say it’s from the 1-2-Taste catalog.

Answer the user's latest query based on these core directives and the conversation history.
"""

@st.cache_resource(ttl=3600)
def get_agent_components():
    async def get_mcp_tools():
        mcp_client = MultiServerMCPClient({"pipedream": {"url": MCP_PIPEDREAM_URL, "transport": "sse"}})
        return await mcp_client.get_tools()

    woocommerce_tools = asyncio.run(get_mcp_tools())
    bound_query_func = partial(_query_pinecone_assistant_with_client, client=pinecone_assistant_client)

    pinecone_assistant_tool = Tool(
        name="get_12taste_product_context",
        func=bound_query_func,
        description="Use this tool to get information about 1-2-Taste products, services, ingredients, flavors, recipes, applications, or any other topic related to the 1-2-Taste catalog or food and beverage industry."
    )

    all_tools = [pinecone_assistant_tool] + woocommerce_tools
    memory = MemorySaver()
    agent_executor = create_react_agent(llm, all_tools, checkpointer=memory)
    all_tool_details = {tool.name: tool.description for tool in all_tools}

    return {
        "agent_executor": agent_executor,
        "memory_instance": memory,
        "pinecone_tool_name": pinecone_assistant_tool.name,
        "all_tool_details_for_prompt": all_tool_details
    }

async def execute_agent_call_with_memory(user_query: str, agent_components: dict):
    assistant_reply = ""
    try:
        agent_executor = agent_components["agent_executor"]
        memory_instance = agent_components["memory_instance"]
        config = {"configurable": {"thread_id": THREAD_ID}}
        system_prompt_content = get_system_prompt(agent_components)

        memory_state = memory_instance.get(config) or {}
        if memory_state.get("messages"):
            token_count = count_tokens(memory_state["messages"])
            if token_count > MAX_HISTORY_TOKENS:
                history = memory_state["messages"][-MESSAGES_TO_KEEP_AFTER_PRUNING:]
                memory_instance.put(config, {"messages": history})

        current_turn_messages = [
            {"role": "system", "content": system_prompt_content},
            {"role": "user", "content": user_query}
        ]
        event = {"messages": current_turn_messages}
        result = await agent_executor.ainvoke(event, config=config)

        if isinstance(result, dict):
            if "final_answer" in result:
                assistant_reply = result["final_answer"]
            elif "messages" in result and result["messages"]:
                assistant_reply = result["messages"][-1].content
        else:
            assistant_reply = "(Unexpected response format from agent.)"

    except Exception as e:
        st.error(f"Error during agent execution: {e}")
        assistant_reply = f"(Error: {e})"

    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
    st.session_state.thinking_for_ui = False
    st.rerun()

def count_tokens(messages: list, model_encoding: str = TOKEN_MODEL_ENCODING) -> int:
    try:
        encoding = tiktoken.get_encoding(model_encoding)
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")
    total = 0
    for message in messages:
        total += 4  # base tokens per message
        for key, value in message.items():
            total += len(encoding.encode(str(value)))
    return total + 2

def handle_new_query_submission(query_text: str):
    if not st.session_state.get('thinking_for_ui', False):
        st.session_state.messages.append({"role": "user", "content": query_text})
        st.session_state.query_to_process = query_text
        st.session_state.thinking_for_ui = True
        st.rerun()

st.title("FiFi Co-Pilot 🚀 (LangGraph Hybrid Agent)")

try:
    agent_components = get_agent_components()
    st.session_state.components_loaded = True
except Exception as e:
    st.error(f"❌ Agent setup failed: {e}")
    st.session_state.components_loaded = False
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if 'thinking_for_ui' not in st.session_state:
    st.session_state.thinking_for_ui = False
if 'query_to_process' not in st.session_state:
    st.session_state.query_to_process = None

st.sidebar.markdown("## Quick Questions")
for q in [
    "Help me with my recipe for a new juice drink",
    "Suggest me some strawberry flavours for beverage",
    "I need vanilla flavours for ice-cream"
]:
    if st.sidebar.button(q, key=q):
        handle_new_query_submission(q)

if st.sidebar.button("🧹 Clear Chat History"):
    st.session_state.clear()
    get_agent_components.clear()
    st.rerun()

if st.session_state.messages:
    chat_export_data_txt = "\n\n".join([
        f"{m['role'].capitalize()}: {str(m['content'])}" for m in st.session_state.messages
    ])
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    st.sidebar.download_button(
        "📥 Download Chat (TXT)", chat_export_data_txt,
        file_name=f"fifi_chat_{now}.txt", mime="text/plain"
    )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(str(msg["content"]))

if st.session_state.get("thinking_for_ui"):
    with st.chat_message("assistant"):
        st.markdown("⌛ FiFi is thinking...")

if st.session_state.get("thinking_for_ui") and st.session_state.get("query_to_process"):
    asyncio.run(execute_agent_call_with_memory(
        st.session_state.query_to_process,
        agent_components
    ))

user_prompt = st.chat_input("Ask FiFi Co-Pilot...", key="main_chat_input", disabled=not st.session_state.get("components_loaded", False))
if user_prompt:
    handle_new_query_submission(user_prompt)
