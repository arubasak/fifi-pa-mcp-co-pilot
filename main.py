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

MAX_HISTORY_TOKENS = 90000
MESSAGES_TO_KEEP_AFTER_PRUNING = 6
TOKEN_MODEL_ENCODING = "cl100k_base"

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
MCP_PIPEDREAM_URL = st.secrets.get("MCP_PIPEDREAM_URL")
PINECONE_PLUGIN_API_KEY = st.secrets.get("PINECONE_API_KEY")
PINECONE_ASSISTANT_NAME = st.secrets.get("PINECONE_ASSISTANT_NAME", "fifi")

if not all([OPENAI_API_KEY, MCP_PIPEDREAM_URL, PINECONE_PLUGIN_API_KEY]):
    st.error("One or more critical secrets are missing (OpenAI, Pipedream, Pinecone Plugin).")
    st.stop()

llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY, temperature=0.2)
THREAD_ID = "fifi_streamlit_session"

def count_tokens(messages: list, model_encoding: str = TOKEN_MODEL_ENCODING) -> int:
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

def prune_history_if_needed(memory_instance, thread_config, current_system_prompt_content, max_tokens, keep_last_n_interactions):
    checkpoint_value = memory_instance.get(thread_config)
    if not checkpoint_value or "messages" not in checkpoint_value or not isinstance(checkpoint_value.get("messages"), list):
        return False
    current_messages_in_history = checkpoint_value["messages"]
    if not current_messages_in_history: return False
    current_token_count = count_tokens(current_messages_in_history)
    if current_token_count > max_tokens:
        user_assistant_messages = [m for m in current_messages_in_history if m.get("role") != "system"]
        pruned_user_assistant_messages = user_assistant_messages[-keep_last_n_interactions:]
        new_history_messages = [{"role": "system", "content": current_system_prompt_content}]
        new_history_messages.extend(pruned_user_assistant_messages)
        memory_instance.put(thread_config, {"messages": new_history_messages})
        return True
    return False

def _query_pinecone_assistant_with_client(query: str, client) -> str:
    try:
        if not client:
            return "Error: Pinecone Assistant client was not provided to the tool."

        # ✅ Use raw query instead of prompt wrapping
        sdk_message = Message(role="user", content=query)
        response_from_sdk = client.chat(messages=[sdk_message], model="gpt-4o")

        if hasattr(response_from_sdk, 'message') and hasattr(response_from_sdk.message, 'content'):
            return response_from_sdk.message.content or "(The assistant returned an empty response.)"

        return "(Could not find content in the assistant's response.)"
    except Exception as e:
        print(f"ERROR querying Pinecone Assistant tool: {e}")
        return f"An error occurred while trying to get product information: {str(e)}"

@st.cache_resource(ttl=3600)
def get_agent_components():
    pc = Pinecone(api_key=PINECONE_PLUGIN_API_KEY)
    pinecone_assistant_client = pc.assistant.Assistant(assistant_name=PINECONE_ASSISTANT_NAME)

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
