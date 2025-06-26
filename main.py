import streamlit as st
import datetime
import asyncio
import tiktoken
import os
import uuid  # NEW: Import for generating unique IDs

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from tavily import TavilyClient

# --- Constants for History Summarization & Pruning ---
SUMMARIZE_THRESHOLD_TOKENS = 6000
MESSAGES_TO_KEEP_AFTER_SUMMARIZATION = 4
PRUNE_TOOL_MESSAGE_THRESHOLD_CHARS = 1500
TOKEN_MODEL_ENCODING = "cl100k_base"

# --- Load environment variables from secrets ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MCP_PINECONE_URL = os.environ.get("MCP_PINECONE_URL")
MCP_PINECONE_API_KEY = os.environ.get("MCP_PINECONE_API_KEY")
MCP_PIPEDREAM_URL = os.environ.get("MCP_PIPEDREAM_URL")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

SECRETS_ARE_MISSING = not all([OPENAI_API_KEY, MCP_PINECONE_URL, MCP_PINECONE_API_KEY, MCP_PIPEDREAM_URL, TAVILY_API_KEY])

# --- We no longer need a hardcoded THREAD_ID ---
# THREAD_ID = "fifi_streamlit_session" <-- DELETED

if not SECRETS_ARE_MISSING:
    llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY, temperature=0.2)

# --- Pydantic model for the tool's input schema ---
class TavilySearchInput(BaseModel):
    query: str = Field(description="The search query to pass to the Tavily search engine.")

@tool(args_schema=TavilySearchInput)
def tavily_search_fallback(query: str) -> str:
    """Search the web using Tavily when the knowledge base doesn't have sufficient information."""
    try:
        tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        response = tavily_client.search(query=query, search_depth="advanced", max_results=5, include_answer=True, include_raw_content=False)
        if response.get('answer'):
            result = f"Web Search Results:\n\nSummary: {response['answer']}\n\nSources:\n"
        else:
            result = "Web Search Results:\n\nSources:\n"
        for i, source in enumerate(response.get('results', []), 1):
            result += f"{i}. {source['title']}\n   URL: {source['url']}\n   Content: {source['content'][:300]}...\n\n"
        return result
    except Exception as e:
        return f"Error performing web search: {str(e)}"

# --- System Prompt Definition (Unchanged) ---
def get_system_prompt_content_string(agent_components_for_prompt=None):
    # This is the full, correct system prompt from the previous step
    if agent_components_for_prompt is None:
        agent_components_for_prompt = {
            'pinecone_tool_name': "functions.get_context",
            'all_tool_details_for_prompt': {
                "functions.get_context": "Retrieves relevant document snippets from the assistant knowledge base.",
                "tavily_search_fallback": "Searches the web for fallback information."
            }
        }
    pinecone_tool = agent_components_for_prompt['pinecone_tool_name']
    all_tool_details = agent_components_for_prompt['all_tool_details_for_prompt']
    prompt = f"""You are FiFi, an expert AI assistant for 1-2-Taste... (Full system prompt is unchanged) ... Decision Logic for Fallback: Always try the primary tool first. If, and only if, the results are inadequate, then consider using the web search tool..."""
    return prompt

# --- All Helper Functions (count_tokens, prune_history, summarize_history_if_needed) are unchanged ---
def count_tokens(messages: list, model_encoding: str = TOKEN_MODEL_ENCODING) -> int:
    if not messages: return 0
    try: encoding = tiktoken.get_encoding(model_encoding)
    except Exception: encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = 0
    for message in messages:
        num_tokens += 4 
        if isinstance(message, BaseMessage): content = message.content
        elif isinstance(message, dict): content = message.get("content", "")
        else: content = str(message)
        if content is not None:
            try: num_tokens += len(encoding.encode(str(content)))
            except (TypeError, AttributeError): pass
    num_tokens += 2
    return num_tokens

def prune_history(memory_instance: MemorySaver, thread_config: dict, threshold: int):
    checkpoint = memory_instance.get(thread_config)
    if not checkpoint or "messages" not in checkpoint: return
    messages = checkpoint.get("messages", [])
    modified = False
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], AIMessage): break
        if isinstance(messages[i], ToolMessage) and len(str(messages[i].content)) > threshold:
            messages[i] = ToolMessage(content=f"[Context from tool call pruned. Length: {len(str(messages[i].content))}]", tool_call_id=messages[i].tool_call_id)
            modified = True
    if modified:
        memory_instance.put(thread_config, {"messages": messages})
        print(f"INFO: Pruned oversized ToolMessage from memory for thread {thread_config['configurable']['thread_id']}")

async def summarize_history_if_needed(memory_instance: MemorySaver, thread_config: dict, main_system_prompt_content_str: str, summarize_threshold_tokens: int, keep_last_n_interactions: int, llm_for_summary: ChatOpenAI):
    checkpoint = memory_instance.get(thread_config)
    if not checkpoint: return False
    current_stored_messages = checkpoint.get("messages", [])
    cleaned_messages = [m for m in current_stored_messages if not (isinstance(m, SystemMessage) and m.content == main_system_prompt_content_str)]
    current_token_count = count_tokens(cleaned_messages)
    st.sidebar.markdown(f"**Conv. Tokens:** `{current_token_count}` / `{summarize_threshold_tokens}`")
    if current_token_count > summarize_threshold_tokens:
        if len(cleaned_messages) <= keep_last_n_interactions: return False
        messages_to_summarize = cleaned_messages[:-keep_last_n_interactions]
        messages_to_keep_raw = cleaned_messages[-keep_last_n_interactions:]
        if messages_to_summarize:
            summarization_prompt = [SystemMessage(content="Summarize this conversation..."), HumanMessage(content="\n".join(f"{m.type}: {m.content}" for m in messages_to_summarize))]
            summary_response = await llm_for_summary.ainvoke(summarization_prompt)
            new_messages = [SystemMessage(content=f"Summary: {summary_response.content}")] + messages_to_keep_raw
            memory_instance.put(thread_config, {"messages": new_messages})
            return True
    return False

# --- Agent Initialization (Unchanged) ---
async def run_async_initialization():
    print("@@@ ASYNC run_async_initialization...")
    client = MultiServerMCPClient({"pinecone": {"url": MCP_PINECONE_URL, "transport": "sse", "headers": {"Authorization": f"Bearer {MCP_PINECONE_API_KEY}"}}, "pipedream": {"url": MCP_PIPEDREAM_URL, "transport": "sse"}})
    mcp_tools = await client.get_tools()
    all_tools = list(mcp_tools) + [tavily_search_fallback]
    memory = MemorySaver()
    pinecone_tool_name = "functions.get_context"
    all_tool_details = {tool.name: tool.description for tool in all_tools}
    system_prompt_content_value = get_system_prompt_content_string({'pinecone_tool_name': pinecone_tool_name, 'all_tool_details_for_prompt': all_tool_details})
    agent_executor = create_react_agent(llm, all_tools, checkpointer=memory)
    print("@@@ ASYNC run_async_initialization: Complete.")
    return {"agent_executor": agent_executor, "memory_instance": memory, "llm_for_summary": llm, "main_system_prompt_content_str": system_prompt_content_value}

@st.cache_resource(ttl=3600)
def get_agent_components_cached():
    print("@@@ Populating cache by running async initialization...")
    return asyncio.run(run_async_initialization())

# --- Agent Execution Call (MODIFIED to use session_state.thread_id) ---
async def execute_agent_call_with_memory(user_query: str):
    assistant_reply = ""
    agent_components = st.session_state.agent_components
    try:
        config = {"configurable": {"thread_id": st.session_state.thread_id}} # Use session's thread_id
        main_system_prompt_content_str = agent_components["main_system_prompt_content_str"]
        prune_history(agent_components["memory_instance"], config, PRUNE_TOOL_MESSAGE_THRESHOLD_CHARS)
        await summarize_history_if_needed(agent_components["memory_instance"], config, main_system_prompt_content_str, SUMMARIZE_THRESHOLD_TOKENS, MESSAGES_TO_KEEP_AFTER_SUMMARIZATION, agent_components["llm_for_summary"])
        current_checkpoint = agent_components["memory_instance"].get(config)
        history_messages = current_checkpoint.get("messages", []) if current_checkpoint else []
        event_messages = [SystemMessage(content=main_system_prompt_content_str)] + history_messages + [HumanMessage(content=user_query)]
        event = {"messages": event_messages}
        result = await agent_components["agent_executor"].ainvoke(event, config=config)
        if isinstance(result, dict) and "messages" in result and result["messages"]:
            for msg in reversed(result["messages"]):
                if isinstance(msg, AIMessage):
                    assistant_reply = msg.content
                    break
            if not assistant_reply: assistant_reply = "(Error: No AI message found)"
        else:
            assistant_reply = f"(Error: Unexpected agent response format)"
    except Exception as e:
        assistant_reply = f"(Error: {e})"
    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
    st.session_state.thinking_for_ui = False
    st.rerun()

def handle_new_query_submission(query_text: str):
    if not st.session_state.get('thinking_for_ui', False):
        st.session_state.messages.append({"role": "user", "content": query_text})
        st.session_state.query_to_process = query_text
        st.session_state.thinking_for_ui = True
        st.rerun()

# --- Streamlit App UI ---
st.title("FiFi Co-Pilot 🚀")

if SECRETS_ARE_MISSING:
    st.error("Secrets are missing. Please configure them in Streamlit secrets.")
    st.stop()

# --- INITIALIZE SESSION STATE (Now includes the unique thread_id) ---
if "agent_components" not in st.session_state: st.session_state.agent_components = None
if "messages" not in st.session_state: st.session_state.messages = []
if 'thinking_for_ui' not in st.session_state: st.session_state.thinking_for_ui = False
if 'query_to_process' not in st.session_state: st.session_state.query_to_process = None
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    print(f"@@@ New session started with Thread ID: {st.session_state.thread_id}")

# --- Agent Initialization ---
try:
    if st.session_state.agent_components is None:
        st.session_state.agent_components = get_agent_components_cached()
except Exception as e:
    st.error(f"Failed to initialize agent. Please refresh. Error: {e}")
    if "agent_components" in st.session_state: del st.session_state.agent_components
    st.stop()

# --- SIDEBAR UI ---
st.sidebar.markdown("## Memory Debugger")
st.sidebar.markdown("---")
st.sidebar.markdown("## Quick Questions")
preview_questions = ["Help me with my recipe...", "Suggest strawberry flavours...", "I need vanilla flavours...", "Latest trends in plant-based proteins?"]
for question in preview_questions:
    if st.sidebar.button(question, key=f"preview_{question}", use_container_width=True):
        handle_new_query_submission(question)

st.sidebar.markdown("---")
# --- NEW "Clear Chat" Logic (Safe and Correct) ---
if st.sidebar.button("🧹 Clear Chat History", use_container_width=True):
    st.session_state.messages = []
    st.session_state.query_to_process = None
    st.session_state.thinking_for_ui = False
    # Generate a new thread ID to start a fresh conversation
    st.session_state.thread_id = str(uuid.uuid4())
    print(f"@@@ New conversation started with new Thread ID: {st.session_state.thread_id}")
    st.rerun()

# --- MAIN CHAT UI ---
for message in st.session_state.get("messages", []):
    with st.chat_message(message["role"]):
        st.markdown(str(message.get("content", "")))

if st.session_state.get('thinking_for_ui', False):
    with st.chat_message("assistant"):
        st.markdown("⌛ FiFi is thinking...")

if st.session_state.get('query_to_process'):
    query_to_run = st.session_state.query_to_process
    st.session_state.query_to_process = None
    asyncio.run(execute_agent_call_with_memory(query_to_run))

user_prompt = st.chat_input("Ask FiFi Co-Pilot...", key="main_chat_input", disabled=st.session_state.get('thinking_for_ui', False))
if user_prompt:
    handle_new_query_submission(user_prompt)
