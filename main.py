import streamlit as st
import datetime
import asyncio
import tiktoken
import os
import traceback
import uuid

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool
from tavily import TavilyClient

# --- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constants for History Summarization ---
SUMMARIZE_THRESHOLD_TOKENS = 500
MESSAGES_TO_KEEP_AFTER_SUMMARIZATION = 12
TOKEN_MODEL_ENCODING = "cl100k_base"

# --- Load environment variables from secrets ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MCP_PINECONE_URL = os.environ.get("MCP_PINECONE_URL")
MCP_PINECONE_API_KEY = os.environ.get("MCP_PINECONE_API_KEY")
MCP_PIPEDREAM_URL = os.environ.get("MCP_PIPEDREAM_URL")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

SECRETS_ARE_MISSING = not all([OPENAI_API_KEY, MCP_PINECONE_URL, MCP_PINECONE_API_KEY, MCP_PIPEDREAM_URL, TAVILY_API_KEY])

if not SECRETS_ARE_MISSING:
    llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY, temperature=0.2)
    if 'thread_id' not in st.session_state:
        st.session_state.thread_id = f"fifi_streamlit_session_{uuid.uuid4()}"
    THREAD_ID = st.session_state.thread_id

# --- Custom Tavily Fallback & General Search Tool ---
@tool
def tavily_search_fallback(query: str) -> str:
    """
    Search the web using Tavily. Use this as your first choice for queries about broader, public-knowledge topics like recent industry news, market trends, or general food science questions.
    """
    try:
        tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        response = tavily_client.search(
            query=query, search_depth="advanced", max_results=5, include_answer=True, include_raw_content=False
        )
        if response.get('answer'):
            result = f"Web Search Results:\n\nSummary: {response['answer']}\n\nSources:\n"
        else:
            result = "Web Search Results:\n\nSources:\n"
        for i, source in enumerate(response.get('results', []), 1):
            result += f"{i}. {source['title']}\n   URL: {source['url']}\n   Content: {source['content'][:300]}...\n\n"
        return result
    except Exception as e:
        return f"Error performing web search: {str(e)}"

# --- System Prompt Definition (CORRECTED) ---
def get_system_prompt_content_string(agent_components_for_prompt=None):
    if agent_components_for_prompt is None:
        agent_components_for_prompt = { 'pinecone_tool_name': "functions.get_context" }
    pinecone_tool = agent_components_for_prompt['pinecone_tool_name']
    prompt = f"""You are FiFi, the expert AI assistant for 1-2-Taste.
Your role is strictly limited to inquiries about 1-2-Taste products, the food/beverage industry, relevant food science, B2B support, and specific e-commerce tasks. Politely decline all out-of-scope questions.
**Intelligent Tool Selection Framework:**
Your first step is to analyze the user's query to determine the best tool. Do not just follow a rigid order; select the tool that best fits the user's intent.
1.  **When to use `{pinecone_tool}` (Knowledge Base):**
    *   Use this tool as your **first choice** for queries about 1-2-Taste's internal information.
    *   **Primary Use Cases:** Specific product details, product recommendations, applications of specific ingredients, and information found in your internal documents.
    *   **Required Parameters:** You MUST use `top_k=5` and `snippet_size=1024`.
2.  **When to use `tavily_search_fallback` (Web Search):**
    *   Use this tool as your **first choice** for queries about broader, public-knowledge topics.
    *   **Primary Use Cases:** Recent industry news or market trends, general food science questions, and high-level questions about ingredient categories.
3.  **Using Web Search as a Fallback:**
    *   If you tried the `{pinecone_tool}` for a query that seemed product-specific but it returned no relevant results, you should then use the web search tool.
4.  **E-commerce Tools:**
    *   Use these tools ONLY for explicit user requests about "WooCommerce", "orders", "customer accounts", or "shipping status".
**Response Formatting Rules (Strictly Enforced):**
*   **Citations are MANDATORY:**
    *   For knowledge base results, cite `productURL`, `source_url`, or `sourceURL`.
    *   For web results, state the information is from a web search and cite the source URL.
*   **Product Rules:**
    *   You MUST NOT mention any product from tool outputs that lacks a verifiable URL.
    *   You MUST NOT provide product prices. Direct the user to the product page or the correct contact.
*   **Failure:** If all tools fail, politely state that the information could not be found.
Based on the conversation history and these instructions, answer the user's last query."""
    return prompt

# --- Helper function to count tokens ---
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

# --- Function to summarize history if needed ---
async def summarize_history_if_needed(
    memory_instance: MemorySaver, thread_config: dict, main_system_prompt_content_str: str,
    summarize_threshold_tokens: int, keep_last_n_interactions: int, llm_for_summary: ChatOpenAI
):
    checkpoint = memory_instance.get(thread_config)
    current_stored_messages = checkpoint.get("messages", []) if checkpoint else []
    cleaned_messages = [m for m in current_stored_messages if not (isinstance(m, SystemMessage) and m.content == main_system_prompt_content_str)]
    conversational_messages_only = cleaned_messages
    current_token_count = count_tokens(conversational_messages_only)
    if current_token_count > summarize_threshold_tokens:
        st.info(f"Summarization Triggered...")
        if len(conversational_messages_only) <= keep_last_n_interactions: return False
        messages_to_summarize = conversational_messages_only[:-keep_last_n_interactions]
        messages_to_keep_raw = conversational_messages_only[-keep_last_n_interactions:]
        if messages_to_summarize:
            summarization_prompt_messages = [SystemMessage(content="Summarize this conversation."), HumanMessage(content="\n".join([f"{m.type.capitalize()}: {m.content}" for m in messages_to_summarize]))]
            try:
                summary_response = await llm_for_summary.ainvoke(summarization_prompt_messages)
                summary_content = summary_response.content
                new_messages_for_checkpoint = [SystemMessage(content=f"Summary: {summary_content}")] + messages_to_keep_raw
                if checkpoint is None: checkpoint = {"messages": []}
                checkpoint["messages"] = new_messages_for_checkpoint
                memory_instance.put(thread_config, checkpoint)
                return True
            except Exception as e:
                st.error(f"Failed to generate summary: {e}")
                return False
    return False

# --- Async handler for agent initialization ---
@st.cache_resource(ttl=3600)
def get_agent_components():
    async def run_async_initialization():
        print("@@@ ASYNC: Initializing resources...")
        client = MultiServerMCPClient({
            "pinecone": {"url": MCP_PINECONE_URL, "transport": "sse", "headers": {"Authorization": f"Bearer {MCP_PINECONE_API_KEY}"}},
            "pipedream": {"url": MCP_PIPEDREAM_URL, "transport": "sse"}
        })
        mcp_tools = await client.get_tools()
        all_tools = list(mcp_tools) + [tavily_search_fallback]
        memory = MemorySaver()
        pinecone_tool_name = "functions.get_context"
        system_prompt_content_value = get_system_prompt_content_string({'pinecone_tool_name': pinecone_tool_name})
        agent_executor = create_react_agent(llm, all_tools, checkpointer=memory)
        print("@@@ ASYNC: Initialization complete.")
        return {"agent_executor": agent_executor, "memory_instance": memory, "llm_for_summary": llm, "main_system_prompt_content_str": system_prompt_content_value}
    print("@@@ get_agent_components: Populating cache...")
    return asyncio.run(run_async_initialization())

# --- Async handler for user queries ---
async def execute_agent_call_with_memory(user_query: str, agent_components: dict):
    assistant_reply = ""
    try:
        config = {"configurable": {"thread_id": THREAD_ID}}
        main_system_prompt_content_str = agent_components["main_system_prompt_content_str"]
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
            if not assistant_reply:
                assistant_reply = f"(Error: No AI message found for query: '{user_query}')"
        else:
            assistant_reply = f"(Error: Unexpected response format: {type(result)} - {result})"
    except Exception as e:
        st.error(f"Error during agent invocation: {e}\n{traceback.format_exc()}")
        assistant_reply = f"(Error: {e})"
    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
    st.session_state.thinking_for_ui = False
    st.rerun()

# --- Input Handling Function ---
def handle_new_query_submission(query_text: str):
    if not st.session_state.get('thinking_for_ui', False):
        st.session_state.messages.append({"role": "user", "content": query_text})
        st.session_state.query_to_process = query_text
        st.session_state.thinking_for_ui = True
        st.rerun()

# --- Streamlit App Starts Here ---
st.markdown("""
<style>
    /* This is the container for the chat input */
    .st-emotion-cache-1629p8f {
        border: 1px solid #cccccc; /* Set a default light grey border */
        border-radius: 7px; /* Optional: adds rounded corners like in the screenshot */
    }
    /* This applies when the user clicks into the text input */
    .st-emotion-cache-1629p8f:focus-within {
        border-color: #e6007e; /* Change border to Mexican Pink on focus */
    }
</style>
""", unsafe_allow_html=True)
st.markdown("<h1 style='font-size: 24px;'>FiFi Co-Pilot</h1>", unsafe_allow_html=True)
st.caption("Hello, I am FiFi, your AI-powered assistant, designed to support you across the sourcing and product development journey. Find the right ingredients, explore recipe ideas, receive real-time assistance with your orders and more.")

if SECRETS_ARE_MISSING:
    st.error("Secrets missing. Please configure OPENAI_API_KEY, MCP_PINECONE_URL, MCP_PINECONE_API_KEY, MCP_PIPEDREAM_URL, and TAVILY_API_KEY.")
    st.stop()

# Initialize session state variables
if "messages" not in st.session_state: st.session_state.messages = []
if 'thinking_for_ui' not in st.session_state: st.session_state.thinking_for_ui = False
if 'query_to_process' not in st.session_state: st.session_state.query_to_process = None
if 'components_loaded' not in st.session_state: st.session_state.components_loaded = False
if 'active_question' not in st.session_state: st.session_state.active_question = None

try:
    agent_components = get_agent_components()
    st.session_state.components_loaded = True
except Exception as e:
    st.error(f"Failed to initialize agent. Please refresh. Error: {e}")
    st.session_state.components_loaded = False
    st.stop()

# --- UI Rendering ---
st.sidebar.markdown("## Quick questions")
preview_questions = [
    "Suggest some natural strawberry flavours for a beverage",
    "Latest trends in plant-based proteins for 2025?",
    "Get order status"
]
for question in preview_questions:
    button_type = "primary" if st.session_state.active_question == question else "secondary"
    if st.sidebar.button(question, key=f"preview_{question}", use_container_width=True, type=button_type):
        st.session_state.active_question = question
        handle_new_query_submission(question)

st.sidebar.markdown("---")
if st.sidebar.button("🧹 Reset chat session", use_container_width=True):
    st.session_state.messages = []
    st.session_state.thinking_for_ui = False
    st.session_state.query_to_process = None
    st.session_state.thread_id = f"fifi_streamlit_session_{uuid.uuid4()}"
    st.session_state.active_question = None
    print(f"@@@ New chat session started. Thread ID: {st.session_state.thread_id}")
    st.rerun()

# Display chat messages with custom assistant avatar
for message in st.session_state.get("messages", []):
    if message["role"] == "assistant":
        with st.chat_message("assistant", avatar="assets/fifi-avatar.png"):
            st.markdown(message.get("content", ""))
    else:
        # MODIFIED: Add the user avatar
        with st.chat_message("user", avatar="assets/user-avatar.png"):
            st.markdown(message.get("content", ""))

if st.session_state.get('thinking_for_ui', False):
    with st.chat_message("assistant", avatar="assets/fifi-avatar.png"):
        st.markdown("⌛ FiFi is thinking...")

# Process new queries
if st.session_state.get('query_to_process'):
    query_to_run = st.session_state.query_to_process
    st.session_state.query_to_process = None
    asyncio.run(execute_agent_call_with_memory(query_to_run, agent_components))

# Chat input
user_prompt = st.chat_input("I'm FiFi, your AI assistant for sourcing and product development. Ask me for ingredients, recipes, or order support—in any language.", key="main_chat_input",
                            disabled=st.session_state.get('thinking_for_ui', False) or not st.session_state.get("components_loaded", False))
if user_prompt:
    st.session_state.active_question = None
    handle_new_query_submission(user_prompt)
