# --- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
import streamlit as st

# Configuration is set to "auto" to ensure sidebar collapses on mobile, as intended.
st.set_page_config(
    page_title="FiFi",
    page_icon="assets/fifi-avatar.png",
    layout="wide",
    initial_sidebar_state="auto" # Correct for mobile sidebar behavior
)

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

# --- FINAL: Robust Memory Management Constants ---
HISTORY_MESSAGE_THRESHOLD = 6
HISTORY_TOKEN_THRESHOLD = 25000
MESSAGES_TO_RETAIN_AFTER_SUMMARY = 2
MAX_INPUT_TOKENS = 25904
TOKEN_MODEL_ENCODING = "cl100k_base"

# --- Load environment variables from secrets ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MCP_PINECONE_URL = os.environ.get("MCP_PINECONE_URL")
MCP_PINECONE_API_KEY = os.environ.get("MCP_PINECONE_API_KEY")
MCP_PIPEDREAM_URL = os.environ.get("MCP_PIPEDREAM_URL")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

SECRETS_ARE_MISSING = not all([OPENAI_API_KEY, MCP_PINECONE_URL, MCP_PINECONE_API_KEY, MCP_PIPEDREAM_URL, TAVILY_API_KEY])

if not SECRETS_ARE_MISSING:
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0.2)
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

# --- System Prompt Definition (Preserved from your code) ---
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
    *   If you tried the `{pinecone_tool}` for a query that seemed product-specific but it returned no relevant results, you should then use `tavily_search_fallback` (Web Search).
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

# --- Layer 1: History Management Function ---
async def manage_history_with_summary(memory: MemorySaver, config: dict, llm_for_summary: ChatOpenAI):
    checkpoint = memory.get(config)
    if not checkpoint: return
    history = checkpoint.get("messages", [])
    conversational_history = [msg for msg in history if isinstance(msg, (AIMessage, HumanMessage))]
    token_count = count_tokens(conversational_history)
    message_count = len(conversational_history)
    if message_count > HISTORY_MESSAGE_THRESHOLD or token_count > HISTORY_TOKEN_THRESHOLD:
        st.info("Conversation history is long. Summarizing older messages...")
        print(f"@@@ MEMORY MGMT: Triggered. Msgs: {message_count}, Tokens: {token_count}")
        if len(conversational_history) <= MESSAGES_TO_RETAIN_AFTER_SUMMARY: return
        messages_to_summarize = conversational_history[:-MESSAGES_TO_RETAIN_AFTER_SUMMARY]
        messages_to_keep = conversational_history[-MESSAGES_TO_RETAIN_AFTER_SUMMARY:]
        summarization_prompt = [SystemMessage(content="You are an expert at creating concise, third-person summaries of conversations. Extract all key entities, topics, and user intentions mentioned."), HumanMessage(content="\n".join([f"{m.type.capitalize()}: {m.content}" for m in messages_to_summarize]))]
        try:
            summary_response = await llm_for_summary.ainvoke(summarization_prompt)
            summary_text = summary_response.content
            new_history = [SystemMessage(content=f"Summary of preceding conversation: {summary_text}"), *messages_to_keep]
            checkpoint["messages"] = new_history
            memory.put(config, checkpoint)
            print("@@@ MEMORY MGMT: History successfully summarized.")
        except Exception as e:
            st.error(f"Could not summarize history: {e}")

# --- Layer 2: Final Prompt Safety Net ---
def truncate_prompt_if_needed(messages: list, max_tokens: int) -> list:
    total_tokens = count_tokens(messages)
    if total_tokens <= max_tokens:
        return messages
    st.warning(f"Request is too large ({total_tokens} tokens). Shortening conversation to fit within limits.")
    print(f"@@@ SAFETY NET: Payload size {total_tokens} > {max_tokens}. Truncating.")
    system_message = messages[0]
    user_query = messages[-1]
    history = messages[1:-1]
    while count_tokens([system_message] + history + [user_query]) > max_tokens and history:
        history.pop(0)
    return [system_message] + history + [user_query]

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
        await manage_history_with_summary(agent_components["memory_instance"], config, agent_components["llm_for_summary"])
        main_system_prompt_content_str = agent_components["main_system_prompt_content_str"]
        current_checkpoint = agent_components["memory_instance"].get(config)
        history_messages = current_checkpoint.get("messages", []) if current_checkpoint else []
        event_messages = [SystemMessage(content=main_system_prompt_content_str)] + history_messages + [HumanMessage(content=user_query)]
        final_messages = truncate_prompt_if_needed(event_messages, MAX_INPUT_TOKENS)
        event = {"messages": final_messages}
        result = await agent_components["agent_executor"].ainvoke(event, config=config)
        if isinstance(result, dict) and "messages" in result and result["messages"]:
            for msg in reversed(result["messages"]):
                if isinstance(msg, AIMessage):
                    assistant_reply = msg.content
                    break
            if not assistant_reply:
                assistant_reply = f"(Error: No AI message found for query: '{user_query}')"
        else:
            assistant_reply = f"(Error: Unexpected response format: {type(result)})"
    except Exception as e:
        st.error(f"Error during agent invocation: {e}\n{traceback.format_exc()}")
        assistant_reply = f"(Error: {e})"
    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
    st.session_state.thinking_for_ui = False
    st.rerun()

# --- Input Handling Function ---
def handle_new_query_submission(query_text: str):
    if not st.session_state.get('thinking_for_ui', False) and query_text:
        st.session_state.active_question = query_text
        st.session_state.messages.append({"role": "user", "content": query_text})
        st.session_state.query_to_process = query_text
        st.session_state.thinking_for_ui = True
        st.rerun()

# --- Streamlit App Starts Here ---

# This CSS creates the custom "floating" input bar, applies original styling,
# and positions the terms text.
st.markdown("""
<style>
    /* 1. Increase the font size for the introductory caption */
    [data-testid="stCaptionContainer"] p {
        font-size: 1.1em !important;
    }

    /* 2. Add padding to the bottom of the chat message list.
          This value must be larger than the total height of our fixed input container
          (input + button + terms text + margins + 10px lift) to prevent overlap. */
    .main .st-emotion-cache-1gulkj5 {
         padding-bottom: 9rem; /* Approx. 144px needed, 9rem is 144px at 16px base font */
    }

    /* 3. The main container for our custom fixed input bar and terms text.
          This lifts the entire block 10px from the bottom and centers it. */
    .custom-fixed-input-container {
        position: fixed;
        bottom: 10px; /* Exact 10px lift you requested */
        left: 50%;
        transform: translateX(-50%);
        width: 100%;
        max-width: 736px; /* Streamlit's default wide layout main column width */
        background-color: #ffffff;
        padding: 1rem; /* Padding inside this container around the input/terms */
        box-sizing: border-box; /* Include padding in width calculation */
        z-index: 99;
        /* No border here, as the input field itself will have the border */
    }

    /* 4. Styling for the st.text_input element to match the original st.chat_input's border */
    /* Target the internal div that holds the actual input field and has the border */
    .custom-fixed-input-container div[data-testid="stTextInput"] > div[data-baseweb="input"] {
        border: 1px solid #cccccc; /* Original light grey border */
        border-radius: 7px; /* Original rounded corners */
        padding: 0.75rem 1rem; /* Adjust padding to match original appearance */
        line-height: 1.5; /* Ensure text sits nicely */
    }
    /* Style for focus state - when the input is active */
    .custom-fixed-input-container div[data-testid="stTextInput"] > div[data-baseweb="input"]:focus-within {
        border-color: #e6007e; /* Original Mexican Pink on focus */
        box-shadow: none; /* Remove default blue glow if present */
    }

    /* 5. Styling for the "Send" button to be compact and use the arrow icon */
    .custom-fixed-input-container button[data-testid="baseButton-secondary"] {
        background-color: transparent !important; /* No background */
        border: none !important; /* No border */
        color: #8c8c8c !important; /* Grey color for icon, like original */
        font-size: 1.8rem; /* Make icon larger */
        padding: 0.1rem 0.5rem; /* Minimal padding for button */
        height: auto; /* Allow height to adapt to icon */
        margin-top: -0.2rem; /* Minor vertical alignment adjustment */
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .custom-fixed-input-container button[data-testid="baseButton-secondary"]:hover {
        color: #e6007e !important; /* Pink on hover */
    }
    .custom-fixed-input-container button[data-testid="baseButton-secondary"] div { /* Target text within button */
        display: none; /* Hide button text label if "Send" is present */
    }

    /* 6. Style for the "Terms and Conditions" text directly below the input */
    .terms-text {
        text-align: center;
        color: grey;
        font-size: 0.75rem;
        margin-top: 8px; /* Minimum margin requested */
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='font-size: 24px;'>FiFi, AI sourcing assistant</h1>", unsafe_allow_html=True)
st.caption("Hello, I am FiFi, your AI-powered assistant, designed to support you across the sourcing and product development journey. Find the right ingredients, explore recipe ideas, technical data, and more.")

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
    "Suggest some natural strawberry flavours for beverage",
    "Latest trends in plant-based proteins for 2025?",
    "Suggest me some vanilla flavours for ice-cream"
]
for question in preview_questions:
    button_type = "primary" if st.session_state.active_question == question else "secondary"
    if st.sidebar.button(question, key=f"preview_{question}", use_container_width=True, type=button_type):
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

# Display chat messages
for message in st.session_state.get("messages", []):
    if message["role"] == "assistant":
        with st.chat_message("assistant", avatar="assets/fifi-avatar.png"):
            st.markdown(message.get("content", ""))
    else:
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

# --- CUSTOM FLOATING INPUT BAR with "Terms and Conditions" below it ---
# This block replaces the original st.chat_input and provides the requested layout.
st.markdown('<div class="custom-fixed-input-container">', unsafe_allow_html=True)

# Using st.form ensures 'Enter' key submission functionality
with st.form(key='chat_form', clear_on_submit=True):
    col1, col2 = st.columns([9, 1]) # Column ratio for input field and button
    with col1:
        user_prompt = st.text_input(
            "Ask me for ingredients, recipes, or order support—in any language.",
            label_visibility="collapsed", # Hide the label to mimic st.chat_input
            key="user_query_input", # Unique key for the text input
            disabled=st.session_state.get('thinking_for_ui', False) or not st.session_state.get("components_loaded", False)
        )
    with col2:
        # The submit button for the form, using a right-arrow to mimic original "Enter" icon visually
        submit_button = st.form_submit_button(
            "➤", # Unicode character for right arrow
            use_container_width=True,
            # key="send_message_button", # REMOVED: This was causing the TypeError when inside a form.
            disabled=st.session_state.get('thinking_for_ui', False) or not st.session_state.get("components_loaded", False)
        )

# The "Terms and Conditions" text, placed directly below the input field within the same fixed container
st.markdown("""
<div class="terms-text">
    By using this agent, you agree to our <a href="https://www.12taste.com/terms-conditions/" target="_blank">Terms of Service</a>.
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True) # Close the custom-fixed-input-container div

# Logic to handle the submission from our custom form
if submit_button and user_prompt:
    st.session_state.active_question = None
    handle_new_query_submission(user_prompt)
