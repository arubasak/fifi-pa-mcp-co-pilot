# --- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
import streamlit as st
import base64
from pathlib import Path
import asyncio
import tiktoken
import os
import traceback
import uuid
from typing import Any, List

# --- LangGraph and LangChain Imports ---
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool
# Using a general utility for token counting, as tiktoken needs explicit encoding
from langchain_core.messages.utils import count_tokens_approximately 

# --- The correct summarization tool from langmem ---
from langmem.short_term import SummarizationNode

# --- Helper function to load and Base64-encode images for stateless deployment ---
@st.cache_data
def get_image_as_base64(file_path):
    """Loads an image file and returns it as a Base64 encoded string."""
    try:
        path = Path(file_path)
        with path.open("rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception as e:
        print(f"Error loading image {file_path}: {e}")
        return None

# Load images once using the helper function
FIFI_AVATAR_B64 = get_image_as_base64("assets/fifi-avatar.png")
USER_AVATAR_B64 = get_image_as_base64("assets/user-avatar.png")

st.set_page_config(
    page_title="FiFi",
    page_icon=f"data:image/png;base64,{FIFI_AVATAR_B64}" if FIFI_AVATAR_B64 else "🤖",
    layout="wide",
    initial_sidebar_state="auto"
)

# Robust asyncio helper function that works in any environment
def get_or_create_eventloop():
    """Gets the active asyncio event loop or creates a new one."""
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

# --- FINAL: "Balanced" Memory Strategy Constants ---
MAX_HISTORY_TOKENS = 20000
MAX_SUMMARY_TOKENS = 2000
TOKEN_MODEL_ENCODING = "cl100k_base" # Still for Tiktoken if used elsewhere directly

# --- NEW: Define a custom state to hold the summarizer's context ---
class State(AgentState):
    context: dict[str, Any]

# --- Load environment variables from secrets ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MCP_PINECONE_URL = os.environ.get("MCP_PINECONE_URL")
MCP_PINECONE_API_KEY = os.environ.get("MCP_PINECONE_API_KEY")
MCP_PIPEDREAM_URL = os.environ.get("MCP_PIPEDREAM_URL")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

SECRETS_ARE_MISSING = not all([OPENAI_API_KEY, MCP_PINECONE_URL, MCP_PINECONE_API_KEY, MCP_PIPEDREAM_URL, TAVILY_API_KEY])

if not SECRETS_ARE_MISSING:
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0.2)
    summarization_llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0)
    
    if 'thread_id' not in st.session_state:
        st.session_state.thread_id = f"fifi_streamlit_session_{uuid.uuid4()}"
    THREAD_ID = st.session_state.thread_id

# --- Custom Tavily Fallback & General Search Tool ---
@tool
def tavily_search_fallback(query: str) -> str:
    """Search the web using Tavily. Use this for queries about broader, public-knowledge topics."""
    try:
        tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        response = tavily_client.search(query=query, search_depth="advanced", max_results=5, include_answer=True, include_raw_content=False)
        if response.get('answer'):
            result = f"Web Search Results:\n\nSummary: {response['answer']}\n\nSources:\n"
        else:
            result = "Web Search Results:\n\nSources:\n"
        for i, source in enumerate(response.get('results', [])):
            result += f"{i}. {source['title']}\n   URL: {source['url']}\n   Content: {source['content'][:300]}...\n\n"
        return result
    except Exception as e:
        return f"Error performing web search: {str(e)}"

# --- System Prompt Definition ---
def get_system_prompt_content_string():
    pinecone_tool = "functions.get_context"
    return f"""You are FiFi, the expert AI assistant for 1-2-Taste, specializing in food and beverage ingredients. Your role is to assist with product inquiries, industry trends, food science, and B2B support. Politely decline out-of-scope questions.

**Tool Selection Framework:**
1.  **`{pinecone_tool}` (Knowledge Base):** This is your **first choice** for all internal information. Use it for specific product details, ingredient recommendations, applications, and technical data. You MUST use `top_k=5` and `snippet_size=1024`.
2.  **`tavily_search_fallback` (Web Search):** Use this as a **fallback** if the knowledge base has no relevant results, or for broad topics like recent market news or general food science questions.
3.  **E-commerce Tools:** Use only for explicit requests about orders, accounts, or shipping.

**Response Rules (Strictly Enforced):**
*   **Citations are MANDATORY:** For knowledge base results, cite `productURL` or `source_url`. For web results, cite the source URL.
*   **Product Safety:** NEVER mention a product that lacks a verifiable URL. NEVER provide prices; direct users to the product page or sales contact.
*   **Failure:** If all tools fail, state that the information could not be found.

Based on the conversation history and these instructions, answer the user's last query."""

# --- Helper function for tiktoken-based token counting (for reference, but count_tokens_approximately is used by node) ---
# Keeping this for accuracy reference. Note: SummarizationNode uses `count_tokens_approximately` by default.
def count_tokens_tiktoken(messages: list, model_encoding: str = TOKEN_MODEL_ENCODING) -> int:
    """Calculates the total number of tokens for a list of messages using the official tiktoken library."""
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

# --- Async handler for agent initialization (Correct LangGraph Pattern) ---
@st.cache_resource(ttl=3600)
def get_agent_components():
    """Initializes and caches the agent components using the correct LangGraph pattern."""
    print("@@@ get_agent_components: Populating cache...")
    loop = get_or_create_eventloop()
    client = MultiServerMCPClient({
        "pinecone": {"url": MCP_PINECONE_URL, "transport": "sse", "headers": {"Authorization": f"Bearer {MCP_PINECONE_API_KEY}"}},
        "pipedream": {"url": MCP_PIPEDREAM_URL, "transport": "sse"}
    })
    mcp_tools = loop.run_until_complete(client.get_tools())
    all_tools = list(mcp_tools) + [tavily_search_fallback]
    
    checkpointer = MemorySaver()
    system_prompt_content = get_system_prompt_content_string()

    # 1. Instantiate the official SummarizationNode.
    # Note: `SummarizationNode` uses `count_tokens_approximately` by default,
    # which is less accurate than tiktoken but provided by LangChain.
    # We are explicitly stating parameters.
    summarization_node = SummarizationNode(
        model=summarization_llm,
        max_tokens=MAX_HISTORY_TOKENS,
        max_summary_tokens=MAX_SUMMARY_TOKENS,
        input_messages_key="messages",
        output_messages_key="messages",
        # system_message is NOT a constructor argument for SummarizationNode.
        # It is handled by the custom_pre_model_hook.
    )

    # 2. Define our own custom hook that chains the summarizer and system prompt.
    def custom_pre_model_hook(state: State) -> dict:
        """
        A custom hook that first runs the summarization node, and then
        prepends the static system message to the processed message list.
        """
        # Ensure 'context' is initialized in the state for SummarizationNode's internal use.
        # This makes the hook robust even if the very first state doesn't have it.
        if "context" not in state:
            state["context"] = {}

        # Run the official summarization logic first.
        summarized_state = summarization_node(state)
        
        # Manually prepend the system message to the result.
        summarized_state["messages"] = [SystemMessage(content=system_prompt_content)] + summarized_state["messages"]
        return summarized_state

    # 3. Create the agent, passing our custom hook.
    agent_executor = create_react_agent(
        llm,
        all_tools,
        checkpointer=checkpointer,
        state_schema=State,
        pre_model_hook=custom_pre_model_hook,
        # system_message is NOT a valid argument for create_react_agent.
    )
    
    print("@@@ Initialization complete.")
    return {"agent_executor": agent_executor, "checkpointer": checkpointer}

# --- SIMPLIFIED: Agent Orchestrator ---
async def execute_agent_call(user_query: str, agent_components: dict):
    """Runs the agent. Memory and system prompt are handled automatically by the pre_model_hook."""
    try:
        config = {"configurable": {"thread_id": THREAD_ID}}
        agent_executor = agent_components["agent_executor"]
        
        # The input to the agent is just the human message.
        # The pre_model_hook will ensure the system prompt and managed history are prepended.
        event = {"messages": [HumanMessage(content=user_query)]}
        result = await agent_executor.ainvoke(event, config=config)

        assistant_reply = ""
        if isinstance(result, dict) and "messages" in result:
            for msg in reversed(result["messages"]):
                if isinstance(msg, AIMessage) and msg.content:
                    assistant_reply = msg.content
                    break
            else:
                 assistant_reply = "(Error: No valid AI response content was found.)"
        else:
            assistant_reply = f"(Error: Unexpected response format from agent: {type(result)})"
        return assistant_reply

    except Exception as e:
        print(f"Error during agent invocation: {e}\n{traceback.format_exc()}")
        return f"(An error occurred during processing. Please try again.)"

# --- Input Handling Function for Streamlit ---
def handle_new_query_submission(query_text: str):
    if not st.session_state.get('thinking_for_ui', False):
        st.session_state.active_question = query_text
        st.session_state.messages.append({"role": "user", "content": query_text})
        st.session_state.query_to_process = query_text
        st.session_state.thinking_for_ui = True
        st.rerun()

# --- Streamlit App UI and Main Execution Logic ---
st.markdown("""
<style>
    /* CSS remains unchanged */
    .st-emotion-cache-1629p8f { border: 1px solid #ffffff; border-radius: 7px; bottom: 5px; position: fixed; width: 100%; max-width: 736px; left: 50%; transform: translateX(-50%); z-index: 101; }
    .st-emotion-cache-1629p8f:focus-within { border-color: #e6007e; }
    [data-testid="stCaptionContainer"] p { font-size: 1.3em !important; }
    .terms-footer { position: fixed; bottom: 10px; left: 50%; transform: translateX(-50%); width: 100%; max-width: 736px; text-align: center; color: grey; font-size: 0.90rem; z-index: 100; }
    [data-testid="stVerticalBlock"] { padding-bottom: 40px; }
    [data-testid="stChatMessage"] { margin-top: 0.1rem !important; margin-bottom: 0.1rem !important; }
    .stApp { overflow-y: auto !important; }
    .st-scroll-to-bottom { display: none !important; }
    .st-emotion-cache-1fplawd { display: none !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='font-size: 24px;'>FiFi, AI sourcing assistant</h1>", unsafe_allow_html=True)
st.caption("Hello, I am FiFi, your AI-powered assistant, designed to support you across the sourcing and product development journey. Find the right ingredients, explore recipe ideas, technical data, and more.")

if SECRETS_ARE_MISSING:
    st.error("Secrets missing. Please configure necessary environment variables.")
    st.stop()

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
    # Retrieve the checkpointer and clear the history for the current thread_id
    checkpointer = agent_components.get("checkpointer")
    if checkpointer:
        checkpointer.put({"configurable": {"thread_id": THREAD_ID}}, {"messages": []})
        print(f"@@@ Chat history for thread ID {THREAD_ID} cleared.")
        
    st.session_state.messages = []
    st.session_state.thinking_for_ui = False
    st.session_state.query_to_process = None
    st.session_state.thread_id = f"fifi_streamlit_session_{uuid.uuid4()}" # Generate new thread ID
    st.session_state.active_question = None
    print(f"@@@ New chat session started. New Thread ID: {st.session_state.thread_id}")
    st.rerun()

fifi_avatar_icon = f"data:image/png;base64,{FIFI_AVATAR_B64}" if FIFI_AVATAR_B64 else "🤖"
user_avatar_icon = f"data:image/png;base64,{USER_AVATAR_B64}" if USER_AVATAR_B64 else "🧑‍💻"
for message in st.session_state.get("messages", []):
    avatar_icon = fifi_avatar_icon if message["role"] == "assistant" else user_avatar_icon
    with st.chat_message(message["role"], avatar=avatar_icon):
        st.markdown(message.get("content", ""))

if st.session_state.get('thinking_for_ui', False):
    with st.chat_message("assistant", avatar=fifi_avatar_icon):
        st.markdown("⌛ FiFi is thinking...")

st.markdown("""
<div class="terms-footer">
    By using this agent, you agree to our <a href="https://www.12taste.com/terms-conditions/" target="_blank">Terms of Service</a>.
</div>
""", unsafe_allow_html=True)

user_prompt = st.chat_input("Ask me for ingredients, recipes, or product development—in any language.", key="main_chat_input",
                            disabled=st.session_state.get('thinking_for_ui', False) or not st.session_state.get("components_loaded", False))
if user_prompt:
    st.session_state.active_question = None
    handle_new_query_submission(user_prompt)

if st.session_state.get('query_to_process'):
    query_to_run = st.session_state.query_to_process
    
    loop = get_or_create_eventloop()
    assistant_reply = loop.run_until_complete(execute_agent_call(query_to_run, agent_components))

    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
    st.session_state.thinking_for_ui = False
    st.session_state.query_to_process = None
    st.rerun()
