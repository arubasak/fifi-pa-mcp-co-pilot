# --- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
import streamlit as st
import base64
from pathlib import Path
import asyncio
import tiktoken
import os
import traceback
import uuid
from typing import List, Sequence

# --- LangGraph and LangChain Imports ---
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool
from tavily import TavilyClient

# --- The correct, canonical way to manage history ---
from langchain_core.runnables.history import RunnableWithMessageHistory
# THIS IS THE CRITICAL CORRECTION: Inheriting from the correct base class
from langchain_core.chat_history import BaseChatMessageHistory

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
MESSAGES_TO_RETAIN_AFTER_SUMMARY = 6
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

# --- Accurate token counting function ---
def count_tokens(messages: list, model_encoding: str = TOKEN_MODEL_ENCODING) -> int:
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

# --- NEW: Custom Message History Class with Summarization Logic ---
class SummarizedMessageHistory(BaseChatMessageHistory):
    """
    A custom message history class that summarizes the history when it exceeds a token limit.
    This class wraps a standard checkpointer and applies summarization on-the-fly.
    """
    def __init__(self, checkpointer: MemorySaver, thread_id: str):
        self.checkpointer = checkpointer
        self.thread_id = thread_id

    @property
    def messages(self) -> List[BaseMessage]:
        """Retrieve messages from the underlying checkpointer and summarize if needed."""
        config = {"configurable": {"thread_id": self.thread_id}}
        checkpoint = self.checkpointer.get(config)
        history = checkpoint.get("messages", []) if checkpoint else []

        token_count = count_tokens(history)

        if token_count <= MAX_HISTORY_TOKENS:
            return history

        print(f"@@@ MEMORY MGMT: History ({token_count} tokens) exceeds threshold ({MAX_HISTORY_TOKENS}). Summarizing.")
        
        messages_to_keep = history[-MESSAGES_TO_RETAIN_AFTER_SUMMARY:]
        messages_to_summarize = history[:-MESSAGES_TO_RETAIN_AFTER_SUMMARY]

        summarization_prompt = [
            SystemMessage(content=f"You are an expert at creating concise, third-person summaries of multi-turn agentic conversations. Extract all key entities, topics, questions, and critical information from tool outputs. The final summary must be under {MAX_SUMMARY_TOKENS} tokens."),
            HumanMessage(content="Please summarize the following conversation:\n\n" + "\n".join([f"{type(m).__name__}: {m.content}" for m in messages_to_summarize]))
        ]
        
        try:
            summary_response = summarization_llm.invoke(summarization_prompt)
            summary_text = summary_response.content
            new_history = [SystemMessage(content=f"This is a summary of the preceding conversation: {summary_text}")] + messages_to_keep
            
            # Persist the summarized history back to the checkpointer
            checkpoint["messages"] = new_history
            self.checkpointer.put(config, checkpoint)
            
            print("@@@ MEMORY MGMT: History successfully summarized.")
            return new_history
        except Exception as e:
            print(f"Error during summarization, returning full history as a fallback: {e}")
            return history

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Add new messages to the underlying checkpointer."""
        config = {"configurable": {"thread_id": self.thread_id}}
        checkpoint = self.checkpointer.get(config) or {"messages": []}
        checkpoint["messages"].extend(messages)
        self.checkpointer.put(config, checkpoint)

    def clear(self) -> None:
        """Clear history from the underlying checkpointer."""
        self.checkpointer.put({"configurable": {"thread_id": self.thread_id}}, {"messages": []})

# --- System Prompt Definition ---
def get_system_prompt_content_string():
    pinecone_tool = "functions.get_context" # Hardcoded as we are simplifying
    return f"""You are FiFi, the expert AI assistant for 1-2-Taste... (rest of your prompt)"""

# --- Async handler for agent initialization (CHANGED TO SYNC) ---
@st.cache_resource(ttl=3600)
def get_agent_components():
    """Initializes and caches the expensive agent components using the new architecture."""
    print("@@@ get_agent_components: Populating cache...")
    loop = get_or_create_eventloop()
    client = MultiServerMCPClient({
        "pinecone": {"url": MCP_PINECONE_URL, "transport": "sse", "headers": {"Authorization": f"Bearer {MCP_PINECONE_API_KEY}"}},
        "pipedream": {"url": MCP_PIPEDREAM_URL, "transport": "sse"}
    })
    mcp_tools = loop.run_until_complete(client.get_tools())
    all_tools = list(mcp_tools) + [tavily_search_fallback]
    
    checkpointer = MemorySaver()
    system_prompt = get_system_prompt_content_string()
    
    # 1. Create a prompt template that includes a placeholder for chat history
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ]
    )
    
    # 2. Create a "stateless" agent by binding tools to the LLM.
    agent = create_react_agent(llm, all_tools)

    # 3. Wrap the stateless agent with the history management runnable.
    agent_with_history = RunnableWithMessageHistory(
        agent,
        # The lambda function creates a new instance of our custom history class for each session_id
        lambda session_id: SummarizedMessageHistory(
            checkpointer=checkpointer,
            thread_id=session_id
        ),
        input_messages_key="messages",
        history_messages_key="chat_history",
    )
    
    print("@@@ Initialization complete.")
    return {"agent_with_history": agent_with_history, "checkpointer": checkpointer}

# --- SIMPLIFIED: Agent Orchestrator ---
def execute_agent_call(user_query: str, agent_components: dict):
    """Runs the agent. Memory is handled by the RunnableWithMessageHistory wrapper."""
    try:
        config = {"configurable": {"thread_id": THREAD_ID}}
        agent_with_history = agent_components["agent_with_history"]
        
        # The input schema for the default react agent is a dict with 'messages'
        event = {"messages": [HumanMessage(content=user_query)]}
        result = agent_with_history.invoke(event, config=config)

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
st.markdown("""<style>...</style>""", unsafe_allow_html=True) # CSS hidden for brevity

st.markdown("<h1 style='font-size: 24px;'>FiFi, AI sourcing assistant</h1>", unsafe_allow_html=True)
st.caption("Hello, I am FiFi...")

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
preview_questions = ["..."] # Shortened for brevity
for question in preview_questions:
    if st.sidebar.button(question, key=f"preview_{question}", use_container_width=True):
        handle_new_query_submission(question)

st.sidebar.markdown("---")
if st.sidebar.button("🧹 Reset chat session", use_container_width=True):
    checkpointer = agent_components.get("checkpointer")
    if checkpointer:
        history = SummarizedMessageHistory(checkpointer, THREAD_ID)
        history.clear()
        
    st.session_state.messages = []
    st.session_state.thinking_for_ui = False
    st.session_state.query_to_process = None
    st.session_state.thread_id = f"fifi_streamlit_session_{uuid.uuid4()}"
    st.session_state.active_question = None
    print(f"@@@ New chat session started. Thread ID: {st.session_state.thread_id}")
    st.rerun()

# Display chat history from session state
for message in st.session_state.get("messages", []):
    with st.chat_message(message["role"]):
        st.markdown(message.get("content", ""))

if st.session_state.get('thinking_for_ui', False):
    with st.chat_message("assistant"):
        st.markdown("⌛ FiFi is thinking...")

st.markdown("""<div class="terms-footer">...</div>""", unsafe_allow_html=True)

user_prompt = st.chat_input("Ask me...", key="main_chat_input",
                            disabled=st.session_state.get('thinking_for_ui', False) or not st.session_state.get("components_loaded", False))
if user_prompt:
    st.session_state.active_question = None
    handle_new_query_submission(user_prompt)

if st.session_state.get('query_to_process'):
    query_to_run = st.session_state.query_to_process
    
    # The call is now synchronous
    assistant_reply = execute_agent_call(query_to_run, agent_components)

    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
    st.session_state.thinking_for_ui = False
    st.session_state.query_to_process = None
    st.rerun()
