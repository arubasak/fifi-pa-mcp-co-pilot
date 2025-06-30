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

# --- LangChain Imports ---
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage, BaseMessage, FunctionMessage
from langchain_core.tools import tool
from tavily import TavilyClient

# --- The correct, canonical way to manage history and build agents ---
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- Helper function to load and Base64-encode images ---
@st.cache_data
def get_image_as_base64(file_path):
    try:
        path = Path(file_path)
        with path.open("rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception as e:
        print(f"Error loading image {file_path}: {e}")
        return None

FIFI_AVATAR_B64 = get_image_as_base64("assets/fifi-avatar.png")
USER_AVATAR_B64 = get_image_as_base64("assets/user-avatar.png")

st.set_page_config(page_title="FiFi", page_icon=f"data:image/png;base64,{FIFI_AVATAR_B64}" if FIFI_AVATAR_B64 else "🤖", layout="wide")

def get_or_create_eventloop():
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

# --- Memory Strategy Constants ---
MAX_HISTORY_TOKENS = 20000
MAX_SUMMARY_TOKENS = 2000
MESSAGES_TO_RETAIN_AFTER_SUMMARY = 6
TOKEN_MODEL_ENCODING = "cl100k_base"

# --- Load Environment Variables ---
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

# --- Tools Definition ---
@tool
def tavily_search_fallback(query: str) -> str:
    """Search the web using Tavily."""
    try:
        tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        response = tavily_client.search(query=query, search_depth="advanced", max_results=5, include_answer=True)
        return str(response.get('results', []))
    except Exception as e:
        return f"Error performing web search: {str(e)}"

# --- Token Counting Function ---
def count_tokens(messages: list, model_encoding: str = TOKEN_MODEL_ENCODING) -> int:
    # ... (implementation remains the same)
    if not messages: return 0
    try: encoding = tiktoken.get_encoding(model_encoding)
    except Exception: encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = 0
    for message in messages:
        num_tokens += 4
        content = message.content if isinstance(message, BaseMessage) else str(message)
        if content:
            try: num_tokens += len(encoding.encode(content))
            except TypeError: pass
    return num_tokens + 2

# --- Custom History Class with Summarization ---
class SummarizedMessageHistory(BaseChatMessageHistory):
    """A custom message history class that summarizes history on-the-fly."""
    def __init__(self, checkpointer: MemorySaver, thread_id: str):
        self.checkpointer = checkpointer
        self.thread_id = thread_id

    @property
    def messages(self) -> List[BaseMessage]:
        config = {"configurable": {"thread_id": self.thread_id}}
        checkpoint = self.checkpointer.get(config)
        history = checkpoint.get("messages", []) if checkpoint else []
        token_count = count_tokens(history)

        if token_count <= MAX_HISTORY_TOKENS:
            return history

        print(f"@@@ MEMORY MGMT: History ({token_count}) > threshold. Summarizing.")
        messages_to_keep = history[-MESSAGES_TO_RETAIN_AFTER_SUMMARY:]
        messages_to_summarize = history[:-MESSAGES_TO_RETAIN_AFTER_SUMMARY]
        prompt = [
            SystemMessage(content=f"Summarize this conversation concisely. The summary must be under {MAX_SUMMARY_TOKENS} tokens."),
            HumanMessage(content="\n".join([f"{type(m).__name__}: {m.content}" for m in messages_to_summarize]))
        ]
        try:
            summary_text = summarization_llm.invoke(prompt).content
            new_history = [SystemMessage(content=f"Summary of conversation: {summary_text}")] + messages_to_keep
            checkpoint["messages"] = new_history
            self.checkpointer.put(config, checkpoint)
            return new_history
        except Exception as e:
            print(f"Error during summarization: {e}")
            return history

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        config = {"configurable": {"thread_id": self.thread_id}}
        checkpoint = self.checkpointer.get(config) or {"messages": []}
        checkpoint["messages"].extend(messages)
        self.checkpointer.put(config, checkpoint)

    def clear(self) -> None:
        self.checkpointer.put({"configurable": {"thread_id": self.thread_id}}, {"messages": []})

# --- Agent Initialization ---
@st.cache_resource(ttl=3600)
def get_agent_components():
    """Initializes and caches the agent components."""
    print("@@@ get_agent_components: Populating cache...")
    loop = get_or_create_eventloop()
    client = MultiServerMCPClient({"pinecone": {...}, "pipedream": {...}}) # Details omitted
    mcp_tools = loop.run_until_complete(client.get_tools())
    all_tools = list(mcp_tools) + [tavily_search_fallback]
    
    checkpointer = MemorySaver()
    
    system_prompt = f"""You are FiFi, the expert AI assistant... (rest of your prompt)"""

    # 1. Manually create the prompt template with the correct placeholders
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # 2. Create the core agent runnable using the modern `create_tool_calling_agent`
    agent = create_tool_calling_agent(llm, all_tools, prompt)
    
    # 3. Create the AgentExecutor which runs the agent loop
    agent_executor = AgentExecutor(agent=agent, tools=all_tools, verbose=True)
    
    # 4. Wrap the AgentExecutor with the history management runnable
    agent_with_history = RunnableWithMessageHistory(
        agent_executor,
        lambda session_id: SummarizedMessageHistory(
            checkpointer=checkpointer,
            thread_id=session_id
        ),
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    
    print("@@@ Initialization complete.")
    return {"agent_with_history": agent_with_history, "checkpointer": checkpointer}

# --- Main Agent Orchestrator ---
def execute_agent_call(user_query: str, agent_components: dict):
    """Runs the agent. Memory is handled by the RunnableWithMessageHistory wrapper."""
    try:
        config = {"configurable": {"thread_id": THREAD_ID}}
        agent_with_history = agent_components["agent_with_history"]
        
        # The input is now a simple dict matching the prompt placeholders
        result = agent_with_history.invoke(
            {"input": user_query},
            config=config
        )

        # The output from AgentExecutor is a dict with the 'output' key
        return result.get("output", "(Error: No 'output' key found in agent response.)")
    except Exception as e:
        print(f"Error during agent invocation: {e}\n{traceback.format_exc()}")
        return "(An error occurred during processing. Please try again.)"

# --- UI Section ---
# ... (The Streamlit UI code below this line remains the same as your last working version) ...

# ... (Input Handling, Sidebar, Chat Display, etc.) ...
def handle_new_query_submission(query_text: str):
    if not st.session_state.get('thinking_for_ui', False):
        st.session_state.active_question = query_text
        st.session_state.messages.append({"role": "user", "content": query_text})
        st.session_state.query_to_process = query_text
        st.session_state.thinking_for_ui = True
        st.rerun()

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
preview_questions = ["..."] 
for question in preview_questions:
    if st.sidebar.button(question, key=f"preview_{question}", use_container_width=True):
        handle_new_query_submission(question)

st.sidebar.markdown("---")
if st.sidebar.button("🧹 Reset chat session", use_container_width=True):
    checkpointer = agent_components.get("checkpointer")
    if checkpointer:
        # Pass dummy system message, it's not used by clear()
        history = SummarizedMessageHistory(checkpointer, THREAD_ID)
        history.clear()
    st.session_state.messages = []
    st.session_state.thinking_for_ui = False
    st.session_state.query_to_process = None
    st.session_state.thread_id = f"fifi_streamlit_session_{uuid.uuid4()}"
    st.session_state.active_question = None
    st.rerun()

for message in st.session_state.get("messages", []):
    with st.chat_message(message["role"]):
        st.markdown(message.get("content", ""))

if st.session_state.get('thinking_for_ui', False):
    with st.chat_message("assistant"):
        st.markdown("⌛ FiFi is thinking...")

st.markdown("""<div class="terms-footer">...</div>""", unsafe_allow_html=True)

user_prompt = st.chat_input("Ask me...", disabled=st.session_state.get('thinking_for_ui', False))
if user_prompt:
    handle_new_query_submission(user_prompt)

if st.session_state.get('query_to_process'):
    query_to_run = st.session_state.query_to_process
    assistant_reply = execute_agent_call(query_to_run, agent_components)
    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
    st.session_state.thinking_for_ui = False
    st.session_state.query_to_process = None
    st.rerun()
