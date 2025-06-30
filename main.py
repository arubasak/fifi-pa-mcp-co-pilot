# --- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
import streamlit as st
import base64
from pathlib import Path
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

# Use the Base64 string for the page_icon to avoid MediaFileStorageError
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

# --- FINAL: "Balanced" Memory Strategy Constants (Implements ConversationSummaryBufferMemory logic) ---
HISTORY_MESSAGE_THRESHOLD = 100       # Secondary trigger, token threshold will likely hit first.
HISTORY_TOKEN_THRESHOLD = 20000       # BALANCED: Trigger summarization when history exceeds this many tokens.
MESSAGES_TO_RETAIN_AFTER_SUMMARY = 6  # ESSENTIAL: Retain the last 6 messages verbatim for agent context.
MAX_INPUT_TOKENS = 60000              # Emergency brake limit for the entire payload.
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
def get_system_prompt_content_string(agent_components_for_prompt=None):
    if agent_components_for_prompt is None:
        agent_components_for_prompt = { 'pinecone_tool_name': "functions.get_context" }
    pinecone_tool = agent_components_for_prompt['pinecone_tool_name']
    prompt = f"""You are FiFi, the expert AI assistant for 1-2-Taste, specializing in food and beverage ingredients. Your role is to assist with product inquiries, industry trends, food science, and B2B support. Politely decline out-of-scope questions.

**Tool Selection Framework:**
1.  **`{pinecone_tool}` (Knowledge Base):** This is your **first choice** for all internal information. Use it for specific product details, ingredient recommendations, applications, and technical data. You MUST use `top_k=5` and `snippet_size=1024`.
2.  **`tavily_search_fallback` (Web Search):** Use this as a **fallback** if the knowledge base has no relevant results, or for broad topics like recent market news or general food science questions.
3.  **E-commerce Tools:** Use only for explicit requests about orders, accounts, or shipping.

**Response Rules (Strictly Enforced):**
*   **Citations are MANDATORY:** For knowledge base results, cite `productURL` or `source_url`. For web results, cite the source URL.
*   **Product Safety:** NEVER mention a product that lacks a verifiable URL. NEVER provide prices; direct users to the product page or sales contact.
*   **Failure:** If all tools fail, state that the information could not be found.

Based on the conversation history and these instructions, answer the user's last query."""
    return prompt

# --- Helper function to count tokens ---
def count_tokens(messages: list, model_encoding: str = TOKEN_MODEL_ENCODING) -> int:
    """Calculates the total number of tokens for a list of messages."""
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

# --- Core Memory Management Function ---
async def manage_history_with_summary_buffer_logic(memory: MemorySaver, config: dict, llm_for_summary: ChatOpenAI):
    """This function is called ONLY when history exceeds a token threshold. It summarizes old messages."""
    st.info("Conversation history is long. Summarizing older messages to save context...")
    print(f"@@@ MEMORY MGMT: Summarization triggered.")
    
    checkpoint = memory.get(config)
    history = checkpoint.get("messages", [])

    if len(history) <= MESSAGES_TO_RETAIN_AFTER_SUMMARY:
        print("@@@ MEMORY MGMT: Not enough messages to summarize while retaining the required buffer.")
        return

    messages_to_summarize = history[:-MESSAGES_TO_RETAIN_AFTER_SUMMARY]
    messages_to_keep = history[-MESSAGES_TO_RETAIN_AFTER_SUMMARY:]

    summarization_prompt_list = [
        SystemMessage(content="You are an expert at creating concise, third-person summaries of multi-turn agentic conversations. Extract all key entities, topics, questions, and critical information from tool outputs. Focus on the core facts and actions. **The final summary must be under 2000 tokens.**"),
        HumanMessage(content="Please summarize the following conversation:\n\n" + "\n".join([f"{type(m).__name__}: {m.content}" for m in messages_to_summarize]))
    ]
    try:
        summary_response = await llm_for_summary.ainvoke(summarization_prompt_list)
        summary_text = summary_response.content
        new_history = [SystemMessage(content=f"This is a summary of the preceding conversation: {summary_text}")] + messages_to_keep
        checkpoint["messages"] = new_history
        memory.put(config, checkpoint)
        print("@@@ MEMORY MGMT: History successfully summarized.")
    except Exception as e:
        st.error(f"Could not summarize history: {e}")
        print(f"Error during summarization: {e}\n{traceback.format_exc()}")

# --- Emergency Safety Net Function ---
def truncate_prompt_if_needed(messages: list, max_tokens: int) -> list:
    """This is an EMERGENCY brake, triggered only if the payload exceeds the absolute max token limit."""
    total_tokens = count_tokens(messages)
    if total_tokens <= max_tokens:
        return messages
    
    st.warning(f"CRITICAL: Request payload ({total_tokens} tokens) exceeds the absolute limit. This may be due to a summarization failure. Truncating history to prevent a crash. Context may be lost.")
    print(f"@@@ EMERGENCY SAFETY NET: Payload size {total_tokens} > {max_tokens}. Truncating.")
    
    system_message = messages[0]
    user_query = messages[-1]
    history = messages[1:-1]
    while count_tokens([system_message] + history + [user_query]) > max_tokens and history:
        history.pop(0)
    return [system_message] + history + [user_query]

# --- Async handler for agent initialization (Cached) ---
@st.cache_resource(ttl=3600)
def get_agent_components():
    """Initializes and caches the expensive agent components."""
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
    loop = get_or_create_eventloop()
    return loop.run_until_complete(run_async_initialization())

# --- ** CORRECTED AND DEBUGGED ** Main Agent Orchestrator ---
async def execute_agent_call_with_memory(user_query: str, agent_components: dict):
    """
    Runs the agent, managing history efficiently by checking the size of the
    ENTIRE potential payload before invocation. This prevents the bug where the
    final payload could exceed the threshold without triggering summarization.
    """
    try:
        config = {"configurable": {"thread_id": THREAD_ID}}
        memory_instance = agent_components["memory_instance"]
        main_system_prompt_content_str = agent_components["main_system_prompt_content_str"]

        # 1. Get the current history.
        current_checkpoint = memory_instance.get(config)
        history_messages = current_checkpoint.get("messages", []) if current_checkpoint else []

        # 2. Construct the FULL potential payload for the NEXT call.
        # This includes the system prompt, the current history, and the new user query.
        potential_payload = [SystemMessage(content=main_system_prompt_content_str)] + history_messages + [HumanMessage(content=user_query)]
        
        # 3. Check the token count of this entire potential payload.
        potential_payload_token_count = count_tokens(potential_payload)
        
        # This debug line is helpful for testing the logic.
        print(f"@@@ DEBUG: Potential payload token count is {potential_payload_token_count}. Threshold is {HISTORY_TOKEN_THRESHOLD}.")

        # 4. If the potential payload exceeds the threshold, trigger summarization NOW.
        if potential_payload_token_count > HISTORY_TOKEN_THRESHOLD:
            await manage_history_with_summary_buffer_logic(memory_instance, config, agent_components["llm_for_summary"])
            
            # 5. After summarizing, we MUST refresh the history to get the new, smaller list.
            current_checkpoint = memory_instance.get(config)
            history_messages = current_checkpoint.get("messages", [])

        # 6. Re-assemble the final payload using the (potentially summarized) history.
        event_messages = [SystemMessage(content=main_system_prompt_content_str)] + history_messages + [HumanMessage(content=user_query)]
        
        # 7. Apply the final emergency safety net.
        final_messages = truncate_prompt_if_needed(event_messages, MAX_INPUT_TOKENS)
        
        # 8. Invoke the agent.
        event = {"messages": final_messages}
        result = await agent_components["agent_executor"].ainvoke(event, config=config)

        # 9. Robustly parse the response.
        assistant_reply = ""
        if isinstance(result, dict) and "messages" in result and result["messages"]:
            for msg in reversed(result["messages"]):
                if isinstance(msg, AIMessage) and msg.content:
                    assistant_reply = msg.content
                    break
            else:
                assistant_reply = "(Error: No valid AI response content was found in the agent's output.)"
        else:
            assistant_reply = f"(Error: Unexpected response format from agent: {type(result)})"
        return assistant_reply

    except Exception as e:
        print(f"Error during agent invocation: {e}\n{traceback.format_exc()}")
        return f"(An error occurred during processing. Please try again.)"


# --- Input Handling Function for Streamlit ---
def handle_new_query_submission(query_text: str):
    """Manages session state for a new user query."""
    if not st.session_state.get('thinking_for_ui', False):
        st.session_state.active_question = query_text
        st.session_state.messages.append({"role": "user", "content": query_text})
        st.session_state.query_to_process = query_text
        st.session_state.thinking_for_ui = True
        st.rerun()

# --- Streamlit App UI and Main Execution Logic ---
st.markdown("""
<style>
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
    st.session_state.messages = []
    st.session_state.thinking_for_ui = False
    st.session_state.query_to_process = None
    st.session_state.thread_id = f"fifi_streamlit_session_{uuid.uuid4()}"
    st.session_state.active_question = None
    print(f"@@@ New chat session started. Thread ID: {st.session_state.thread_id}")
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
    assistant_reply = loop.run_until_complete(execute_agent_call_with_memory(query_to_run, agent_components))

    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
    st.session_state.thinking_for_ui = False
    st.session_state.query_to_process = None
    st.rerun()
