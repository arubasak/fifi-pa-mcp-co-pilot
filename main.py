# --- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
import streamlit as st
import base64
from pathlib import Path
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

# Helper function to load and Base64-encode images for stateless deployment
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

# ✅✅✅ --- THIS IS THE MODIFIED SECTION --- ✅✅✅

# --- List of domains to exclude from general web searches ---
# This helps filter out competitors, marketplaces, and other noisy sources.
DEFAULT_EXCLUDED_DOMAINS = [
    "ingredientsnetwork.com",
    "csmingredients.com",
    "batafood.com",
    "nccingredients.com",
    "prinovaglobal.com",
    "ingrizo.com",
    "solina.com",
    "opply.com",
    "brusco.co.uk",
    "lehmanningredients.co.uk",
    "nccingredients.com",
    "i-ingredients.com",
    "fciltd.com",
    "lupafoods.com",
    "tradeingredients.com",
    "peterwhiting.co.uk",
    "globalgrains.co.uk",
    "tradeindia.com",
    "udaan.com",
    "ofbusiness.com",
    "indiamart.com",
    "symega.com",
    "meviveinternational.com",
    "amazon.com",
    "podfoods.co",
    "gocheetah.com",
    "foodmaven.com",
    "connect.kehe.com",
    "knowde.com",
    "ingredientsonline.com",
    "sourcegoodfood.com"
]

# --- Custom Tavily Fallback & General Search Tool ---
@tool
def tavily_search_fallback(query: str) -> str:
    """
    Search the web using Tavily. Use this for queries about broader, public-knowledge topics.
    This tool automatically excludes a predefined list of competitor and marketplace domains.
    """
    try:
        tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        
        # The exclude_domains parameter is now included in the search call
        response = tavily_client.search(
            query=query, 
            search_depth="advanced", 
            max_results=5, 
            include_answer=True, 
            include_raw_content=False,
            exclude_domains=DEFAULT_EXCLUDED_DOMAINS
        )
        
        if response.get('answer'):
            result = f"Web Search Results:\n\nSummary: {response['answer']}\n\nSources:\n"
        else:
            result = "Web Search Results:\n\nSources:\n"
            
        for i, source in enumerate(response.get('results', []), 1):
            result += f"{i}. {source['title']}\n   URL: {source['url']}\n   Content: {source['content']}\n\n"
            
        return result
    except Exception as e:
        return f"Error performing web search: {str(e)}"

# ✅✅✅ --- END OF MODIFIED SECTION --- ✅✅✅


# --- System Prompt Definition ---
def get_system_prompt_content_string(agent_components_for_prompt=None):
    if agent_components_for_prompt is None:
        agent_components_for_prompt = { 'pinecone_tool_name': "functions.get_context" }
    pinecone_tool = agent_components_for_prompt['pinecone_tool_name']

    # This prompt includes robust rules for anti-repetition and a mandatory disclaimer.
    prompt = f"""<instructions>
<system_role>
You are FiFi, a helpful and expert AI assistant for 1-2-Taste. Your primary goal is to be helpful within your designated scope. Your role is to assist with product and service inquiries, flavours, industry trends, food science, and B2B support. Politely decline out-of-scope questions. You must follow the tool protocol exactly as written to gather information.
</system_role>

<core_mission_and_scope>
Your mission is to provide information and support on 1-2-Taste products, the food and beverage industry, food science, and related B2B support. Use the conversation history to understand the user's intent, especially for follow-up questions.
</core_mission_and_scope>

<tool_protocol>
Your process for gathering information is a mandatory, sequential procedure. Do not deviate.

1.  **Step 1: Primary Tool Execution.**
    *   For any user query, your first and only initial action is to call the `{pinecone_tool}`.
    *   **Parameters:** Unless specified by a different rule (like the Anti-Repetition Rule), you MUST use `top_k=5` and `snippet_size=1024`.

2.  **Step 2: Mandatory Result Analysis.**
    *   After the primary tool returns a result, you MUST analyze it against the failure conditions below.

3.  **Step 3: Conditional Fallback Execution.**
    *   **If** the primary tool fails (because the result is empty, irrelevant, or lacks a `sourceURL`/`productURL`), then your next and only action **MUST** be to call the `tavily_search_fallback` tool with the original user query.
    *   Do not stop or apologize after a primary tool failure. The fallback call is a required part of the procedure.

4.  **Step 4: Final Answer Formulation.**
    *   Formulate your answer based on the data from the one successful tool call (either the primary or the fallback).
    *   **Disclaimer Rule:** If your answer is based on results from `tavily_search_fallback`, you **MUST** begin your response with this exact disclaimer, enclosed in a markdown quote block:
        > I could not find specific results within the 1-2-Taste EU product database. The following information is from a general web search and may point to external sites not affiliated with 1-2-Taste.
    *   If both tools fail, only then should you state that you could not find the information.
</tool_protocol>

<formatting_rules>
- **Citations are Mandatory:** Always cite the URL from the tool you used. When using tavily_search_fallback, you MUST include every source URL provided in the search results.
- **Source Format:** Present sources as a numbered list with both title and URL for each result.
- **Complete Attribution - CRITICAL RULE:** You MUST display ALL sources returned by the tool. If the tool provides 5 sources, your response MUST reference all 5 sources. If the tool provides 3 sources, show all 3. NEVER omit any sources from your response. This is a mandatory requirement.
- **Source Display Requirements:** 
  * List every single source with its title and URL
  * Use the exact format: "1. **[Title]**: [URL]"
  * Do not summarize or condense the source list
  * Include all sources even if they seem similar or redundant
- **Product Rules:** Do not mention products without a URL. NEVER provide product prices; direct users to the product page or ask to contact Sales Team at: sales-eu@12taste.com
- **Anti-Repetition Rule:**
    *   When a user asks for "more," "other," or "different" suggestions on a topic you have already discussed, you MUST alter your search strategy.
    *   **Action:** Your next call to `{pinecone_tool}` for this topic MUST use a larger `top_k` parameter, for example, `top_k=10`. This is to ensure you get a wider selection of potential results.
    *   **Filtering:** Before presenting the new results, you MUST review the conversation history and filter out any products or `sourceURL`s that you have already suggested.
    *   **Response:** If you have new, unique products after filtering, present them. If the larger search returns only products you have already mentioned, you MUST inform the user that you have no new suggestions on this topic. Do not list the old products again.
</formatting_rules>

<final_instruction>
Adhering to your core mission and the mandatory tool protocol, provide a helpful and context-aware response to the user's query.
</final_instruction>
</instructions>"""
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
    loop = get_or_create_eventloop()
    return loop.run_until_complete(run_async_initialization())

# --- FIX: MODIFIED ASYNC HANDLER ---
async def execute_agent_call_with_memory(user_query: str, agent_components: dict):
    """
    Runs the agent and returns the assistant's reply or an error string.
    """
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
        
        assistant_reply = ""
        if isinstance(result, dict) and "messages" in result and result["messages"]:
            for msg in reversed(result["messages"]):
                if isinstance(msg, AIMessage):
                    assistant_reply = msg.content
                    break
            if not assistant_reply:
                assistant_reply = f"(Error: No AI message found for query: '{user_query}')"
        else:
            assistant_reply = f"(Error: Unexpected response format: {type(result)})"
        
        return assistant_reply

    except Exception as e:
        print(f"Error during agent invocation: {e}\n{traceback.format_exc()}")
        return f"(An error occurred during processing. Please try again.)"

# --- Input Handling Function ---
def handle_new_query_submission(query_text: str):
    if not st.session_state.get('thinking_for_ui', False):
        st.session_state.active_question = query_text
        st.session_state.messages.append({"role": "user", "content": query_text})
        st.session_state.query_to_process = query_text
        st.session_state.thinking_for_ui = True
        st.rerun()

# --- Streamlit App UI ---
st.markdown("""
<style>
    .st-emotion-cache-1629p8f {
        border: 1px solid #ffffff;
        border-radius: 7px;
        bottom: 5px;
        position: fixed;
        width: 100%;
        max-width: 736px;
        left: 50%;
        transform: translateX(-50%);
        z-index: 101;
    }
    .st-emotion-cache-1629p8f:focus-within {
        border-color: #e6007e;
    }
    [data-testid="stCaptionContainer"] p {
        font-size: 1.3em !important;
    }
    [data-testid="stVerticalBlock"] {
        padding-bottom: 40px;
    }
    [data-testid="stChatMessage"] {
        margin-top: 0.1rem !important;
        margin-bottom: 0.1rem !important;
    }
    .stApp {
        overflow-y: auto !important;
    }
    .st-scroll-to-bottom {
        display: none !important;
    }
    .st-emotion-cache-1fplawd {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='font-size: 24px;'>FiFi, AI sourcing assistant</h1>", unsafe_allow_html=True)
st.caption("Hello, I am FiFi, your AI-powered assistant, designed to support you across the sourcing and product development journey. Find the right ingredients, explore recipe ideas, technical data, and more.")

if SECRETS_ARE_MISSING:
    st.error("Secrets missing. Please configure necessary environment variables.")
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

st.sidebar.markdown('By using this agent, you agree to our <a href="https://www.12taste.com/terms-conditions/" target="_blank">Terms of Service</a>.', unsafe_allow_html=True)

# Display chat messages with Base64 avatars
fifi_avatar_icon = f"data:image/png;base64,{FIFI_AVATAR_B64}" if FIFI_AVATAR_B64 else "🤖"
user_avatar_icon = f"data:image/png;base64,{USER_AVATAR_B64}" if USER_AVATAR_B64 else "🧑‍💻"
for message in st.session_state.get("messages", []):
    avatar_icon = fifi_avatar_icon if message["role"] == "assistant" else user_avatar_icon
    with st.chat_message(message["role"], avatar=avatar_icon):
        st.markdown(message.get("content", ""))

if st.session_state.get('thinking_for_ui', False):
    with st.chat_message("assistant", avatar=fifi_avatar_icon):
        st.markdown("⌛ FiFi is thinking...")

user_prompt = st.chat_input("Ask me for ingredients, recipes, or product development—in any language.", key="main_chat_input",
                            disabled=st.session_state.get('thinking_for_ui', False) or not st.session_state.get("components_loaded", False))
if user_prompt:
    st.session_state.active_question = None
    handle_new_query_submission(user_prompt)

# --- FIX: MODIFIED PROCESSING LOGIC ---
if st.session_state.get('query_to_process'):
    query_to_run = st.session_state.query_to_process
    
    loop = get_or_create_eventloop()
    assistant_reply = loop.run_until_complete(execute_agent_call_with_memory(query_to_run, agent_components))

    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
    st.session_state.thinking_for_ui = False
    st.session_state.query_to_process = None
    st.rerun()
