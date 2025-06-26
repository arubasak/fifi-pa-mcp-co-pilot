import streamlit as st
import datetime
import asyncio
import tiktoken
import os
import traceback

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool
from tavily import TavilyClient

# --- Constants for History Summarization ---
SUMMARIZE_THRESHOLD_TOKENS = 500
MESSAGES_TO_KEEP_AFTER_SUMMARIZATION = 12
TOKEN_MODEL_ENCODING = "cl100k_base"

# --- Load environment variables from secrets ---
# For Streamlit Cloud, set these in the secrets management.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MCP_PINECONE_URL = os.environ.get("MCP_PINECONE_URL")
MCP_PINECONE_API_KEY = os.environ.get("MCP_PINECONE_API_KEY")
MCP_PIPEDREAM_URL = os.environ.get("MCP_PIPEDREAM_URL")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

# A single check at the start to ensure all keys are present.
SECRETS_ARE_MISSING = not all([OPENAI_API_KEY, MCP_PINECONE_URL, MCP_PINECONE_API_KEY, MCP_PIPEDREAM_URL, TAVILY_API_KEY])

if not SECRETS_ARE_MISSING:
    llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY, temperature=0.2)
    THREAD_ID = "fifi_streamlit_session"

# --- Custom Tavily Fallback & General Search Tool ---
@tool
def tavily_search_fallback(query: str) -> str:
    """
    Search the web using Tavily. Use this as your first choice for queries about broader, public-knowledge topics like recent industry news, market trends, or general food science questions.
    """
    try:
        tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        response = tavily_client.search(
            query=query,
            search_depth="advanced",
            max_results=5,
            include_answer=True,
            include_raw_content=False
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

# --- System Prompt Definition (OPTIMIZED) ---
def get_system_prompt_content_string(agent_components_for_prompt=None):
    if agent_components_for_prompt is None:
        agent_components_for_prompt = {
            'pinecone_tool_name': "functions.get_context",
        }
    pinecone_tool = agent_components_for_prompt['pinecone_tool_name']
    
    # This prompt uses the "Intelligent Tool Selection Framework" for better efficiency.
    prompt = f"""You are FiFi, the expert AI assistant for 1-2-Taste.
Your role is strictly limited to inquiries about 1-2-Taste products, the food/beverage industry, relevant food science, B2B support, and specific e-commerce tasks. Politely decline all out-of-scope questions.

**Intelligent Tool Selection Framework:**

Your first step is to analyze the user's query to determine the best tool. Do not just follow a rigid order; select the tool that best fits the user's intent.

1.  **When to use `{pinecone_tool}` (Knowledge Base):**
    *   Use this tool as your **first choice** for queries about 1-2-Taste's internal information.
    *   **Primary Use Cases:** Specific product details (e.g., "What is the shelf life of your vanilla extract?"), product recommendations from the 1-2-Taste catalog, applications of specific ingredients, and information found in your internal documents.
    *   **Required Parameters:** You MUST use `top_k=5` and `snippet_size=1024`.

2.  **When to use `tavily_search_fallback` (Web Search):**
    *   Use this tool as your **first choice** for queries about broader, public-knowledge topics.
    *   **Primary Use Cases:** Recent industry news or market trends (e.g., "What are the latest developments in plant-based proteins?"), general food science questions (e.g., "How does the Maillard reaction work?"), and high-level questions about ingredient categories not specific to a 1-2-Taste product.

3.  **Using Web Search as a Fallback:**
    *   If you tried the `{pinecone_tool}` for a query that seemed product-specific but it returned no relevant results, you should then use the web search tool to find an answer.

4.  **E-commerce Tools:**
    *   Use these tools ONLY for explicit user requests about "WooCommerce", "orders", "customer accounts", or "shipping status".

**Response Formatting Rules (Strictly Enforced):**

*   **Citations are MANDATORY:**
    *   For knowledge base results, cite the `productURL`, `source_url`, or `sourceURL`.
    *   For web results, state that the information is from a web search and cite the source URL.
*   **Product Rules:**
    *   You MUST NOT mention any product from tool outputs that lacks a verifiable URL.
    *   You MUST NOT provide product prices. Direct the user to the product page or the correct contact (sales-eu@12taste.com or the quote request page).
*   **Failure:** If all tools fail to find a relevant answer, politely state that the information could not be found.

Based on the conversation history and these instructions, answer the user's last query."""
    return prompt

# --- Helper function to count tokens ---
def count_tokens(messages: list, model_encoding: str = TOKEN_MODEL_ENCODING) -> int:
    if not messages: return 0
    try: encoding = tiktoken.get_encoding(model_encoding)
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")
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
    st.sidebar.markdown(f"**Conv. Tokens:** `{current_token_count}` / `{summarize_threshold_tokens}`")

    if current_token_count > summarize_threshold_tokens:
        st.info(f"Summarization Triggered: History ({current_token_count}) > threshold ({summarize_threshold_tokens}).")
        if len(conversational_messages_only) <= keep_last_n_interactions: return False

        messages_to_summarize = conversational_messages_only[:-keep_last_n_interactions]
        messages_to_keep_raw = conversational_messages_only[-keep_last_n_interactions:]
        
        if messages_to_summarize:
            summarization_prompt_messages = [
                SystemMessage(content="Please summarize the following conversation history concisely..."),
                HumanMessage(content="\n".join([f"{m.type.capitalize()}: {m.content}" for m in messages_to_summarize]))
            ]
            try:
                summary_response = await llm_for_summary.ainvoke(summarization_prompt_messages)
                summary_content = summary_response.content
                new_messages_for_checkpoint = [SystemMessage(content=f"Previous conversation summary: {summary_content}")] + messages_to_keep_raw
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
    # This is a synchronous wrapper for the async initialization function.
    async def run_async_initialization():
        print("@@@ ASYNC run_async_initialization: Starting resource initialization...")
        # Reminder: If you get a TaskGroup error here, it's almost always a bad URL or API key for Pinecone/Pipedream.
        client = MultiServerMCPClient({
            "pinecone": {"url": MCP_PINECONE_URL, "transport": "sse", "headers": {"Authorization": f"Bearer {MCP_PINECONE_API_KEY}"}},
            "pipedream": {"url": MCP_PIPEDREAM_URL, "transport": "sse"}
        })
        mcp_tools = await client.get_tools()
        
        # Combine MCP tools with local Python tools
        all_tools = list(mcp_tools) + [tavily_search_fallback]
        
        memory = MemorySaver()
        
        pinecone_tool_name = "functions.get_context"
        
        system_prompt_content_value = get_system_prompt_content_string({
            'pinecone_tool_name': pinecone_tool_name,
        })
        
        agent_executor = create_react_agent(llm, all_tools, checkpointer=memory)
        print("@@@ ASYNC run_async_initialization: Initialization complete.")
        
        return {
            "agent_executor": agent_executor,
            "memory_instance": memory,
            "llm_for_summary": llm,
            "main_system_prompt_content_str": system_prompt_content_value
        }
    
    print("@@@ get_agent_components: Populating cache by running async initialization...")
    return asyncio.run(run_async_initialization())

# --- Async handler for user queries ---
async def execute_agent_call_with_memory(user_query: str, agent_components: dict):
    assistant_reply = ""
    try:
        config = {"configurable": {"thread_id": THREAD_ID}}
        main_system_prompt_content_str = agent_components["main_system_prompt_content_str"]

        await summarize_history_if_needed(
            agent_components["memory_instance"], config, main_system_prompt_content_str,
            SUMMARIZE_THRESHOLD_TOKENS, MESSAGES_TO_KEEP_AFTER_SUMMARIZATION, agent_components["llm_for_summary"]
        )

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
                assistant_reply = f"(Error: No AI message found in result for user query: '{user_query}')"
        else:
            assistant_reply = f"(Error: Unexpected agent response format: {type(result)} - {result})"

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
st.title("FiFi Co-Pilot 🚀 (Optimized Agent)")

if SECRETS_ARE_MISSING:
    st.error("One or more secrets are missing. Please configure OPENAI_API_KEY, MCP_PINECONE_URL, MCP_PINECONE_API_KEY, MCP_PIPEDREAM_URL, and TAVILY_API_KEY in Streamlit secrets.")
    st.stop()

# Initialize session state variables
if "messages" not in st.session_state: st.session_state.messages = []
if 'thinking_for_ui' not in st.session_state: st.session_state.thinking_for_ui = False
if 'query_to_process' not in st.session_state: st.session_state.query_to_process = None
if 'components_loaded' not in st.session_state: st.session_state.components_loaded = False

# Load agent components
try:
    agent_components = get_agent_components()
    st.session_state.components_loaded = True
except Exception as e:
    st.error(f"Failed to initialize agent. Please refresh. Error: {e}")
    st.session_state.components_loaded = False
    st.stop()

# --- UI Rendering ---
st.sidebar.markdown("## Memory Debugger")
st.sidebar.markdown("---")
st.sidebar.markdown("## Quick Questions")
preview_questions = [
    "Suggest me some strawberry flavours for beverage",
    "I need vanilla flavours for ice-cream",
    "What are the latest trends in plant-based proteins for 2025?"
]
for question in preview_questions:
    if st.sidebar.button(question, key=f"preview_{question}", use_container_width=True):
        handle_new_query_submission(question)

st.sidebar.markdown("---")
if st.sidebar.button("🧹 Clear Chat History", use_container_width=True):
    memory = agent_components.get("memory_instance")
    if memory:
        memory.put({"configurable": {"thread_id": THREAD_ID}}, {"messages": []})
    st.session_state.messages = []
    st.session_state.thinking_for_ui = False
    st.session_state.query_to_process = None
    print("@@@ Chat history cleared.")
    st.rerun()

# Display chat messages
for message in st.session_state.get("messages", []):
    with st.chat_message(message["role"]):
        st.markdown(str(message.get("content", "")))

if st.session_state.get('thinking_for_ui', False):
    with st.chat_message("assistant"):
        st.markdown("⌛ FiFi is thinking...")

# Process new queries
if st.session_state.get('query_to_process'):
    query_to_run = st.session_state.query_to_process
    st.session_state.query_to_process = None
    asyncio.run(execute_agent_call_with_memory(query_to_run, agent_components))

# Chat input
user_prompt = st.chat_input("Ask FiFi Co-Pilot...", key="main_chat_input",
                            disabled=st.session_state.get('thinking_for_ui', False) or not st.session_state.get("components_loaded", False))
if user_prompt:
    handle_new_query_submission(user_prompt)
