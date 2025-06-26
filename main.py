import streamlit as st
import datetime
import asyncio
import tiktoken
import os

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool
from tavily import TavilyClient
# NEW: Import BaseModel and Field from Pydantic
from pydantic import BaseModel, Field

# --- Constants for History Summarization ---
SUMMARIZE_THRESHOLD_TOKENS = 6000
MESSAGES_TO_KEEP_AFTER_SUMMARIZATION = 4
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
    THREAD_ID = "fifi_streamlit_session"

# --- NEW: Define a Pydantic model for the tool's input schema ---
class TavilySearchInput(BaseModel):
    query: str = Field(description="The search query to pass to the Tavily search engine.")

# --- UPDATED: Custom Tavily Fallback Tool with the explicit schema ---
@tool(args_schema=TavilySearchInput)
def tavily_search_fallback(query: str) -> str:
    """
    Search the web using Tavily when the knowledge base doesn't have sufficient information.
    Use this as a fallback when the primary knowledge base search returns insufficient results for topics like recent industry trends or general food science questions.
    """
    try:
        tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        # The 'query' argument is now accessed directly as it is passed by the decorator
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

# --- System Prompt Definition (Unchanged) ---
def get_system_prompt_content_string(agent_components_for_prompt=None):
    if agent_components_for_prompt is None:
        agent_components_for_prompt = {
            'pinecone_tool_name': "functions.get_context",
            'all_tool_details_for_prompt': { "functions.get_context": "...", "tavily_search_fallback": "..." }
        }

    pinecone_tool = agent_components_for_prompt['pinecone_tool_name']
    all_tool_details = agent_components_for_prompt['all_tool_details_for_prompt']
    
    prompt = f"""You are FiFi, an expert AI assistant for 1-2-Taste. Your **sole purpose** is to assist users with inquiries related to 1-2-Taste's products, the food and beverage ingredients industry, food science topics relevant to 1-2-Taste's offerings, B2B inquiries, recipe development support using 1-2-Taste ingredients, and specific e-commerce functions related to 1-2-Taste's WooCommerce platform.

**Core Mission:**
*   Provide accurate, **cited** information about 1-2-Taste's offerings using your product information capabilities.
*   Assist with relevant e-commerce tasks if explicitly requested by the user.
*   When your primary knowledge base doesn't have sufficient information, use the web search tool as a fallback to provide helpful information.
*   Politely decline to answer questions that are outside of your designated scope.

**Tool Usage Priority and Guidelines (Internal Instructions for You, the LLM):**

1.  **Primary Product & Industry Information Tool (Internally known as `{pinecone_tool}`):**
    *   For ANY query that could relate to 1-2-Taste product details, ingredients, flavors, etc., you **MUST ALWAYS PRIORITIZE** using this specialized tool. Its description is: "{all_tool_details.get(pinecone_tool, 'Retrieves relevant document snippets from the assistant knowledge base.')}"
    *   To manage token usage and control the amount of context returned, you MUST include the `top_k` and `snippet_size` parameters in your arguments. Use the following values:
        *   `top_k`: 5
        *   `snippet_size`: 1024
    *   For example, a correct tool call would look like: `get_context(query='some query about ingredients', top_k=5, snippet_size=1024)`

2.  **Web Search Fallback Tool (Internally, `tavily_search_fallback`):**
    *   You should **ONLY** use the web search tool if the primary knowledge base tool returns insufficient, irrelevant, or no useful information for a query that is still relevant to the food and beverage industry.

3.  **E-commerce and Order Management Tools (Internally, WooCommerce tools):**
    *   You should **ONLY** use these tools if the user's query EXPLICITLY mentions "WooCommerce", "orders", "customer accounts", or other clear e-commerce tasks.

**Response Guidelines & Output Format:**
*   **Strict Inclusion Policy:** You **MUST ONLY** include products in your answer that have a verifiable `productURL` or `source_url` in the tool's output. If a product appears in the tool's context but lacks a URL, you **MUST ignore it.**
*   **Mandatory Citations:** For every product you mention, you **MUST ALWAYS** cite the `productURL` or `source_url`.
*   **Web Source Citation:** When using information from the web search tool, clearly state that the information is from a web search and cite the source URLs provided by the tool.
*   **Pricing:** Do not provide product prices. Direct users to the product page, to contact sales, or to the quote request page for (QUOTE ONLY) products.
*   If both your knowledge base and web search fail, politely explain that you could not find the information.

Answer the user's last query based on these instructions and the conversation history."""
    return prompt

# --- All remaining functions and the Streamlit UI are unchanged ---

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

async def summarize_history_if_needed(
    memory_instance: MemorySaver, thread_config: dict, main_system_prompt_content_str: str,
    summarize_threshold_tokens: int, keep_last_n_interactions: int, llm_for_summary: ChatOpenAI
):
    checkpoint = memory_instance.get(thread_config)
    current_stored_messages = checkpoint.get("messages", []) if checkpoint else []
    cleaned_messages = [ m for m in current_stored_messages if not (isinstance(m, SystemMessage) and m.content == main_system_prompt_content_str)]
    conversational_messages_only = cleaned_messages
    current_token_count = count_tokens(conversational_messages_only)
    st.sidebar.markdown(f"**Conv. Tokens (w/ summaries):** `{current_token_count}` / `{summarize_threshold_tokens}`")
    if current_token_count > summarize_threshold_tokens:
        st.info(f"Summarization Triggered: History ({current_token_count} tokens) > threshold ({summarize_threshold_tokens}).")
        if len(conversational_messages_only) <= keep_last_n_interactions: return False
        messages_to_summarize = conversational_messages_only[:-keep_last_n_interactions]
        messages_to_keep_raw = conversational_messages_only[-keep_last_n_interactions:]
        if messages_to_summarize:
            summarization_prompt_messages = [
                SystemMessage(content="Please summarize the following conversation history concisely..."),
                HumanMessage(content="\n".join([f"{m.type.capitalize()}: {m.content}" for m in messages_to_summarize]))]
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

async def run_async_initialization():
    print("@@@ ASYNC run_async_initialization: Starting actual resource initialization...")
    client = MultiServerMCPClient({
        "pinecone": {"url": MCP_PINECONE_URL, "transport": "sse", "headers": {"Authorization": f"Bearer {MCP_PINECONE_API_KEY}"}},
        "pipedream": {"url": MCP_PIPEDREAM_URL, "transport": "sse"}})
    mcp_tools = await client.get_tools()
    all_tools = list(mcp_tools) + [tavily_search_fallback]
    memory = MemorySaver()
    pinecone_tool_name = "functions.get_context"
    all_tool_details = {tool.name: tool.description for tool in all_tools}
    system_prompt_content_value = get_system_prompt_content_string({
        'pinecone_tool_name': pinecone_tool_name, 'all_tool_details_for_prompt': all_tool_details})
    agent_executor = create_react_agent(llm, all_tools, checkpointer=memory)
    print("@@@ ASYNC run_async_initialization: Initialization complete.")
    return {
        "agent_executor": agent_executor, "memory_instance": memory, "llm_for_summary": llm,
        "main_system_prompt_content_str": system_prompt_content_value}

@st.cache_resource(ttl=3600)
def get_agent_components():
    print("@@@ get_agent_components: Populating cache by running the async initialization...")
    return asyncio.run(run_async_initialization())

async def execute_agent_call_with_memory(user_query: str, agent_components: dict):
    assistant_reply = ""
    try:
        config = {"configurable": {"thread_id": THREAD_ID}}
        main_system_prompt_content_str = agent_components["main_system_prompt_content_str"]
        await summarize_history_if_needed(
            agent_components["memory_instance"], config, main_system_prompt_content_str,
            SUMMARIZE_THRESHOLD_TOKENS, MESSAGES_TO_KEEP_AFTER_SUMMARIZATION, agent_components["llm_for_summary"])
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
        import traceback
        st.error(f"Error during agent invocation: {e}\n{traceback.format_exc()}")
        assistant_reply = f"(Error: {e})"
    if "messages" not in st.session_state:
        st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
    st.session_state.thinking_for_ui = False
    st.rerun()

def handle_new_query_submission(query_text: str):
    if not st.session_state.get('thinking_for_ui', False):
        st.session_state.messages.append({"role": "user", "content": query_text})
        st.session_state.query_to_process = query_text
        st.session_state.thinking_for_ui = True
        st.rerun()

st.title("FiFi Co-Pilot 🚀")
if SECRETS_ARE_MISSING:
    st.error("One or more secrets are missing. Please configure them in Streamlit secrets.")
    st.stop()

if "messages" not in st.session_state: st.session_state.messages = []
if 'thinking_for_ui' not in st.session_state: st.session_state.thinking_for_ui = False
if 'query_to_process' not in st.session_state: st.session_state.query_to_process = None
if 'components_loaded' not in st.session_state: st.session_state.components_loaded = False

try:
    agent_components = get_agent_components()
    st.session_state.components_loaded = True
    st.success("Agent Initialized Successfully!")
except Exception as e:
    st.error(f"Failed to initialize agent. Please refresh. Error: {e}")
    st.session_state.components_loaded = False
    st.stop()

st.sidebar.markdown("## Memory Debugger")
st.sidebar.markdown("---")
st.sidebar.markdown("## Quick Questions")
preview_questions = ["Help me with my recipe for a new juice drink", "Suggest me some strawberry flavours for beverage", "I need vanilla flavours for ice-cream", "What are the latest trends in plant-based proteins for 2025?"]
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
    print("@@@ Chat history cleared from UI and memory checkpoint.")
    st.rerun()

for message in st.session_state.get("messages", []):
    with st.chat_message(message["role"]):
        st.markdown(str(message.get("content", "")))

if st.session_state.get('thinking_for_ui', False):
    with st.chat_message("assistant"):
        st.markdown("⌛ FiFi is thinking...")

if st.session_state.get('query_to_process'):
    query_to_run = st.session_state.query_to_process
    st.session_state.query_to_process = None
    asyncio.run(execute_agent_call_with_memory(query_to_run, agent_components))

user_prompt = st.chat_input("Ask FiFi Co-Pilot...", key="main_chat_input", disabled=st.session_state.get('thinking_for_ui', False) or not st.session_state.get("components_loaded", False))
if user_prompt:
    handle_new_query_submission(user_prompt)
