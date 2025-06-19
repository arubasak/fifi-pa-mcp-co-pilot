import streamlit as st
import datetime
import asyncio
import tiktoken
from functools import partial

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import Tool

# --- Imports for the Pinecone Assistant Tool ---
from pinecone import Pinecone
from pinecone_plugins.assistant.models.chat import Message
from pinecone_plugins.assistant.control.core.client.exceptions import PineconeApiException

# --- Constants for History Pruning ---
MAX_HISTORY_TOKENS = 90000
MESSAGES_TO_KEEP_AFTER_PRUNING = 6
TOKEN_MODEL_ENCODING = "cl100k_base"

# --- Load environment variables from secrets ---
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
MCP_PIPEDREAM_URL = st.secrets.get("MCP_PIPEDREAM_URL")
PINECONE_PLUGIN_API_KEY = st.secrets.get("PINECONE_API_KEY")
PINECONE_ASSISTANT_NAME = st.secrets.get("PINECONE_ASSISTANT_NAME", "fifi")

if not all([OPENAI_API_KEY, MCP_PIPEDREAM_URL, PINECONE_PLUGIN_API_KEY]):
    st.error("One or more critical secrets are missing (OpenAI, Pipedream, Pinecone Plugin).")
    st.stop()

# --- LLM for the LangGraph Agent (Using gpt-4o for best reasoning) ---
llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY, temperature=0.2)
THREAD_ID = "fifi_streamlit_session"

# --- Helper Functions (Unchanged) ---
def count_tokens(messages: list, model_encoding: str = TOKEN_MODEL_ENCODING) -> int:
    # ... (function is unchanged) ...
    if not messages: return 0
    try: encoding = tiktoken.get_encoding(model_encoding)
    except Exception: encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = 0
    for message in messages:
        num_tokens += 4
        for key, value in message.items():
            if value is not None:
                try: num_tokens += len(encoding.encode(str(value)))
                except TypeError: pass
    num_tokens += 2
    return num_tokens

def prune_history_if_needed(memory_instance: MemorySaver, thread_config: dict, current_system_prompt_content: str, max_tokens: int, keep_last_n_interactions: int):
    # ... (function is unchanged) ...
    checkpoint_value = memory_instance.get(thread_config)
    if not checkpoint_value or "messages" not in checkpoint_value or not isinstance(checkpoint_value.get("messages"), list):
        return False
    current_messages_in_history = checkpoint_value["messages"]
    if not current_messages_in_history: return False
    current_token_count = count_tokens(current_messages_in_history)
    if current_token_count > max_tokens:
        print(f"INFO: History token count ({current_token_count}) > max ({max_tokens}). Pruning...")
        user_assistant_messages = [m for m in current_messages_in_history if m.get("role") != "system"]
        pruned_user_assistant_messages = user_assistant_messages[-keep_last_n_interactions:]
        new_history_messages = [{"role": "system", "content": current_system_prompt_content}]
        new_history_messages.extend(pruned_user_assistant_messages)
        memory_instance.put(thread_config, {"messages": new_history_messages})
        print(f"INFO: History pruned. New token count: {count_tokens(new_history_messages)}.")
        return True
    return False

# --- Custom Tool Definition with Allulose Guardrail ---
def _query_pinecone_assistant_with_client(query: str, client) -> str:
    try:
        if not client:
            return "Error: Pinecone Assistant client was not provided to the tool."

        # Send the clean query directly for best semantic search results
        sdk_message = Message(role="user", content=query)
        response_from_sdk = client.chat(messages=[sdk_message], model="gpt-4o") 
        
        response_text = ""
        if hasattr(response_from_sdk, 'message') and hasattr(response_from_sdk.message, 'content'):
            response_text = response_from_sdk.message.content or ""
        else:
            return "(Could not find content in the assistant's response.)"

        # --- KEY CHANGE: ALLULOSE GUARDRAIL ---
        # Check if the query is about Allulose and Europe/EFSA/approval
        query_lower = query.lower()
        if 'allulose' in query_lower and ('europe' in query_lower or 'efsa' in query_lower or 'approve' in query_lower):
            print("INFO: Allulose regulatory guardrail triggered.")
            # Start with the definitive, correct answer.
            corrected_response = "Allulose is not yet approved for use as a novel food in the European Union (EU) and the UK. Its regulatory status is still pending with authorities like EFSA."
            
            # If the original response contained useful product info, append it.
            if "allsweet" in response_text.lower():
                # We add a transition phrase to make the combined response logical.
                corrected_response += "\n\nWhile it awaits full approval, here is some information on available Allulose products for other regions or for R&D purposes:\n\n" + response_text
            
            return corrected_response
        # --- END OF GUARDRAIL ---

        # If the guardrail is not triggered, return the original response.
        return response_text

    except Exception as e:
        print(f"ERROR querying Pinecone Assistant tool: {e}")
        return f"An error occurred while trying to get product information: {str(e)}"

# --- Agent Initialization ---
@st.cache_resource(ttl=3600)
def get_agent_components():
    print("@@@ get_agent_components: Populating cache...")
    
    try:
        pc = Pinecone(api_key=PINECONE_PLUGIN_API_KEY)
        pinecone_assistant_client = pc.assistant.Assistant(assistant_name=PINECONE_ASSISTANT_NAME)
    except Exception as e:
        print(f"@@@ FATAL ERROR initializing Pinecone Assistant client: {e}")
        raise e 

    async def get_mcp_tools():
        mcp_client = MultiServerMCPClient({"pipedream": {"url": MCP_PIPEDREAM_URL, "transport": "sse"}})
        return await mcp_client.get_tools()
    
    woocommerce_tools = asyncio.run(get_mcp_tools())

    bound_query_func = partial(_query_pinecone_assistant_with_client, client=pinecone_assistant_client)

    pinecone_assistant_tool = Tool(
        name="get_12taste_product_context",
        func=bound_query_func,
        description="Use this tool for all questions about 1-2-Taste products, services, ingredients, flavors, recipes, applications, regulatory status, or any topic related to the 1-2-Taste catalog or food and beverage industry."
    )

    all_tools = [pinecone_assistant_tool] + woocommerce_tools
    memory = MemorySaver()
    agent_executor = create_react_agent(llm, all_tools, checkpointer=memory)
    all_tool_details = {tool.name: tool.description for tool in all_tools}
    
    print("@@@ get_agent_components: Initialization complete.")
    return {
        "agent_executor": agent_executor,
        "memory_instance": memory,
        "pinecone_tool_name": pinecone_assistant_tool.name,
        "all_tool_details_for_prompt": all_tool_details
    }

# --- Simplified System Prompt Definition ---
def get_system_prompt(agent_components):
    pinecone_tool = agent_components['pinecone_tool_name']
    prompt = f"""You are FiFi, a specialized AI assistant for 1-2-Taste.

**Primary Directives:**
1.  **Tool Prioritization:**
    *   For any query about 1-2-Taste products, services, ingredients, recipes, or industry topics, you **MUST** use the `{pinecone_tool}` tool.
    *   **This includes creative requests like "help me with a recipe." In such cases, use the tool to find relevant ingredients and then use that information to provide recipe suggestions.**
    *   For e-commerce tasks like orders or customer accounts, use the appropriate WooCommerce tool based on its description.
    *   For any other topic, you **MUST** politely decline, stating you specialize in 1-2-Taste topics.
2.  **User-Facing Persona:**
    *   When asked about your capabilities, describe your functions simply (e.g., "I can answer questions about 1-2-Taste products and ingredients."). **NEVER reveal internal tool names.**
    *   **Do not state product prices.** If asked, direct users to the product page or a sales contact.
    *   **Cite your sources.** When the `{pinecone_tool}` tool provides a source URL, you must include it in your response. If no URL is available from the tool, state that the info is from the 1-2-Taste catalog.

Answer the user's latest query based on these core directives and the conversation history.
"""
    return prompt

# --- Async handler for user queries ---
async def execute_agent_call_with_memory(user_query: str, agent_components: dict):
    # ... (function is unchanged) ...
    assistant_reply = ""
    try:
        agent_executor = agent_components["agent_executor"]
        memory_instance = agent_components["memory_instance"]
        
        config = {"configurable": {"thread_id": THREAD_ID}}
        system_prompt_content = get_system_prompt(agent_components)

        prune_history_if_needed(
            memory_instance, config, system_prompt_content,
            MAX_HISTORY_TOKENS, MESSAGES_TO_KEEP_AFTER_PRUNING
        )

        current_turn_messages = [
            {"role": "system", "content": system_prompt_content},
            {"role": "user", "content": user_query}
        ]
        event = {"messages": current_turn_messages}
        result = await agent_executor.ainvoke(event, config=config)

        if isinstance(result, dict) and "messages" in result and result["messages"]:
            assistant_reply = result["messages"][-1].content
        else:
            assistant_reply = f"(Error: Unexpected agent response format: {type(result)} - {result})"
            st.error(f"Unexpected agent response: {result}")

    except Exception as e:
        import traceback
        st.error(f"Error during agent invocation: {e}\n{traceback.format_exc()}")
        assistant_reply = f"(Error: {e})"

    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
    st.session_state.thinking_for_ui = False
    st.rerun()

# --- Input Handling Function ---
def handle_new_query_submission(query_text: str):
    # ... (function is unchanged) ...
    if not st.session_state.get('thinking_for_ui', False):
        st.session_state.messages.append({"role": "user", "content": query_text})
        st.session_state.query_to_process = query_text
        st.session_state.thinking_for_ui = True
        st.rerun()

# --- Streamlit App Starts Here ---
st.title("FiFi Co-Pilot 🚀 (LangGraph Hybrid Agent)")

try:
    agent_components = get_agent_components()
    st.session_state.components_loaded = True 
except Exception as e:
    st.error(f"Failed to initialize agent components. The app cannot continue. Please refresh. Error: {e}")
    print(f"@@@ CRITICAL FAILURE during get_agent_components(): {e}")
    st.session_state.components_loaded = False
    st.stop() 

# --- UI Rendering ---
st.sidebar.markdown("## Quick Questions")
preview_questions = ["Help me with my recipe for a new juice drink", "Suggest me some strawberry flavours for beverage", "I need vanilla flavours for ice-cream", "is allulose approved in Europe?"]
for question in preview_questions:
    if st.sidebar.button(question, key=f"preview_{question}", use_container_width=True):
        handle_new_query_submission(question)

st.sidebar.markdown("---")
if st.sidebar.button("🧹 Clear Chat History", use_container_width=True):
    # ... (function is unchanged) ...
    st.session_state.messages = []
    st.session_state.thinking_for_ui = False
    st.session_state.query_to_process = None
    
    get_agent_components.clear() 
    if "components_loaded" in st.session_state:
        del st.session_state["components_loaded"]
    
    print("@@@ Chat history cleared, cache cleared.")
    st.rerun()

if st.session_state.messages:
    # ... (function is unchanged) ...
    chat_export_data_txt = "\n\n".join([f"{msg.get('role', 'Unknown').capitalize()}: {msg.get('content', '')}" for msg in st.session_state.messages])
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    st.sidebar.download_button(
        label="📥 Download Chat (TXT)", data=chat_export_data_txt,
        file_name=f"fifi_chat_{current_time}.txt", mime="text/plain", use_container_width=True
    )
st.sidebar.markdown("---")
st.sidebar.info("💡 FiFi uses a hybrid agent for product info and e-commerce tasks!")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(str(message.get("content", "")))

if st.session_state.get('thinking_for_ui', False):
    with st.chat_message("assistant"):
        st.markdown("⌛ FiFi is thinking...")

if st.session_state.get('thinking_for_ui', False) and st.session_state.get('query_to_process') is not None:
    if st.session_state.get("components_loaded"):
        query_to_run = st.session_state.query_to_process
        st.session_state.query_to_process = None
        asyncio.run(execute_agent_call_with_memory(query_to_run, agent_components))
    else:
        st.error("Agent is not ready. Please refresh the page.")
        st.session_state.thinking_for_ui = False
        st.session_state.query_to_process = None
        st.rerun()

user_prompt = st.chat_input("Ask FiFi Co-Pilot...", key="main_chat_input", disabled=not st.session_state.get("components_loaded", False))
if user_prompt:
    handle_new_query_submission(user_prompt)
