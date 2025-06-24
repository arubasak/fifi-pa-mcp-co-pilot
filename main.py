import streamlit as st
from pinecone import Pinecone
from pinecone_plugins.assistant.models.chat import Message
from pinecone_plugins.assistant.control.core.client.exceptions import PineconeApiException
import traceback

# --- Page Configuration ---
st.set_page_config(
    page_title="Pinecone Assistant - Direct Test",
    layout="centered"
)

st.title("Pinecone Assistant Direct Client Test")
st.markdown("""
This app performs a direct query to your Pinecone Assistant, bypassing LangGraph entirely. 
Use it to verify three things:
1.  Are your Streamlit secrets (`PINECONE_API_KEY`, `PINECONE_REGION`, `PINECONE_ASSISTANT_NAME`) correct?
2.  Can the app successfully connect to your specific Pinecone Assistant?
3.  Does your assistant return a valid, knowledge-based answer for a specific question?
""")

# --- Load Secrets ---
try:
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
    # Use .get() for optional secrets to provide a default value safely
    PINECONE_ENVIRONMENT = st.secrets.get("PINECONE_REGION", "us")
    ASSISTANT_NAME = st.secrets.get("PINECONE_ASSISTANT_NAME", "fifiv1")
    st.success(f"Secrets loaded for assistant '{ASSISTANT_NAME}' in region '{PINECONE_ENVIRONMENT}'.")
except KeyError as e:
    st.error(f"Missing critical secret: {e}. Please ensure this secret is configured in your Streamlit Cloud app settings or local secrets.toml.")
    st.stop()

# --- User Input ---
st.markdown("---")
user_question = st.text_input(
    "Enter the question to ask your assistant:",
    "is Allulose approved in the EU?"
)

# --- Test Execution ---
if st.button("Run Direct Pinecone Query", type="primary"):
    if not user_question:
        st.warning("Please enter a question to test.")
        st.stop()

    with st.spinner(f"Connecting to assistant '{ASSISTANT_NAME}' and sending query..."):
        try:
            # 1. Initialize Pinecone Client and Assistant
            st.write("Initializing Pinecone client...")
            pc = Pinecone(api_key=PINECONE_API_KEY)
            
            st.write(f"Connecting to assistant: '{ASSISTANT_NAME}'...")
            assistant = pc.assistant.Assistant(assistant_name=ASSISTANT_NAME)

            # 2. Prepare the message
            test_messages = [Message(role="user", content=user_question)]
            st.write(f"Sending message: `user: {user_question}`")

            # 3. Query the assistant
            response = assistant.chat(messages=test_messages, model="gpt-4o")

            # 4. Display results
            st.markdown("---")
            st.subheader("✅ Query Successful!")
            
            st.markdown("### Raw Response from Pinecone SDK")
            st.write("This is the complete object returned by `assistant.chat()`.")
            st.write(response)

            st.markdown("### Extracted Content")
            st.write("This is the final content extracted from `response.message.content`.")
            
            # Robustly get content, as in your script
            content = getattr(getattr(response, "message", None), "content", None)

            if content:
                st.success(content)
            else:
                st.error("Extraction Failed: The 'content' field is empty or missing in the response.")

        except PineconeApiException as e:
            st.error(f"Pinecone API Error: {e}", icon="🚨")
            st.write("This usually means the `ASSISTANT_NAME` is incorrect, or there's an issue with your API key or environment.")
        except Exception as e:
            st.error(f"An Unexpected Error Occurred: {e}", icon="🔥")
            st.code(traceback.format_exc())
