import streamlit as st
import sys
import os
import re
import uuid
import pandas as pd

# Add project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache

set_llm_cache(SQLiteCache(database_path="data/llm_cache/langchain.db"))

# Set page configuration
st.set_page_config(
    page_title="Agentic Table Playground",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded"
)

from backend.data_processing.table_operation.table_operation_workflow import DataFrameWorkflow
from frontend.ui_components import apply_common_styles, display_project_info

# Apply custom styles
apply_common_styles()

# Initialize session state
if "workflow" not in st.session_state:
    st.session_state.workflow = DataFrameWorkflow()
if "files_uploaded" not in st.session_state:
    st.session_state.files_uploaded = False
if "operation_result" not in st.session_state:
    st.session_state.operation_result = None
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "operation_steps" not in st.session_state:
    st.session_state.operation_steps = []


def main():
    """Main application entry point."""
    st.title("🧮 Agentic Table Playground")
    
    # Add description
    st.markdown("""
    ### Welcome to the AI Agent-Powered Data Processing Playground
    
    **Agentic Table Playground** leverages AI agents to enable complex data processing through natural language interaction.
    Simply upload your CSV or Excel files and describe what you want to do in plain English!
    
    **Key Features:**
    - 🔗 **Table Merging**: Join related tables like Excel VLOOKUP
    - 🔄 **Data Reshaping**: Convert between wide and long formats  
    - ⚖️ **Data Comparison**: Find differences between datasets
    - 📊 **Vertical Stacking**: Combine similar tables
    - 🧹 **Data Deduplication**: Remove duplicate records
    - 🤖 **Multi-step Processing**: Handle complex workflows with AI agents
    """)

    # Project information
    display_project_info()

    st.markdown("---")

    # Privacy notice
    st.warning(
        "**Privacy Notice**: The system only accesses file names and column headers—it does not read or store your actual table data. "
        "Table previews are temporarily loaded in memory and cleared when you refresh the page."
    )

    # Core functionality
    handle_file_upload()
    if st.session_state.files_uploaded:
        # Only show Dataset Operations section if user has started a conversation
        if st.session_state.conversation_history:
            process_user_query()
        else:
            # Show only the input box without the Dataset Operations section
            user_query = st.chat_input("Enter your dataset operation requirements:")
            if user_query:
                # Add the user query to conversation history and then show the full interface
                st.session_state.conversation_history.append({"role": "user", "content": user_query})
                st.rerun()

        if st.session_state.get("operation_result"):
            display_operation_result()
            display_feedback()


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean column names by removing special characters and converting to lowercase.

    Supports both English and Chinese characters in column names.

    Args:
        df: Input DataFrame with potentially messy column names.

    Returns:
        DataFrame with cleaned column names.
    """
    df = df.copy()
    df.columns = df.columns.str.replace(r"[^\w\u4e00-\u9fff]+", "_", regex=True)
    df.columns = df.columns.str.lower()
    return df


def handle_file_upload():
    """
    Handle file upload logic for CSV and Excel files.

    Processes uploaded files, cleans column names, and loads them into the workflow.
    """
    st.markdown("## Data Upload")
    with st.container(border=True):
        uploaded_files = st.file_uploader(
            "Select CSV or Excel files (multiple files supported)",
            type=["csv", "xlsx", "xls"],
            accept_multiple_files=True,
            help="Upload CSV or Excel files. Multiple files can be uploaded for operations like merging or comparison."
        )

        if uploaded_files:
            try:
                dataframes = {}
                for uploaded_file in uploaded_files:
                    file_name = uploaded_file.name
                    base_name = os.path.splitext(file_name)[0]

                    # Clean filename for use as variable name (supports Chinese characters)
                    clean_name = re.sub(r'[^\w\u4e00-\u9fff]+', '_', base_name).lower()

                    if file_name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)

                    # Clean column names to ensure consistency
                    df = clean_column_names(df)
                    dataframes[clean_name] = df

                # Load processed dataframes into workflow
                st.session_state.workflow.load_dataframes(dataframes)
                st.session_state.files_uploaded = True

                st.success(f"Successfully uploaded {len(uploaded_files)} file(s)!")

            except Exception as e:
                st.error(f"Error uploading files: {str(e)}")
                st.session_state.files_uploaded = False
        else:
            st.session_state.files_uploaded = False

        if st.session_state.files_uploaded:
            st.markdown("---")
            st.markdown("#### Uploaded Dataset Preview")
            display_loaded_dataframes()


def display_loaded_dataframes():
    """
    Display preview of loaded original datasets using tabs.

    Shows dataset information including shape and preview data in a tabbed interface
    for multiple datasets or a single view for one dataset.
    """
    original_dataframes = st.session_state.workflow.get_original_dataframe_info()

    if not original_dataframes:
        st.info("No datasets uploaded yet. Please upload data files first.")
        return

    df_names = list(original_dataframes.keys())
    if len(df_names) == 1:
        df_name = df_names[0]
        df_info = original_dataframes[df_name]
        st.markdown(f"**{df_name}** - {df_info['shape'][0]} rows × {df_info['shape'][1]} columns")
        st.dataframe(df_info["preview"], use_container_width=True)
    else:
        tabs = st.tabs([f"{name} ({info['shape'][0]}×{info['shape'][1]})" for name, info in original_dataframes.items()])
        for tab, (df_name, df_info) in zip(tabs, original_dataframes.items()):
            with tab:
                st.dataframe(df_info["preview"], use_container_width=True)

    st.caption(
        "Note: To ensure data processing accuracy, column names containing special characters or spaces will be automatically cleaned."
    )


def process_user_query():
    """
    Process user queries and display results in a chat interface.

    Handles conversation history, user input, and AI responses for dataset operations.
    """
    st.markdown("## Dataset Operations")

    chat_container = st.container(border=True)
    input_placeholder = st.empty()

    display_conversation_history(chat_container)

    # Check if there's an unprocessed user message (first message after page reload)
    if (st.session_state.conversation_history and
        len(st.session_state.conversation_history) == 1 and
        st.session_state.conversation_history[-1]["role"] == "user"):
        # Process the first message automatically
        user_query = st.session_state.conversation_history[-1]["content"]
        process_and_display_response(chat_container, user_query)

    user_query = input_placeholder.chat_input("Enter your dataset operation requirements:")

    if user_query:
        display_user_input(chat_container, user_query)
        process_and_display_response(chat_container, user_query)


def display_conversation_history(container):
    """Display conversation history in the chat interface."""
    with container:
        for message in st.session_state.conversation_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])


def display_user_input(container, user_query):
    """Display user input in chat interface and save to conversation history."""
    with container:
        with st.chat_message("user"):
            st.markdown(user_query)
    st.session_state.conversation_history.append(
        {"role": "user", "content": user_query}
    )


def process_and_display_response(container, user_query):
    """
    Process user query through AI workflow and display the response.

    Args:
        container: Streamlit container for displaying the response.
        user_query: User's natural language query for data operations.
    """
    thinking_placeholder = st.empty()

    with thinking_placeholder:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.workflow.process_query(user_query)

    thinking_placeholder.empty()

    display_assistant_response(container, result)

    # Save trace_id for feedback tracking
    if "trace_id" in result:
        st.session_state.current_trace_id = result["trace_id"]

    # Store operation result only when operations are executed
    if result["next_step"] == "execute_operation":
        st.session_state.operation_result = result
    else:
        st.session_state.operation_result = None


def display_assistant_response(container, result):
    """
    Display AI assistant response and save to conversation history.

    Handles different response types: need_more_info, execute_operation, out_of_scope.

    Args:
        container: Streamlit container for displaying the response.
        result: AI assistant response containing next_step and message/operation details.
    """
    with container:
        with st.chat_message("assistant"):
            if result["next_step"] == "need_more_info":
                message = result.get("message", "More information is needed to process your request.")
                st.markdown(message)
                st.session_state.conversation_history.append(
                    {"role": "assistant", "content": message}
                )
            elif result["next_step"] == "execute_operation":
                message = "Operation executed successfully! Here are the steps performed:\n"
                st.markdown(message)
                st.session_state.operation_steps = result.get("operation", [])
                for i, step in enumerate(st.session_state.operation_steps, 1):
                    st.markdown(f"Step {i}: {step['tool_name']}")
                full_message = (
                    message
                    + "\n"
                    + "\n".join(
                        [
                            f"Step {i}: {step['tool_name']}"
                            for i, step in enumerate(
                                st.session_state.operation_steps, 1
                            )
                        ]
                    )
                )
                st.session_state.conversation_history.append(
                    {"role": "assistant", "content": full_message}
                )
                st.session_state.operation_result = result
            elif result["next_step"] == "out_of_scope":
                message = result.get("message", "Sorry, your request is outside my processing scope.")
                st.markdown(message)
                st.session_state.conversation_history.append(
                    {"role": "assistant", "content": message}
                )


def display_operation_result():
    """
    Display the results of executed data operations.

    Shows generated DataFrames with download options for each operation step.
    """
    if st.session_state.operation_result:
        st.markdown("---")
        st.markdown("## Operation Results")

        with st.container(border=True):
            for i, step in enumerate(st.session_state.operation_steps, 1):
                output_df_names = step["output_df_names"]
                for df_name in output_df_names:
                    if df_name in st.session_state.workflow.dataframes:
                        df = st.session_state.workflow.dataframes[df_name]
                        st.markdown(f"#### {df_name}")
                        st.caption(f"*Generated by Step {i}: {step['tool_name']}*")
                        st.dataframe(df)
                        provide_csv_download(df, df_name)
                st.markdown("---")


def display_feedback():
    """
    Display feedback interface and handle user feedback collection.

    Allows users to rate the operation results and sends feedback to monitoring system.
    """
    if "current_trace_id" in st.session_state:
        st.markdown("---")
        st.markdown("##### Did this operation meet your requirements?")

        # Initialize feedback status to prevent duplicate submissions
        if "feedback_given" not in st.session_state:
            st.session_state.feedback_given = False

        col1, col2, col3 = st.columns([1, 1, 3])

        with col1:
            yes_button = st.button(
                "👍 Yes",
                key="feedback_yes",
                use_container_width=True,
                disabled=st.session_state.feedback_given,
            )
            if yes_button and not st.session_state.feedback_given:
                st.session_state.workflow.record_feedback(
                    trace_id=st.session_state.current_trace_id, is_useful=True
                )
                st.session_state.feedback_given = True

        with col2:
            no_button = st.button(
                "👎 No",
                key="feedback_no",
                use_container_width=True,
                disabled=st.session_state.feedback_given,
            )
            if no_button and not st.session_state.feedback_given:
                st.session_state.workflow.record_feedback(
                    trace_id=st.session_state.current_trace_id, is_useful=False
                )
                st.session_state.feedback_given = True

        with col3:
            if st.session_state.feedback_given:
                st.success("Thank you for your feedback!")

        if st.session_state.feedback_given:
            del st.session_state.current_trace_id


def provide_csv_download(df: pd.DataFrame, df_name: str):
    """
    Provide CSV download option for individual DataFrame.

    Args:
        df: DataFrame to be downloaded.
        df_name: Name for the downloaded file.
    """
    csv = df.to_csv(index=False)
    st.download_button(
        label=f"Download {df_name} (CSV)",
        data=csv,
        file_name=f"{df_name}.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
