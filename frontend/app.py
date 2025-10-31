import streamlit as st
import sys
import os
import re
import uuid
import pandas as pd

# Add project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Load environment variables before importing other modules
from utils.env_loader import load_env
load_env()

from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache

# Ensure cache directory exists
cache_dir = os.path.join(project_root, "data", "llm_cache")
os.makedirs(cache_dir, exist_ok=True)

set_llm_cache(SQLiteCache(database_path=os.path.join(cache_dir, "langchain.db")))

# Set page configuration
st.set_page_config(
    page_title="Agentic Table Playground",
    page_icon="🧮",
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
    ### 智能表格处理演示平台

    基于 AI Function Calling 技术实现的数据处理演示。上传 CSV/Excel 文件，
    用自然语言描述您的数据处理需求即可完成操作。
    """)

    # System Architecture and Features in an expandable section
    with st.expander("🏗️ **功能介绍与系统架构**", expanded=False):
        st.markdown("### 系统工作流程图")

        # Display the system architecture diagram
        st.image("frontend/assets/system_architecture.png",
                caption="系统工作流程图",
                use_container_width=True)

        st.markdown("""
        ### 技术架构
        - **前端**：Streamlit - 简洁的 Web 界面
        - **后端**：Python + Pandas - 强大的数据处理能力
        - **AI 引擎**：LangChain + OpenAI - 自然语言理解
        - **向量数据库**：Milvus - 工具检索与示例匹配
        - **监控系统**：Langfuse - LLM 调用监控与优化

        **支持的操作类型：**
        表格合并 • 数据重塑 • 数据比较 • 垂直堆叠 • 数据去重
        """)

    # Project information
    display_project_info()

    st.markdown("---")

    # Privacy notice
    st.warning(
        "**隐私提示**：系统仅读取文件名和列名信息，不会读取或存储您的具体表格数据。"
        "表格预览内容临时存储在内存中，刷新页面后会自动清除。"
    )

    # Core functionality
    handle_file_upload()
    if st.session_state.files_uploaded:
        # Only show Dataset Operations section if user has started a conversation
        if st.session_state.conversation_history:
            process_user_query()
        else:
            # Show only the input box without the Dataset Operations section
            user_query = st.chat_input("请输入您的数据处理需求：")
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
    st.markdown("## 数据上传")
    with st.container(border=True):
        uploaded_files = st.file_uploader(
            "选择 CSV 或 Excel 文件（支持多文件上传）",
            type=["csv", "xlsx", "xls"],
            accept_multiple_files=True,
            help="上传 CSV 或 Excel 格式的数据文件。可以上传多个文件进行合并、比较等操作。"
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

                st.success(f"成功上传 {len(uploaded_files)} 个文件！")

            except Exception as e:
                st.error(f"文件上传失败：{str(e)}")
                st.session_state.files_uploaded = False
        else:
            st.session_state.files_uploaded = False

        if st.session_state.files_uploaded:
            st.markdown("---")
            st.markdown("#### 已上传数据集预览")
            display_loaded_dataframes()


def display_loaded_dataframes():
    """
    Display preview of loaded original datasets using tabs.

    Shows dataset information including shape and preview data in a tabbed interface
    for multiple datasets or a single view for one dataset.
    """
    original_dataframes = st.session_state.workflow.get_original_dataframe_info()

    if not original_dataframes:
        st.info("还没有上传数据集，请先上传数据文件。")
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
        "提示：为确保数据处理准确性，包含特殊字符或空格的列名将被自动清理。"
    )


def process_user_query():
    """
    Process user queries and display results in a chat interface.

    Handles conversation history, user input, and AI responses for dataset operations.
    """
    st.markdown("## 数据集操作")

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

    user_query = input_placeholder.chat_input("请输入您的数据处理需求：")

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
            with st.spinner("思考中..."):
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
                message = result.get("message", "需要更多信息来处理您的请求。")
                st.markdown(message)
                st.session_state.conversation_history.append(
                    {"role": "assistant", "content": message}
                )
            elif result["next_step"] == "execute_operation":
                message = "操作已成功执行！以下是执行的步骤：\n"
                st.markdown(message)
                st.session_state.operation_steps = result.get("operation", [])
                for i, step in enumerate(st.session_state.operation_steps, 1):
                    st.markdown(f"步骤 {i}：{step['tool_name']}")
                full_message = (
                    message
                    + "\n"
                    + "\n".join(
                        [
                            f"步骤 {i}：{step['tool_name']}"
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
                message = result.get("message", "抱歉，您的请求超出了我的处理范围。")
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
        st.markdown("## 操作结果")

        with st.container(border=True):
            for i, step in enumerate(st.session_state.operation_steps, 1):
                output_df_names = step["output_df_names"]
                for df_name in output_df_names:
                    if df_name in st.session_state.workflow.dataframes:
                        df = st.session_state.workflow.dataframes[df_name]
                        st.markdown(f"#### {df_name}")
                        st.caption(f"*由步骤 {i} 生成：{step['tool_name']}*")
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
        st.markdown("##### 本次操作是否满足您的需求？")

        # Initialize feedback status to prevent duplicate submissions
        if "feedback_given" not in st.session_state:
            st.session_state.feedback_given = False

        col1, col2, col3 = st.columns([1, 1, 3])

        with col1:
            yes_button = st.button(
                "👍 满足",
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
                "👎 不满足",
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
                st.success("感谢您的反馈！")

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
        label=f"下载 {df_name} (CSV)",
        data=csv,
        file_name=f"{df_name}.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
