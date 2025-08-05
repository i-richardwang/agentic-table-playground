import logging
import os
from typing import List, Dict, Any, Optional

import uuid
import json

# Optional import for Langfuse monitoring - gracefully handle if not available
try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    Langfuse = None

from langchain_core.tools import tool

from backend.data_processing.table_operation.table_operation_models import (
    AssistantResponse,
)
from utils.llm_tools import LanguageModelChain, init_language_model, create_embeddings
from utils.vector_db_utils import (
    connect_to_milvus,
    initialize_vector_store,
    search_in_milvus,
)

# Initialize logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize Langfuse client for monitoring if available
langfuse_client = Langfuse() if LANGFUSE_AVAILABLE else None

# Initialize the language model for AI operations
language_model = init_language_model()

# System message for AI assistant behavior
SYSTEM_MESSAGE = """
You are a professional data analysis assistant specializing in helping users perform table operations. Your task is to understand user requirements and use the provided tool functions to manipulate tables. You can handle simple single-step operations as well as complex user requirements that need multiple steps to complete.

Please carefully analyze user requirements and follow these rules:

1. When users mention "tables," they refer to DataFrame variables already loaded in the environment. For users, these are the tables they uploaded. Assume users may not understand Python programming.

2. If the user's request is unclear or lacks necessary information, prioritize returning "need_more_info" as next_step. Ask for more details using friendly, plain language to help users clarify their requirements. Avoid technical terms like "DataFrame," "inner," "left join," etc. Instead, use more understandable expressions like "table," "keep all data from the left table," etc.

3. Only return "out_of_scope" as next_step when you fully understand the user's requirements and determine that existing tool functions truly cannot complete the task.

4. If the user's request is clear and can be completed using the provided tool functions, return "execute_operation" as next_step and provide a complete list of operation steps.

5. Determine whether multiple steps are needed based on task requirements:
   a. If the user's requirement can be completed with a single tool, don't split into steps.
   b. If the requirement cannot be completed with a single tool but can be accomplished through a combination of multiple tools, break down the task into necessary operation steps.
   c. Generate an OperationStep for each step, including tool_name, tool_args, and output_df_names.

6. When generating tool_args, use original table names in the environment or outputs from previous steps as inputs.

7. For output_df_names:
   a. output_df_names should be meaningful names that comply with file naming conventions, completely and accurately representing the meaning of output tables.
   b. These names will be used as filenames when users download, so they need to combine user intent and original table names for easy user identification.
   c. Some tools (like compare_dataframes) may output multiple DataFrames, so use a list to store output names.
   d. Ensure that based on whether the called function outputs one or multiple DataFrames, provide the corresponding number of names in the output_df_names list.

8. When selecting operations and providing suggestions, fully consider the data type information of each table.

9. Intelligently determine possible field name differences. For example, field names mentioned by users may differ slightly from actual field names in tables. If you can reasonably infer which field the user is referring to, directly use the correct field name in your response. When unable to determine, ask the user for clarification.

10. If the user's request has some ambiguous but inferable parts, you can propose your guess and ask if it's correct.

11. In your response, first restate your understanding of the user's requirements, then provide suggestions or request more information.

Please analyze user input carefully according to these guidelines and provide appropriate responses. Your goal is to help users complete their table operation requirements as much as possible, even if this may require multiple interactions to clarify and refine requirements. Only consider using "out_of_scope" when you're certain you cannot provide any useful help.
"""

# Template for human message input to the AI assistant
HUMAN_MESSAGE_TEMPLATE = """
Available table operation tool functions are as follows:

{tools_description}

Here is an example for reference when executing user requirements (but the user's actual requirements may differ from the example, so please don't directly use the operation steps from the example, but regenerate operation steps based on the user's requirements):

{example}

Current user-uploaded tables and their information:

{dataframe_info}

User input:
{user_input}

Please analyze the user's requirements and provide appropriate responses according to the guidelines in the system message.
"""


def create_langfuse_handler(session_id: str, step: str):
    """
    Create Langfuse CallbackHandler for monitoring AI operations.

    Args:
        session_id: Unique session identifier for tracking.
        step: Current operation step name for context.

    Returns:
        Configured Langfuse CallbackHandler or None if unavailable.
    """
    if not LANGFUSE_AVAILABLE:
        return None

    # Currently disabled - placeholder for future Langfuse integration
    return None


def record_user_feedback(trace_id: str, is_useful: bool):
    """
    Record user feedback for operation quality monitoring.

    Args:
        trace_id: Unique identifier for the operation trace.
        is_useful: Boolean indicating whether the operation was helpful.
    """
    if langfuse_client:
        try:
            langfuse_client.score(
                trace_id=trace_id, name="feedback", value=is_useful, data_type="BOOLEAN"
            )
        except Exception as e:
            logger.warning(f"Failed to record feedback: {e}")


def get_similar_tools(query: str, top_k: int = 3) -> str:
    """
    Retrieve tool descriptions most similar to the query from vector database.

    Args:
        query (str): User query.
        top_k (int): Number of most similar results to return.

    Returns:
        str: Formatted tool description string.
    """
    connect_to_milvus(db_name=os.getenv("VECTOR_DB_DATABASE", "examples"))
    collection = initialize_vector_store("tools_description")

    embeddings = create_embeddings()
    query_vector = embeddings.embed_query(query)

    # Retrieve more results than needed to allow for deduplication
    results = search_in_milvus(collection, query_vector, "description", top_k * 3)

    # Log retrieval results for debugging and monitoring
    for result in results:
        logger.info(
            f"Retrieved tool: {result['tool_name']}, similarity: {result['distance']:.4f}"
        )

    # Remove duplicate tools while preserving order by similarity
    unique_tools = []
    seen_tools = set()
    for result in results:
        tool_name = result["tool_name"]
        if tool_name not in seen_tools and len(unique_tools) < top_k:
            unique_tools.append(result)
            seen_tools.add(tool_name)

    tools_description = ""
    for result in unique_tools:
        tools_description += f"Function Name:\n{result['tool_name']}\n\n"
        tools_description += f"Description:\n{result['full_description']}\n\n"
        tools_description += f"Parameters:\n{result['args']}\n\n"

    return tools_description.strip()


def create_dataframe_assistant():
    """
    Create AI assistant for DataFrame operations.

    Returns:
        Configured language model chain ready for processing user queries.
    """
    assistant_chain = LanguageModelChain(
        AssistantResponse, SYSTEM_MESSAGE, HUMAN_MESSAGE_TEMPLATE, language_model
    )()

    return assistant_chain


def get_similar_example(
    query: str, collection_name: str = "data_operation_examples"
) -> Dict[str, str]:
    """
    Retrieve the most similar example to user query from vector database.

    Args:
        query: User's natural language query.
        collection_name: Name of the Milvus collection containing examples.

    Returns:
        Dictionary containing similar example or empty dict if none found.
    """
    # Connect to Milvus vector database
    connect_to_milvus(db_name=os.getenv("VECTOR_DB_DATABASE", "examples"))

    # Initialize the vector store collection
    collection = initialize_vector_store(collection_name)

    # Generate embedding vector for the user query
    embeddings = create_embeddings()
    query_vector = embeddings.embed_query(query)

    # Search for the most similar example
    results = search_in_milvus(collection, query_vector, "user_query", top_k=1)

    if results:
        similar_example = {
            "User uploaded tables": results[0].get("user_tables"),
            "User query": results[0].get("user_query"),
            "Output": results[0].get("output"),
        }
        return similar_example
    else:
        return {}  # Return empty dict if no similar examples found


def format_example(example: Dict[str, str]) -> str:
    """
    Format example dictionary into readable text format for AI processing.

    Args:
        example: Dictionary containing example data with keys and values.

    Returns:
        Formatted string representation of the example.
    """
    if not example:
        return "{}"

    formatted = "{\n"
    for key, value in example.items():
        formatted += f"{key}: \n{value}\n\n"
    formatted += "}"
    return formatted


def process_user_query(
    assistant_chain,
    user_input: str,
    dataframe_info: Dict[str, Dict],
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Process user query through the AI assistant chain.

    Args:
        assistant_chain: Configured language model chain for processing.
        user_input: User's natural language query.
        dataframe_info: Information about available DataFrames.
        session_id: Optional session identifier for tracking.

    Returns:
        Dictionary containing AI response with next steps and operations.

    Raises:
        ValueError: If query processing fails.
    """
    try:
        if session_id is None:
            session_id = str(uuid.uuid4())

        langfuse_handler = create_langfuse_handler(session_id, "process_user_query")

        # Retrieve relevant tools based on user query
        tools_description = get_similar_tools(user_input)
        dataframe_info_str = format_dataframe_info(dataframe_info)

        # Get similar examples to guide AI response
        similar_example = get_similar_example(user_input)
        example_str = format_example(similar_example)

        input_data = {
            "user_input": user_input,
            "example": example_str,
            "tools_description": tools_description,
            "dataframe_info": dataframe_info_str,
        }

        logger.info(f"Processing user query: {user_input}")

        # Process query through AI assistant with optional monitoring
        if langfuse_handler:
            result = assistant_chain.invoke(
                input_data, config={"callbacks": [langfuse_handler]}
            )
            trace_id = langfuse_handler.get_trace_id()
            result["trace_id"] = trace_id
        else:
            result = assistant_chain.invoke(input_data)
            result["trace_id"] = session_id  # Use session_id as fallback trace identifier

        logger.info(f"Query processed successfully. Result: {result}")

        return result

    except Exception as e:
        logger.error(f"Error processing user query: {str(e)}")
        raise ValueError(f"Error occurred while processing query: {str(e)}")


def format_dataframe_info(dataframe_info: Dict[str, Dict]):
    """
    Format DataFrame information for AI processing.

    Args:
        dataframe_info: Dictionary containing DataFrame metadata and structure.

    Returns:
        Formatted DataFrame information as dictionary items.
    """
    return dataframe_info.items()
