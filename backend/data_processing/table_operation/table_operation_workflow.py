import logging
import uuid
from typing import List, Dict, Any, Optional
import pandas as pd

from backend.data_processing.table_operation.table_operation_core import (
    create_dataframe_assistant,
    process_user_query,
    record_user_feedback,
)
from backend.data_processing.table_operation.table_operations import *

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataFrameWorkflow:
    """
    Manages DataFrame operation workflows with AI-powered natural language processing.

    This class orchestrates the entire workflow from user query processing to operation execution,
    maintaining conversation history and managing multiple DataFrames throughout the process.
    """

    def __init__(self):
        """
        Initialize DataFrameWorkflow instance.

        Sets up the AI assistant, initializes data storage, and prepares the workflow
        for processing user queries and managing DataFrame operations.
        """
        self.assistant_chain = create_dataframe_assistant()
        self.dataframes: Dict[str, pd.DataFrame] = {}
        self.available_tools = [
            tool for tool in globals().values() if callable(tool) and hasattr(tool, 'name')
        ]
        self.conversation_history: List[Dict[str, str]] = []
        self.current_state: str = "initial"
        self.session_id: Optional[str] = None
        self.original_dataframes = set()

    def load_dataframe(self, name: str, df: pd.DataFrame) -> None:
        """
        Load DataFrame into workflow and mark as original dataset.

        Args:
            name: Unique identifier for the DataFrame.
            df: Pandas DataFrame to be loaded.
        """
        self.dataframes[name] = df
        self.original_dataframes.add(name)  # Track as original dataset
        logger.info(f"Loaded DataFrame '{name}' with shape {df.shape}")

    def load_dataframes(self, dataframes: Dict[str, pd.DataFrame]) -> None:
        """
        Load multiple DataFrames into workflow and mark as original datasets.

        Args:
            dataframes: Dictionary mapping DataFrame names to DataFrame objects.
        """
        for name, df in dataframes.items():
            self.load_dataframe(name, df)
        logger.info(f"Loaded {len(dataframes)} DataFrames: {list(dataframes.keys())}")

    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process user query and execute corresponding operations.

        Args:
            query: User's query string.

        Returns:
            Dictionary containing operation results.
        """
        logger.info(f"Processing query: {query}")
        self.conversation_history.append({"role": "user", "content": query})

        if self.session_id is None:
            self.session_id = str(uuid.uuid4())

        dataframe_info = self.get_dataframe_info()

        full_context = "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in self.conversation_history]
        )

        result = process_user_query(
            self.assistant_chain,
            full_context,
            dataframe_info,
            self.session_id,
        )
        logger.info(f"AI response: {result}")

        # Extract next step from AI response (result is a dictionary)
        next_step = result.get("next_step")

        if next_step == "need_more_info":
            self._handle_need_more_info(result)
        elif next_step == "execute_operation":
            self._handle_execute_operation(result)
        else:  # Handle out_of_scope requests
            self._handle_out_of_scope(result)

        return result

    def record_feedback(self, trace_id: str, is_useful: bool):
        """
        Record user feedback for monitoring and improvement.

        Args:
            trace_id: Unique identifier for the operation trace.
            is_useful: Boolean indicating whether the operation was useful.
        """
        record_user_feedback(trace_id, is_useful)

    def _handle_need_more_info(self, result: Dict[str, Any]) -> None:
        """
        Handle cases where AI needs more information from the user.

        Args:
            result: AI assistant's response containing the clarification request.
        """
        self.current_state = "need_more_info"
        self.conversation_history.append(
            {"role": "assistant", "content": result.get("message", "")}
        )

    def _handle_execute_operation(self, result: Dict[str, Any]) -> None:
        """
        Handle operation execution by processing each step sequentially.

        Args:
            result: AI assistant's response containing operation steps to execute.
        """
        self.current_state = "ready"
        operation_steps = result.get("operation", [])

        final_results = {}
        for step in operation_steps:
            tool_name = step.get("tool_name")
            tool_args = step.get("tool_args", {})
            output_df_names = step.get("output_df_names", [])

            logger.info(f"Executing step: {tool_name} with args: {tool_args}")

            if tool_name == "stack_dataframes":
                # Special handling for stack_dataframes which requires name-dataframe pairs
                dataframes_with_names = []
                for df_name in tool_args.get("dataframes", []):
                    if df_name in self.dataframes:
                        dataframes_with_names.append(
                            (df_name, self.dataframes[df_name])
                        )
                tool_args["dataframes"] = dataframes_with_names
            else:
                self._replace_dataframe_names_with_objects(tool_args)

            tool_function = self._get_tool_function(tool_name)

            if tool_function:
                try:
                    step_result = tool_function.invoke(tool_args)
                    self._process_step_result(step_result, output_df_names)

                    # Save result if this is the final step
                    if step == operation_steps[-1]:
                        final_results = self._format_final_results(
                            step_result, output_df_names
                        )
                except Exception as e:
                    logger.error(f"Error executing tool: {str(e)}")
                    raise ValueError(f"Error occurred while executing operation: {str(e)}")
            else:
                error_msg = f"Unknown tool: {tool_name}"
                logger.error(error_msg)
                raise ValueError(error_msg)

        result.update(final_results)

    def _handle_out_of_scope(self, result: Dict[str, Any]) -> None:
        """
        Handle requests that are outside the system's capabilities.

        Args:
            result: AI assistant's response explaining why the request is out of scope.
        """
        self.current_state = "out_of_scope"
        self.conversation_history.append(
            {"role": "assistant", "content": result.get("message", "")}
        )

    def _replace_dataframe_names_with_objects(self, tool_args: Dict[str, Any]) -> None:
        """
        Replace DataFrame names in tool arguments with actual DataFrame objects.

        Args:
            tool_args: Tool arguments dictionary that may contain DataFrame names as strings.
        """
        for arg, value in tool_args.items():
            if isinstance(value, str) and value in self.dataframes:
                tool_args[arg] = self.dataframes[value]
                logger.info(
                    f"Replaced dataframe name '{value}' with actual dataframe. "
                    f"Shape: {tool_args[arg].shape}, columns: {tool_args[arg].columns.tolist()}"
                )

    def _get_tool_function(self, tool_name: str) -> Any:
        """
        Get the corresponding tool function based on tool name.

        Args:
            tool_name: Name of the tool function to retrieve.

        Returns:
            Tool function object, or None if not found.
        """
        return next(
            (tool for tool in self.available_tools if tool.name == tool_name), None
        )

    def _process_step_result(
        self, step_result: Any, output_df_names: List[str]
    ) -> None:
        """
        Process and store the execution result of a single operation step.

        Args:
            step_result: Result from tool execution (DataFrame or tuple of DataFrames).
            output_df_names: List of names for storing the output DataFrames.
        """
        if isinstance(step_result, tuple) and len(step_result) == len(output_df_names):
            for df, name in zip(step_result, output_df_names):
                self.dataframes[name] = df
                logger.info(f"Stored result DataFrame '{name}' with shape {df.shape}")
        elif isinstance(step_result, pd.DataFrame) and len(output_df_names) == 1:
            self.dataframes[output_df_names[0]] = step_result
            logger.info(
                f"Stored result DataFrame '{output_df_names[0]}' with shape {step_result.shape}"
            )
        else:
            logger.warning(
                f"Unexpected step result type or mismatch with output names: {type(step_result)}"
            )

    def _format_final_results(
        self, final_result: Any, output_df_names: List[str]
    ) -> Dict[str, Any]:
        """
        Format final operation results for return to the frontend.

        Args:
            final_result: Execution result of the last operation step.
            output_df_names: List of output DataFrame names (for future use).

        Returns:
            Formatted result dictionary containing the processed DataFrames.
        """
        if isinstance(final_result, tuple) and len(final_result) == 2:
            return {"result_df1": final_result[0], "result_df2": final_result[1]}
        elif isinstance(final_result, pd.DataFrame):
            return {"result_df": final_result}
        else:
            logger.warning(f"Unexpected final result type: {type(final_result)}")
            return {"result": final_result}

    def get_original_dataframe_info(self) -> Dict[str, Dict]:
        """
        Get information about the original uploaded datasets only.

        Returns:
            Dictionary containing information about original datasets (excluding derived ones).
        """
        return {
            name: self.get_dataframe_info()[name] for name in self.original_dataframes
        }

    def get_dataframe(self, name: str) -> pd.DataFrame:
        """
        Get DataFrame by name from the workflow storage.

        Args:
            name: Name of the DataFrame to retrieve.

        Returns:
            DataFrame with the specified name, or None if not found.
        """
        return self.dataframes.get(name)

    def get_dataframe_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get comprehensive information about all DataFrames in the workflow.

        Returns:
            Dictionary containing shape, data types, and preview for each DataFrame.
        """
        return {
            name: {
                "shape": df.shape,
                "dtypes": df.dtypes.to_dict(),
                "preview": df.head(10)  # Show first 10 rows as preview
            }
            for name, df in self.dataframes.items()
        }

    def get_last_message(self) -> str:
        """
        Get the content of the last message in conversation history.

        Returns:
            Content of the last message, or empty string if no conversation exists.
        """
        return (
            self.conversation_history[-1]["content"]
            if self.conversation_history
            else ""
        )

    def reset_conversation(self) -> None:
        """
        Reset conversation history and workflow state to initial conditions.
        """
        self.conversation_history = []
        self.current_state = "initial"
        logger.info("Conversation history and state reset")
