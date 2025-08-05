from typing import Dict, Any, Literal, Optional, List
from pydantic import BaseModel, Field


class OperationStep(BaseModel):
    """Represents a single operation step."""

    tool_name: str = Field(..., description="Name of the tool function to call")
    tool_args: Dict[str, Any] = Field(..., description="Parameters for calling the tool function")
    output_df_names: List[str] = Field(..., description="List of DataFrame names for operation result output")


class AssistantResponse(BaseModel):
    """Represents AI assistant's response."""

    next_step: Literal["need_more_info", "execute_operation", "out_of_scope"] = Field(
        ..., description="Next step operation: need more information, execute operation, or out of scope"
    )
    operation: Optional[List[OperationStep]] = Field(
        None, description="When next_step is 'execute_operation', contains list of operation steps to execute"
    )
    message: Optional[str] = Field(
        None,
        description="When next_step is 'need_more_info' or 'out_of_scope', contains message for user",
    )
