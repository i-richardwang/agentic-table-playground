import os
import logging
from typing import Any, Optional, Type

from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def init_language_model(temperature: float = 0.0, model_name: Optional[str] = None, **kwargs: Any) -> ChatOpenAI:
    """
    Initialize OpenAI language model.

    Args:
        temperature: Temperature for model output, controls randomness. Default is 0.0.
        model_name: Model name, default is gpt-4o-mini.
        **kwargs: Other optional parameters, will be passed to model initialization.

    Returns:
        Initialized ChatOpenAI instance.

    Raises:
        ValueError: Raised when required API key is missing.
    """
    model_name = model_name or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable not found. Please set your OpenAI API key."
        )

    model_params = {
        "model": model_name,
        "temperature": temperature,
        "max_tokens": 1024,
        **kwargs,
    }

    # Add custom API base URL if specified
    openai_api_base = os.environ.get("OPENAI_API_BASE")
    if openai_api_base:
        model_params["base_url"] = openai_api_base

    return ChatOpenAI(**model_params)


class LanguageModelChain:
    """
    Language model chain for processing input and generating output conforming to specified schema.

    Attributes:
        model_cls: Pydantic model class defining the output structure.
        parser: JSON output parser.
        prompt_template: Chat prompt template.
        chain: Complete processing chain.
    """

    def __init__(
        self, model_cls: Type[BaseModel], sys_msg: str, user_msg: str, model: Any
    ):
        """
        Initialize LanguageModelChain instance.

        Args:
            model_cls: Pydantic model class defining the output structure.
            sys_msg: System message.
            user_msg: User message.
            model: Language model instance.

        Raises:
            ValueError: Raised when provided parameters are invalid.
        """
        if not issubclass(model_cls, BaseModel):
            raise ValueError("model_cls must be a subclass of Pydantic BaseModel")
        if not isinstance(sys_msg, str) or not isinstance(user_msg, str):
            raise ValueError("sys_msg and user_msg must be string types")
        if not callable(model):
            raise ValueError("model must be a callable object")

        self.model_cls = model_cls
        self.parser = JsonOutputParser(pydantic_object=model_cls)

        format_instructions = """
Output your answer as a JSON object that conforms to the following schema:
```json
{schema}
```

Important instructions:
1. Ensure your JSON is valid and properly formatted.
2. Do not include the schema definition in your answer.
3. Only output the data instance that matches the schema.
4. Do not include any explanations or comments within the JSON output.
        """

        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", sys_msg + format_instructions),
                ("human", user_msg),
            ]
        ).partial(schema=model_cls.model_json_schema())

        self.chain = self.prompt_template | model | self.parser

    def __call__(self) -> Any:
        """
        Return the configured processing chain for execution.

        Returns:
            Complete LangChain processing pipeline ready for invocation.
        """
        return self.chain




def create_embeddings(model: Optional[str] = None) -> "OpenAIEmbeddings":
    """
    Create OpenAI embedding model instance for vector operations.

    Args:
        model: Embedding model name, defaults to text-embedding-3-small.

    Returns:
        Configured OpenAIEmbeddings instance ready for text embedding.

    Raises:
        ValueError: Raised when required API key is missing.
    """
    from langchain_openai import OpenAIEmbeddings

    model = model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    # Use EMBEDDING_API_KEY for embedding operations
    embedding_api_key = os.environ.get("EMBEDDING_API_KEY")
    if not embedding_api_key:
        raise ValueError(
            "EMBEDDING_API_KEY environment variable not found. Please set your embedding API key."
        )

    embedding_params = {
        "model": model,
        "api_key": embedding_api_key
    }

    # Add custom API base URL if specified
    embedding_api_base = os.environ.get("EMBEDDING_API_BASE")
    if embedding_api_base:
        embedding_params["base_url"] = embedding_api_base

    return OpenAIEmbeddings(**embedding_params)
