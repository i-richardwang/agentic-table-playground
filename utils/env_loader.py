import os
from dotenv import load_dotenv


def load_env():
    """
    Load environment variables from .env file into the current environment.

    Locates the .env file in the project root directory and loads all
    environment variables defined within it.
    """
    # Get project root directory path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # Build .env file path
    dotenv_path = os.path.join(project_root, '.env')

    # Load environment variables from .env file
    load_dotenv(dotenv_path)

    print("Environment variables loaded successfully")