import os
from dotenv import load_dotenv


def load_env():
    """
    Load environment variables from .env file or Streamlit secrets.

    For local development, loads from .env file in the project root directory.
    For Streamlit Cloud deployment, uses st.secrets which automatically loads
    from .streamlit/secrets.toml file.
    """
    try:
        # Try to import streamlit and use secrets if available (Streamlit Cloud)
        import streamlit as st
        if hasattr(st, 'secrets') and st.secrets:
            # Load secrets into environment variables
            for key, value in st.secrets.items():
                if isinstance(value, (str, int, float, bool)):
                    os.environ[key] = str(value)
            print("Environment variables loaded from Streamlit secrets")
            return
    except (ImportError, AttributeError):
        # Streamlit not available or no secrets, continue with .env file
        pass
    
    # Fallback to .env file for local development
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    dotenv_path = os.path.join(project_root, '.env')
    
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path)
        print("Environment variables loaded from .env file")
    else:
        print("Warning: No .env file found and not running in Streamlit Cloud")
        print("Please ensure environment variables are set properly")