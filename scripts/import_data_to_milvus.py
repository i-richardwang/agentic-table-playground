#!/usr/bin/env python3
"""
Simple script to import demo CSV data into Milvus vector database.

This script imports demo datasets for AI tool retrieval and example matching:
1. Tools_Description.csv -> tools_description collection
   - Tool function descriptions for AI to find relevant tools
2. Table-Operation-Tools-and-Examples.csv -> data_operation_examples collection
   - Example user queries and operations for AI reference

Prerequisites:
- Milvus running locally or remotely
- Environment configured with OPENAI_API_KEY, VECTOR_DB_*, and VECTOR_EMBEDDING_DIMENSION settings

Usage:
    python scripts/import_data_to_milvus.py

The script will:
- Connect to Milvus using environment variables
- Create/recreate collections based on data/config/collections_config.json
- Generate embeddings for text fields using OpenAI
- Insert data and create indexes for efficient search
- Load collections for immediate use

Note: Existing collections will be dropped and recreated for clean import.
"""

import os
import sys
import json
import csv
from typing import List, Dict, Any

# Add project root to Python path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from utils.llm_tools import create_embeddings


def load_config() -> Dict[str, Any]:
    """
    Load collection configuration from JSON file.

    Returns:
        Dictionary containing collection schemas and configurations.
    """
    config_path = os.path.join(project_root, "data", "config", "collections_config.json")
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def connect_to_milvus():
    """
    Establish connection to Milvus vector database.

    Uses environment variables for configuration and supports both
    local and cloud deployments.
    """
    # Get connection parameters from environment
    host = os.getenv("VECTOR_DB_HOST", "localhost")
    port = os.getenv("VECTOR_DB_PORT", "19530")
    db_name = os.getenv("VECTOR_DB_DATABASE", "examples")

    # Configure connection for cloud services using URI format
    if host.startswith("http"):
        # Cloud service - use URI format
        uri = host
        connection_params = {
            "alias": "default",
            "uri": uri,
            "db_name": db_name,
        }

        # Add token authentication for cloud services
        token = os.getenv("VECTOR_DB_TOKEN")
        user = os.getenv("VECTOR_DB_USER")
        password = os.getenv("VECTOR_DB_PASSWORD")

        if token:
            connection_params["token"] = token
        elif user and password:
            connection_params["user"] = user
            connection_params["password"] = password
    else:
        # Local service - use host/port format
        connection_params = {
            "alias": "default",
            "host": host,
            "port": int(port),
            "db_name": db_name,
        }

        # Add authentication if provided
        user = os.getenv("VECTOR_DB_USER")
        password = os.getenv("VECTOR_DB_PASSWORD")

        if user and password:
            connection_params["user"] = user
            connection_params["password"] = password

    connections.connect(**connection_params)
    print("✅ Connected to Milvus")


def create_collection_schema(collection_config: Dict[str, Any]) -> CollectionSchema:
    """
    Create Milvus collection schema based on configuration.

    Args:
        collection_config: Dictionary containing field definitions and metadata.

    Returns:
        Configured CollectionSchema ready for collection creation.
    """
    fields = []

    # Get vector dimension from environment configuration
    vector_dim = int(os.getenv("VECTOR_EMBEDDING_DIMENSION", "1024"))

    # Add primary key field with auto-increment
    fields.append(FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True))

    # Process field configurations to create schema fields
    for field_config in collection_config["fields"]:
        field_name = field_config["name"]
        is_vector = field_config.get("is_vector", False)

        if is_vector:
            # Create vector field for embeddings with configured dimension
            fields.append(FieldSchema(name=f"{field_name}_vector", dtype=DataType.FLOAT_VECTOR, dim=vector_dim))

        # Create text field for original content
        fields.append(FieldSchema(name=field_name, dtype=DataType.VARCHAR, max_length=65535))

    return CollectionSchema(fields, collection_config["description"])


def create_collection_if_not_exists(collection_name: str, collection_config: Dict[str, Any]):
    """
    Create Milvus collection, dropping existing one if present.

    Args:
        collection_name: Name of the collection to create.
        collection_config: Configuration dictionary for the collection.

    Returns:
        Created Collection object ready for data insertion.
    """
    if utility.has_collection(collection_name):
        print(f"⚠️  Collection '{collection_name}' already exists, dropping it...")
        utility.drop_collection(collection_name)
    
    schema = create_collection_schema(collection_config)
    collection = Collection(collection_name, schema)
    print(f"✅ Created collection '{collection_name}'")
    return collection


def read_csv_data(csv_path: str) -> List[Dict[str, str]]:
    """
    Read data from CSV file into list of dictionaries.

    Args:
        csv_path: Path to the CSV file to read.

    Returns:
        List of dictionaries, each representing a row from the CSV.
    """
    data = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:  # Handle BOM
        reader = csv.DictReader(f)
        for row in reader:
            # Clean up field names and values for consistent processing
            cleaned_row = {}
            for key, value in row.items():
                # Remove BOM (Byte Order Mark) and whitespace from field names
                clean_key = key.strip().lstrip('\ufeff')
                cleaned_row[clean_key] = value if value is not None else ""
            data.append(cleaned_row)
    print(f"📖 Read {len(data)} records from {csv_path}")
    return data


def prepare_data_for_insertion(data: List[Dict[str, str]], collection_config: Dict[str, Any]) -> Dict[str, List]:
    """
    Prepare data for Milvus insertion by generating embeddings for specified fields.

    Args:
        data: List of records from CSV file.
        collection_config: Configuration specifying which fields need embeddings.

    Returns:
        Dictionary with field names as keys and lists of values ready for insertion.
    """
    embeddings_model = create_embeddings()
    embedding_fields = collection_config.get("embedding_fields", [])

    # Initialize data structure for all fields
    prepared_data = {}
    for field_config in collection_config["fields"]:
        field_name = field_config["name"]
        prepared_data[field_name] = []

        if field_config.get("is_vector", False):
            prepared_data[f"{field_name}_vector"] = []

    # Process each record and generate embeddings
    print("🔄 Generating embeddings...")
    for i, record in enumerate(data):
        if i % 10 == 0:
            print(f"   Processing record {i+1}/{len(data)}")

        for field_config in collection_config["fields"]:
            field_name = field_config["name"]
            # Extract field value from CSV record, handling missing fields
            field_value = record.get(field_name, "")
            if field_value is None:
                field_value = ""
            prepared_data[field_name].append(field_value)

            # Generate embeddings only for specified fields
            if field_name in embedding_fields:
                print(f"   Generating embedding for {field_name}: {field_value[:50]}...")
                embedding = embeddings_model.embed_query(field_value)
                prepared_data[f"{field_name}_vector"].append(embedding)

    return prepared_data


def insert_data_to_collection(collection: Collection, data: Dict[str, List]):
    """
    Insert prepared data into Milvus collection.

    Args:
        collection: Milvus collection object to insert data into.
        data: Dictionary with field names as keys and lists of values.
    """
    # Convert column-based data to row-based format required by Milvus
    num_records = len(next(iter(data.values())))
    rows = []

    for i in range(num_records):
        row = {}
        for field_name, field_values in data.items():
            row[field_name] = field_values[i]
        rows.append(row)

    print(f"📥 Inserting {len(rows)} records into collection...")
    collection.insert(rows)
    collection.flush()
    print("✅ Data inserted successfully")


def create_index(collection: Collection, collection_config: Dict[str, Any]):
    """
    Create indexes for vector fields to enable efficient similarity search.

    Args:
        collection: Milvus collection to create indexes for.
        collection_config: Configuration specifying which fields need indexes.
    """
    embedding_fields = collection_config.get("embedding_fields", [])
    
    for field_name in embedding_fields:
        vector_field = f"{field_name}_vector"
        index_params = {
            "metric_type": "IP",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        
        print(f"🔍 Creating index for {vector_field}...")
        collection.create_index(vector_field, index_params)
    
    collection.load()
    print("✅ Index created and collection loaded")


def import_csv_to_collection(csv_filename: str, collection_name: str, config: Dict[str, Any]):
    """
    Complete workflow to import CSV data into a Milvus collection.

    Args:
        csv_filename: Name of the CSV file in data/datasets directory.
        collection_name: Name of the Milvus collection to create/populate.
        config: Configuration dictionary containing collection schemas.
    """
    print(f"\n🚀 Importing {csv_filename} to {collection_name}")
    
    # Set up file paths and configuration
    csv_path = os.path.join(project_root, "data", "datasets", csv_filename)
    collection_config = config["collections"][collection_name]

    # Create the collection with proper schema
    collection = create_collection_if_not_exists(collection_name, collection_config)

    # Read CSV data and prepare with embeddings
    csv_data = read_csv_data(csv_path)
    prepared_data = prepare_data_for_insertion(csv_data, collection_config)

    # Insert data and create search indexes
    insert_data_to_collection(collection, prepared_data)
    create_index(collection, collection_config)
    
    print(f"✅ Successfully imported {csv_filename} to {collection_name}\n")


def main():
    """
    Main function to import all demo data into Milvus collections.

    Imports tool descriptions and example data for AI-powered table operations.
    """
    print("🎯 Starting Milvus data import...")
    
    # Load collection configuration from JSON
    config = load_config()

    # Establish connection to Milvus database
    connect_to_milvus()

    # Import demo datasets for AI tool retrieval
    import_csv_to_collection("Tools_Description.csv", "tools_description", config)
    import_csv_to_collection("Table-Operation-Tools-and-Examples.csv", "data_operation_examples", config)
    
    print("🎉 All data imported successfully!")
    print("\n📋 Summary:")
    print("   - tools_description: Tool function descriptions for AI retrieval")
    print("   - data_operation_examples: Example queries and operations for AI reference")


if __name__ == "__main__":
    main()
