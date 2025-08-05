import os
from typing import List, Dict, Any
from pymilvus import connections, Collection, utility


def connect_to_milvus(db_name: str = "default"):
    """
    Establish connection to Milvus vector database.

    Args:
        db_name: Name of the database to connect to. Defaults to "default".
    """
    # Retrieve connection configuration from environment
    host = os.getenv("VECTOR_DB_HOST", "localhost")
    port = os.getenv("VECTOR_DB_PORT", "19530")

    # Configure connection parameters based on service type
    if host.startswith("http"):
        # Cloud service configuration - use URI format
        uri = host
        connection_params = {
            "alias": "default",
            "uri": uri,
            "db_name": db_name,
        }

        # Add authentication for cloud services
        token = os.getenv("VECTOR_DB_TOKEN")
        user = os.getenv("VECTOR_DB_USER")
        password = os.getenv("VECTOR_DB_PASSWORD")

        if token:
            connection_params["token"] = token
        elif user and password:
            connection_params["user"] = user
            connection_params["password"] = password
    else:
        # Local service configuration - use host/port format
        connection_params = {
            "alias": "default",
            "host": host,
            "port": int(port),
            "db_name": db_name,
        }

        # Add authentication credentials if provided
        user = os.getenv("VECTOR_DB_USER")
        password = os.getenv("VECTOR_DB_PASSWORD")

        if user and password:
            connection_params["user"] = user
            connection_params["password"] = password

    connections.connect(**connection_params)


def initialize_vector_store(collection_name: str) -> Collection:
    """
    Initialize or load vector store collection from Milvus.

    Args:
        collection_name: Name of the collection to initialize.

    Returns:
        Loaded Milvus collection object ready for operations.

    Raises:
        ValueError: If the specified collection does not exist.
    """
    if not utility.has_collection(collection_name):
        raise ValueError(
            f"Collection {collection_name} does not exist. Please create it first."
        )

    collection = Collection(collection_name)
    collection.load()
    return collection


def search_in_milvus(
    collection: Collection, query_vector: List[float], vector_field: str, top_k: int = 1
) -> List[Dict[str, Any]]:
    """
    Search for most similar vectors in Milvus collection using semantic similarity.

    Args:
        collection: Milvus collection object to search in.
        query_vector: Query vector for similarity search.
        vector_field: Name of the vector field to search against.
        top_k: Number of most similar results to return. Defaults to 1.

    Returns:
        List of search results with metadata and similarity scores.
    """
    search_params = {"metric_type": "IP", "params": {"nprobe": 10}}

    # Get all non-vector fields for output (excluding ID and vector fields)
    output_fields = [
        field.name
        for field in collection.schema.fields
        if not field.name.endswith("_vector") and field.name != "id"
    ]

    results = collection.search(
        data=[query_vector],
        anns_field=f"{vector_field}_vector",
        param=search_params,
        limit=top_k,
        output_fields=output_fields,
    )

    return [
        {
            **{field: getattr(hit.entity, field) for field in output_fields},
            "distance": hit.distance,
        }
        for hit in results[0]
    ]