import os
from fastmcp import FastMCP
from sqlite3 import connect

mcp_database = FastMCP("mcp_sql_server")

def connect_db(path: str):
    """
    Connects to the SQLite database at the specified path.
    Returns a connection object.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Database file '{path}' does not exist.")
    else:
        print(f"Connecting to database at {path}")
        try:
            conn = connect(path)
            print("Connection successful.")
        except Exception as e:
            raise Exception(f"Failed to connect to the database: {e}")
    return conn

@mcp_database.tool()
def execute_read_query(query: str, path:str) -> str:
    """
    Executes a SELECT SQL query and returns the result.
    Returns an error if the query is not a SELECT query or if the query fails.
    """
    allowed_dml_types = ["select"]
    query_type = query.strip().split()[0].lower()
    if query_type not in allowed_dml_types:
        raise ValueError(
            f"Query type '{query_type}' is not allowed. Only {allowed_dml_types} are permitted."
        )
    try:
        conn = connect_db(path)
        cursor = conn.cursor()
        cursor.execute(query)
        result = cursor.fetchall()
        conn.close()
        return str(result)
    except Exception as e:
        raise RuntimeError(f"An error occurred while executing the query: {e}")

@mcp_database.tool()
def execute_read_write_query(query: str, path: str) -> str:
    """
    Executes a SQL query that can read (SELECT) or modify the database (INSERT, UPDATE, DELETE, etc.)
    and returns a success message or an error if the query fails.
    """
    allowed_dml_types = [
        "select",
        "insert",
        "update",
        "delete",
        "create",
        "drop",
        "alter",
        "truncate",
        "replace",
    ]
    query_type = query.strip().split()[0].lower()
    if query_type not in allowed_dml_types:
        raise ValueError(
            f"Query type '{query_type}' is not allowed. Only {allowed_dml_types} are permitted."
        )
    try:
        conn = connect_db(path)
        cursor = conn.cursor()
        cursor.execute(query)
        conn.commit()
        conn.close()
        return "Query executed successfully."
    except Exception as e:
        raise RuntimeError(f"An error occurred while executing the query: {e}")



if __name__ == "__main__":
    mcp_database.run()
