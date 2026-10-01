import os
from dotenv import load_dotenv
from fastmcp import FastMCP
from langchain_community.utilities import GoogleSerperAPIWrapper
import wikipedia
import asyncio
load_dotenv(override=False)

search_mcp = FastMCP("basic_MCP_tools")
@search_mcp.tool()
def web_search(query: str, location: str = "United Kingdom", hl: str = "en") -> str:
    """Useful for searching information on the web."""
    search_class = GoogleSerperAPIWrapper()
    result = search_class.run(query)
    return result

wiki_mcp = FastMCP("wiki_MCP_tools")
@wiki_mcp.tool()
def wiki_scrape(query: str) -> str:
    """Useful for scraping content from wikipedia."""
    try:
        tool_result = wikipedia.summary(query, sentences=5)
    except:
        tool_result = "No relevant Wikipedia summary found."
    return tool_result

# Start the MCP server
main_mcp = FastMCP("main_MCP_server")
async def setup():
    await main_mcp.import_server(search_mcp, prefix="search")
    await main_mcp.import_server(wiki_mcp, prefix="wiki")
if __name__ == "__main__":
    # Initialize and run the server
    asyncio.run(setup())
    main_mcp.run(transport='stdio')