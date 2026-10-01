from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from tool_files.dalle_tool import DallETool
from langchain.tools import Tool, StructuredTool
import wikipedia
from playwright.sync_api import sync_playwright
import os
import random
from PyPDF2 import PdfReader
import numpy as np
import numexpr as ne
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from tool_files.gmail_tools import (
    gmail_list_tool as raw_gmail_list_tool,
    gmail_delete_tool as raw_gmail_delete_tool,
    gmail_send_tool as raw_gmail_send_tool,
    gmail_read_tool as raw_gmail_read_tool,
)
from tool_files.gmail_tools import GmailListInput, GmailDeleteInput, GmailSendInput, GmailReadInput
from tool_files.sql_tool import Database
from tool_files.custom_tools import *
import sys
import hashlib
from typing import Optional, List, Union
import asyncio


def setup_paths():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_directory)
    grandparent_dir = os.path.dirname(parent_dir)

    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    if grandparent_dir not in sys.path:
        sys.path.append(grandparent_dir)

setup_paths()
load_dotenv(override=False)

llm = AzureChatOpenAI(
    deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME"),
    model=os.getenv("AZURE_MODEL_NAME"),
    api_version=os.getenv("AZURE_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_API_KEY"),
        temperature=1,
)

# === Gmail TOOLS ===
async def gmail_list_tool_async(sender: Optional[str] = None, max_results: int = 5, folder: Optional[str] = None) -> list:
    return await asyncio.to_thread(raw_gmail_list_tool, sender=sender, max_results=max_results, folder=folder)

async def gmail_delete_tool_async(
    sender: Optional[str] = None,
    subject_keyword: Optional[str] = None,
    folder: Optional[str] = "inbox",
    permanent: Optional[bool] = False
) -> str:
    return await asyncio.to_thread(raw_gmail_delete_tool, sender=sender, subject_keyword=subject_keyword, folder=folder, permanent=permanent)

async def gmail_send_tool_async(
    to: Optional[str] = None,
    subject: Optional[str] = None,
    body: Optional[str] = None,
    cc: Optional[str] = None,
    bcc: Optional[str] = None
):
    return await asyncio.to_thread(
        raw_gmail_send_tool,
        to=to,
        subject=subject,
        body=body,
        cc=cc,
        bcc=bcc
    )

async def gmail_read_tool_async(sender: Optional[str] = None, max_results: int = 5):
    return await asyncio.to_thread(
        raw_gmail_read_tool,
        sender=sender,
        max_results=max_results
    )
    
gmail_list_tool = StructuredTool(
    func=raw_gmail_list_tool,
    name="gmail_list_tool",
    description="List recent Gmail messages matching a query.",
    args_schema=GmailListInput,
    coroutine=gmail_list_tool_async,
)

gmail_delete_tool = StructuredTool(
    name="gmail_delete_tool",
    description="Delete a Gmail message by its ID.",
    args_schema=GmailDeleteInput,
    func=raw_gmail_delete_tool,
    coroutine=gmail_delete_tool_async,
)

gmail_send_tool = StructuredTool(
    name="gmail_send_tool",
    description="Send an email using Gmail.",
    args_schema=GmailSendInput,
    func=raw_gmail_send_tool,
    coroutine=gmail_send_tool_async,
)

gmail_read_tool = StructuredTool(
    name="gmail_read_tool",
    description="Read a Gmail message by ID or from a thread.",
    args_schema=GmailReadInput,
    func=raw_gmail_read_tool,
    coroutine=gmail_read_tool_async,
)

# === SQL TOOLS ===
async def sql_read_tool_async(query: str) -> str:
    return await asyncio.to_thread(sql_read_sync, query)

sql_read_tool = Tool(
    name="sql_read_tool",
    description="Useful for reading data from a SQL database.",
    args_schema=DBReadQueryInput,
    func=sql_read_sync,
    coroutine=sql_read_tool_async,
)

async def sql_read_write_tool_async(query: str) -> str:
    return await asyncio.to_thread(sql_rw_sync, query)

sql_read_write_tool = Tool(
    name="sql_read_write_tool",
    description="Useful for reading and writing data to a SQL database.",
    args_schema=DBReadWriteQueryInput,
    func = sql_rw_sync,
    coroutine=sql_read_write_tool_async,
)

# === SEARCH TOOL ===
async def search_tool_async(query: str) -> str:
    return await asyncio.to_thread(search_tool_sync, query)

search_tool = Tool(
    name="search_tool",
    description="Useful for searching information on the web.",
    args_schema=SearchInput,
    func = search_tool_sync,
    coroutine=search_tool_async,
)

# ===== Image Generation Tool =====
async def imagegen_tool_async(prompt: str) -> str:
    return await asyncio.to_thread(imagegen_tool_sync, prompt)

imagegen_tool = Tool(
    name="imagegen_tool",
    description="Useful for generating images from text.",
    args_schema=ImagegenInput,
    func = imagegen_tool_sync,
    coroutine=imagegen_tool_async,
)

# ===== Calculator Tool =====
async def calculator_tool_async(expression: str) -> str:
    return await asyncio.to_thread(calculator_tool_sync, expression)

calculator_tool = Tool(
    name="calculator_tool",
    description="Useful for evaluating mathematical expressions.",
    args_schema=CalculatorInput,
    func = calculator_tool_sync,
    coroutine=calculator_tool_async,
)

# ===== Wikipedia Tool =====
async def wiki_tool_async(query: str) -> str:
    return await asyncio.to_thread(wiki_tool_sync, query)

wiki_tool = Tool(
    name="wiki_tool",
    description="Useful for scraping content from Wikipedia.",
    args_schema=WikiInput,
    func = wiki_tool_sync,
    coroutine=wiki_tool_async,
)

# ===== Random Number Generator Tool =====
async def generate_random_async(random_type: str, params: List[float]) -> float:
    return await asyncio.to_thread(generate_random_sync, random_type, params)

random_tool = StructuredTool(
    name="random_tool",
    description="Generates a random number using uniform, gaussian, or exponential distribution.",
    args_schema=RandomInput,
    func=generate_random_sync,
    coroutine=generate_random_async,
)

# ===== Hash Generator Tool =====
async def hash_string_async(text: str, algorithm: str = "sha256") -> str:
    return await asyncio.to_thread(hash_string_sync, text, algorithm)

hash_tool = StructuredTool(
    name="hash_tool",
    description="Computes a cryptographic hash (SHA256, MD5, SHA1) of a string.",
    args_schema=HashInput,
    func=hash_string_sync,
    coroutine=hash_string_async,
)

# ===== URL Screenshot Tool (Playwright) =====
async def screenshot_async(url: str = "https://arxiv.org/abs/2505.06934", output_file: str = "screenshot.png") -> str:
    """
    Async wrapper for Playwright screenshot function.
    """
    return await asyncio.to_thread(screenshot_sync, url, output_file)

screenshot_tool = StructuredTool(
    name="screenshot_tool",
    description="Takes a full-page screenshot of a webpage using a headless browser. Can be called with no url address (has default option)",
    args_schema=ScreenshotInput,
    func=screenshot_sync,
    coroutine=screenshot_async,
)

# ===== PDF Metadata Tool =====
async def pdf_metadata_async(file_path: str) -> dict:
    return await asyncio.to_thread(pdf_metadata_sync, file_path)

pdf_metadata_tool = StructuredTool(
    name="pdf_metadata_tool",
    description="Extracts title, author, and page count from a PDF file without reading the full text.",
    args_schema=PDFMetaInput,
    func=pdf_metadata_sync,
    coroutine=pdf_metadata_async,
)

# ===== PDF Summary Tool =====
async def pdf_summary_async(file_path: str, max_chars: int = 500) -> dict:
    return await asyncio.to_thread(pdf_summary_sync, file_path, max_chars)

pdf_summary_tool = StructuredTool(
    name="pdf_summary_tool",
    description="Extracts a short preview text from the first page of a PDF, useful for LLM summarization.",
    args_schema=PDFSummaryInput,
    func=pdf_summary_sync,
    coroutine=pdf_summary_async,
)
