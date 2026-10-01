"""Deterministic Gmail-compatible tools for the public research artifact.

These functions preserve the original names and Pydantic schemas while avoiding
OAuth credentials, network access, and external side effects.
"""

from typing import Optional

from pydantic import BaseModel, Field


_MESSAGES = [
    {
        "id": "mock-email-001",
        "from": "alice@example.com",
        "to": "researcher@example.com",
        "subject": "Project update",
        "body": "The artifact review is scheduled for Friday.",
        "folder": "inbox",
        "date": "2026-01-15T09:00:00Z",
    },
    {
        "id": "mock-email-002",
        "from": "bob@example.com",
        "to": "researcher@example.com",
        "subject": "Travel receipt",
        "body": "The conference hotel receipt is attached.",
        "folder": "inbox",
        "date": "2026-01-14T16:30:00Z",
    },
]


class GmailListInput(BaseModel):
    sender: Optional[str] = Field(default=None, description="Optional sender email filter.")
    max_results: int = Field(default=5, description="Maximum number of messages to return.")
    folder: Optional[str] = Field(default=None, description="Optional mailbox folder filter.")


class GmailDeleteInput(BaseModel):
    sender: Optional[str] = Field(default=None, description="Optional sender email filter.")
    subject_keyword: Optional[str] = Field(default=None, description="Optional subject filter.")
    folder: Optional[str] = Field(default="inbox", description="Mailbox folder to search.")
    permanent: Optional[bool] = Field(default=False, description="Whether deletion is permanent.")


class GmailSendInput(BaseModel):
    to: Optional[str] = Field(default=None, description="Recipient email address.")
    subject: Optional[str] = Field(default=None, description="Message subject.")
    body: Optional[str] = Field(default=None, description="Message body.")
    cc: Optional[str] = Field(default=None, description="Comma-separated CC recipients.")
    bcc: Optional[str] = Field(default=None, description="Comma-separated BCC recipients.")


class GmailReadInput(BaseModel):
    sender: Optional[str] = Field(default=None, description="Optional sender email filter.")
    max_results: int = Field(default=5, description="Maximum number of messages to return.")


def _matching(sender: Optional[str] = None, folder: Optional[str] = None) -> list[dict]:
    return [
        dict(message)
        for message in _MESSAGES
        if (sender is None or message["from"] == sender)
        and (folder is None or message["folder"] == folder)
    ]


def gmail_list_tool(
    sender: Optional[str] = None,
    max_results: int = 5,
    folder: Optional[str] = None,
) -> list[dict]:
    return _matching(sender, folder)[:max_results]


def gmail_read_tool(sender: Optional[str] = None, max_results: int = 5) -> list[dict]:
    return _matching(sender)[:max_results]


def gmail_send_tool(
    to: Optional[str] = None,
    subject: Optional[str] = None,
    body: Optional[str] = None,
    cc: Optional[str] = None,
    bcc: Optional[str] = None,
) -> str:
    return (
        "Mock email accepted: "
        f"to={to or 'researcher@example.com'}, subject={subject or '(No Subject)'}, "
        f"cc={cc or ''}, bcc={bcc or ''}, body_length={len(body or '')}"
    )


def gmail_delete_tool(
    sender: Optional[str] = None,
    subject_keyword: Optional[str] = None,
    folder: Optional[str] = "inbox",
    permanent: Optional[bool] = False,
) -> str:
    matches = [
        message
        for message in _matching(sender, folder)
        if subject_keyword is None or subject_keyword.lower() in message["subject"].lower()
    ]
    if not matches:
        return "No matching mock emails found to delete."
    operation = "permanently deleted" if permanent else "moved to mock trash"
    return f"Mock email {matches[0]['id']} was {operation}."
