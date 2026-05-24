"""Compatibility wrapper for the new API entrypoint.

Prefer running: uvicorn api.app:app
"""

from api.app import app
