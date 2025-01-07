# utils/__init__.py

"""Utils package initialization."""

from .time_manager import time_manager  # Import time_manager first
# Remove the logger import to prevent circular dependencies
# Let other modules import logger directly when needed

__all__ = ['time_manager']