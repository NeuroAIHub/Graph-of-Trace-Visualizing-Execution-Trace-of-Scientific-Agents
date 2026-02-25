from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class ChatAdapter(ABC):
    """
    Abstract base class for chat adapters.

    Each provider-specific adapter should implement the `chat` method and use
    the configuration dictionary passed in the constructor to perform API calls.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    @abstractmethod
    async def chat(self, prompt: str, **kwargs: Any) -> str:
        """
        Send a prompt to the underlying model and return its textual response.

        Provider-specific adapters can accept additional keyword arguments,
        but they should all return a plain string as the main answer.
        """
        raise NotImplementedError


