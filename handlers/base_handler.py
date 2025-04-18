from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

class Handler(ABC):
    def __init__(self):
        self._next_handler: Optional['Handler'] =  None

    def set_next(self, handler: 'Handler') -> 'Handler':
        """
        handler1.set_next(handler2).set_next(handler3)
        """
        self._next_handler = handler
        return handler
    
    def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        The default chaining behavior. Calls the next handler if it exists.
        Subclasses will override this method with their specific processing logic.
        """
        if self._next_handler:
            return self._next_handler.handle(request)
        return request
    
    @abstractmethod
    def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        This method must be implemented by all concrete handlers.
        It contains the actual processing logic.
        """
        pass