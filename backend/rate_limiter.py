import logging
from typing import Callable, Any
from functools import wraps

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def execute_function(func: Callable, *args: Any, **kwargs: Any) -> Any:
    """
    Simple pass-through function that executes the given function with its arguments
    
    Args:
        func: Function to execute
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
            
    Returns:
        Result of the function
    """
    return func(*args, **kwargs)

def no_rate_limit(func: Callable) -> Callable:
    """
    Decorator that simply passes through to the original function
    without any rate limiting
    
    Args:
        func: Function to decorate
            
    Returns:
        The original function unchanged
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)
    return wrapper 