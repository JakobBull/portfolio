import pytest
from backend.rate_limiter import execute_function, no_rate_limit

class TestSimplifiedRateLimiter:
    """Test the simplified rate limiter functions"""
    
    def test_execute_function(self):
        """Test the execute_function function"""
        # Create a simple test function
        counter = 0
        def test_func(increment=1):
            nonlocal counter
            counter += increment
            return counter
        
        # Execute the function multiple times
        result1 = execute_function(test_func)
        assert result1 == 1
        
        result2 = execute_function(test_func, 2)
        assert result2 == 3
        
        result3 = execute_function(test_func, increment=5)
        assert result3 == 8
        
        # Check that all executions were successful
        assert counter == 8
    
    def test_decorator(self):
        """Test the no_rate_limit decorator"""
        # Create a decorated function
        @no_rate_limit
        def test_func():
            return "success"
        
        # Call the decorated function
        result = test_func()
        assert result == "success"
        
        # Test with arguments
        @no_rate_limit
        def test_func_with_args(arg1, arg2=None):
            return f"{arg1}-{arg2}"
        
        result = test_func_with_args("hello", arg2="world")
        assert result == "hello-world"

# Create a compatibility test to ensure the market_interface.py file works with our changes
def test_compatibility_with_market_interface():
    """Test compatibility with market_interface module"""
    from backend.rate_limiter import no_rate_limit
    
    # Create a function that simulates the scheduled decorator in market_interface
    def scheduled(func):
        @no_rate_limit
        def wrapper(*args, **kwargs):
            return execute_function(func, *args, **kwargs)
        return wrapper
    
    # Test the scheduled decorator
    @scheduled
    def test_func(arg1, arg2=None):
        return f"{arg1}-{arg2}"
    
    result = test_func("hello", arg2="world")
    assert result == "hello-world" 