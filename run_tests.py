#!/usr/bin/env python3
"""
Test runner for the portfolio management system.
Run this script to execute all tests using pytest.
"""

import pytest
import sys
import os

# Add the project root directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_tests():
    """Run all tests in the tests directory using pytest"""
    # Run the tests with pytest
    exit_code = pytest.main([
        'tests',                  # Test directory
        '-v',                     # Verbose output
        '--tb=short',             # Shorter traceback format
        '--no-header',            # No header
        '--no-summary',           # No summary
        '--showlocals',           # Show local variables in tracebacks
        '-xvs',                   # Exit on first failure, verbose, no capture
    ])
    
    return exit_code

if __name__ == '__main__':
    # Run the tests
    exit_code = run_tests()
    
    # Exit with the pytest exit code
    sys.exit(exit_code) 