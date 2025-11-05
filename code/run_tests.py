#!/usr/bin/env python3
"""Run tests for the signal research project"""

import pytest
import sys
import os

def main():
    # Add code directory to Python path
    code_dir = os.path.join(os.path.dirname(__file__), 'code')
    sys.path.insert(0, code_dir)
    
    # Run tests
    pytest_args = [
        '-v',
        '--tb=short',
        os.path.join(code_dir, 'tests'),
        os.path.join(code_dir, 'signals', 'tests')
    ]
    
    return pytest.main(pytest_args)

if __name__ == '__main__':
    sys.exit(main())