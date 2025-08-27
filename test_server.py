#!/usr/bin/env python3
"""
Test server for FreeDurov application.
Runs the Flask app on port 5000 for testing purposes.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logger.info("Starting FreeDurov test server on port 5000")
    logger.info("Open http://localhost:5000 in your browser")
    
    # Run in debug mode on port 5000 for testing
    app.run(host='127.0.0.1', port=5000, debug=True, threaded=True)