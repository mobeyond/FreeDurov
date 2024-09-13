import socketserver
from server import MyHandler
import config
import logging
import sys

def setup_logging():
    logging.basicConfig(
        filename=config.LOG_FILE,
        level=getattr(logging, config.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    # Set up argument parsing
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
            config.PORT = port
        except ValueError:
            print(f"Invalid port number: {sys.argv[1]}. Using default port {config.PORT}")

    setup_logging()
    print(f"Starting server on {config.HOST}:{config.PORT}")
    print(f"QR Code embedding is {'enabled' if config.INCLUDE_QR_CODE else 'disabled'}")
    
    try:
        with socketserver.TCPServer((config.HOST, config.PORT), MyHandler) as httpd:
            print(f"Server running on port {config.PORT}")
            httpd.serve_forever()
    except OSError as e:
        if e.errno == 98:  # Address already in use
            print(f"Error: Port {config.PORT} is already in use. Please choose a different port.")
        else:
            print(f"Error starting server: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()