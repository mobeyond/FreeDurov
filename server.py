import io, traceback, http.server, ssl, threading, logging, os, tempfile
import config
from image_processor import process_avasplit
from html_generator import generate_html_response, generate_no_gifs_html
from email.parser import BytesParser
from email.policy import default

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MyHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        logging.info("Handling GET request")
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        
        html = '''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Group Goes GIF</title>
        </head>
        <body>
            <div class="container">
                <h1>Group Goes GIF</h1>
                <form id="upload-form" enctype="multipart/form-data" method="post">
                    <input type="file" id="file-input" name="imageFile" accept="image/*" required>
                    <div id="file-name">No file chosen</div>
                    <button type="submit" id="submit-btn">Convert to GIF</button>
                </form>
            </div>
        </body>
        </html>
        '''
        self.wfile.write(html.encode())

    def do_POST(self):
        logging.info("Handling POST request")
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        logging.info(f"Received POST data of length: {content_length}")

        try:
            # Parse multipart form data using email parser
            content_type = self.headers.get('Content-Type', '')
            if 'multipart/form-data' not in content_type:
                raise ValueError("Invalid content type")
            
            # Create a proper email message from the post data
            env_data = f"Content-Type: {content_type}\r\n\r\n".encode() + post_data
            parser = BytesParser(policy=default)
            msg = parser.parsebytes(env_data)
            
            # Extract image file from form data
            image_data = None
            for part in msg.iter_parts():
                if part.get_content_disposition() == 'form-data':
                    name = part.get_param('name', header='Content-Disposition')
                    if name == 'imageFile':
                        image_data = part.get_payload(decode=True)
                        break
            
            if not image_data:
                raise ValueError("No image data received")

            # Save image to temporary file for processing
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
                temp_file.write(image_data)
                temp_image_path = temp_file.name
            
            # Create output directory
            output_dir = tempfile.mkdtemp()
            
            try:
                gif_files = process_avasplit(temp_image_path, output_dir)
            finally:
                # Clean up temporary image file
                os.unlink(temp_image_path)
            
            if not gif_files:
                logging.warning("No GIF files generated. Sending disclaimer response.")
                html_response = generate_no_gifs_html()
            else:
                html_response = generate_html_response(gif_files, self.server.server_port)

            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(html_response.encode())

        except Exception as e:
            logging.error(f"Error in do_POST: {str(e)}")
            traceback.print_exc()
            error_html = self.generate_error_html(f"An error occurred: {str(e)}")
            self.send_response(500)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(error_html.encode())

    def generate_error_html(self, error_message):
        return f'''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Error - GROUP GOES GIF</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    margin: 0;
                    background-color: #f0f0f0;
                }}
                .error-container {{
                    background-color: white;
                    padding: 2rem;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    text-align: center;
                }}
                h1 {{
                    color: #d32f2f;
                }}
                .back-link {{
                    display: inline-block;
                    margin-top: 1rem;
                    color: #4CAF50;
                    text-decoration: none;
                    font-weight: bold;
                }}
            </style>
        </head>
        <body>
            <div class="error-container">
                <h1>Error</h1>
                <p>{error_message}</p>
                <a href="/" class="back-link">Back to Upload</a>
            </div>
        </body>
        </html>
        '''

def run_server(port, use_https=False):
    server_address = ('', port)
    httpd = http.server.HTTPServer(server_address, MyHandler)
    
    if use_https:
        try:
            # Check if SSL files exist before loading
            if not os.path.exists(config.SSL_CERT_FILE) or not os.path.exists(config.SSL_KEY_FILE):
                logging.warning(f"SSL certificate files not found: {config.SSL_CERT_FILE}, {config.SSL_KEY_FILE}")
                logging.info("Continuing with HTTP only")
                use_https = False
            else:
                context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                context.load_cert_chain(certfile=config.SSL_CERT_FILE, keyfile=config.SSL_KEY_FILE)
                httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
                logging.info(f"Serving HTTPS on port {port}")
        except Exception as e:
            logging.error(f"Failed to initialize HTTPS: {str(e)}")
            logging.info("Continuing with HTTP only")
            use_https = False
    
    if not use_https:
        logging.info(f"Serving HTTP on port {port}")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logging.info("Server is shutting down.")
    finally:
        httpd.server_close()
        logging.info("Server closed.")

if __name__ == "__main__":
    # Ensure upload directories exist
    os.makedirs(config.TEMP_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Only start HTTP server if SSL files don't exist
    if os.path.exists(config.SSL_CERT_FILE) and os.path.exists(config.SSL_KEY_FILE):
        http_thread = threading.Thread(target=run_server, args=(config.HTTP_PORT,))
        https_thread = threading.Thread(target=run_server, args=(config.HTTPS_PORT, True))
        
        http_thread.start()
        https_thread.start()
        
        http_thread.join()
        https_thread.join()
    else:
        logging.info("SSL certificates not found, starting HTTP server only")
        run_server(config.HTTP_PORT, False)