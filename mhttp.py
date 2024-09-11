import http.server
import socketserver
import cgi
import base64
import io
import os
import glob
from urllib.parse import parse_qs

# Import the image processing function
from avas7 import process_image_for_web

PORT = 9000

class MyHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        
        html = '''
        <html>
        <body>
            <h1>Image Upload and Processing</h1>
            <form enctype="multipart/form-data" method="post">
                <input type="file" name="imageFile" accept="image/*">
                <input type="submit" value="Upload and Process">
            </form>
        </body>
        </html>
        '''
        self.wfile.write(html.encode())

    def do_POST(self):
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={'REQUEST_METHOD': 'POST',
                     'CONTENT_TYPE': self.headers['Content-Type'],
                     })

        image_item = form['imageFile']
        image_data = image_item.file.read()

        # Process the image
        gif_files = process_image_for_web(image_data)
        print(f"GIF files returned: {gif_files}")

        # Look for additional GIF files in the current directory
        additional_gif_files = glob.glob('*.gif')
        print(f"Additional GIF files found: {additional_gif_files}")

        # Combine both lists of GIF files
        all_gif_files = gif_files + additional_gif_files
        print(f"All GIF files: {all_gif_files}")

        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

        html = f'''
        <html>
        <body>
            <h1>Processing Results</h1>
            <h2>Generated GIFs:</h2>
        '''

        for gif_file in all_gif_files:
            try:
                if isinstance(gif_file, str):  # It's a filename
                    with open(gif_file, 'rb') as f:
                        gif_content = f.read()
                else:  # It's already file content
                    gif_content = gif_file
                gif_base64 = base64.b64encode(gif_content).decode('utf-8')
                html += f'<img src="data:image/gif;base64,{gif_base64}" alt="Generated GIF"><br>'
            except Exception as e:
                print(f"Error processing GIF file {gif_file}: {str(e)}")

        html += '''
        <br>
        <a href="/">Back to Upload</a>
        </body>
        </html>
        '''

        self.wfile.write(html.encode())

        # Clean up GIF files after sending the response
        for gif_file in all_gif_files:
            if isinstance(gif_file, str):
                try:
                    os.remove(gif_file)
#                    print(f"Removed {gif_file}")
                except Exception as e:
                    print(f"Error removing {gif_file}: {str(e)}")

with socketserver.TCPServer(("0.0.0.0", PORT), MyHandler) as httpd:
    print(f"Server running on port {PORT}")
    httpd.serve_forever()