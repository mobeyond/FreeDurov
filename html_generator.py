import base64
import config
import time
from PIL import Image
import os, io

def generate_html_response(gif_files, port):
    protocol = "https" if port == config.HTTPS_PORT else "http"
    #base_url = f"{protocol}://{config.DOMAIN_NAME}"
    
    html = f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{config.HTML_TITLE}</title>
        <style>
            * {{
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }}
            html, body {{
                height: 100%;
                width: 100%;
                overflow-x: hidden;
            }}
            body {{
                font-family: Arial, sans-serif;
                background-color: {config.HTML_BACKGROUND_COLOR};
                color: #333;
                padding-top: 60px; /* Height of the header */
            }}
            header {{
                background-color: {config.HTML_HEADER_COLOR};
                color: white;
                text-align: center;
                padding: 1rem;
                width: 100%;
                position: fixed;-*
                top: 0;
                left: 0;
                height: 60px; /* Fixed height for header */
                z-index: 1000;
            }}
            h1 {{
                margin: 0;
                font-size: 24px;
                line-height: 28px; /* Center text vertically in header */
            }}
            .content {{
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
            }}
            .gif-column {{
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 20px;
            }}
            .gif-item {{
                width: 100%;
                max-width: 600px;
            }}
            .gif-item img {{
                width: 100%;
                height: auto;
                display: block;
                border: 1px solid #ddd;
                border-radius: 4px;
            }}
            .back-link {{
                display: block;
                text-align: center;
                margin-top: 20px;
                color: {config.HTML_HEADER_COLOR};
                text-decoration: none;
                font-weight: bold;
            }}
            .download-link {{
                display: block;
                margin-top: 10px;
                text-align: center;
                color: #007bff;
                text-decoration: none;
            }}
        </style>
    </head>
    <body>
        <header>
            <h1>{config.HTML_TITLE}</h1>
        </header>
        <div class="content">
            <div class="gif-column">
    '''

    # Add base image as the first item
    base_image_path = config.BASE_IMAGE_PATH
    if os.path.exists(base_image_path):
        with Image.open(base_image_path) as img:
            # Resize base image to match the size of the first GIF
            if gif_files:
                with Image.open(gif_files[0]) as first_gif:
                    img = img.resize(first_gif.size, Image.LANCZOS)
            
            # Convert to base64
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            encoded_string = base64.b64encode(buffered.getvalue()).decode()
        
        html += f'''
                <div class="gif-item">
                    <img src="data:image/png;base64,{encoded_string}" alt="Base Image">
                    <a href="data:image/png;base64,{encoded_string}" download="base_image.png" class="download-link">Download Base Image</a>
                </div>
        '''

    # Add GIF files
    for i, gif_file in enumerate(gif_files):
        with open(gif_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        html += f'''
                <div class="gif-item">
                    <img src="data:image/gif;base64,{encoded_string}" alt="Generated GIF">
                    <a href="data:image/gif;base64,{encoded_string}" download="generated_gif_{i+1}.gif" class="download-link">Download GIF {i+1}</a>
                </div>
        '''

    html += f'''
            </div>
            <a href="/" class="back-link">Back to Upload</a>
        </div>
    </body>
    </html>
    '''
    
    # Save the generated HTML to a log file
    #log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    #os.makedirs(log_dir, exist_ok=True)
    #timestamp = time.strftime("%Y%m%d-%H%M%S")
    #log_file_path = os.path.join(log_dir, f'generated_html_{timestamp}.html')
    #with open(log_file_path, 'w', encoding='utf-8') as log_file:
    #   log_file.write(html)
    
    #print(f"Generated HTML saved to: {log_file_path}")

    return html

def get_gif_base64(gif_file):
    with open(gif_file, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def generate_no_gifs_html():
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>No GIFs Generated - GROUP GOES GIF</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
                background-color: #f0f0f0;
            }
            .message-container {
                background-color: white;
                padding: 2rem;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                text-align: center;
            }
            h1 {
                color: #4CAF50;
            }
            .back-link {
                display: inline-block;
                margin-top: 1rem;
                color: #4CAF50;
                text-decoration: none;
                font-weight: bold;
            }
        </style>
    </head>
    <body>
        <div class="message-container">
            <h1>No GIFs Generated</h1>
            <p>We couldn't detect available profiles in the uploaded image.</p>
            <p>Please try uploading a different image with clear profiles.</p>
            <a href="/" class="back-link">Back to Upload</a>
        </div>
    </body>
    </html>
    '''