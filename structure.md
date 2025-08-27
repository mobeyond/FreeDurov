# FreeDurov: Image Splitting & GIF Conversion Service

## Overview
FreeDurov is a web-based application that splits collage images into individual components and combines them into animated GIF files. The service provides both a web interface and backend processing capabilities for handling image uploads, splitting, and GIF generation.

## Key Features
- Image splitting functionality to separate collage components
- GIF creation from split image components
- Web-based user interface
- File upload processing
- Configurable output parameters

## Technical Architecture

### Core Components
- <mcfile name="avasplit.py" path="c:\Local\FreeDurov\avasplit.py"></mcfile> - Image splitting logic
- <mcfile name="gif_maker.py" path="c:\Local\FreeDurov\gif_maker.py"></mcfile> - GIF generation functionality
- <mcfile name="image_processor.py" path="c:\Local\FreeDurov\image_processor.py"></mcfile> - Image processing utilities
- <mcfile name="server.py" path="c:\Local\FreeDurov\server.py"></mcfile> - Web server implementation
- <mcfile name="app.py" path="c:\Local\FreeDurov\app.py"></mcfile> - Application entry point

### Web Interface
- <mcfolder name="templates" path="c:\Local\FreeDurov\templates"></mcfolder> - HTML templates
  - <mcfile name="index.html" path="c:\Local\FreeDurov\templates\index.html"></mcfile> - Main upload page
  - <mcfile name="result.html" path="c:\Local\FreeDurov\templates\result.html"></mcfile> - Results display page
- <mcfolder name="static" path="c:\Local\FreeDurov\static"></mcfolder> - Web assets
  - <mcfile name="style.css" path="c:\Local\FreeDurov\static\css\style.css"></mcfile> - Styling
  - <mcfile name="script.js" path="c:\Local\FreeDurov\static\js\script.js"></mcfile> - Client-side functionality

### Configuration
- <mcfile name="config.py" path="c:\Local\FreeDurov\config.py"></mcfile> - Application settings

### File Storage
- <mcfolder name="static/uploads" path="c:\Local\FreeDurov\static\uploads"></mcfolder> - Processed files storage

## Installation
1. Clone the repository
2. Install required dependencies (not specified in current project files)
3. Configure settings in <mcfile name="config.py" path="c:\Local\FreeDurov\config.py"></mcfile>

## Usage
1. Start the server: `python server.py`
2. Access the web interface through your browser
3. Upload an image collage
4. Configure splitting and GIF parameters
5. Generate and download the resulting GIF

## Technical Details
- Image splitting is handled by <mcfile name="avasplit.py" path="c:\Local\FreeDurov\avasplit.py"></mcfile>
- GIF creation is implemented in <mcfile name="gif_maker.py" path="c:\Local\FreeDurov\gif_maker.py"></mcfile>
- Web server uses templates from <mcfolder name="templates" path="c:\Local\FreeDurov\templates"></mcfolder> to render pages
- Uploaded files are processed and stored in the uploads directory