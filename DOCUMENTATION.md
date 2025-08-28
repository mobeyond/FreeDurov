# FreeDurov: Image Processing & GIF Creation Service

## Overview

FreeDurov is a sophisticated web-based application that automatically splits collage images into individual avatar profiles and creates animated GIFs from them. The service features an advanced two-stage workflow with user selection capabilities, providing both a Flask-based web interface and backend processing capabilities for handling image uploads, intelligent image splitting, and selective GIF generation.

## Features

### Core Functionality
- **Intelligent Image Splitting**: Automatically detects and separates individual profiles from group photos/collages using advanced computer vision
- **Profile Recognition**: Uses sophisticated algorithms to identify circular and square profile shapes with configurable parameters
- **Interactive Selection Interface**: Two-stage workflow allowing users to select which profiles to include in GIF generation
- **Selective GIF Generation**: Creates animated GIFs only from user-selected profiles with customizable settings
- **Real-time Processing Status**: Live updates during image processing and GIF creation phases
- **Secure File Handling**: Comprehensive security validation and safe file operations

### Technical Features
- **Dual Server Architecture**: Both Flask and basic HTTP server implementations with enhanced routing
- **Security-First Design**: Comprehensive input validation, path traversal protection, and secure file handling
- **Modular Processing Pipeline**: Separated profile extraction and GIF creation phases for better user control
- **Configurable Processing**: Environment-based configuration for all processing parameters
- **Comprehensive Logging**: Detailed logging for debugging and monitoring with structured error handling
- **Responsive Web Interface**: Modern UI with real-time updates and interactive elements

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd FreeDurov
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   # On Windows:
   venv\\Scripts\\activate
   # On Linux/Mac:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up directories**
   ```bash
   mkdir -p static/uploads output
   ```

5. **Optional: Configure SSL (for HTTPS)**
   ```bash
   # Generate self-signed certificate (for development)
   openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes
   ```

## Configuration

### Environment Variables

The application supports configuration through environment variables:

#### Directories
- `FREEDUROV_TEMP_DIR`: Temporary upload directory (default: `static/uploads`)
- `FREEDUROV_OUTPUT_DIR`: Output directory for processed files (default: `output`)
- `FREEDUROV_BASE_IMAGE`: Base image for GIF backgrounds (default: `base.png`)

#### Server Settings
- `FREEDUROV_HTTP_PORT`: HTTP server port (default: 80)
- `FREEDUROV_HTTPS_PORT`: HTTPS server port (default: 443)
- `FREEDUROV_SSL_CERT_FILE`: SSL certificate file (default: `cert.pem`)
- `FREEDUROV_SSL_KEY_FILE`: SSL key file (default: `key.pem`)

#### Image Processing
- `FREEDUROV_MIN_CONTOUR_RATIO`: Minimum contour ratio (default: 0.0005)
- `FREEDUROV_MIN_CONTOUR_AREA`: Minimum contour area (default: 225)
- `FREEDUROV_CORNER_RADIUS`: Corner rounding radius (default: 15)

#### GIF Settings
- `FREEDUROV_GIF_DURATION`: Frame duration in milliseconds (default: 1000)
- `FREEDUROV_MAX_PROFILES_PER_GIF`: Max profiles per GIF (default: 100)
- `FREEDUROV_INCLUDE_QR_CODE`: Include QR codes (default: false)
- `FREEDUROV_QR_CODE_URL`: QR code URL (default: "https://douban.com")

#### Application Settings
- `FREEDUROV_DEBUG`: Enable debug mode (default: false)
- `FREEDUROV_LOG_LEVEL`: Logging level (default: "INFO")
- `FREEDUROV_EXPORT_INTERMEDIARY`: Export intermediate images (default: true)

## Usage

### Starting the Application

#### Option 1: Flask Application (Recommended)
```bash
python app.py
```

#### Option 2: Basic HTTP Server
```bash
python server.py
```

### Web Interface

1. **Access the application**
   - HTTP: `http://localhost:80`
   - HTTPS: `https://localhost:443` (if SSL configured)
   - Test Server: `http://localhost:5000` (recommended for development)

2. **Upload an image**
   - Click "Choose File" and select a collage/group photo
   - Supported formats: PNG, JPG, JPEG, GIF, WebP
   - Maximum file size: 16MB
   - Comprehensive security validation performed automatically

3. **Profile Extraction Phase**
   - The application automatically detects and extracts individual profiles
   - Real-time processing status with loading animations
   - Extracted profiles are saved as individual image files

4. **Interactive Selection Phase**
   - View all extracted profiles as selectable thumbnails
   - Use checkboxes to select desired profiles for GIF creation
   - "Select All" and "Select None" buttons for convenience
   - Dynamic selection counter shows number of selected profiles
   - Visual feedback with checkmark overlays and highlighting

5. **GIF Generation Phase**
   - GIFs created only from user-selected profiles
   - Processing status updates in real-time
   - Support for keyboard shortcuts (Ctrl+A to select all)

6. **Download Results**
   - Generated GIFs display with individual download links
   - Secure file serving through dedicated download routes
   - Professional result layout with file information

### API Usage

#### Upload File
```bash
curl -X POST -F "file=@your_image.jpg" http://localhost:5000/
```

#### Check Job Status
```bash
curl http://localhost:5000/job_status/<job_id>
```

#### Profile Selection Workflow
```bash
# 1. Upload returns job_id and redirects to selection interface
# 2. Access selection page
curl http://localhost:5000/selection/<job_id>

# 3. Submit selected profiles
curl -X POST -d "selected_profiles=profile_001.jpg&selected_profiles=profile_003.jpg" \
     http://localhost:5000/confirm_selection/<job_id>

# 4. Check results
curl http://localhost:5000/result/<job_id>
```

#### Secure File Download
```bash
# Download generated GIFs
curl http://localhost:5000/download/<timestamp>/<filename>
```

## Architecture

### Core Components

#### Web Layer
- **app.py**: Flask application with enhanced routing, job management, and secure file handling
- **server.py**: Alternative basic HTTP server implementation for simple deployments
- **templates/**: HTML templates including new selection.html for interactive profile selection
  - **base.html**: Common layout and styling
  - **index.html**: File upload interface
  - **selection.html**: Interactive profile selection interface
  - **result.html**: Enhanced results display with download options
- **static/**: Enhanced CSS, JavaScript, and secure file storage

#### Processing Layer
- **image_processor.py**: Modular processing coordinator with separated extraction and GIF creation phases
- **avasplit.py**: Advanced image splitting and profile detection with clustering algorithms
- **gif_maker.py**: Sophisticated GIF creation with image polishing and animation effects

#### Security & Utilities
- **security_utils.py**: Comprehensive security validation, path traversal protection, and secure file operations
- **config.py**: Environment-based configuration management with extensive parameter support
- **html_generator.py**: Dynamic HTML generation for results (legacy compatibility)

### Enhanced Data Flow

1. **File Upload**: Multi-layer security validation, safe filename generation, and secure storage
2. **Image Analysis**: Advanced edge detection, contour analysis, and shape recognition
3. **Profile Detection**: Machine learning-based clustering and geometric analysis
4. **Profile Extraction**: Individual profile isolation and secure file saving
5. **User Selection**: Interactive interface for profile selection with real-time feedback
6. **Selective GIF Creation**: Animation generation only from user-selected profiles
7. **Secure Result Delivery**: Protected file serving through dedicated download routes

### Job Management System

- **Asynchronous Processing**: Background threads for non-blocking operations
- **Status Tracking**: Real-time job status updates (PROCESSING → PROFILES_EXTRACTED → CREATING_GIFS → COMPLETED)
- **Session Management**: Secure job ID generation and validation
- **Resource Cleanup**: Automatic cleanup of old jobs and temporary files

## Development

### Running Tests
```bash
# Run comprehensive application tests
python -m pytest test_application.py -v

# Run specific test modules
python test_server.py              # Server functionality tests
python test_file_serving.py        # File serving security tests
python test_complete_pipeline.py   # End-to-end pipeline tests
python integration_test.py         # Integration tests

# Run manual testing
python manual_test.py              # Interactive manual testing

# Run all tests with test runner
python run_tests.py                # Comprehensive test suite
```

### Code Quality
The project follows PEP 8 standards and includes:
- Type hints for better code documentation
- Comprehensive error handling and logging
- Security-first design principles with input validation
- Modular architecture with separation of concerns
- Detailed logging for debugging and monitoring
- Automated testing framework with multiple test levels

### Adding Features

1. **Image Processing**: Extend [`avasplit.py`](c:\Local\FreeDurov\avasplit.py) for new detection algorithms
2. **GIF Effects**: Modify [`gif_maker.py`](c:\Local\FreeDurov\gif_maker.py) for new animation effects
3. **Web Interface**: Update templates and add new routes in [`app.py`](c:\Local\FreeDurov\app.py)
4. **Selection Interface**: Enhance [`selection.html`](c:\Local\FreeDurov\templates\selection.html) for new selection features
5. **Security**: Extend [`security_utils.py`](c:\Local\FreeDurov\security_utils.py) for additional validation
6. **Configuration**: Add new environment variables in [`config.py`](c:\Local\FreeDurov\config.py)

### Route Structure

- `GET /` - Main upload page with enhanced UI
- `POST /` - Secure file upload handler with comprehensive validation
- `GET /selection/<job_id>` - **NEW** - Interactive profile selection interface
- `POST /confirm_selection/<job_id>` - **NEW** - Handle user profile selections
- `GET /result/<job_id>` - Enhanced results page with secure file URLs
- `GET /download/<path:filename>` - **NEW** - Secure file download with validation
- `GET /job_status/<job_id>` - AJAX status endpoint for real-time updates
- `GET /static/uploads/<path:filename>` - Legacy compatibility route
- `GET /debug/files` - Development debugging endpoint

## Security

### Implemented Security Measures
- **Input Validation**: Comprehensive file and filename validation using [`security_utils.py`](c:\Local\FreeDurov\security_utils.py)
- **Path Traversal Protection**: Secure path joining and validation to prevent directory traversal attacks
- **File Type Validation**: MIME type and content verification for uploaded files
- **Size Limits**: Configurable file size restrictions (16MB default)
- **Secure File Handling**: Safe file operations, cleanup, and temporary file management
- **Content Validation**: Binary content verification for image files
- **Secure Routing**: Protected download routes with filename validation
- **Job ID Validation**: Regex-based job ID format validation to prevent injection attacks
- **Safe File Serving**: Controlled file serving from designated directories only

### Security Configuration
- Change default ports for production deployment (use [`test_server.py`](c:\Local\FreeDurov\test_server.py) for development)
- Use proper SSL certificates (not self-signed) in production
- Configure firewall rules appropriately
- Regular security updates for dependencies
- Review and update security validation rules in [`security_utils.py`](c:\Local\FreeDurov\security_utils.py)

## Troubleshooting

### Common Issues

#### "No profiles detected"
- Ensure the image has clear, distinct profiles with sufficient contrast
- Try adjusting `MIN_CONTOUR_RATIO` and `MIN_CONTOUR_AREA` in configuration
- Check image quality, resolution, and profile clarity
- Verify profiles have recognizable circular or square shapes

#### "File not found" or download errors
- Check that GIF files were successfully generated in the results phase
- Verify file permissions in the upload directories
- Ensure [`security_utils.py`](c:\Local\FreeDurov\security_utils.py) validation is not blocking legitimate files
- Check browser console for network errors

#### "File too large" errors
- Reduce image size or adjust `MAX_CONTENT_LENGTH` in [`app.py`](c:\Local\FreeDurov\app.py)
- Consider image compression before upload
- Check available disk space for processing

#### SSL/HTTPS issues
- Verify SSL certificate files exist and are valid
- Check file permissions on certificate files
- Ensure ports 80/443 are available and not blocked
- Use [`test_server.py`](c:\Local\FreeDurov\test_server.py) for development (port 5000)

#### Processing failures or selection interface issues
- Check logs for detailed error information
- Verify all dependencies are installed correctly (run `pip install -r requirements.txt`)
- Ensure sufficient disk space for processing and temporary files
- Test with [`manual_test.py`](c:\Local\FreeDurov\manual_test.py) for isolated testing
- Check JavaScript console for frontend errors in selection interface

### Debugging

Enable debug mode for development:
```bash
export FREEDUROV_DEBUG=true
export FREEDUROV_LOG_LEVEL=DEBUG
python app.py
```

## Performance Optimization

### Recommendations
- Use SSD storage for faster I/O operations
- Adjust processing parameters based on typical image sizes
- Consider implementing caching for repeated operations
- Monitor memory usage for large images

### Scaling
- Use a reverse proxy (nginx) for production deployment
- Consider containerization with Docker
- Implement load balancing for multiple instances
- Use external storage for processed files

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Ensure all tests pass
5. Submit a pull request

## License

This project is available under the specified license terms. See LICENSE file for details.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review logs for error details
3. Submit issues with detailed reproduction steps
4. Include environment information and configuration