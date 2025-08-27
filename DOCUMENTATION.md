# FreeDurov: Image Processing & GIF Creation Service

## Overview

FreeDurov is a sophisticated web-based application that automatically splits collage images into individual avatar profiles and creates animated GIFs from them. The service provides both a Flask-based web interface and backend processing capabilities for handling image uploads, intelligent image splitting, and GIF generation.

## Features

### Core Functionality
- **Intelligent Image Splitting**: Automatically detects and separates individual profiles from group photos/collages
- **Profile Recognition**: Uses advanced computer vision algorithms to identify circular and square profile shapes
- **GIF Generation**: Creates animated GIFs from extracted profiles with customizable settings
- **Web Interface**: User-friendly upload and result viewing interface
- **Batch Processing**: Handles multiple profiles and creates multiple GIFs as needed

### Technical Features
- **Dual Server Architecture**: Both Flask and basic HTTP server implementations
- **Security-First Design**: Comprehensive input validation and secure file handling
- **Configurable Processing**: Environment-based configuration for all processing parameters
- **Comprehensive Logging**: Detailed logging for debugging and monitoring
- **Error Recovery**: Robust error handling with graceful degradation

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

2. **Upload an image**
   - Click "Choose File" and select a collage/group photo
   - Supported formats: PNG, JPG, JPEG, GIF, WebP
   - Maximum file size: 16MB

3. **Processing**
   - The application will automatically detect profiles
   - Processing status is shown in real-time
   - Results are displayed when processing completes

4. **Download Results**
   - Generated GIFs are available for download
   - Multiple GIFs may be created for large groups

### API Usage

#### Upload File
```bash
curl -X POST -F "file=@your_image.jpg" http://localhost/
```

#### Check Job Status
```bash
curl http://localhost/job_status/<job_id>
```

## Architecture

### Core Components

#### Web Layer
- **app.py**: Flask application with secure file handling
- **server.py**: Alternative basic HTTP server implementation
- **templates/**: HTML templates for web interface
- **static/**: CSS, JavaScript, and uploaded files

#### Processing Layer
- **image_processor.py**: Main processing coordinator
- **avasplit.py**: Core image splitting and profile detection algorithms
- **gif_maker.py**: GIF creation and animation functionality

#### Utilities
- **config.py**: Configuration management with environment variable support
- **security_utils.py**: Security validation and file handling utilities
- **html_generator.py**: Dynamic HTML generation for results

### Data Flow

1. **File Upload**: Secure validation and storage
2. **Image Analysis**: Edge detection and contour analysis
3. **Profile Detection**: Clustering and shape recognition
4. **Profile Extraction**: Individual profile isolation
5. **GIF Creation**: Animation generation with effects
6. **Result Delivery**: Secure file serving and download

## Development

### Running Tests
```bash
python -m pytest test_application.py -v
```

### Code Quality
The project follows PEP 8 standards and includes:
- Type hints for better code documentation
- Comprehensive error handling
- Security-first design principles
- Detailed logging for debugging

### Adding Features

1. **Image Processing**: Extend `avasplit.py` for new detection algorithms
2. **GIF Effects**: Modify `gif_maker.py` for new animation effects
3. **Web Interface**: Update templates and add new routes in `app.py`
4. **Configuration**: Add new environment variables in `config.py`

## Security

### Implemented Security Measures
- **Input Validation**: Comprehensive file and filename validation
- **Path Traversal Protection**: Secure path joining and validation
- **File Type Validation**: MIME type and content verification
- **Size Limits**: Configurable file size restrictions
- **Secure File Handling**: Safe file operations and cleanup

### Security Configuration
- Change default ports for production deployment
- Use proper SSL certificates (not self-signed) in production
- Configure firewall rules appropriately
- Regular security updates for dependencies

## Troubleshooting

### Common Issues

#### "No profiles detected"
- Ensure the image has clear, distinct profiles
- Try adjusting `MIN_CONTOUR_RATIO` and `MIN_CONTOUR_AREA`
- Check image quality and contrast

#### "File too large" errors
- Reduce image size or adjust `MAX_CONTENT_LENGTH`
- Consider image compression before upload

#### SSL/HTTPS issues
- Verify SSL certificate files exist and are valid
- Check file permissions on certificate files
- Ensure ports 80/443 are available

#### Processing failures
- Check logs in `app.log` for detailed error information
- Verify all dependencies are installed correctly
- Ensure sufficient disk space for processing

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