import os
import re
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory, abort
from werkzeug.utils import secure_filename
from datetime import datetime
import threading
import time
import logging
from collections import defaultdict
from image_processor import process_avasplit
import ssl
import config
from security_utils import (
    validate_filename, 
    validate_file_content, 
    generate_safe_filename, 
    secure_path_join,
    safe_file_removal
)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Create upload directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
if isinstance(config.OUTPUT_DIR, str):
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
MAX_FILE_SIZE_MB = 16

# Security configuration
ALLOWED_BASE_DIRS = [app.config['UPLOAD_FOLDER'], config.OUTPUT_DIR]

job_data = defaultdict(lambda: {
    'status': 'PENDING', 
    'output_files': [], 
    'profile_filenames': [],
    'selected_profiles': [],
    'created_at': time.time()
})

# Clean up old job data periodically
def cleanup_old_jobs():
    """Remove job data older than 1 hour."""
    current_time = time.time()
    old_jobs = [job_id for job_id, data in job_data.items() 
                if current_time - data.get('created_at', 0) > 3600]
    for job_id in old_jobs:
        del job_data[job_id]
    logger.info(f"Cleaned up {len(old_jobs)} old job records")

def allowed_file(filename):
    """Check if filename has an allowed extension and passes security validation."""
    if not filename:
        return False
    
    # Basic extension check
    has_valid_ext = '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
    # Security validation
    is_secure = validate_filename(filename)
    
    return has_valid_ext and is_secure

def validate_uploaded_file(file):
    """Comprehensive validation of uploaded file."""
    if not file or not file.filename:
        return False, "No file selected"
    
    if not allowed_file(file.filename):
        return False, "Invalid file type or unsafe filename"
    
    # Check file size (this is also handled by Flask's MAX_CONTENT_LENGTH)
    if hasattr(file, 'content_length') and file.content_length:
        if file.content_length > MAX_FILE_SIZE_MB * 1024 * 1024:
            return False, f"File too large (max {MAX_FILE_SIZE_MB}MB)"
    
    return True, "File is valid"

def image_processor(input_path, output_dir, job_id):
    try:
        time.sleep(5)

        # Extract profiles without creating GIFs yet
        profile_regions, profile_filenames = process_avasplit(input_path, output_dir)
        
        if profile_filenames:
            # Store extracted profile information for selection
            job_data[job_id]['profile_filenames'] = profile_filenames
            job_data[job_id]['status'] = 'PROFILES_EXTRACTED'
        else:
            job_data[job_id]['status'] = 'FAILED'
            logger.error(f"No profiles extracted for job {job_id}")
            
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        job_data[job_id]['status'] = 'FAILED'

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Clean up old jobs periodically
        cleanup_old_jobs()
        
        if 'file' not in request.files:
            logger.warning("No file part in request")
            return render_template('index.html', error="No file selected")
            
        file = request.files['file']
        
        # Validate the uploaded file
        is_valid, error_message = validate_uploaded_file(file)
        if not is_valid:
            logger.warning(f"File validation failed: {error_message}")
            return render_template('index.html', error=error_message)
        
        try:
            # Generate safe filename
            original_filename = file.filename
            if not original_filename:
                return render_template('index.html', error="Invalid filename")
                
            safe_filename = generate_safe_filename(original_filename, unique=True)
            
            # Create timestamped job ID and directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            job_id = f"{timestamp}_{safe_filename}"
            
            # Create secure upload directory
            upload_dir = secure_path_join(app.config['UPLOAD_FOLDER'], timestamp)
            os.makedirs(upload_dir, exist_ok=True)
            
            # Save file securely
            input_path = secure_path_join(upload_dir, safe_filename)
            file.save(input_path)
            
            # Validate the saved file content
            if not validate_file_content(input_path):
                safe_file_removal(input_path, ALLOWED_BASE_DIRS)
                logger.warning(f"Invalid file content: {original_filename}")
                return render_template('index.html', error="Invalid or corrupted image file")
            
            # Initialize job data
            job_data[job_id]['status'] = 'PROCESSING'
            job_data[job_id]['created_at'] = time.time()
            
            logger.info(f"Processing file: {original_filename} -> {safe_filename}")
            
            # Start processing in background thread
            threading.Thread(
                target=image_processor, 
                args=(input_path, upload_dir, job_id),
                daemon=True
            ).start()

            return redirect(url_for('selection', job_id=job_id))
            
        except ValueError as ve:
            logger.error(f"Security violation in upload: {str(ve)}")
            return render_template('index.html', error="Security error: Invalid file path")
        except Exception as e:
            logger.error(f"Error processing upload: {str(e)}")
            return render_template('index.html', error="An error occurred while processing your upload")
            
    return render_template('index.html')

@app.route('/result/<job_id>')
def result(job_id):
    # Validate job_id format for security
    if not re.match(r'^[0-9]{8}_[0-9]{6}_[a-zA-Z0-9_.-]+$', job_id):
        logger.warning(f"Invalid job_id format: {job_id}")
        abort(404)
    
    if job_id not in job_data:
        logger.warning(f"Job not found: {job_id}")
        abort(404)
    
    job_info = job_data[job_id]
    status = job_info['status']
    output_files = job_info['output_files']
    
    logger.info(f"Result page for job {job_id}, status: {status}, files: {len(output_files)}")
    
    # Validate and create file URLs for template
    safe_file_urls = []
    for path in output_files:
        try:
            # Ensure path is within allowed directories
            abs_path = os.path.abspath(path)
            if any(abs_path.startswith(os.path.abspath(base_dir)) for base_dir in ALLOWED_BASE_DIRS):
                # Create relative path from upload folder for the download route
                rel_path = os.path.relpath(path, app.config['UPLOAD_FOLDER'])
                # Normalize path separators for URLs (Windows uses backslashes)
                rel_path_url = rel_path.replace('\\', '/').replace('\\', '/')
                
                # Use our custom download route instead of static route
                file_url = url_for('download_file', filename=rel_path_url)
                safe_file_urls.append({
                    'url': file_url,
                    'filename': os.path.basename(path)
                })
                logger.info(f"Generated file URL: {file_url} for {path}")
            else:
                logger.warning(f"Unsafe output file path: {path}")
        except Exception as e:
            logger.error(f"Error processing output path {path}: {e}")
    
    logger.info(f"Generated {len(safe_file_urls)} file URLs for template")
    
    return render_template('result.html', 
                         job_id=job_id, 
                         status=status, 
                         output_files=safe_file_urls)

@app.route('/selection/<job_id>')
def selection(job_id):
    """Display profile selection page."""
    # Validate job_id format for security
    if not re.match(r'^[0-9]{8}_[0-9]{6}_[a-zA-Z0-9_.-]+$', job_id):
        logger.warning(f"Invalid job_id format: {job_id}")
        abort(404)
    
    if job_id not in job_data:
        logger.warning(f"Job not found: {job_id}")
        abort(404)
    
    job_info = job_data[job_id]
    status = job_info['status']
    
    logger.info(f"Selection page for job {job_id}, status: {status}")
    
    if status == 'PROFILES_EXTRACTED':
        profile_filenames = job_info.get('profile_filenames', [])
        
        # Create URLs for profile images
        profile_urls = []
        for filename in profile_filenames:
            try:
                # Extract timestamp from job_id to build relative path
                timestamp = job_id.split('_')[0] + '_' + job_id.split('_')[1]
                rel_path = f"{timestamp}/{filename}"
                rel_path_url = rel_path.replace('\\', '/')
                
                file_url = url_for('download_file', filename=rel_path_url)
                profile_urls.append({
                    'url': file_url,
                    'filename': filename,
                    'display_name': filename.replace('.jpg', '').replace('profile_', 'Profile ')
                })
                logger.info(f"Generated profile URL: {file_url} for {filename}")
            except Exception as e:
                logger.error(f"Error processing profile {filename}: {e}")
        
        return render_template('selection.html', 
                             job_id=job_id, 
                             profiles=profile_urls)
    
    elif status == 'PROCESSING':
        return render_template('selection.html', 
                             job_id=job_id, 
                             processing=True)
    else:
        # Redirect to result if not in profile extraction phase
        return redirect(url_for('result', job_id=job_id))

@app.route('/confirm_selection/<job_id>', methods=['POST'])
def confirm_selection(job_id):
    """Handle user profile selection and trigger GIF generation."""
    # Validate job_id format for security
    if not re.match(r'^[0-9]{8}_[0-9]{6}_[a-zA-Z0-9_.-]+$', job_id):
        logger.warning(f"Invalid job_id format: {job_id}")
        abort(404)
    
    if job_id not in job_data:
        logger.warning(f"Job not found: {job_id}")
        abort(404)
    
    try:
        # Get selected profiles from form data
        selected_profiles = request.form.getlist('selected_profiles')
        
        if not selected_profiles:
            logger.warning(f"No profiles selected for job {job_id}")
            return redirect(url_for('selection', job_id=job_id))
        
        logger.info(f"Selected {len(selected_profiles)} profiles for job {job_id}: {selected_profiles}")
        
        # Update job status and start GIF generation
        job_data[job_id]['status'] = 'CREATING_GIFS'
        job_data[job_id]['selected_profiles'] = selected_profiles
        
        # Start GIF creation in background thread
        threading.Thread(
            target=gif_generator, 
            args=(job_id, selected_profiles),
            daemon=True
        ).start()
        
        return redirect(url_for('result', job_id=job_id))
        
    except Exception as e:
        logger.error(f"Error processing selection for job {job_id}: {str(e)}")
        job_data[job_id]['status'] = 'FAILED'
        return redirect(url_for('result', job_id=job_id))

def gif_generator(job_id, selected_profile_filenames):
    """Generate GIFs from selected profiles."""
    try:
        # Get job information
        job_info = job_data[job_id]
        
        # Extract timestamp from job_id to build output directory path
        timestamp = job_id.split('_')[0] + '_' + job_id.split('_')[1]
        output_dir = secure_path_join(app.config['UPLOAD_FOLDER'], timestamp)
        
        # Import the function for creating GIFs from selected profiles
        from image_processor import create_gifs_from_selected_profiles
        
        # Create GIFs from selected profiles
        gif_files = create_gifs_from_selected_profiles(selected_profile_filenames, output_dir)
        
        if gif_files:
            for gif_file in gif_files:
                job_data[job_id]['output_files'].append(gif_file)
            job_data[job_id]['status'] = 'COMPLETED'
            logger.info(f"Successfully generated {len(gif_files)} GIF files for job {job_id}")
        else:
            job_data[job_id]['status'] = 'FAILED'
            logger.error(f"Failed to generate GIF files for job {job_id}")
            
    except Exception as e:
        logger.error(f"Error generating GIFs for job {job_id}: {str(e)}")
        job_data[job_id]['status'] = 'FAILED'

@app.route('/job_status/<job_id>')
def job_status(job_id):
    # Validate job_id format
    if not re.match(r'^[0-9]{8}_[0-9]{6}_[a-zA-Z0-9_.-]+$', job_id):
        abort(404)
    
    if job_id not in job_data:
        abort(404)
    
    status = job_data[job_id]['status']
    return jsonify({'status': status})

@app.route('/download/<path:filename>')
def download_file(filename):
    """Serve files from the upload directory for download."""
    logger.info(f"Download request for: {filename}")
    
    # Validate filename for security
    if not validate_filename(os.path.basename(filename)):
        logger.warning(f"Unsafe filename requested: {filename}")
        abort(404)
    
    try:
        # Ensure the file path is within the upload folder
        safe_path = secure_path_join(app.config['UPLOAD_FOLDER'], filename)
        logger.info(f"Attempting to serve file: {safe_path}")
        
        # Check if file exists and is actually a file
        if not os.path.exists(safe_path):
            logger.warning(f"File not found: {safe_path}")
            abort(404)
            
        if not os.path.isfile(safe_path):
            logger.warning(f"Path is not a file: {safe_path}")
            abort(404)
        
        # For GIF files, skip content validation to avoid issues with binary content
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext not in ['.gif', '.png', '.jpg', '.jpeg', '.webp']:
            logger.warning(f"Attempted to serve non-image file: {filename}")
            abort(404)
        
        logger.info(f"Successfully serving file: {filename}")
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
        
    except ValueError as ve:
        logger.warning(f"Path traversal attempt: {filename} - {ve}")
        abort(404)
    except Exception as e:
        logger.error(f"Error serving file {filename}: {e}")
        abort(500)

@app.route('/static/uploads/<path:filename>')
def serve_static_upload(filename):
    """Legacy route for backward compatibility."""
    return download_file(filename)

@app.route('/debug/files')
def debug_files():
    """Debug endpoint to list available files."""
    if not app.debug:
        abort(404)
    
    try:
        files_info = []
        upload_dir = app.config['UPLOAD_FOLDER']
        
        for root, dirs, files in os.walk(upload_dir):
            for file in files:
                if file.endswith(('.gif', '.png', '.jpg', '.jpeg', '.webp')):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, upload_dir)
                    size = os.path.getsize(file_path)
                    files_info.append({
                        'filename': file,
                        'relative_path': rel_path.replace('\\', '/'),  # Normalize path separators
                        'size': size,
                        'download_url': url_for('download_file', filename=rel_path.replace('\\', '/'))
                    })
        
        return jsonify({
            'upload_folder': upload_dir,
            'files': files_info,
            'total_files': len(files_info)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def run_app_http(port):
    app.run(host='0.0.0.0', port=port, threaded=True)

def run_app_https(port, certfile, keyfile):
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context.load_cert_chain(certfile, keyfile)
    app.run(host='0.0.0.0', port=port, ssl_context=context, threaded=True)

if __name__ == '__main__':
    http_port = config.HTTP_PORT if isinstance(config.HTTP_PORT, int) else 80
    https_port = config.HTTPS_PORT if isinstance(config.HTTPS_PORT, int) else 443
    cert_path = config.SSL_CERT_FILE if isinstance(config.SSL_CERT_FILE, str) else "cert.pem"
    key_path = config.SSL_KEY_FILE if isinstance(config.SSL_KEY_FILE, str) else "key.pem"

    # Only start HTTPS if SSL files exist
    if os.path.exists(cert_path) and os.path.exists(key_path):
        http_thread = threading.Thread(target=run_app_http, args=(http_port,))
        https_thread = threading.Thread(target=run_app_https, args=(https_port, cert_path, key_path))
        
        http_thread.start()
        https_thread.start()
        
        http_thread.join()
        https_thread.join()
    else:
        logger.info("SSL certificates not found, starting HTTP server only")
        run_app_http(http_port)
