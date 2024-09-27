import os
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from datetime import datetime
import threading
import time
import logging
from collections import defaultdict
from image_processor import process_avasplit
import ssl

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

job_data = defaultdict(lambda: {'status': 'PENDING', 'output_files': []})

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def image_processor(input_path, output_dir, job_id):
    try:
        time.sleep(5)

        gif_files = process_avasplit(input_path, output_dir)
        for gif_file in gif_files:
            output_path = gif_file
            job_data[job_id]['output_files'].append(output_path)

        job_data[job_id]['status'] = 'COMPLETED'
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        job_data[job_id]['status'] = 'FAILED'

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                job_id = f"{timestamp}_{filename}"
                upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], timestamp)
                os.makedirs(upload_dir, exist_ok=True)
                input_path = os.path.join(upload_dir, filename)
                file.save(input_path)

                job_data[job_id]['status'] = 'PROCESSING'

                threading.Thread(target=image_processor, args=(input_path, upload_dir, job_id)).start()

                return redirect(url_for('result', job_id=job_id))
            except Exception as e:
                logger.error(f"Error processing upload: {str(e)}")
                return "An error occurred while processing your upload", 500
    return render_template('index.html')

@app.route('/result/<job_id>')
def result(job_id):
    job_info = job_data[job_id]
    status = job_info['status']
    output_files = job_info['output_files']
    relative_paths = [os.path.relpath(path, 'static') for path in output_files]
    return render_template('result.html', job_id=job_id, status=status, output_files=relative_paths)

@app.route('/job_status/<job_id>')
def job_status(job_id):
    status = job_data[job_id]['status']
    return jsonify({'status': status})

@app.route('/static/uploads/<path:filename>')
def serve_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

import argparse
import config


def run_app_http(port):
    app.run(host='0.0.0.0', port=http_port, threaded=True)

def run_app_https(port, certfile, keyfile):
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context.load_cert_chain(cert_path, key_path)
    app.run(host='0.0.0.0', port=https_port, ssl_context=context, threaded=True)

if __name__ == '__main__':
    http_port = config.HTTP_PORT
    https_port = config.HTTPS_PORT
    cert_path = config.SSL_CERT_FILE
    key_path = config.SSL_KEY_FILE

    http_thread = threading.Thread(target=run_app_http, args=(http_port,))
    https_thread = threading.Thread(target=run_app_https, args=(https_port, cert_path, key_path))

    http_thread.start()
    https_thread.start()

    http_thread.join()
    https_thread.join()
