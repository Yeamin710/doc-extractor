import io
import requests
import fitz # PyMuPDF
from flask import Flask, request, jsonify

app = Flask(__name__)

# Define allowed extensions for direct file uploads
ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    """
    A simple home route to confirm the service is running.
    """
    return 'Your PDF Extractor API is running. Send a POST request to /extract-pdf.'

@app.route('/extract-pdf', methods=['POST'])
def extract_pdf():
    pdf_file_data = None
    filename = None

    # --- Handle File Upload ---
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if file and allowed_file(file.filename):
            pdf_file_data = file.read()
            filename = file.filename
        else:
            return jsonify({'error': 'Invalid file type. Only PDF files are allowed.'}), 400
    # --- Handle URL Input ---
    elif request.is_json:
        data = request.get_json()
        pdf_url = data.get('pdf_url')
        if pdf_url:
            try:
                response = requests.get(pdf_url, stream=True)
                response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)

                # Check if the content type is PDF
                if 'application/pdf' not in response.headers.get('Content-Type', ''):
                    return jsonify({'error': 'URL does not point to a PDF file.'}), 400

                pdf_file_data = response.content
                filename = pdf_url.split('/')[-1] # Basic filename from URL
                if not filename.lower().endswith('.pdf'):
                    filename = "downloaded_pdf.pdf" # Default if URL has no .pdf extension
            except requests.exceptions.RequestException as e:
                return jsonify({'error': f'Failed to download PDF from URL: {str(e)}'}), 400
        else:
            return jsonify({'error': 'No file uploaded or "pdf_url" provided in JSON payload.'}), 400
    else:
        return jsonify({'error': 'Unsupported media type. Please upload a file or send JSON with "pdf_url".'}), 415

    if not pdf_file_data:
        return jsonify({'error': 'No PDF data received.'}), 400

    extracted_text_per_page = []
    try:
        # Open the PDF from in-memory bytes
        doc = fitz.open(stream=pdf_file_data, filetype="pdf")

        for page_num in range(doc.page_count):
            page = doc[page_num]
            text = page.get_text()
            extracted_text_per_page.append({
                'page_number': page_num + 1,
                'text': text
            })
        doc.close()

        return jsonify({
            'status': 'success',
            'filename': filename,
            'pages': extracted_text_per_page,
            'message': 'Text extracted successfully. Ready for summarization, highlighting, and MCQ generation.'
        }), 200

    except fitz.FileDataError as e:
        return jsonify({'error': f'Invalid PDF file or corrupted data: {str(e)}'}), 400
    except Exception as e:
        # Catch any other unexpected errors during processing
        return jsonify({'error': f'An unexpected error occurred during PDF processing: {str(e)}'}), 500

if __name__ == '__main__':
    # This block is for local development only. Gunicorn will handle this on Render.
    app.run(debug=True, host='0.0.0.0', port=5000)
