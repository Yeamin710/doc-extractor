from flask import Flask, request, jsonify
import requests
import tempfile
import os
import fitz  # PyMuPDF for PDF
import docx2txt

app = Flask(__name__)

def extract_text(file_path, extension):
    if extension == 'pdf':
        text = ""
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text()
        return text
    elif extension == 'docx':
        return docx2txt.process(file_path)
    elif extension == 'txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        return "Unsupported file type."

@app.route('/extract-text', methods=['POST'])
def extract():
    data = request.json
    file_url = data.get('fileUrl')
    extension = data.get('extension')

    if not file_url or not extension:
        return jsonify({"error": "Missing fileUrl or extension"}), 400

    try:
        response = requests.get(file_url)
        if response.status_code != 200:
            return jsonify({"error": "Failed to download file"}), 400

        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{extension}") as tmp_file:
            tmp_file.write(response.content)
            tmp_file_path = tmp_file.name

        extracted = extract_text(tmp_file_path, extension)

        os.remove(tmp_file_path)

        return jsonify({"text": extracted.strip()[:20000]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
