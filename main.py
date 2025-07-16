from flask import Flask, request, jsonify
import requests
import tempfile
import os
import fitz  # PyMuPDF for PDF
import docx2txt
import logging
import json

# --- Configuration ---
# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# --- LLM Integration Configuration (Qwen via OpenRouter as an example) ---
# IMPORTANT: Access API Key from environment variables (recommended for production)
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "qwen/qwen-2-72b-instruct" # Example Qwen model on OpenRouter

# Ensure API Key is available
if not OPENROUTER_API_KEY:
    logging.error("OPENROUTER_API_KEY environment variable not set. LLM features will not work.")
    # For local development, you could hardcode it here for testing ONLY
    # OPENROUTER_API_KEY = "sk-or-YOUR_OPENROUTER_API_KEY"
    # But remove this line before pushing to a public repo or deploying to production without env var set.


# --- Helper Functions ---

def call_llm_api(prompt_messages, model=LLM_MODEL, max_tokens=1000, temperature=0.7):
    """
    Helper function to call a generic LLM API (e.g., OpenRouter).
    `prompt_messages` should be in the format: [{"role": "user", "content": "Your message"}]
    """
    if not OPENROUTER_API_KEY:
        raise ValueError("LLM API key is not set. Cannot call LLM.")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": prompt_messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    try:
        logging.info(f"Calling LLM API with model: {model}")
        response = requests.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers=headers,
            data=json.dumps(data)
        )
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        llm_response = response.json()
        logging.info("LLM API call successful.")
        return llm_response['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        logging.error(f"Error calling LLM API: {e}")
        raise ValueError(f"Failed to communicate with LLM API: {e}")
    except KeyError as e:
        logging.error(f"Unexpected LLM response format: {e}, Response: {llm_response}")
        raise ValueError(f"Unexpected LLM response format: {e}")

def extract_text_from_file(file_path, extension):
    """
    Extracts text from various document types.
    """
    try:
        if extension == 'pdf':
            text = ""
            with fitz.open(file_path) as doc:
                for page in doc:
                    text += page.get_text()
            logging.info(f"Text extracted from PDF: {file_path}")
            return text
        elif extension == 'docx':
            text = docx2txt.process(file_path)
            logging.info(f"Text extracted from DOCX: {file_path}")
            return text
        elif extension == 'txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            logging.info(f"Text extracted from TXT: {file_path}")
            return text
        else:
            logging.warning(f"Unsupported file type for extraction: {extension}")
            raise ValueError("Unsupported file type for extraction.")
    except Exception as e:
        logging.error(f"Error during text extraction from {file_path}: {e}")
        raise

# --- API Endpoints ---

@app.route('/extract-text', methods=['POST'])
def extract_document_text():
    """
    Extracts text from a document provided via URL or raw binary upload.
    Expects JSON: {"fileUrl": "...", "extension": "..."} OR raw binary data with headers.
    """
    tmp_file_path = None
    try:
        # Handle JSON payload (for fileUrl)
        if request.is_json:
            data = request.json
            file_url = data.get('fileUrl')
            extension = data.get('extension')
            
            if not file_url or not extension:
                logging.warning("Missing fileUrl or extension in JSON payload for /extract-text.")
                return jsonify({"error": "Missing fileUrl or extension"}), 400
            
            logging.info(f"Downloading file from URL: {file_url}")
            response = requests.get(file_url)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            file_content = response.content

        # Handle raw binary upload (e.g., from n8n's "Send Binary Data" option)
        elif request.data:
            file_content = request.data
            # Try to get extension from Content-Type, or rely on client to send it in a header
            content_type = request.headers.get('Content-Type', '').lower()
            if 'pdf' in content_type: extension = 'pdf'
            elif 'word' in content_type or 'msword' in content_type or 'document' in content_type: extension = 'docx' # Expanded for common word mimetypes
            elif 'text' in content_type: extension = 'txt'
            else:
                # If Content-Type isn't clear, client *must* send an 'X-File-Extension' header
                extension = request.headers.get('X-File-Extension', '').lower()
                if not extension:
                    logging.warning("Raw binary upload without detectable Content-Type or X-File-Extension.")
                    return jsonify({"error": "Unsupported Content-Type or missing X-File-Extension header for raw upload"}), 400
            logging.info(f"Received raw binary data for extraction, assumed extension: {extension}")
        else:
            logging.warning("No fileUrl (JSON) or raw data found for /extract-text.")
            return jsonify({"error": "No file URL or raw data provided"}), 400

        # Write content to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{extension}") as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name

        extracted_text = extract_text_from_file(tmp_file_path, extension)
        
        # Limit the response size to avoid issues with very large texts
        return jsonify({"text": extracted_text.strip()[:20000], "status": "success"})

    except ValueError as ve:
        logging.error(f"Client error in /extract-text: {ve}", exc_info=True)
        return jsonify({"error": str(ve)}), 400
    except requests.exceptions.RequestException as re:
        logging.error(f"Download failed for /extract-text: {re}", exc_info=True)
        return jsonify({"error": f"Failed to download file: {re}"}), 400
    except Exception as e:
        logging.error(f"Unhandled server error in /extract-text: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred during extraction."}), 500
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
            logging.info(f"Removed temporary file: {tmp_file_path}")


@app.route('/summarize', methods=['POST'])
def summarize_text():
    """
    Summarizes provided text using LLM.
    Expects JSON: {"text": "...", "sessionId": "..."}
    """
    data = request.json
    text = data.get('text')
    session_id = data.get('sessionId')

    if not text:
        return jsonify({"error": "Missing 'text' for summarization"}), 400

    prompt = f"Summarize the following document content concisely:\n\n{text}"
    messages = [{"role": "user", "content": prompt}]

    try:
        summary = call_llm_api(messages)
        logging.info(f"Text summarized for session: {session_id}")
        return jsonify({"summary": summary, "sessionId": session_id, "status": "success"})
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 500
    except Exception as e:
        logging.error(f"Unhandled server error in /summarize: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred during summarization."}), 500


@app.route('/highlight', methods=['POST'])
def highlight_text():
    """
    Identifies and returns key sentences/phrases from the provided text using LLM.
    Expects JSON: {"text": "...", "sessionId": "..."}
    """
    data = request.json
    text = data.get('text')
    session_id = data.get('sessionId')

    if not text:
        return jsonify({"error": "Missing 'text' for highlighting"}), 400

    prompt = f"Identify and list the 5 most important sentences or key phrases from the following text:\n\n{text}"
    messages = [{"role": "user", "content": prompt}]

    try:
        highlights = call_llm_api(messages)
        logging.info(f"Text highlighted for session: {session_id}")
        return jsonify({"highlights": highlights, "sessionId": session_id, "status": "success"})
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 500
    except Exception as e:
        logging.error(f"Unhandled server error in /highlight: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred during highlighting."}), 500


@app.route('/generate-mcq', methods=['POST'])
def generate_mcq():
    """
    Generates Multiple Choice Questions (MCQs) from text using LLM.
    Expects JSON: {"text": "...", "difficulty": "easy/medium/hard", "sessionId": "..."}
    """
    data = request.json
    text = data.get('text')
    difficulty = data.get('difficulty', 'medium')
    session_id = data.get('sessionId')

    if not text:
        return jsonify({"error": "Missing 'text' for MCQ generation"}), 400

    prompt = f"""Generate 3 multiple-choice questions (MCQs) from the following text, suitable for a {difficulty} difficulty level.
    For each question, provide 4 options (A, B, C, D) and clearly state the correct answer.
    Format your response as:
    1. Question?
    A) Option A
    B) Option B
    C) Option C
    D) Option D
    Correct Answer: X

    2. ...

    Text:
    {text}
    """
    messages = [{"role": "user", "content": prompt}]

    try:
        mcqs = call_llm_api(messages, max_tokens=1500)
        logging.info(f"MCQs generated for session: {session_id}, difficulty: {difficulty}")
        return jsonify({"mcqs": mcqs, "sessionId": session_id, "status": "success"})
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 500
    except Exception as e:
        logging.error(f"Unhandled server error in /generate-mcq: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred during MCQ generation."}), 500


@app.route('/chat', methods=['POST'])
def chat_with_document():
    """
    Handles conversational chat based on document content using LLM.
    Expects JSON: {"sessionId": "...", "message": "...", "document_text": "..."}
    """
    data = request.json
    session_id = data.get('sessionId')
    user_message = data.get('message')
    document_text = data.get('document_text') # Assume n8n sends this for context

    if not user_message or not document_text:
        return jsonify({"error": "Missing 'message' or 'document_text' for chat"}), 400

    # In a real app, you might fetch document_text from a database based on sessionId
    # For this example, we assume n8n sends the relevant text context.

    prompt = f"""You are an AI assistant specialized in answering questions based on provided document text.
    Answer the user's question concisely using ONLY the information from the document text provided.
    If the answer is not in the document, state that you don't have enough information from the document.

    Document Text:
    {document_text}

    User's Question:
    {user_message}
    """
    messages = [{"role": "user", "content": prompt}]

    try:
        chat_response = call_llm_api(messages, max_tokens=500)
        logging.info(f"Chat response generated for session: {session_id}")
        return jsonify({"response": chat_response, "sessionId": session_id, "status": "success"})
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 500
    except Exception as e:
        logging.error(f"Unhandled server error in /chat: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred during chat."}), 500

# --- Main Application Run ---
if __name__ == '__main__':
    # When deploying to Render, Gunicorn or another WSGI server will run the app,
    # so app.run() is mainly for local development.
    logging.info("Starting Flask application...")
    app.run(host='0.0.0.0', port=5000)
