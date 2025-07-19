import os
import io
import json
import uuid # Import uuid for generating session IDs
import requests # Needed for OpenRouter API calls
import base64 # Import base64 for decoding

from flask import Flask, request, jsonify, g # Import g for app context
from flask_cors import CORS

import fitz # PyMuPDF
from dotenv import load_dotenv

# Firebase Admin SDK imports
import firebase_admin
from firebase_admin import credentials, firestore

# Load environment variables from .env file (for local development)
load_dotenv()

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# --- Configure OpenRouter API ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    print("WARNING: OPENROUTER_API_KEY environment variable not set. LLM features will not work.")

OPENROUTER_API_BASE = "https://openrouter.ai/api/v1/chat/completions"

# Define your preferred models in order of preference (Primary, Fallback 1, Fallback 2)
PREFERRED_LLM_MODELS = [
    "qwen/qwen-2.5-72b-instruct:free",  # Primary choice
    "deepseek/deepseek-coder-6.7b-instruct:free", # Corrected Deepseek model ID based on common OpenRouter free models
    "google/gemma-2-9b-it:free" # Corrected Gemma model ID based on common OpenRouter free models
]

# --- Initialize Firestore (only once per app instance) ---
# Use Flask's app context to ensure Firestore is initialized only once
def get_firestore_db():
    if 'firestore_db' not in g:
        try:
            firebase_credentials_json = os.getenv('FIREBASE_SERVICE_ACCOUNT_KEY')
            # DEBUG: Print first 100 characters of the loaded environment variable
            print(f"DEBUG: Value of FIREBASE_SERVICE_ACCOUNT_KEY: {firebase_credentials_json[:100]}...") 
            
            if firebase_credentials_json:
                # Check if app is already initialized to avoid re-initialization errors
                if not firebase_admin._apps:
                    try:
                        # Attempt to parse the JSON string
                        cred = credentials.Certificate(json.loads(firebase_credentials_json))
                    except json.JSONDecodeError as e:
                        # If JSON parsing fails, print a specific error and set db to None
                        print(f"ERROR: JSON decoding failed for Firebase credentials: {e}")
                        g.firestore_db = None
                        return g.firestore_db # Exit early if credentials are bad
                    
                    firebase_admin.initialize_app(cred)
                g.firestore_db = firestore.client()
                print("Firestore initialized successfully!")
            else:
                print("FIREBASE_SERVICE_ACCOUNT_KEY environment variable not set. Firestore will not be available.")
                g.firestore_db = None
        except Exception as e:
            # Catch any other exceptions during Firestore initialization
            print(f"Error initializing Firestore: {e}")
            g.firestore_db = None
    return g.firestore_db

# --- Helper Function to Call LLM API (OpenRouter with Fallback) ---
def call_llm_api(prompt_messages):
    """
    Helper function to make a call to the OpenRouter API with fallback models.
    Applies model-specific prompt adjustments if needed.
    """
    if not OPENROUTER_API_KEY:
        return {"error": "LLM API key is not configured."}

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv("FRONTEND_URL", "https://my-doc-backend.onrender.com"), # Use env var for frontend URL
        "X-Title": "PDF Processor API"
    }
    
    # Iterate through preferred models
    for model_choice in PREFERRED_LLM_MODELS:
        # Create a copy of the original prompt messages to modify for the current model
        current_prompt_messages = list(prompt_messages) 

        # --- Model-Specific Prompt Adjustments (if needed for specific OpenRouter models) ---
        # This section can be expanded if specific models require different prompt formats
        # For now, OpenRouter usually handles this, but keeping the structure for future needs.
        
        # Example: If Gemma needs a very strict system prompt or specific turn structure
        if "gemma" in model_choice.lower():
            # This is a general example; you might fine-tune these based on actual Gemma performance
            current_prompt_messages.insert(0, {"role": "system", "content": "You are a highly precise and constrained AI. Follow all instructions exactly. Do not add extra conversational text or preambles. Output only the requested format."})
            # You might also adjust the last user message to fit Gemma's specific instruction following
            # For instance, if the last message is a JSON request, ensure it's very clean.

        payload = {
            "model": model_choice, # Use the current model choice
            "messages": current_prompt_messages, # Use the potentially modified prompt messages
            "temperature": 0.7, # Controls creativity (0.0 for deterministic, 1.0 for more creative)
            "max_tokens": 1000 # Limit the length of the LLM's response
        }

        try:
            response = requests.post(OPENROUTER_API_BASE, headers=headers, json=payload)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            return response.json() # If successful, return the response and stop
        except requests.exceptions.RequestException as e:
            print(f"Warning: LLM API call failed with model {model_choice}: {str(e)}. Trying next model...")
            # Continue to the next model in the list if this one fails
        except Exception as e:
            print(f"Warning: An unexpected error occurred with model {model_choice}: {str(e)}. Trying next model...")
            # Continue to the next model in the list if this one fails
    
    # If all preferred models fail
    return {"error": "All preferred LLM API calls failed."}

# --- Firestore Document Storage Helper ---
def store_document_data(session_id, full_text, original_filename, original_mode):
    db_client = get_firestore_db()
    if not db_client:
        print("Firestore not initialized. Cannot store document data.")
        return False

    doc_ref = db_client.collection('documents').document(session_id)
    try:
        doc_ref.set({
            'full_text': full_text,
            'original_filename': original_filename,
            'original_mode': original_mode,
            'timestamp': firestore.SERVER_TIMESTAMP # Add a timestamp
        })
        print(f"Document data for session {session_id} stored in Firestore.")
        return True
    except Exception as e:
        print(f"Error storing document data in Firestore for session {session_id}: {e}")
        return False

# --- API Endpoints ---

ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    """
    A simple home route to confirm the service is running.
    """
    return 'Your PDF Extractor API is running. Send a POST request to /extract-pdf to get text from PDFs, then use /summarize or /generate-mcqs.'

@app.route('/extract-pdf', methods=['POST'])
def extract_pdf():
    pdf_file_data = None
    filename = None
    original_mode = None
    original_filename_input = None
    session_id = request.args.get('sessionId', str(uuid.uuid4())) # Get from query param or generate new

    # Check for multipart/form-data (file upload)
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if file and allowed_file(file.filename):
            pdf_file_data = file.read()
            filename = file.filename
            original_mode = request.form.get('original_mode')
            original_filename_input = request.form.get('original_filename')
        else:
            return jsonify({'error': 'Invalid file type. Only PDF files are allowed.'}), 400
    # Check for JSON payload
    elif request.is_json:
        data = request.get_json()
        pdf_url = data.get('pdf_url')
        # NEW: Handle Base64 encoded PDF content
        pdf_content_base64 = data.get('file_content_base64') 
        original_mode = data.get('original_mode')
        original_filename_input = data.get('original_filename')
        
        # Prioritize sessionId from JSON body if present
        session_id = data.get('sessionId', session_id)

        if pdf_content_base64:
            try:
                # Decode the Base64 string to bytes
                pdf_file_data = base64.b64decode(pdf_content_base64)
                # Use the provided filename or default
                filename = original_filename_input or "uploaded_base64_pdf.pdf" 
                if not allowed_file(filename):
                    return jsonify({'error': 'Invalid file extension for Base64 PDF. Must be .pdf'}), 400
            except Exception as e:
                return jsonify({'error': f'Failed to decode Base64 PDF content: {str(e)}'}), 400
        elif pdf_url:
            try:
                response = requests.get(pdf_url, stream=True)
                response.raise_for_status()

                if 'application/pdf' not in response.headers.get('Content-Type', ''):
                    return jsonify({'error': 'URL does not point to a PDF file.'}), 400

                pdf_file_data = response.content
                filename = pdf_url.split('/')[-1]
                if not filename.lower().endswith('.pdf'):
                    filename = "downloaded_pdf.pdf"
            except requests.exceptions.RequestException as e:
                return jsonify({'error': f'Failed to download PDF from URL: {str(e)}'}), 400
        else:
            return jsonify({'error': 'No file uploaded, "pdf_url", or "file_content_base64" provided in JSON payload.'}), 400
    else:
        return jsonify({'error': 'Unsupported media type. Please upload a file or send JSON with "pdf_url" or "file_content_base64".'}), 415

    if not pdf_file_data:
        return jsonify({'error': 'No PDF data received.'}), 400

    extracted_text_per_page = []
    try:
        doc = fitz.open(stream=pdf_file_data, filetype="pdf")

        for page_num in range(doc.page_count):
            page = doc[page_num]
            text = page.get_text()
            extracted_text_per_page.append({
                'page_number': page_num + 1,
                'text': text
            })
        doc.close()

        full_extracted_text = "\n".join([p['text'] for p in extracted_text_per_page])

        # Store full_text in Firestore
        if not store_document_data(session_id, full_extracted_text, original_filename_input or filename, original_mode):
            return jsonify({"error": "Failed to store document data in Firestore."}), 500

        response_payload = {
            'status': 'success',
            'filename': original_filename_input if original_filename_input else filename,
            'full_text': full_extracted_text,
            'pages': extracted_text_per_page,
            'message': 'Text extracted successfully. Use the full_text for summarization or MCQ generation.',
            'sessionId': session_id # Include session ID in response
        }
        
        if original_mode is not None:
            response_payload['original_mode'] = original_mode
        # Include difficulty in response if present in request data from JSON body
        if request.is_json and 'difficulty' in data:
            response_payload['difficulty'] = data.get('difficulty')


        return jsonify(response_payload), 200

    except fitz.FileDataError as e:
        return jsonify({'error': f'Invalid PDF file or corrupted data: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'An unexpected error occurred during PDF processing: {str(e)}'}), 500

@app.route('/summarize', methods=['POST'])
def summarize_text():
    data = request.get_json()
    full_text = data.get('text') # Renamed from text_to_summarize for consistency
    original_filename = data.get('original_filename', 'unknown_file')
    original_mode = data.get('original_mode', 'summarize')
    session_id = data.get('sessionId', str(uuid.uuid4())) # Get existing or generate new session ID

    if not full_text:
        return jsonify({'error': 'No text provided for summarization.'}), 400

    # Store full_text in Firestore (if not already stored by /extract-pdf)
    # This ensures that if summarization is called directly, context is saved.
    # If called after /extract-pdf, it will just overwrite with the same data.
    if not store_document_data(session_id, full_text, original_filename, original_mode):
        return jsonify({"error": "Failed to store document data for session"}), 500

    summarize_prompt = f"""
    You are an expert summarizer. Your task is to create a comprehensive and accurate summary of the provided document text.
    The summary should capture all main points and critical information without unnecessary details.
    Aim for a summary that is typically 5-8 sentences long, ensuring good coverage of the document.

    Document Text:
    ---
    {full_text}
    ---

    Please provide the summary directly.
    """

    prompt_messages = [
        {"role": "system", "content": "You are a helpful assistant that summarizes text concisely."},
        {"role": "user", "content": summarize_prompt}
    ]

    llm_response = call_llm_api(prompt_messages)

    if "error" in llm_response:
        return jsonify(llm_response), 500
    
    summary = llm_response.get("choices", [{}])[0].get("message", {}).get("content", "Could not generate summary.")
    
    return jsonify({
        'status': 'success',
        'summary': summary,
        'original_filename': original_filename,
        'original_mode': original_mode,
        'sessionId': session_id, # Include session ID in response
        'full_text': full_text # <<< ADDED THIS LINE for summarize
    }), 200

@app.route('/highlight', methods=['POST'])
def highlight_text():
    data = request.get_json()
    full_text = data.get('text') # Renamed from text_to_highlight for consistency
    original_filename = data.get('original_filename', 'unknown_file')
    original_mode = data.get('original_mode', 'highlight')
    session_id = data.get('sessionId', str(uuid.uuid4())) # Get existing or generate new session ID

    if not full_text:
        return jsonify({'error': 'No text provided for highlighting.'}), 400

    if not store_document_data(session_id, full_text, original_filename, original_mode):
        return jsonify({"error": "Failed to store document data for session"}), 500

    highlight_prompt = f"""
    You are an intelligent text analyzer. Your task is to extract the most important and salient sentences or key phrases from the provided document text.
    Focus on information that is critical to understanding the core content.
    Present the extracted highlights as a clear, concise bulleted list. Do not add any introductory or concluding remarks.

    Document Text:
    ---
    {full_text}
    ---

    Please provide the highlights as a bulleted list.
    """

    prompt_messages = [
        {"role": "system", "content": "You are a helpful assistant that identifies and extracts the most important sentences or phrases from a given text. Return only the highlighted sentences/phrases as a bulleted list."},
        {"role": "user", "content": highlight_prompt}
    ]

    llm_response = call_llm_api(prompt_messages)

    if "error" in llm_response:
        return jsonify(llm_response), 500

    highlights = llm_response.get("choices", [{}])[0].get("message", {}).get("content", "Could not generate highlights.")

    return jsonify({
        'status': 'success',
        'highlights': highlights,
        'original_filename': original_filename,
        'original_mode': original_mode,
        'sessionId': session_id, # Include session ID in response
        'full_text': full_text # <<< ADDED THIS LINE for highlight
    }), 200

@app.route('/generate-mcqs', methods=['POST'])
def generate_mcqs():
    data = request.get_json()
    full_text = data.get('text') # Renamed from text_for_mcq for consistency
    original_filename = data.get('original_filename', 'unknown_file')
    original_mode = data.get('original_mode', 'mcq')
    session_id = data.get('sessionId', str(uuid.uuid4())) # Get existing or generate new session ID
    
    # NEW: Get difficulty and number of questions
    difficulty = data.get('difficulty', 'medium').lower() # Default to medium
    num_questions = int(data.get('num_questions', 5)) # Default to 5 questions

    if not full_text:
        return jsonify({'error': 'No text provided for MCQ generation.'}), 400
    
    if not isinstance(num_questions, int) or num_questions <= 0:
        return jsonify({'error': 'num_questions must be a positive integer.'}), 400

    if not store_document_data(session_id, full_text, original_filename, original_mode):
        return jsonify({"error": "Failed to store document data for session"}), 500

    mcq_prompt = f"""
    You are an expert at creating {difficulty} multiple-choice questions (MCQs) from provided text.
    Generate exactly {num_questions} MCQs. Each question must have 4 distinct options (A, B, C, D) and only one correct answer.
    Ensure the questions are directly answerable from the text and cover different aspects of the content.
    Your response MUST be a valid JSON array of objects. Do not include any other text, explanations, or formatting outside the JSON.

    Text to generate MCQs from:
    ---
    {full_text}
    ---

    JSON Output Format:
    [
      {{
        "question": "The question text here?",
        "options": ["A) Option A", "B) Option B", "C) Option C", "D) Option D"],
        "correct_answer": "A"
      }},
      {{
        "question": "Another question?",
        "options": ["A) Yes", "B) No", "C) Maybe", "D) Unsure"],
        "correct_answer": "C"
      }}
    ]
    """

    prompt_messages = [
        {"role": "system", "content": "You are an expert at creating multiple-choice questions from provided text, always responding in valid JSON and nothing else."},
        {"role": "user", "content": mcq_prompt}
    ]

    llm_response = call_llm_api(prompt_messages)

    if "error" in llm_response:
        return jsonify(llm_response), 500
    
    try:
        mcqs_json_str = llm_response.get("choices", [{}])[0].get("message", {}).get("content", "{}")
        mcqs_data = json.loads(mcqs_json_str)
        if not isinstance(mcqs_data, list):
            raise ValueError("LLM did not return a JSON array.")
        
        response_payload = { 
            'status': 'success',
            'mcqs': mcqs_data,
            'original_filename': original_filename,
            'original_mode': original_mode,
            'sessionId': session_id,
            'difficulty': difficulty,
            'full_text': full_text 
        }
        return jsonify(response_payload), 200 
    except json.JSONDecodeError:
        return jsonify({'error': 'LLM response was not valid JSON for MCQs. Please check the LLM output format.'}), 500
    except ValueError as e:
        return jsonify({'error': f'Error parsing LLM response: {e}. Raw response: {mcqs_json_str}'}), 500
    except Exception as e:
        return jsonify({'error': f'An unexpected error occurred while processing LLM MCQ response: {str(e)}'}), 500

@app.route('/chat', methods=['POST'])
def chat_with_document():
    data = request.get_json()
    session_id = data.get('sessionId')
    user_query = data.get('query')

    if not session_id or not user_query:
        return jsonify({"error": "Session ID and query are required"}), 400

    db_client = get_firestore_db()
    if not db_client:
        return jsonify({"error": "Firestore not initialized. Cannot retrieve document context."}), 500

    try:
        doc_ref = db_client.collection('documents').document(session_id)
        doc = doc_ref.get()

        if not doc.exists:
            return jsonify({"error": "Document context not found for this session."}), 404

        full_text = doc.to_dict().get('full_text')
        if not full_text:
            return jsonify({"error": "Full text not found in document context."}), 404

        # Construct chat prompt with document context
        chat_prompt = f"""You are an AI assistant helping a user understand a document.
The document content is:
---
{full_text}
---

Based on the document above, answer the following question:
User: {user_query}
Assistant:"""
        
        prompt_messages = [
            {"role": "system", "content": "You are a helpful assistant that answers questions based on provided document content."},
            {"role": "user", "content": chat_prompt}
        ]

        llm_response = call_llm_api(prompt_messages)

        if "error" in llm_response:
            return jsonify(llm_response), 500

        assistant_response = llm_response.get("choices", [{}])[0].get("message", {}).get("content", "Could not generate response.")

        return jsonify({
            "response": assistant_response,
            "sessionId": session_id # Echo session ID
        })
    except Exception as e:
        print(f"Error during chat interaction: {e}")
        return jsonify({"error": f"Error during chat interaction: {e}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
