import os
import io
import json
import uuid # Import uuid for generating session IDs
import requests # Needed for OpenRouter API calls
import base64 # Import base64 for decoding

from flask import Flask, request, jsonify, g # Import g for app context
from flask_cors import CORS

import fitz # PyMuPDF - ONLY for PDF processing
from PIL import Image # For image manipulation (needed for OCR)
import pytesseract # For OCR
from dotenv import load_dotenv

# Firebase Admin SDK imports
import firebase_admin
from firebase_admin import credentials, firestore

# --- Configure Tesseract Path (for local development/testing only) ---
# On Render, Tesseract will be in your PATH automatically if installed via apt-get
# If you run locally and Tesseract is not found, you might need to set this:
# pytesseract.pytesseract.tesseract_cmd = r'/path/to/tesseract' # e.g., r'C:\Program Files\Tesseract-OCR\tesseract.exe' for Windows

# Load environment variables from .env file (for local development)
load_dotenv()

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# --- Configure OpenRouter API ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    print("WARNING: OPENROUTER_API_KEY environment variable not set. LLM features will not work.")

OPENROUTER_API_BASE = "https://openrouter.ai/api/v1/chat/completions"

# --- Configure AIMLAPI.com API ---
AIML_API_KEY = os.getenv("AIML_API_KEY")
if not AIML_API_KEY:
    print("WARNING: AIML_API_KEY environment variable not set. AIMLAPI.com LLM features will not work.")

# --- Configure Google AI Studio (Gemini) API ---
GOOGLE_AI_STUDIO_API_KEY = os.getenv("GOOGLE_AI_STUDIO_API_KEY")
# Base URL for Google Gemini API (generative language API)
GOOGLE_AI_STUDIO_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models/" 

if not OPENROUTER_API_KEY:
    print("WARNING: OPENROUTER_API_KEY environment variable not set. OpenRouter LLM features will not work.")
if not AIML_API_KEY:
    print("WARNING: AIML_API_KEY environment variable not set. AIMLAPI.com LLM features will not work.")
if not GOOGLE_AI_STUDIO_API_KEY:
    print("WARNING: GOOGLE_AI_STUDIO_API_KEY environment variable not set. Google AI Studio LLM features will not work.")


# Define your preferred models for general use (summarize, highlight, mcq)
# Now includes AIMLAPI.com Gemma 3 models and Google Gemini models as fallback options.
PREFERRED_LLM_MODELS = [
    "qwen/qwen-2.5-72b-instruct:free",           # OpenRouter: Primary choice
    "moonshotai/kimi-dev-72b:free",              # OpenRouter: Second choice
    "mistralai/devstral-small-2505:free",        # OpenRouter: Third choice
    "deepseek/deepseek-coder-6.7b-instruct:free", # OpenRouter: Existing Deepseek
    "google/gemma-2-9b-it:free",                 # OpenRouter: Existing Gemma
    # AIMLAPI.com Gemma 3 models (free tier)
    "google/gemma-3-1b-it",                      # AIMLAPI.com
    "google/gemma-3-4b-it",                      # AIMLAPI.com
    "google/gemma-3-12b-it",                     # AIMLAPI.com
    "google/gemma-3-27b-it",                     # AIMLAPI.com
    "google/gemma-3n-e4b-it",                    # AIMLAPI.com
    # Google AI Studio (Gemini) models
    "gemini-1.5-flash-latest",                   # Google AI Studio: Fast and cost-effective
    "gemini-1.5-pro-latest",                     # Google AI Studio: More capable, higher cost
]

# --- Initialize Firestore (only once per app instance) ---
def get_firestore_db():
    if 'firestore_db' not in g:
        try:
            firebase_credentials_json = os.getenv('FIREBASE_SERVICE_ACCOUNT_KEY')
            print(f"DEBUG: Value of FIREBASE_SERVICE_ACCOUNT_KEY: {firebase_credentials_json[:100]}...") 
            
            if firebase_credentials_json:
                if not firebase_admin._apps:
                    try:
                        cred = credentials.Certificate(json.loads(firebase_credentials_json))
                    except json.JSONDecodeError as e:
                        print(f"ERROR: JSON decoding failed for Firebase credentials: {e}")
                        g.firestore_db = None
                        return g.firestore_db
                    
                    firebase_admin.initialize_app(cred)
                g.firestore_db = firestore.client()
                print("Firestore initialized successfully!")
            else:
                print("FIREBASE_SERVICE_ACCOUNT_KEY environment variable not set. Firestore will not be available.")
                g.firestore_db = None
        except Exception as e:
            print(f"Error initializing Firestore: {e}")
            g.firestore_db = None
    return g.firestore_db

# --- Helper Function to Call LLM API (OpenRouter, AIMLAPI.com, Google AI Studio Fallback) ---
def call_llm_api(prompt_messages, primary_models_to_try=None):
    """
    Helper function to make a call to LLM APIs with fallback models.
    Prioritizes primary_models_to_try (if provided), then PREFERRED_LLM_MODELS.
    Distinguishes between OpenRouter, AIMLAPI.com, and Google AI Studio models.
    """
    
    models_to_attempt = []
    if primary_models_to_try:
        models_to_attempt.extend(primary_models_to_try)
    models_to_attempt.extend(PREFERRED_LLM_MODELS) # These are the ultimate fallbacks

    for model_choice in models_to_attempt:
        current_prompt_messages = list(prompt_messages) 

        api_key = None
        api_full_url = None 
        request_payload = {}
        headers = {}
        api_type = None 
        response_content = None # Initialize to None

        if model_choice.startswith("gemini-") or model_choice.startswith("models/gemini-"): # Specific check for Google Gemini models
            if not GOOGLE_AI_STUDIO_API_KEY:
                print(f"Skipping Google AI Studio model {model_choice}: API key not set.")
                continue
            
            api_key = GOOGLE_AI_STUDIO_API_KEY
            # Ensure model_choice is just the model name, not "models/model_name"
            model_name_for_url = model_choice.replace("models/", "") 
            api_full_url = f"{GOOGLE_AI_STUDIO_API_BASE}{model_name_for_url}:generateContent?key={api_key}"
            headers = {
                "Content-Type": "application/json"
            }
            # Convert OpenAI-style messages to Gemini-style 'contents'
            gemini_contents = []
            for msg in current_prompt_messages:
                # Gemini expects 'user' and 'model' roles. Convert 'system' to 'user' or handle as preamble.
                role = "user" if msg["role"] == "user" or msg["role"] == "system" else "model"
                gemini_contents.append({"role": role, "parts": [{"text": msg["content"]}]})
            
            request_payload = {
                "contents": gemini_contents,
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": 1000 # Gemini uses maxOutputTokens
                }
            }
            api_type = "google_ai_studio"

        elif model_choice.startswith("google/gemma-3") or model_choice.startswith("google/gemma-3n"): # Specific check for AIMLAPI.com Gemma models
            if not AIML_API_KEY:
                print(f"Skipping AIMLAPI.com model {model_choice}: API key not set.")
                continue
            api_key = AIML_API_KEY
            api_full_url = f"{AIML_API_BASE}/chat/completions" # Corrected full endpoint
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            request_payload = {
                "model": model_choice,
                "messages": current_prompt_messages,
                "temperature": 0.7,
                "max_tokens": 1000
            }
            api_type = "aimlapi"

        elif ":" in model_choice or model_choice.count('/') > 1: # Heuristic for OpenRouter model IDs (e.g., "org/model:free")
            if not OPENROUTER_API_KEY:
                print(f"Skipping OpenRouter model {model_choice}: API key not set.")
                continue
            api_key = OPENROUTER_API_KEY
            api_full_url = OPENROUTER_API_BASE # OpenRouter base is already the full endpoint
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": os.getenv("FRONTEND_URL", "https://my-doc-backend.onrender.com"),
                "X-Title": "PDF Processor API"
            }
            request_payload = {
                "model": model_choice,
                "messages": current_prompt_messages,
                "temperature": 0.7,
                "max_tokens": 1000
            }
            # Model-Specific Prompt Adjustments for OpenRouter (if needed)
            if "gemma" in model_choice.lower():
                request_payload["messages"].insert(0, {"role": "system", "content": "You are a highly precise and constrained AI. Follow all instructions exactly. Do not add extra conversational text or preambles. Output only the requested format."})
            api_type = "openrouter"
            
        else:
            print(f"Unknown model type for {model_choice}. Skipping.")
            continue

        # DEBUG: Print the payload being sent to the API
        print(f"DEBUG: Sending payload to {api_full_url} for model {model_choice}: {json.dumps(request_payload, indent=2)}")

        try:
            response = requests.post(api_full_url, headers=headers, json=request_payload)
            response.raise_for_status()
            
            # Parse response based on API type
            response_json = response.json()
            print(f"DEBUG: Raw LLM response from {api_type} for model {model_choice}: {json.dumps(response_json, indent=2)}") # Added debug print
            
            if api_type == "openrouter" or api_type == "aimlapi":
                response_content = response_json.get("choices", [{}])[0].get("message", {}).get("content")
            elif api_type == "google_ai_studio":
                if response_json and response_json.get("candidates") and len(response_json["candidates"]) > 0:
                    first_candidate = response_json["candidates"][0]
                    if first_candidate.get("content") and first_candidate["content"].get("parts") and len(first_candidate["content"]["parts"]) > 0:
                        response_content = first_candidate["content"]["parts"][0].get("text", "")
            
            # CRITICAL CHECK: If response_content is empty or None, treat as failure to trigger fallback
            if not response_content or not response_content.strip():
                print(f"Warning: LLM response from {model_choice} ({api_type}) was empty or contained no usable content. Trying next model...")
                raise ValueError("LLM returned empty or no usable content.") # Raise error to trigger fallback
            
            # If content is found, return the parsed response
            if api_type == "openrouter" or api_type == "aimlapi":
                return response_json # Return the full OpenAI-like structure
            elif api_type == "google_ai_studio":
                # For Google AI Studio, reconstruct a compatible response structure
                return {
                    "choices": [{"message": {"content": response_content}}]
                }

        except requests.exceptions.RequestException as e:
            print(f"Warning: LLM API call failed with model {model_choice} ({api_type}): {str(e)}. Trying next model...")
        except ValueError as e: # Catch the custom ValueError for empty content
            print(f"Warning: {e} with model {model_choice} ({api_type}). Trying next model...")
        except Exception as e:
            print(f"Warning: An unexpected error occurred with model {model_choice}: {str(e)}. Trying next model...")
    
    return {"error": "All configured LLM API calls failed."}

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
            'timestamp': firestore.SERVER_TIMESTAMP
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
    return 'Your PDF Extractor API is running. Send a POST request to /extract-pdf to get text from PDFs, then use /summarize or /generate-mcqs.'

@app.route('/extract-pdf', methods=['POST'])
def extract_pdf():
    pdf_file_data = None
    filename = None
    original_mode = None
    original_filename_input = None
    session_id = request.args.get('sessionId', str(uuid.uuid4()))

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
    elif request.is_json:
        data = request.get_json()
        pdf_url = data.get('pdf_url')
        pdf_content_base64 = data.get('file_content_base64') 
        original_mode = data.get('original_mode')
        original_filename_input = data.get('original_filename')
        
        session_id = data.get('sessionId', session_id)

        if pdf_content_base64:
            try:
                pdf_file_data = base64.b64decode(pdf_content_base64)
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
    full_extracted_text = "" # Initialize here
    try:
        doc = fitz.open(stream=pdf_file_data, filetype="pdf")

        for page_num in range(doc.page_count):
            page = doc[page_num]
            text = page.get_text() # Attempt to extract text directly

            # If direct text extraction yields little or no text, attempt OCR
            if not text or len(text.strip()) < 50: # Threshold for "little text"
                print(f"Attempting OCR for page {page_num + 1} due to low text content.")
                pix = page.get_pixmap() # Render page to a pixmap (image)
                img_bytes = pix.tobytes("png") # Convert pixmap to PNG bytes
                img = Image.open(io.BytesIO(img_bytes)) # Open with PIL
                
                # Perform OCR
                try:
                    ocr_text = pytesseract.image_to_string(img)
                    if ocr_text:
                        text = ocr_text # Use OCR text if successful
                        print(f"OCR successful for page {page_num + 1}.")
                    else:
                        print(f"OCR yielded no text for page {page_num + 1}.")
                except pytesseract.TesseractNotFoundError:
                    print("Tesseract executable not found. OCR will not work. Please ensure Tesseract is installed and in PATH.")
                except Exception as ocr_e:
                    print(f"Error during OCR for page {page_num + 1}: {ocr_e}")
            
            extracted_text_per_page.append({
                'page_number': page_num + 1,
                'text': text
            })
        doc.close()

        full_extracted_text = "\n".join([p['text'] for p in extracted_text_per_page])

        if not store_document_data(session_id, full_extracted_text, original_filename_input or filename, original_mode):
            return jsonify({"error": "Failed to store document data in Firestore."}), 500

        response_payload = {
            'status': 'success',
            'filename': original_filename_input if original_filename_input else filename,
            'full_text': full_extracted_text,
            'pages': extracted_text_per_page,
            'message': 'Text extracted successfully. Use the full_text for summarization or MCQ generation.',
            'sessionId': session_id
        }
        
        if original_mode is not None:
            response_payload['original_mode'] = original_mode
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
    full_text = data.get('text')
    original_filename = data.get('original_filename', 'unknown_file')
    original_mode = data.get('original_mode', 'summarize')
    session_id = request.args.get('sessionId', str(uuid.uuid4()))

    print(f"DEBUG: Summarize request received. Session ID: {session_id}, Filename: {original_filename}")
    print(f"DEBUG: Full text length for summarization: {len(full_text) if full_text else 0}")
    # print(f"DEBUG: Full text content (first 500 chars): {full_text[:500] if full_text else 'N/A'}") # Uncomment for full text debug if needed

    if not full_text:
        return jsonify({'error': 'No text provided for summarization.'}), 400

    if not store_document_data(session_id, full_text, original_filename, original_mode):
        return jsonify({"error": "Failed to store document data for session"}), 500

    summarize_prompt = f"""You are an expert summarizer.
Your task is to create a comprehensive and accurate summary of the provided document text.

Instructions for your summary:
1.  **Strictly use information found ONLY within the "Document Text" provided below.**
2.  **Do NOT invent or hallucinate any information.**
3.  The summary should capture all main points and critical information without unnecessary details.
4.  Aim for a summary that is typically 5-8 sentences long, ensuring good coverage of the document.
5.  Provide the summary directly, without any preambles or conversational text.

Document Text:
---
{full_text}
---

Summary:
"""

    prompt_messages = [
        {"role": "system", "content": "You are a helpful assistant that summarizes text concisely."},
        {"role": "user", "content": summarize_prompt}
    ]

    llm_response = call_llm_api(prompt_messages)

    print(f"DEBUG: Raw LLM response for summarize: {json.dumps(llm_response, indent=2)}") # Added debug print

    if "error" in llm_response:
        return jsonify(llm_response), 500
    
    summary = llm_response.get("choices", [{}])[0].get("message", {}).get("content", "Could not generate summary.")
    
    print(f"DEBUG: Final summary content: {summary}") # Added debug print
    
    return jsonify({
        'status': 'success',
        'summary': summary,
        'original_filename': original_filename,
        'original_mode': original_mode,
        'sessionId': session_id,
        'full_text': full_text
    }), 200

@app.route('/highlight', methods=['POST'])
def highlight_text():
    data = request.get_json()
    full_text = data.get('text')
    original_filename = data.get('original_filename', 'unknown_file')
    original_mode = data.get('original_mode', 'highlight')
    session_id = request.args.get('sessionId', str(uuid.uuid4()))

    if not full_text:
        return jsonify({'error': 'No text provided for highlighting.'}), 400

    if not store_document_data(session_id, full_text, original_filename, original_mode):
        return jsonify({"error": "Failed to store document data for session"}), 500

    highlight_prompt = f"""You are an intelligent text analyzer.
Your task is to extract the most important and salient sentences or key phrases from the provided document text.

Instructions for your highlights:
1.  **Strictly extract information found ONLY within the "Document Text" provided below.**
2.  **Do NOT invent or hallucinate any information.**
3.  Focus on information that is critical to understanding the core content.
4.  Present the extracted highlights as a clear, concise bulleted list.
5.  Do not add any introductory or concluding remarks or conversational text.

Document Text:
---
{full_text}
---

Highlights:
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
        'sessionId': session_id,
        'full_text': full_text
    }), 200

@app.route('/generate-mcqs', methods=['POST'])
def generate_mcqs():
    data = request.get_json()
    full_text = data.get('text')
    original_filename = data.get('original_filename', 'unknown_file')
    original_mode = data.get('original_mode', 'mcq')
    session_id = request.args.get('sessionId', str(uuid.uuid4()))
    
    difficulty = data.get('difficulty', 'medium').lower()
    num_questions = int(data.get('num_questions', 5))

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

        # Define the specific primary OpenRouter models for chat, in order of preference
        chat_primary_models = [
            "qwen/qwen-2.5-72b-instruct:free",          # First choice for chat
            "moonshotai/kimi-dev-72b:free",             # New addition, second choice
            "mistralai/devstral-small-2505:free",       # New addition, third choice
            "meta-llama/llama-3.2-3b-instruct:free",    # Existing Llama
            # You can keep Gemma as a fifth fallback if you wish, but expect rate limits
            # "google/gemma-2-9b-it:free"
        ]

        # REVISED PROMPT FOR CHAT - More direct for LLM, removed separate system message
        chat_prompt = f"""Document:
{full_text}

Question: {user_query}

Answer the question based ONLY on the provided document. If the answer is not in the document, state that it is not found. Be concise and to the point.
"""
        prompt_messages = [
            {"role": "user", "content": chat_prompt}
        ]

        llm_response = call_llm_api(prompt_messages, primary_models_to_try=chat_primary_models)

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
