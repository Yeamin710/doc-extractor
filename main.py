import io
import os
import requests
import fitz # PyMuPDF
import json # Added for JSON parsing in MCQ generation
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
    return 'Your PDF Extractor API is running. Send a POST request to /extract-pdf to get text from PDFs, then use /summarize or /generate-mcqs.'

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

                if 'application/pdf' not in response.headers.get('Content-Type', ''):
                    return jsonify({'error': 'URL does not point to a PDF file.'}), 400

                pdf_file_data = response.content
                filename = pdf_url.split('/')[-1]
                if not filename.lower().endswith('.pdf'):
                    filename = "downloaded_pdf.pdf"
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
        doc = fitz.open(stream=pdf_file_data, filetype="pdf")

        for page_num in range(doc.page_count):
            page = doc[page_num]
            text = page.get_text()
            extracted_text_per_page.append({
                'page_number': page_num + 1,
                'text': text
            })
        doc.close()

        # Added for direct use by LLM functions
        full_extracted_text = "\n".join([p['text'] for p in extracted_text_per_page])

        return jsonify({
            'status': 'success',
            'filename': filename,
            'full_text': full_extracted_text, # This is the key for LLM input
            'pages': extracted_text_per_page,
            'message': 'Text extracted successfully. Use the full_text for summarization or MCQ generation.'
        }), 200

    except fitz.FileDataError as e:
        return jsonify({'error': f'Invalid PDF file or corrupted data: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'An unexpected error occurred during PDF processing: {str(e)}'}), 500

# --- LLM Integration Functions ---

# Get your OpenRouter API Key from environment variables
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    print("WARNING: OPENROUTER_API_KEY environment variable not set. LLM features will not work.")

OPENROUTER_API_BASE = "https://openrouter.ai/api/v1/chat/completions"
# You can choose a different model if desired, e.g., "openai/gpt-3.5-turbo"
# Check OpenRouter.ai/models for available models and their costs
LLM_MODEL = "mistralai/mistral-7b-instruct" # A good, generally cost-effective choice

def call_llm_api(prompt_messages):
    """
    Helper function to make a call to the OpenRouter API.
    """
    if not OPENROUTER_API_KEY:
        return {"error": "LLM API key is not configured."}

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        # Optional: For OpenRouter leaderboard, replace with your site details
        "HTTP-Referer": "https://my-doc-backend.onrender.com",
        "X-Title": "PDF Processor API"
    }
    payload = {
        "model": LLM_MODEL,
        "messages": prompt_messages,
        "temperature": 0.7, # Controls creativity (0.0 for deterministic, 1.0 for more creative)
        "max_tokens": 1000 # Limit the length of the LLM's response
    }

    try:
        response = requests.post(OPENROUTER_API_BASE, headers=headers, json=payload)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"LLM API call failed: {str(e)}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred during LLM processing: {str(e)}"}

@app.route('/summarize', methods=['POST'])
def summarize_text():
    """
    Endpoint to summarize text provided in the request body.
    Expects JSON: {"text": "Your long text here"}
    """
    data = request.get_json()
    text_to_summarize = data.get('text')

    if not text_to_summarize:
        return jsonify({'error': 'No text provided for summarization.'}), 400

    prompt_messages = [
        {"role": "system", "content": "You are a helpful assistant that summarizes text concisely."},
        {"role": "user", "content": f"Summarize the following document text:\n\n{text_to_summarize}"}
    ]

    llm_response = call_llm_api(prompt_messages)

    if "error" in llm_response:
        return jsonify(llm_response), 500
    
    summary = llm_response.get("choices", [{}])[0].get("message", {}).get("content", "Could not generate summary.")
    
    return jsonify({
        'status': 'success',
        'summary': summary
    }), 200

@app.route('/generate-mcqs', methods=['POST'])
def generate_mcqs():
    """
    Endpoint to generate MCQs from text provided in the request body.
    Expects JSON: {"text": "Your document text here", "num_questions": 5}
    """
    data = request.get_json()
    text_for_mcq = data.get('text')
    num_questions = data.get('num_questions', 3) # Default to 3 questions

    if not text_for_mcq:
        return jsonify({'error': 'No text provided for MCQ generation.'}), 400
    
    if not isinstance(num_questions, int) or num_questions <= 0:
        return jsonify({'error': 'num_questions must be a positive integer.'}), 400

    mcq_prompt = f"""
    Generate {num_questions} multiple-choice questions (MCQs) with 4 options (A, B, C, D) and indicate the correct answer.
    The questions should be based on the following text:

    ---
    {text_for_mcq}
    ---

    Format the output as a JSON array of objects, where each object has:
    "question": "The question text",
    "options": ["A) option A", "B) option B", "C) option C", "D) option D"],
    "correct_answer": "A" (or B, C, D)

    Example JSON structure:
    [
      {{
        "question": "What is the capital of France?",
        "options": ["A) Berlin", "B) Paris", "C) Rome", "D) Madrid"],
        "correct_answer": "B"
      }},
      {{
        "question": "Which planet is known as the Red Planet?",
        "options": ["A) Earth", "B) Mars", "C) Jupiter", "D) Venus"],
        "correct_answer": "B"
      }}
    ]
    """

    prompt_messages = [
        {"role": "system", "content": "You are an expert at creating multiple-choice questions from provided text, always responding in valid JSON."},
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
        
        return jsonify({
            'status': 'success',
            'mcqs': mcqs_data
        }), 200
    except json.JSONDecodeError:
        return jsonify({'error': 'LLM response was not valid JSON for MCQs. Please check the LLM output format.'}), 500
    except ValueError as e:
        return jsonify({'error': f'Error parsing LLM response: {e}. Raw response: {mcqs_json_str}'}), 500
    except Exception as e:
        return jsonify({'error': f'An unexpected error occurred while processing LLM MCQ response: {str(e)}'}), 500
