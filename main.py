import io
import os
import requests
import fitz # PyMuPDF
import json
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
    # Variables to capture original_mode and original_filename from request
    original_mode = None
    original_filename_input = None

    # --- Handle File Upload (form-data) ---
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if file and allowed_file(file.filename):
            pdf_file_data = file.read()
            filename = file.filename # This is the actual uploaded filename

            # Capture original_mode and original_filename if sent as form data along with the file
            original_mode = request.form.get('original_mode')
            original_filename_input = request.form.get('original_filename')

        else:
            return jsonify({'error': 'Invalid file type. Only PDF files are allowed.'}), 400
    # --- Handle URL Input (JSON payload) ---
    elif request.is_json:
        data = request.get_json()
        pdf_url = data.get('pdf_url')
        
        # Capture original_mode and original_filename if sent in JSON payload
        original_mode = data.get('original_mode')
        original_filename_input = data.get('original_filename')

        if pdf_url:
            try:
                response = requests.get(pdf_url, stream=True)
                response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)

                if 'application/pdf' not in response.headers.get('Content-Type', ''):
                    return jsonify({'error': 'URL does not point to a PDF file.'}), 400

                pdf_file_data = response.content
                filename = pdf_url.split('/')[-1] # Extract filename from URL
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

        full_extracted_text = "\n".join([p['text'] for p in extracted_text_per_page])

        # Include original_mode and original_filename in the response
        response_payload = {
            'status': 'success',
            'filename': original_filename_input if original_filename_input else filename, # Use original_filename_input if available
            'full_text': full_extracted_text,
            'pages': extracted_text_per_page,
            'message': 'Text extracted successfully. Use the full_text for summarization or MCQ generation.'
        }
        
        if original_mode is not None:
            response_payload['original_mode'] = original_mode
        if original_filename_input is not None:
            response_payload['original_filename'] = original_filename_input

        return jsonify(response_payload), 200

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

# Define your preferred models in order of preference (Primary, Fallback 1, Fallback 2)
PREFERRED_LLM_MODELS = [
    "qwen/qwen-2.5-72b-instruct:free",  # Primary choice
    "deepseek/deepseek-r1:free", # Second choice (fallback 1)
    "google/gemma-2-9b-it:free" # Third choice (fallback 2)
]

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
        "HTTP-Referer": "https://my-doc-backend.onrender.com",
        "X-Title": "PDF Processor API"
    }
    
    # Iterate through preferred models
    for model_choice in PREFERRED_LLM_MODELS:
        # Create a copy of the original prompt messages to modify for the current model
        current_prompt_messages = list(prompt_messages) 

        # --- Model-Specific Prompt Adjustments ---
        if model_choice == "google/gemma-2-9b-it:free":
            # Add more explicit instructions for Gemma
            current_prompt_messages.insert(0, {"role": "system", "content": "You are a highly precise and constrained AI. Follow all instructions exactly. Do not add extra conversational text or preambles. Output only the requested format."})
            
            if "summarize" in current_prompt_messages[-1]["content"].lower():
                current_prompt_messages[-1]["content"] += "\n\nProvide the summary directly, without any introductory phrases like 'Here is the summary:'."
            elif "extract the key highlights" in current_prompt_messages[-1]["content"].lower():
                current_prompt_messages[-1]["content"] += "\n\nEnsure output is ONLY a bulleted list of sentences/phrases. No other text."
            elif "generate multiple-choice questions" in current_prompt_messages[-1]["content"].lower():
                current_prompt_messages[-1]["content"] += "\n\nYour response MUST be a valid JSON array. Do not include any text before or after the JSON. Strictly adhere to the example format."


        payload = {
            "model": model_choice, # Use the current model choice
            "messages": current_prompt_messages, # Use the potentially modified prompt messages
            "temperature": 0.7,
            "max_tokens": 1000
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


@app.route('/summarize', methods=['POST'])
def summarize_text():
    data = request.get_json()
    text_to_summarize = data.get('text')
    # Capture original_filename and original_mode from the request
    original_filename = data.get('original_filename')
    original_mode = data.get('original_mode')
    full_text_input = data.get('full_text') # Capture full_text if sent

    if not text_to_summarize:
        return jsonify({'error': 'No text provided for summarization.'}), 400

    summarize_prompt = f"""
    You are an expert summarizer. Your task is to create a comprehensive and accurate summary of the provided document text.
    The summary should capture all main points and critical information without unnecessary details.
    Aim for a summary that is typically 5-8 sentences long, ensuring good coverage of the document.

    Document Text:
    ---
    {text_to_summarize}
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
    
    response_payload = {
        'status': 'success',
        'summary': summary
    }
    # Echo back the original data
    if original_filename is not None:
        response_payload['original_filename'] = original_filename
    if original_mode is not None:
        response_payload['original_mode'] = original_mode
    if full_text_input is not None:
        response_payload['full_text'] = full_text_input

    return jsonify(response_payload), 200

@app.route('/highlight', methods=['POST'])
def highlight_text():
    data = request.get_json()
    text_to_highlight = data.get('text')
    # Capture original_filename and original_mode from the request
    original_filename = data.get('original_filename')
    original_mode = data.get('original_mode')
    full_text_input = data.get('full_text') # Capture full_text if sent

    if not text_to_highlight:
        return jsonify({'error': 'No text provided for highlighting.'}), 400

    highlight_prompt = f"""
    You are an intelligent text analyzer. Your task is to extract the most important and salient sentences or key phrases from the provided document text.
    Focus on information that is critical to understanding the core content.
    Present the extracted highlights as a clear, concise bulleted list. Do not add any introductory or concluding remarks.

    Document Text:
    ---
    {text_to_highlight}
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

    response_payload = {
        'status': 'success',
        'highlights': highlights
    }
    # Echo back the original data
    if original_filename is not None:
        response_payload['original_filename'] = original_filename
    if original_mode is not None:
        response_payload['original_mode'] = original_mode
    if full_text_input is not None:
        response_payload['full_text'] = full_text_input

    return jsonify(response_payload), 200

@app.route('/generate-mcqs', methods=['POST'])
def generate_mcqs():
    data = request.get_json()
    text_for_mcq = data.get('text')
    num_questions = data.get('num_questions', 5)
    # Capture original_filename and original_mode from the request
    original_filename = data.get('original_filename')
    original_mode = data.get('original_mode')
    full_text_input = data.get('full_text') # Capture full_text if sent

    if not text_for_mcq:
        return jsonify({'error': 'No text provided for MCQ generation.'}), 400
    
    if not isinstance(num_questions, int) or num_questions <= 0:
        return jsonify({'error': 'num_questions must be a positive integer.'}), 400

    mcq_prompt = f"""
    You are an expert at creating challenging and clear multiple-choice questions (MCQs) from provided text.
    Generate exactly {num_questions} MCQs. Each question must have 4 distinct options (A, B, C, D), and only one correct answer.
    Ensure the questions are directly answerable from the text and cover different aspects of the content.
    Your response MUST be a valid JSON array of objects. Do not include any other text, explanations, or formatting outside the JSON.

    Text to generate MCQs from:
    ---
    {text_for_mcq}
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
            'mcqs': mcqs_data
        }
        # Echo back the original data
        if original_filename is not None:
            response_payload['original_filename'] = original_filename
        if original_mode is not None:
            response_payload['original_mode'] = original_mode
        if full_text_input is not None:
            response_payload['full_text'] = full_text_input

        return jsonify(response_payload), 200
    except json.JSONDecodeError:
        return jsonify({'error': 'LLM response was not valid JSON for MCQs. Please check the LLM output format.'}), 500
    except ValueError as e:
        return jsonify({'error': f'Error parsing LLM response: {e}. Raw response: {mcqs_json_str}'}), 500
    except Exception as e:
        return jsonify({'error': f'An unexpected error occurred while processing LLM MCQ response: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=os.environ.get("PORT", 5000))
