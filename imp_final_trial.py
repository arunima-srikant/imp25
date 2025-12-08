import os
import json
import glob
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai.types import Content, GenerateContentConfig, Part
from dotenv import load_dotenv

# 1. Setup Flask and Environment
app = Flask(__name__)
CORS(app)
load_dotenv()

# Initialize Client
client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
print("Gemini Client initialized successfully.")

MODEL_NAME = 'gemini-2.5-flash'
DIRECTORY = os.getcwd()

# Global variable to store context so we don't reload files on every request
FULL_INTERVIEW_CONTEXT = ""

# --- Helper Functions (Same as your original code) ---

def load_and_parse_json(file_path):
    """Parses interview transcripts."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    parts = [f"=== Interview File: {os.path.basename(file_path)} ==="]
    for entry in data:
        speaker = entry.get("speaker", "Unknown")
        question = entry.get("question", "")
        answer = entry.get("answer", "")
        parts.append(f"\n{speaker}:\nQ: {question}\nA: {answer}")
    return "\n".join(parts)

def load_survey_json(file_path):
    """Parses survey analytics."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    parts = [f"=== Survey Analytics File: {os.path.basename(file_path)} ==="]
    
    if "survey_summary" in data:
        parts.append("\n--- SURVEY SUMMARY ---")
        for section, values in data["survey_summary"].items():
            parts.append(f"\n[{section.upper()}]")
            parts.append(json.dumps(values, indent=2))

    if "free_text_insights" in data:
        parts.append("\n--- FREE TEXT INSIGHTS ---")
        for question, summary in data["free_text_insights"].items():
            parts.append(f"\nQ: {question}\nSummary: {summary}")

    return "\n".join(parts)

def load_all_json_from_folder(folder_path):
    """Loads all .json files in the supplied folder."""
    json_paths = sorted(glob.glob(os.path.join(folder_path, "*.json")))
    parts = []

    for path in json_paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                parts.append(load_and_parse_json(path))
            elif isinstance(data, dict) and ("survey_summary" in data or "free_text_insights" in data):
                parts.append(load_survey_json(path))
            else:
                print(f"Skipping unrecognized JSON format: {path}")
        except Exception as e:
            print(f"Error reading {path}: {e}")

    print(f"\nLoaded {len(parts)} JSON context blocks from '{folder_path}'")
    return "\n\n".join(parts)

# --- Initialization Block ---
# We load the data immediately when the app starts
with app.app_context():
    print("Loading context files...")
    FULL_INTERVIEW_CONTEXT = load_all_json_from_folder(DIRECTORY)


# --- Flask Routes ---

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "active", "model": MODEL_NAME}), 200

@app.route('/chat', methods=['POST'])
def chat():
    """
    Expects JSON input: { "query": "your question here" }
    """
    data = request.get_json()
    user_query = data.get('query')

    if not user_query:
        return jsonify({"error": "No 'query' provided in request body"}), 400

    # Construct System Instruction
    system_instruction_text = f"""
    You are an analytical assistant, built to answer questions that cafe entrepreneurs have when starting a new cafe. 
    You DO NOT just repeat or summarize the context provided.

    Your goals:
    1. Use the interview transcript contexts as background knowledge.
    2. Use the survey analytics as a customer perspective to cafe-going.
    3. Use both transcripts and survey to give holistic answers.
    4. Think beyond explicit text.
    5. Infer patterns, motives, insights, and deeper meanings.
    6. Provide thoughtful, evaluative, and analytical answers.

    --- INTERVIEW CONTEXT START ---
    {FULL_INTERVIEW_CONTEXT}
    --- INTERVIEW CONTEXT END ---
    """

    try:
        # Construct the content payload
        # Note: We pass the system instruction as the first user part here based on your original logic.
        # Ideally, system instructions should go into the config, but this works for context injection.
        contents = [
            Content(
                role="user",
                parts=[Part.from_text(text=system_instruction_text)]
            ),
            Content(
                role="user",
                parts=[Part.from_text(text=user_query)]
            )
        ]

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=GenerateContentConfig(temperature=0.2)
        )

        return jsonify({
            "response": response.text
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run the Flask app on port 5000
    app.run(debug=True, port=5000)