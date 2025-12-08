import os
import json
import glob
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai.types import Content, GenerateContentConfig # <-- Import the parent config class
from dotenv import load_dotenv

#1. Setup

# Load variables from the .env file in the current directory
load_dotenv()

app = Flask(__name__)
CORS(app) # Enable Cross-Origin requests

# The client automatically finds and uses the key
client = genai.Client()
print("Gemini Client initialized successfully using key from .env file.")

MODEL_NAME = 'gemini-2.5-flash'
CONTEXT_FILE_PATH = "IMP" # <-- Using the JSON file name

directory = os.getcwd()

def load_and_parse_json(file_path):
    """
    For the interview transcripts of following structure:
    [
        {"speaker": "...", "question": "...", "answer": "..."},
        ...
    ]
    """
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
    """
    Loads survey analytics JSON:
    {
        "survey_summary": {...},
        "free_text_insights": {...}
    }

    Converts it into readable text for LLM context.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    parts = [f"=== Survey Analytics File: {os.path.basename(file_path)} ==="]

    # Survey Summary
    if "survey_summary" in data:
        parts.append("\n--- SURVEY SUMMARY ---")
        for section, values in data["survey_summary"].items():
            parts.append(f"\n[{section.upper()}]")
            parts.append(json.dumps(values, indent=2))

    # Free Text Insights
    if "free_text_insights" in data:
        parts.append("\n--- FREE TEXT INSIGHTS ---")
        for question, summary in data["free_text_insights"].items():
            parts.append(f"\nQ: {question}\nSummary: {summary}")

    return "\n".join(parts)


def load_all_json_from_folder(folder_path):
    """
    Loads all .json files in the supplied folder: interview and survey.
    """
    json_paths = sorted(glob.glob(os.path.join(folder_path, "*.json")))
    parts = []

    for path in json_paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # A: Interview JSON (list)
        if isinstance(data, list):
            parts.append(load_and_parse_json(path))

        # B: Survey JSON (dict)
        elif isinstance(data, dict) and (
            "survey_summary" in data or "free_text_insights" in data
        ):
            parts.append(load_survey_json(path))

        else:
            print(f"Skipping unrecognized JSON format: {path}")

    print(f"\nLoaded {len(parts)} JSON context blocks from '{folder_path}'")
    return "\n\n".join(parts)


full_interview_context = load_all_json_from_folder(directory)

#3. Construct the Prompt
SYSTEM_INSTRUCTION = """
You are an analytical assistant, built to answer questions that cafe entreprenuers have when starting a new cafe. 
You DO NOT just repeat or summarize the context provided.

Your goals:
1. Use the interview transcript contexts as background knowledge.
2. Use the survey analytics as a customer perspective to cafe-going.
3. Use both transcripts and survey to give holistic answers.
4. Think beyond explicit text.
5. Infer patterns, motives, insights, and deeper meanings.
6. Provide thoughtful, evaluative, and analytical answers.
7. If the user asks about something subjective (e.g., fonts, design decisions), use the context to think of an answer.

Be concise, analytical, and insight-driven.

--- INTERVIEW CONTEXT START ---
{full_interview_context}
--- INTERVIEW CONTEXT END ---
"""

#4. User Query
user_query = "when did the fat labrador cafe open?"

contents = [
    # System instruction + context combined as user message
    Content(
        role="user",
        parts=[genai.types.Part.from_text(text=SYSTEM_INSTRUCTION)]
    ),

    # Context
    Content(
        role="user",
        parts=[genai.types.Part.from_text(text=full_interview_context)]
    ),

    # Actual query
    Content(
        role="user",
        parts=[genai.types.Part.from_text(text=user_query)]
    )
]

response = client.models.generate_content(
    model=MODEL_NAME,
    contents=contents,
    config=GenerateContentConfig(temperature=0.2)
)

print("Chatbot Response:\n", response.text)