import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai
from pypdf import PdfReader
import requests

# Load our keys from the .env file
load_dotenv()

GEMINI_KEY = os.getenv("GEMINI_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
PDF_PATH = os.getenv("PDF_PATH", "knowledge_base.pdf")

# Configure the Gemini model (same setup as the weather agent demo)
genai.configure(api_key=GEMINI_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

# Flask is what lets a website talk to this Python file over the internet
# (input()/print() only work in a terminal, a website can't use those)
app = Flask(__name__)
CORS(app)


# Step 1: read the whole PDF one time when the server starts
def read_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    full_text = ""
    for page in reader.pages:
        full_text = full_text + page.extract_text()
    return full_text


pdf_text = read_pdf_text(PDF_PATH)
print("PDF loaded. Characters:", len(pdf_text))


# Step 2: answer the user's question using the PDF content as context
# same idea as final_output() in the weather agent - stuff the data into
# the prompt and let Gemini turn it into a friendly answer
def answer_question(user_message: str) -> str:
    prompt = f"""
Here is the Sahara Net knowledge base:

{pdf_text}

User question: "{user_message}"

Answer using ONLY the information above. If the answer is not in the
knowledge base, say you are not sure and suggest contacting human support
instead of guessing. Keep the answer short and friendly. Answer in the
same language the user used (Arabic or English).
"""
    response = model.generate_content(prompt)
    answer = response.text.strip()
    return answer


# Step 3: classify the question, same pattern as the routing agent
def classify_category(user_message: str) -> str:
    prompt = f"""
Classify this message into exactly one category:
Billing, Technical, Account, Sales, Cybersecurity, General

Reply with ONLY the category name, nothing else.

Message: "{user_message}"
"""
    response = model.generate_content(prompt)
    category = response.text.strip()
    return category


# Step 4: save the conversation to the database
def save_chat_log(session_id, user_message, ai_reply, category, customer_id) -> None:
    url = f"{SUPABASE_URL}/rest/v1/ai_chat_logs"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "session_id": session_id,
        "customer_id": customer_id,
        "user_message": user_message,
        "ai_reply": ai_reply,
        "category": category
    }
    requests.post(url, headers=headers, json=data)


# A simple route to check the server is alive
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "pdf_characters": len(pdf_text)})


# The main route the website's chat popup calls
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message")
    session_id = data.get("session_id")
    customer_id = data.get("customer_id")

    if not user_message:
        return jsonify({"error": "message is required"}), 400

    ai_reply = answer_question(user_message)
    category = classify_category(user_message)
    save_chat_log(session_id, user_message, ai_reply, category, customer_id)

    return jsonify({"reply": ai_reply, "category": category})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
