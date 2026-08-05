import os
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai
from pypdf import PdfReader
import requests

# =========================================================
# SETUP
# =========================================================

# Load our keys from the .env file (locally) or from Render's
# Environment Variables (when deployed)
load_dotenv()

GEMINI_KEY = os.getenv("GEMINI_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
PDF_PATH = os.getenv("PDF_PATH", "knowledge_base.pdf")

# Configure the Gemini model (same setup as the weather agent demo)
genai.configure(api_key=GEMINI_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

# Flask is what lets a website talk to this Python file over the internet.
# input()/print() only work in a terminal, a website can't use those,
# so Flask gives us "routes" (URLs) the website can send requests to.
app = Flask(__name__)
CORS(app)  # allows our website (a different domain) to call this API


# =========================================================
# STEP 1: READ THE PDF AND SPLIT IT INTO SMALL CHUNKS
# =========================================================
# Why we split it into chunks instead of sending the WHOLE pdf every time:
# 1) A big PDF might be too long to send in every single message.
# 2) If we send the whole PDF, the AI sometimes gets confused and mixes
#    information together or "hallucinates" (makes things up).
# Splitting it into small chunks lets us pick only the chunks that are
# actually relevant to the question the user asked - this is called
# "Retrieval-Augmented Generation" (RAG): we RETRIEVE the relevant text
# first, then GENERATE the answer using only that text.

def read_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    full_text = ""
    for page in reader.pages:
        full_text = full_text + page.extract_text()
    return full_text


def split_into_chunks(text: str, chunk_size: int = 600) -> list[str]:
    # This is a simple way to split text: just cut it every 600 characters.
    # It's not perfect (it might cut a sentence in half) but it's simple
    # and works fine for a knowledge base like ours.
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


pdf_text = read_pdf_text(PDF_PATH)
pdf_chunks = split_into_chunks(pdf_text)
print("PDF loaded. Characters:", len(pdf_text), "| Chunks:", len(pdf_chunks))


# =========================================================
# STEP 2: FIND THE CHUNKS THAT ACTUALLY MATCH THE QUESTION
# =========================================================
# We are NOT using embeddings or vector math here (that needs extra
# libraries like numpy that weren't covered in the course). Instead we
# use a simple and easy-to-understand method: count how many of the
# SAME WORDS appear in both the question and each chunk. The chunk with
# the most matching words is probably the most relevant one.

def get_words(text: str) -> set:
    # turns a sentence into a set of lowercase words, ignoring punctuation
    # example: "What's the price?" -> {"what", "s", "the", "price"}
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return set(words)


def find_relevant_chunks(user_message: str, top_n: int = 3) -> list:
    question_words = get_words(user_message)

    scored_chunks = []
    for chunk in pdf_chunks:
        chunk_words = get_words(chunk)
        # "&" between two sets gives us the words that appear in BOTH
        overlap_count = len(question_words & chunk_words)
        scored_chunks.append((overlap_count, chunk))

    # sort so the chunk with the MOST matching words comes first
    scored_chunks.sort(key=lambda pair: pair[0], reverse=True)

    return scored_chunks[:top_n]


# =========================================================
# STEP 3: ANSWER THE QUESTION USING ONLY THE RELEVANT CHUNKS
# =========================================================
# Same idea as final_output() in the weather agent demo: we take real
# data (this time, chunks of text instead of weather JSON) and ask
# Gemini to turn it into a friendly, human-sounding answer - but we
# tell it VERY clearly to only use what we give it, and not to guess.

def answer_question(user_message: str, context_text: str) -> str:
    prompt = f"""
You are a support assistant for Sahara Net. Here is some information from
the official Sahara Net knowledge base that might help answer the question:

{context_text}

User question: "{user_message}"

Rules:
- Answer using ONLY the information above. Do not use anything you
  already know from outside this text.
- If the answer is not fully covered by the information above, say you
  are not sure and suggest contacting human support instead of guessing.
- Keep the answer short and friendly, 2-4 sentences max.
- Answer in the same language the user used (Arabic or English).
"""
    response = model.generate_content(prompt)
    answer = response.text.strip()
    return answer


# =========================================================
# STEP 4: CLASSIFY THE QUESTION (same pattern as the routing agent)
# =========================================================

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


# =========================================================
# STEP 5: SAVE THE CONVERSATION TO THE DATABASE
# =========================================================
# This uses requests.post the same way the weather agent used
# requests.get - just talking to a REST API, except this time it's
# Supabase's API instead of OpenWeatherMap's.

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


# =========================================================
# ROUTES (these are the "doors" the website can knock on)
# =========================================================

# A simple route to check the server is alive
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "pdf_characters": len(pdf_text), "chunks": len(pdf_chunks)})


# The main route the website's chat popup calls
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message")
    session_id = data.get("session_id")
    customer_id = data.get("customer_id")

    if not user_message:
        return jsonify({"error": "message is required"}), 400

    # Step A: find the chunks that best match this question
    top_chunks = find_relevant_chunks(user_message)
    best_score = top_chunks[0][0]  # how many matching words the BEST chunk had

    # Step B: OUT-OF-SCOPE GUARD
    # If not even ONE word from the question matches anything in our
    # knowledge base, the question is almost certainly not about Sahara
    # Net at all (example: "what's the capital of France?"). In that
    # case we don't even call Gemini - we just refuse right away. This
    # saves an API call AND guarantees we never answer off-topic
    # questions using the model's general knowledge.
    if best_score == 0:
        ai_reply = ("I can only help with questions about Sahara Net's services, "
                     "plans, billing, and support. Please ask something related to "
                     "Sahara Net, or use the Customer Support option for anything else.")
        category = "General"
        save_chat_log(session_id, user_message, ai_reply, category, customer_id)
        return jsonify({"reply": ai_reply, "category": category})

    # Step C: build the context text out of only the relevant chunks
    # (NOT the whole PDF - this is what keeps the answer grounded and
    # reduces hallucination)
    context_text = "\n\n---\n\n".join(chunk for score, chunk in top_chunks)

    # Step D: get the answer and the category
    ai_reply = answer_question(user_message, context_text)
    category = classify_category(user_message)

    # Step E: save it to the database
    save_chat_log(session_id, user_message, ai_reply, category, customer_id)

    return jsonify({"reply": ai_reply, "category": category})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
