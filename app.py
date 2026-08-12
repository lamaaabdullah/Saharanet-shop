import os
import re
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai
from pypdf import PdfReader
import requests

# =========================================================
# SETUP
# =========================================================

load_dotenv()

GEMINI_KEY = os.getenv("GEMINI_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
PDF_PATH = os.getenv("PDF_PATH", "knowledge_base.pdf")

genai.configure(api_key=GEMINI_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

app = Flask(__name__)
CORS(app)


# =========================================================
# STEP 1: READ THE PDF AND SPLIT IT INTO SMALL CHUNKS
# =========================================================

def read_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    full_text = ""
    for page in reader.pages:
        full_text = full_text + page.extract_text()
    return full_text


def split_into_chunks(text: str, chunk_size: int = 600) -> list[str]:
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

def get_words(text: str) -> set:
    words = re.findall(r"[a-zA-Z\u0600-\u06FF]+", text.lower())
    return set(words)


def find_relevant_chunks(user_message: str, top_n: int = 3) -> list:
    question_words = get_words(user_message)
    scored_chunks = []
    for chunk in pdf_chunks:
        chunk_words = get_words(chunk)
        overlap_count = len(question_words & chunk_words)
        scored_chunks.append((overlap_count, chunk))
    scored_chunks.sort(key=lambda pair: pair[0], reverse=True)
    return scored_chunks[:top_n]


# =========================================================
# STEP 2b: DETECT SMALL TALK
# =========================================================

SMALL_TALK_WORDS = {
    "hi", "hello", "hey", "hiya", "yo",
    "thanks", "thank", "thankyou", "ok", "okay", "cool", "great", "nice",
    "bye", "goodbye", "morning", "evening",
    "مرحبا", "هلا", "السلام", "عليكم", "وعليكم", "شكرا", "شكرًا",
    "تمام", "طيب", "اوك", "أوك", "يعطيك", "العافية", "مساء", "صباح", "الخير", "النور"
}


def is_small_talk(user_message: str) -> bool:
    words = get_words(user_message)
    text = user_message.strip()

    if len(words) <= 4 and len(words & SMALL_TALK_WORDS) > 0:
        return True
    if len(text) <= 15 and "?" not in text and "؟" not in text:
        return True
    return False


def answer_small_talk(user_message: str) -> str:
    prompt = f"""
You are a warm, friendly customer support assistant for Sahara Net, a
Saudi telecom and cloud services company. The user just sent a casual
message (a greeting, thanks, etc.), not a real question:

"{user_message}"

Reply warmly and naturally in 1-2 short sentences, in the SAME language
the user used. Mention you're happy to help with Sahara Net's services,
billing, or support.
"""
    response = model.generate_content(prompt)
    return response.text.strip()


# =========================================================
# STEP 3: ANSWER THE QUESTION USING ONLY THE RELEVANT CHUNKS
# =========================================================

def answer_question(user_message: str, context_text: str) -> str:
    try:
        prompt = f"""
You are a warm, helpful support assistant for Sahara Net. Here is some
information from the official Sahara Net knowledge base that might help
answer the question:

{context_text}

User question: "{user_message}"

Rules:
- Answer using ONLY the information above. Do not use anything you
  already know from outside this text.
- Sound natural and friendly, not robotic. Short and clear, 2-4
  sentences max.
- Answer in the SAME language the user used (Arabic or English).
- IMPORTANT: if the information above does NOT fully answer the
  question, your reply must START with the exact word UNSURE: (followed
  by a short, friendly message stating you couldn't find a direct answer and that a member of our support team will be happy to follow up with you).
"""
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Gemini Error: {e}")
        return "UNSURE: An error occurred while generating the answer."


# =========================================================
# STEP 3b: TAKE ACTION - open a real support ticket
# =========================================================

def create_support_ticket(customer_id, user_message: str, ai_reply: str) -> bool:
    if not customer_id:
        print("Skipped ticket creation: no customer_id (user not logged in).")
        return False

    url = f"{SUPABASE_URL}/rest/v1/support_logs"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "title": "AI Assistant could not fully answer a question",
        "content": f"Customer asked: \"{user_message}\"\n\nAI's partial answer: {ai_reply}",
        "status": "open",
        "customer_id": int(customer_id)
    }
    response = requests.post(url, headers=headers, json=data)

    if response.status_code != 201:
        print("Ticket creation failed:", response.status_code, response.text)
    return response.status_code == 201


# =========================================================
# STEP 4: CLASSIFY THE QUESTION
# =========================================================

CATEGORIES = [
    "Shared Hosting Linux",
    "Windows Web Hosting",
    "General Hosting",
    "Billing",
    "DNS - Nameservers",
    "Saudi Domains",
    "SSL Certificate Support",
    "Mail Settings",
    "Sahara Website Builder",
    "Mobile and Device Settings",
    "Internet Services",
    "General"
]


def classify_category(user_message: str) -> str:
    try:
        prompt = f"""
Classify this message into EXACTLY ONE of these categories:
{", ".join(CATEGORIES)}

Reply with ONLY the category name from that list, nothing else.

Message: "{user_message}"
"""
        response = model.generate_content(prompt)
        category = response.text.strip()
        return category if category in CATEGORIES else "General"
    except Exception:
        return "General"


# =========================================================
# STEP 5: SAVE THE CONVERSATION TO THE DATABASE
# =========================================================

def save_chat_log(session_id, user_message, ai_reply, category, customer_id) -> None:
    url = f"{SUPABASE_URL}/rest/v1/ai_chat_logs"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "session_id": session_id or str(uuid.uuid4()),
        "customer_id": int(customer_id) if customer_id else None,
        "user_message": user_message,
        "ai_reply": ai_reply,
        "category": category
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code != 201:
        print("Chat log save failed:", response.status_code, response.text)


# =========================================================
# ROUTES
# =========================================================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "pdf_characters": len(pdf_text), "chunks": len(pdf_chunks)})


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message")
    session_id = data.get("session_id")
    customer_id = data.get("customer_id")

    if not user_message:
        return jsonify({"error": "message is required"}), 400

    # Step A: greeting / small talk
    if is_small_talk(user_message):
        ai_reply = answer_small_talk(user_message)
        category = "General"
        save_chat_log(session_id, user_message, ai_reply, category, customer_id)
        return jsonify({"reply": ai_reply, "category": category, "ticket_created": False})

    # Step B: find the chunks that best match this real question
    top_chunks = find_relevant_chunks(user_message)
    best_score = top_chunks[0][0]

    # Step C: OUT-OF-SCOPE GUARD - score is 0
    if best_score == 0:
        ai_reply = ("I'm sorry, I couldn't find a direct answer to your question in the information provided. "
                    "A member of our support team will be happy to follow up with you to help you get these details.")
        category = "General"
        
        ticket_created = create_support_ticket(customer_id, user_message, ai_reply)
        
        save_chat_log(session_id, user_message, ai_reply, category, customer_id)
        return jsonify({"reply": ai_reply, "category": category, "ticket_created": ticket_created})

    # Step D: build context and get the answer from Gemini
    context_text = "\n\n---\n\n".join(chunk for score, chunk in top_chunks)
    ai_reply = answer_question(user_message, context_text)
    category = classify_category(user_message)

    # Step E: Comprehensive check for English and Arabic unsure/apology scenarios
    ticket_created = False
    lower_reply = ai_reply.lower()

    is_unsure = (
        ai_reply.startswith("UNSURE:") or 
        # English Scenarios
        "couldn't find" in lower_reply or 
        "can't find" in lower_reply or 
        "not found" in lower_reply or 
        "i'm sorry" in lower_reply or
        "sorry" in lower_reply or
        "an error occurred" in lower_reply or
        # Arabic Scenarios
        "عذرا" in ai_reply or
        "عذرآ" in ai_reply or
        "آسف" in ai_reply or
        "عفو" in ai_reply or
        "لا أستطيع" in ai_reply or
        "لا يمكنني" in ai_reply or
        "لا توفر" in ai_reply or
        "لا يتوفر" in ai_reply or
        "لا يوجد" in ai_reply or
        "لم أجد" in ai_reply or
        "لم يتم العثور" in ai_reply or
        "معلومات كافية" in ai_reply or
        "فريق الدعم" in ai_reply
    )

    if is_unsure:
        if ai_reply.startswith("UNSURE:"):
            ai_reply = ai_reply.replace("UNSURE:", "", 1).strip()
            
        ticket_created = create_support_ticket(customer_id, user_message, ai_reply)

    # Step F: save the conversation
    save_chat_log(session_id, user_message, ai_reply, category, customer_id)

    return jsonify({"reply": ai_reply, "category": category, "ticket_created": ticket_created})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
