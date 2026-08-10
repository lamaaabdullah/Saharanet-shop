import os
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai
from pypdf import PdfReader
import requests
from datetime import datetime
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

def read_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    full_text = ""
    for page in reader.pages:
        full_text = full_text + page.extract_text()
    return full_text


def split_into_chunks(text: str, chunk_size: int = 2000) -> list[str]:
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
# STEP 2b: DETECT SMALL TALK (greetings, thanks, etc.)
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

    # Short message, no question mark, in ANY language - safely catches
    # greetings in languages we didn't hardcode a word list for.
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
the user used (match their language exactly, whatever it is). You can
mention you're happy to help with anything about Sahara Net's .
"""
    response = model.generate_content(prompt)
    return response.text.strip()


# =========================================================
# STEP 3: ANSWER THE QUESTION USING ONLY THE RELEVANT CHUNKS
# =========================================================


def answer_question(user_message: str, context_text: str) -> str:
    prompt = f"""
You are a warm, helpful support assistant for Sahara Net. Here is some
information from the official Sahara Net knowledge base that might help
answer the question:

{context_text}

User question: "{user_message}"

Rules:
- Answer using ONLY the information above. Do not use anything you
  already know from outside this text.
- Sound natural and friendly, not robotic - like a helpful human agent,
  not a legal document. Short and clear, 2-4 sentences max.
- Answer in the SAME language the user used (Arabic or English) - if
  they wrote in Arabic, reply fully in Arabic.
- IMPORTANT: if the information above does NOT fully answer the
  question, start your reply with the exact word UNSURE: (followed by a
  short, friendly message saying you're not fully sure and a member of
  the support team will follow up). Only use UNSURE: when you genuinely
  can't answer from the text above - not just because the question is
  worded a bit differently than the knowledge base.
"""
    response = model.generate_content(prompt)
    answer = response.text.strip()
    return answer


# =========================================================
# STEP 3b: TAKE ACTION - open a real support ticket
# =========================================================


def create_support_ticket(
    customer_id, user_message: str, ai_reply: str
) -> bool:
    
    if not customer_id:
        return False

    url = f"{SUPABASE_URL}/rest/v1/support_logs"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }

    now = datetime.utcnow().isoformat()

    # 2
    data = {
        "support_id": support_id",
        "title": "طلب دعم فني تلقائي من البوت",
        "content": f"سؤال العميل: {user_message}\n\nرد البوت: {ai_reply}",
        "status": "open",
        "created_date": now,
        "updated_date": now,
        "customer_id": customer_id,
        "admin": admin_id,
        "phone": "",  # إرسال نص فارغ لتفادي خطأ NOT NULL
        "contact_email": "",  # إرسال نص فارغ لتفادي خطأ NOT NULL
    }

    response = requests.post(url, headers=headers, json=data)

   
    print("Supabase Status:", response.status_code)
    print("Supabase Response:", response.text)

    return response.status_code in [200, 201]


# =========================================================
# STEP 4: CLASSIFY THE QUESTION (same pattern as the routing agent)
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
    "General"  # fallback for anything that doesn't clearly fit above
]


def classify_category(user_message: str) -> str:
    prompt = f"""
Classify this message into EXACTLY ONE of these categories:
{", ".join(CATEGORIES)}

Reply with ONLY the category name from that list, nothing else.

Message: "{user_message}"
"""
    response = model.generate_content(prompt)
    category = response.text.strip()

    return category if category in CATEGORIES else "General"


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

    # Step A: greeting / small talk gets a relaxed, friendly reply
    if is_small_talk(user_message):
        ai_reply = answer_small_talk(user_message)
        category = "General"
        save_chat_log(session_id, user_message, ai_reply, category, customer_id)
        return jsonify({"reply": ai_reply, "category": category, "ticket_created": False})

    # Step B: find the chunks that best match this real question
    top_chunks = find_relevant_chunks(user_message)
    best_score = top_chunks[0][0]

    # Step C: OUT-OF-SCOPE GUARD - not even one matching word means this
    # isn't about Sahara Net at all, so we refuse without calling Gemini
    # and WITHOUT opening a ticket (this isn't a real business issue).
    if best_score == 0:
        ai_reply = ("I can only help with questions about Sahara Net's services, "
                     "plans, billing, and support. Please ask something related to "
                     "Sahara Net, or use the Customer Support option for anything else.")
        category = "General"
        save_chat_log(session_id, user_message, ai_reply, category, customer_id)
        return jsonify({"reply": ai_reply, "category": category, "ticket_created": False})

    # Step D: build context from only the relevant chunks, get the answer
    context_text = "\n\n---\n\n".join(chunk for score, chunk in top_chunks)
    ai_reply = answer_question(user_message, context_text)
    category = classify_category(user_message)

    # Step E: NEW - check if the agent marked itself unsure, and if so,
    # decide on its own to open a real support ticket
    ticket_created = False
    if (
        "UNSURE:" in ai_reply
        or "فريق الدعم" in ai_reply
        or "غير متأكد" in ai_reply
    ):
        clean_reply = ai_reply.replace("UNSURE:", "").strip()
        ticket_created = create_support_ticket(
            customer_id, user_message, clean_reply
        )
        
    # Step F: save the conversation either way
    save_chat_log(session_id, user_message, ai_reply, category, customer_id)

    return jsonify({"reply": ai_reply, "category": category, "ticket_created": ticket_created})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
