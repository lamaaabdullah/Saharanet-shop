from datetime import datetime
import logging
import os
import re
import traceback
import uuid
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
import google.generativeai as genai
from pypdf import PdfReader
import requests

# =========================================================
# SETUP & CONFIGURATION
# =========================================================

load_dotenv()

# إعداد الـ Logging لرؤية جميع التفاصيل في Render Logs
logging.basicConfig(level=logging.INFO)

GEMINI_KEY = os.getenv("GEMINI_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
PDF_PATH = os.getenv("PDF_PATH", "knowledge_base.pdf")

if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)
    # استخدام gemini-1.5-flash لضمان أعلى توافقية واستقرار
    model = genai.GenerativeModel("gemini-1.5-flash")
else:
    model = None

app = Flask(__name__)
CORS(app)


# =========================================================
# STEP 1: READ PDF & SPLIT INTO CHUNKS
# =========================================================


def read_pdf_text(path: str) -> str:
    try:
        reader = PdfReader(path)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() or ""
        return full_text
    except Exception as e:
        logging.error(f"Error reading PDF: {e}")
        return ""


def split_into_chunks(text: str, chunk_size: int = 2000) -> list:
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i : i + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


try:
    pdf_text = read_pdf_text(PDF_PATH)
    pdf_chunks = split_into_chunks(pdf_text)
    logging.info(
        f"PDF loaded successfully. Characters: {len(pdf_text)} | Chunks: {len(pdf_chunks)}"
    )
except Exception as e:
    logging.warning(f"Could not load PDF on startup: {e}")
    pdf_text = ""
    pdf_chunks = []


# =========================================================
# STEP 2: HELPER FUNCTIONS & SMALL TALK DETECTION
# =========================================================


def get_words(text: str) -> set:
    words = re.findall(r"[a-zA-Z\u0600-\u06FF]+", text.lower())
    return set(words)


def find_relevant_chunks(user_message: str, top_n: int = 3) -> list:
    try:
        question_words = get_words(user_message)
        scored_chunks = []
        for chunk in pdf_chunks:
            chunk_words = get_words(chunk)
            overlap_count = len(question_words & chunk_words)
            scored_chunks.append((overlap_count, chunk))

        # الفرز بطريقة آمنة
        scored_chunks.sort(key=lambda item: item[0], reverse=True)
        return scored_chunks[:top_n]
    except Exception as e:
        logging.error(f"Error in find_relevant_chunks: {e}")
        return [(0, chunk) for chunk in pdf_chunks[:top_n]]


SMALL_TALK_WORDS = {
    "hi",
    "hello",
    "hey",
    "hiya",
    "yo",
    "thanks",
    "thank",
    "thankyou",
    "ok",
    "okay",
    "cool",
    "great",
    "nice",
    "bye",
    "goodbye",
    "morning",
    "evening",
    "مرحبا",
    "هلا",
    "السلام",
    "عليكم",
    "وعليكم",
    "شكرا",
    "شكرًا",
    "تمام",
    "طيب",
    "اوك",
    "أوك",
    "يعطيك",
    "العافية",
    "مساء",
    "صباح",
    "الخير",
    "النور",
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
    try:
        if not model:
            return "Hello! How can I help you today with Sahara Net services?"

        prompt = f"""
You are a warm, friendly customer support assistant for Sahara Net.
The user sent a greeting or casual message: "{user_message}"
Reply warmly in 1-2 short sentences in the SAME language.
"""
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logging.error(f"Error in answer_small_talk: {e}")
        return "Hello! How can I help you today with Sahara Net services?"


# =========================================================
# STEP 3: ANSWER QUESTION VIA GEMINI
# =========================================================


def answer_question(user_message: str, context_text: str) -> str:
    try:
        if not model:
            return "UNSURE: Model not initialized."

        # تنظيف وقص النصوص الطويلة جداً لضمان عدم حدوث خطأ في API
        clean_context = context_text[:4000].strip()

        prompt = f"""
You are a warm, helpful support assistant for Sahara Net.

Context Information:
{clean_context}

User Question: "{user_message}"

Rules:
- Answer using ONLY the information provided in the context above.
- Be clear and concise (2-4 sentences max).
- Reply in the SAME language used by the user.
- If the context does not contain enough information to answer, start your reply with 'UNSURE:'.
"""
        response = model.generate_content(prompt)

        if response and hasattr(response, "text") and response.text:
            return response.text.strip()
        else:
            return "UNSURE: No response generated."

    except Exception as e:
        logging.error(f"GEMINI ERROR DETAILS:\n{traceback.format_exc()}")
        return "UNSURE: An error occurred while generating the answer."


# =========================================================
# STEP 3b: CREATE SUPPORT TICKET IN SUPABASE
# =========================================================


def create_support_ticket(
    customer_id,
    user_message: str,
    ai_reply: str,
    admin_id=None,
) -> bool:
    try:
        url = f"{SUPABASE_URL}/rest/v1/support_logs"
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        }

        now = datetime.utcnow().isoformat()

        data = {
            "title": "Automated Support Request from Bot",
            "content": f"Customer Question: {user_message}\n\nBot Reply: {ai_reply}",
            "status": "open",
            "created_date": now,
            "updated_date": now,
            "phone": "",
            "contact_email": "",
        }

        if customer_id and str(customer_id).isdigit():
            data["customer_id"] = int(customer_id)

        if admin_id and str(admin_id).isdigit():
            data["admin_id"] = int(admin_id)

        response = requests.post(url, headers=headers, json=data, timeout=10)
        logging.info(
            f"Ticket Creation Status: {response.status_code} | Body: {response.text}"
        )
        return response.status_code in [200, 201]
    except Exception as e:
        logging.error(f"Error creating support ticket:\n{traceback.format_exc()}")
        return False


# =========================================================
# STEP 4: CLASSIFY CATEGORY
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
    "General",
]


def classify_category(user_message: str) -> str:
    try:
        if not model:
            return "General"
        prompt = f"Classify this message into EXACTLY ONE: {', '.join(CATEGORIES)}\nMessage: \"{user_message}\""
        response = model.generate_content(prompt)
        cat = response.text.strip()
        return cat if cat in CATEGORIES else "General"
    except Exception:
        return "General"


# =========================================================
# STEP 5: SAVE CHAT LOG TO SUPABASE
# =========================================================


def save_chat_log(
    session_id, user_message, ai_reply, category, customer_id
) -> None:
    try:
        if not SUPABASE_URL or not SUPABASE_KEY:
            return

        url = f"{SUPABASE_URL}/rest/v1/ai_chat_logs"
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        }
        data = {
            "session_id": session_id if session_id else str(uuid.uuid4()),
            "user_message": user_message,
            "ai_reply": ai_reply,
            "category": category,
        }

        if customer_id:
            data["customer_id"] = customer_id

        requests.post(url, headers=headers, json=data, timeout=5)
    except Exception as e:
        logging.error(f"Failed to save chat log:\n{traceback.format_exc()}")


# =========================================================
# ROUTES
# =========================================================


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "pdf_characters": len(pdf_text),
            "chunks": len(pdf_chunks),
        }
    )


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json() or {}
        user_message = data.get("message")
        session_id = data.get("session_id")
        customer_id = data.get("customer_id")

        if not user_message:
            return jsonify({"error": "message is required"}), 400

        # Step A: Handle Small Talk
        if is_small_talk(user_message):
            ai_reply = answer_small_talk(user_message)
            category = "General"
            save_chat_log(
                session_id, user_message, ai_reply, category, customer_id
            )
            return jsonify(
                {
                    "reply": ai_reply,
                    "category": category,
                    "ticket_created": False,
                }
            )

        # Step B: Match Relevant PDF Chunks
        top_chunks = find_relevant_chunks(user_message)
        best_score = top_chunks[0][0] if top_chunks else 0

        # Step C: Out-Of-Scope Guard
        if best_score == 0 and len(pdf_chunks) > 0:
            ai_reply = (
                "I can only help with questions about Sahara Net's services, "
                "plans, billing, and support. Please ask something related to "
                "Sahara Net, or use the Customer Support option for anything else."
            )
            category = "General"
            save_chat_log(
                session_id, user_message, ai_reply, category, customer_id
            )
            return jsonify(
                {
                    "reply": ai_reply,
                    "category": category,
                    "ticket_created": False,
                }
            )

        # Step D: Generate Answer using PDF Context
        context_text = "\n---\n".join(chunk for score, chunk in top_chunks if score > 0)
        if not context_text:
            context_text = "\n---\n".join(chunk for score, chunk in top_chunks)

        ai_reply = answer_question(user_message, context_text)
        category = classify_category(user_message)

        # Step E: Trigger Support Ticket if Unsure
        ticket_created = False
        unsure_keywords = [
            "UNSURE:",
            "فريق الدعم",
            "غير متأكد",
            "لست متأكداً",
            "لست متأكد",
            "غير متاكد",
            "تواصل مع الدعم",
            "سيتابع معك",
            "An error occurred",
        ]

        if any(keyword in ai_reply for keyword in unsure_keywords):
            clean_reply = ai_reply.replace("UNSURE:", "").strip()
            ticket_created = create_support_ticket(
                customer_id, user_message, clean_reply
            )

        # Step F: Save Chat Log
        save_chat_log(
            session_id, user_message, ai_reply, category, customer_id
        )

        return jsonify(
            {
                "reply": ai_reply,
                "category": category,
                "ticket_created": ticket_created,
            }
        )

    except Exception as e:
        logging.error(f"Unhandled Error in /chat route:\n{traceback.format_exc()}")
        return (
            jsonify(
                {
                    "reply": "An issue occurred while processing your request.",
                    "category": "General",
                    "ticket_created": False,
                }
            ),
            500,
        )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
