from datetime import datetime
import os
import re
import uuid
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
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
    try:
        reader = PdfReader(path)
        full_text = ""
        for page in reader.pages:
            full_text = full_text + page.extract_text()
        return full_text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""


def split_into_chunks(text: str, chunk_size: int = 2000) -> list[str]:
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i : i + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


try:
    pdf_text = read_pdf_text(PDF_PATH)
    pdf_chunks = split_into_chunks(pdf_text)
    print(
        "PDF loaded. Characters:",
        len(pdf_text),
        "| Chunks:",
        len(pdf_chunks),
    )
except Exception as e:
    print(f"Warning: Could not load PDF on startup: {e}")
    pdf_text = ""
    pdf_chunks = []


# =========================================================
# STEP 2: FIND MATCHING CHUNKS & SMALL TALK
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
    prompt = f"""
You are a warm, friendly customer support assistant for Sahara Net, a
Saudi telecom and cloud services company. The user just sent a casual
message (a greeting, thanks, etc.), not a real question:

"{user_message}"

Reply warmly and naturally in 1-2 short sentences, in the SAME language
the user used.
"""
    response = model.generate_content(prompt)
    return response.text.strip()


# =========================================================
# STEP 3: ANSWER QUESTION
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
    return response.text.strip()


# =========================================================
# STEP 3b: CREATE SUPPORT TICKET (UPDATED DATA FORMAT)
# =========================================================


def create_support_ticket(
    customer_id,
    user_message: str,
    ai_reply: str,
    admin_id=None,
    support_id=None,
) -> bool:
    try:
        # إنشاء ID للتذكرة إن لم يوجد
        if not support_id:
            support_id = str(uuid.uuid4())

        url = f"{SUPABASE_URL}/rest/v1/support_logs"
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        }

        now = datetime.utcnow().isoformat()

        # بناء البيانات وتجنب إرسال None للحقول الحساسة
        data = {
            "support_id": support_id,
            "title": "Automated Support Request from Bot",
            "content": f"Customer Question: {user_message}\n\nBot Reply: {ai_reply}",
            "status": "open",
            "created_date": now,
            "updated_date": now,
            "customer_id": customer_id,
            "admin": admin_id,
            "phone": "",  # إرسال نص فارغ لتفادي خطأ NOT NULL
            "contact_email": "",  # إرسال نص فارغ لتفادي خطأ NOT NULL
        }

        response = requests.post(url, headers=headers, json=data, timeout=5)

        print("Supabase Ticket Status:", response.status_code)
        print("Supabase Ticket Response:", response.text)

        return response.status_code in [200, 201]
    except Exception as e:
        print("Error creating support ticket:", str(e))
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
# STEP 5: SAVE CHAT LOG
# =========================================================


def save_chat_log(
    session_id, user_message, ai_reply, category, customer_id
) -> None:
    try:
        url = f"{SUPABASE_URL}/rest/v1/ai_chat_logs"
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
        }
        data = {
            "session_id": session_id or str(uuid.uuid4()),
            "user_message": user_message,
            "ai_reply": ai_reply,
            "category": category,
        }

        if customer_id:
            data["customer_id"] = customer_id

        requests.post(url, headers=headers, json=data, timeout=5)
    except Exception as e:
        print(f"Failed to save chat log: {e}")


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

        # Step A: Small talk
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

        # Step B: Match chunks
        top_chunks = find_relevant_chunks(user_message)
        best_score = top_chunks[0][0] if top_chunks else 0

        # Step C: Out of scope guard
        if best_score == 0:
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

        # Step D: Answer using context
        context_text = "\n\n---\n\n".join(chunk for score, chunk in top_chunks)
        ai_reply = answer_question(user_message, context_text)
        category = classify_category(user_message)

        # Step E: Check if unsure (English or Arabic indicators)
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
        ]

        if any(keyword in ai_reply for keyword in unsure_keywords):
            clean_reply = ai_reply.replace("UNSURE:", "").strip()
            ticket_created = create_support_ticket(
                customer_id, user_message, clean_reply
            )

        # Step F: Save log
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
        print(f"Unhandled Error in /chat route: {e}")
        return (
            jsonify(
                {
                    "error": "Internal server error",
                    "details": str(e),
                }
            ),
            500,
        )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
