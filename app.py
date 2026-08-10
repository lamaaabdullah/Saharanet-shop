import os
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
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

nltk.download("stopwords", quiet=True)
AR_STOPWORDS = set(stopwords.words("arabic"))
EN_STOPWORDS = set(stopwords.words("english"))
ALL_STOPWORDS = AR_STOPWORDS.union(EN_STOPWORDS)


def get_words(text: str) -> set:
    words = re.findall(r"[a-zA-Z\u0600-\u06FF]+", text.lower())
    return {
        word for word in words if word not in ALL_STOPWORDS and len(word) > 1
    }


model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

-
def find_relevant_chunks_semantic(
    user_message: str,
    pdf_chunks: list[str],
    chunk_embeddings=None,
    top_n: int = 3,
) -> list: )
    if chunk_embeddings is None:
        chunk_embeddings = model.encode(pdf_chunks, convert_to_tensor=True)

    query_embedding = model.encode(user_message, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, chunk_embeddings)[0]
    top_results = cosine_scores.argsort(descending=True)[:top_n]

    results = []
    for idx in top_results:
        results.append((float(cosine_scores[idx]), pdf_chunks[idx]))

    return results
# =========================================================
# STEP 2b: DETECT SMALL TALK (greetings, thanks, etc.)
# =========================================================

SMALL_TALK_WORDS = {
    # English
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
    # Arabic
    "مرحبا",
    "مرحباً",
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
    "أهلا",
    "اهلا",
}


def is_small_talk(user_message: str) -> bool:
    words = get_words(user_message)
    text = user_message.strip()

    if not text:
        return False

    #1
    if len(words) <= 4 and len(words & SMALL_TALK_WORDS) > 0:
        return True

    # 2.
    if len(words) <= 2 and "?" not in text and "؟" not in text:
        # التأكد من عدم وجود كلمات مفتاحية استعلامية شائعة
        keywords = {
            "أسعار",
            "سعر",
            "دعم",
            "خدمة",
            "باقة",
            "فايبر",
            "عروض",
            "price",
            "cost",
            "support",
        }
        if not (words & keywords):
            return True

    return False


def answer_small_talk(user_message: str) -> str:
    prompt = f"""
You are a warm, friendly customer support assistant for Sahara Net, a Saudi telecom and cloud services company. 
The user just sent a casual message (a greeting, thanks, etc.), not a specific inquiry:

"{user_message}"

Reply warmly and naturally in 1-2 short sentences, in the EXACT same language the user used. 
Let them know you're happy to assist them with any of Sahara Net's services
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

def create_support_ticket(customer_id, user_message: str, ai_reply: str) -> bool:
    if not customer_id:
        # We can only attach a ticket to a real logged-in customer,
        # because support_logs.customer_id is required in our database.
        # (An anonymous visitor's question just won't get auto-escalated.)
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
        "customer_id": customer_id
    }
    response = requests.post(url, headers=headers, json=data)
    return response.status_code == 201


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
    if ai_reply.startswith("UNSURE:"):
        ai_reply = ai_reply.replace("UNSURE:", "", 1).strip()
        ticket_created = create_support_ticket(customer_id, user_message, ai_reply)

    # Step F: save the conversation either way
    save_chat_log(session_id, user_message, ai_reply, category, customer_id)

    return jsonify({"reply": ai_reply, "category": category, "ticket_created": ticket_created})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
