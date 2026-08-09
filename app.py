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
# Splitting the PDF into small chunks (instead of sending the WHOLE pdf
# every time) lets us pick only the chunks that are actually relevant
# to the question the user asked - this is called "Retrieval-Augmented
# Generation" (RAG): we RETRIEVE the relevant text first, then GENERATE
# the answer using only that text. This keeps answers grounded and
# reduces hallucination.

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
# Simple word-overlap matching (no embeddings/vector math - those need
# extra libraries like numpy that weren't covered in the course). The
# word pattern below matches BOTH English letters (a-z, A-Z) AND Arabic
# letters (\u0600-\u06FF), so this works correctly for Arabic questions
# too, not just English ones.

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
# Greetings and thank-yous don't share any words with our knowledge
# base PDF, so without this check they'd wrongly get treated as
# "out of scope". This handles them separately with a relaxed, warm
# reply instead of the strict grounded rules.

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
mention you're happy to help with anything about Sahara Net's services,
billing, or support.
"""
    response = model.generate_content(prompt)
    return response.text.strip()


# =========================================================
# STEP 3: ANSWER THE QUESTION USING ONLY THE RELEVANT CHUNKS
# =========================================================
# Same idea as final_output() in the weather agent demo: take real data
# (chunks of text instead of weather JSON) and ask Gemini to turn it
# into a friendly, human-sounding answer, grounded strictly in that data.
#
# NEW: we also ask Gemini to self-report when it ISN'T confident, by
# starting its reply with the word UNSURE:. We use that self-report in
# the next step to decide whether the agent should take action on its
# own (open a real support ticket) instead of just replying with text.

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
# This is the piece that makes this more than "just a chatbot that
# talks". When the agent judges that it can't confidently answer
# something, it doesn't just say "I'm not sure" and stop - it DECIDES
# on its own to take a real action: creating an actual support ticket
# in the database so a human takes over. Nobody told the code "create a
# ticket for THIS specific message" - the agent makes that call itself,
# based on judging the confidence of its own answer. This is the
# difference between generating text and taking action.

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
# requests.post, same idea as the weather agent's requests.get - just
# talking to a REST API, except this time it's Supabase's API instead
# of OpenWeatherMap's.

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
