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


def split_into_chunks(text: str, chunk_size: int = 1000) -> list[str]:
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
#
# IMPORTANT FIX: the word pattern below now matches BOTH English letters
# (a-z, A-Z) AND Arabic letters (the \u0600-\u06FF range). Before this
# fix, get_words() only understood English, so any Arabic question ended
# up with an EMPTY word set, which made it look "out of scope" every
# single time - even for perfectly normal Arabic questions about pricing
# or billing. This one line is what makes Arabic actually work.

def get_words(text: str) -> set:
    # turns a sentence into a set of lowercase words, ignoring punctuation
    # works for both English and Arabic text
    words = re.findall(r"[a-zA-Z\u0600-\u06FF]+", text.lower())
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
# STEP 2b: DETECT SMALL TALK (greetings, thanks, etc.)
# =========================================================
# The strict "only answer from the PDF" rule is great for real questions,
# but it made the assistant refuse even simple things like "hi" or
# "thank you" (since those words don't appear anywhere in our knowledge
# base). That felt cold and robotic. So we handle small talk separately,
# with a relaxed, friendly prompt instead of the strict grounded one.
#
# Two ways a message counts as small talk:
#   1) It contains a known greeting/small-talk word (Arabic or English).
#   2) OR it's just a very short message with no question mark - this
#      covers greetings in ANY other language (French, Urdu, etc.)
#      without us having to hardcode every language's word for "hi".

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

    # Case 1: a known greeting/small-talk word, in a short message
    if len(words) <= 4 and len(words & SMALL_TALK_WORDS) > 0:
        return True

    # Case 2: short message, no question mark, in ANY language.
    # A real question almost always has a "?" or "؟", or is longer than
    # a quick greeting - so this safely catches "hi" / "salut" /
    # "namaste" / etc. without needing a word list for every language.
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
# Same idea as final_output() in the weather agent demo: we take real
# data (this time, chunks of text instead of weather JSON) and ask
# Gemini to turn it into a friendly, human-sounding answer - but we
# tell it VERY clearly to only use what we give it, and not to guess.

def answer_question(user_message: str, context_text: str) -> str:
    prompt = f"""
You are Sahara Net's official AI customer support assistant, but you have a smart, witty, slightly philosophical, and highly conversational personality. You love language and enjoy adapting seamlessly to whatever dialect, language, or mix (like Arabic, English, or Arabizi) the user writes in.

The following information was retrieved from Sahara Net's official knowledge base:

--------------------
{context_text}
--------------------

Customer question:
{user_message}

Instructions:
1. Base your answers on the provided knowledge, but express them with style, intelligence, and a conversational flair (feel free to "get philosophical" or playfully comment on the language or phrasing if it fits naturally!).
2. If the user asks about personal account details or invoices ("فواتيري") that aren't in the text, smartly guide them on how to access their customer portal or billing section instead of just giving a dry error.
3. Never invent fake technical prices or policies, but maintain an engaging, human-like, and clever tone.
4. Always reply in the exact same language, tone, or dialect used by the customer.
5. Keep it concise yet expressive.

Return ONLY the final answer.
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

    # Step A: is this just a greeting / small talk? Handle it separately
    # with a relaxed, friendly reply - skip the strict knowledge-base
    # rules entirely for these.
    if is_small_talk(user_message):
        ai_reply = answer_small_talk(user_message)
        category = "General"
        save_chat_log(session_id, user_message, ai_reply, category, customer_id)
        return jsonify({"reply": ai_reply, "category": category})

    # Step B: find the chunks that best match this real question
    top_chunks = find_relevant_chunks(user_message)
    best_score = top_chunks[0][0]  # how many matching words the BEST chunk had

    # Step C: OUT-OF-SCOPE GUARD
    # If not even ONE word from the question matches anything in our
    # knowledge base, the question is almost certainly not about Sahara
    # Net at all (example: "what's the capital of France?"). In that
    # case we don't even call Gemini - we just refuse right away. This
    # saves an API call AND guarantees we never answer off-topic
    # questions using the model's general knowledge.
    if best_score == 0:
        prompt = f"""
The user asked: "{user_message}"
This doesn't seem to directly match our knowledge base about Sahara Net.
Reply politely, cleverly, and with a bit of personality in the EXACT same language/dialect the user used (whether Arabic, English, or mixed). Let them know you specialize in Sahara Net's services, plans, billing, and support, and invite them to rephrase or ask something related to the company. Keep it natural and engaging (2-3 sentences).
        """
        response = model.generate_content(prompt)
        ai_reply = response.text.strip()
        category = "General"
        save_chat_log(session_id, user_message, ai_reply, category, customer_id)
        return jsonify({"reply": ai_reply, "category": category})

    # Step D: build the context text out of only the relevant chunks
    # (NOT the whole PDF - this is what keeps the answer grounded and
    # reduces hallucination)
    context_text = "\n\n---\n\n".join(chunk for score, chunk in top_chunks)

    # Step E: get the answer and the category
    ai_reply = answer_question(user_message, context_text)
    category = classify_category(user_message)

    # Step F: save it to the database
    save_chat_log(session_id, user_message, ai_reply, category, customer_id)

    return jsonify({"reply": ai_reply, "category": category})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
