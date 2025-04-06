import streamlit as st
from duckduckgo_search import DDGS
import markdown
import re
import random
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gtts import gTTS
from io import BytesIO
import base64

# Load conversation dataset
@st.cache_resource
def load_data():
    try:
        with open('conversations.json') as f:
            conv_data = json.load(f)
        conversations = []
        responses = []
        for conv in conv_data:
            conversations.append(conv['question'].lower())
            responses.append(conv['answer'])
        return conversations, responses
    except:
        return [], []

conversations, responses = load_data()

# Initialize TF-IDF vectorizer
vectorizer = TfidfVectorizer()
if conversations:
    conv_vectors = vectorizer.fit_transform(conversations)

# Knowledge base
KNOWLEDGE = {
    "bikes": [
        "🚲 Bikes are awesome! I know about:\n- Road bikes\n- Mountain bikes\n- Electric bikes",
        "Cycling fan here! Ask me about bike maintenance or types!"
    ],
    "python": [
        "🐍 Python is my favorite! Try:\n- [Python Docs](https://docs.python.org)\n- [Real Python](https://realpython.com)"
    ]
}

# Text-to-speech function
def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang='en')
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp
    except:
        return None

# Response generation
def generate_response(message):
    # 1. Try dataset match
    if conversations:
        input_vec = vectorizer.transform([message.lower()])
        similarities = cosine_similarity(input_vec, conv_vectors)
        best_match_idx = similarities.argmax()
        if similarities[0][best_match_idx] > 0.6:
            return responses[best_match_idx]
    
    # 2. Check knowledge base
    lower_msg = message.lower()
    for topic, replies in KNOWLEDGE.items():
        if re.search(r'\b' + re.escape(topic) + r'\b', lower_msg):
            return random.choice(replies)
    
    # 3. Try web search for questions
    if any(q_word in lower_msg for q_word in ['what','when','where','how','why','who']):
        try:
            with DDGS() as ddgs:
                results = [r for r in ddgs.text(message, max_results=3)]
            if results:
                response = "🔍 Here's what I found:\n\n"
                for i, r in enumerate(results, 1):
                    response += f"{i}. [{r['title']}]({r['href']})\n{r['body'][:100]}...\n\n"
                return response
        except:
            pass
    
    # 4. Final fallback
    return random.choice([
        "Interesting question! Could you rephrase that?",
        "I'm still learning about that topic.",
        "Hmm, I'm not sure. Try asking about Python or bikes."
    ])

# Streamlit UI
st.title("🤖 Advanced AI Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        response = generate_response(prompt)
        st.markdown(response)
        
        # Add audio playback if possible
        audio = text_to_speech(response)
        if audio:
            audio_base64 = base64.b64encode(audio.read()).decode('utf-8')
            st.audio(f"data:audio/mp3;base64,{audio_base64}", format='audio/mp3')
    
    st.session_state.messages.append({"role": "assistant", "content": response})

