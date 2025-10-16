# app.py
import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import requests
import time
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Starscape AI Chatbot",
    page_icon="🚀",
    layout="wide",
)

st.markdown("""
<style>
    .stChatMessage {
        background-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# API Configuration
# -----------------------------
def check_api_config():
    api_key = os.getenv("API_KEY")
    api_url = os.getenv("API_URL")
    
    if not api_key or not api_url:
        st.error("⚠️ **API Configuration Missing!**")
        st.error("Please create a `.env` file with:")
        st.code("API_KEY=your_api_key\nAPI_URL=your_api_url")
        st.stop()
    
    return api_key, api_url

API_KEY, API_URL = check_api_config()

# -----------------------------
# Load Database
# -----------------------------
@st.cache_data
def load_database(csv_file="Starscape_Database.csv"):
    if not os.path.exists(csv_file):
        st.error(f"❌ File not found: {csv_file}")
        return pd.DataFrame()

    df = pd.read_csv(csv_file)
    df.columns = [col.strip() for col in df.columns]
    
    numeric_cols = ["Damage_Per_Second", "Shield_Points", "Hull_Points", 
                    "Energy_Capacity", "Speed", "Acceleration", "Agility", "TIER"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    df = df.dropna(subset=["SHIP_NAME", "Speed", "Damage_Per_Second"])
    return df

# -----------------------------
# Initialize Model
# -----------------------------
@st.cache_resource
def get_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# Stream Text
# -----------------------------
def stream_text(text, delay=0.02):
    words = text.split()
    streamed = ""
    placeholder = st.empty()
    
    for word in words:
        streamed += word + " "
        placeholder.markdown(streamed + "▌")
        time.sleep(delay)
    
    placeholder.markdown(streamed)
    return streamed

# -----------------------------
# Detect Detail Request
# -----------------------------
def wants_details(query):
    """Strict detection - only when explicitly asking for details"""
    detail_phrases = [
        "tell me about", "details about", "detail on", "describe",
        "what is the", "explain", "information about", "info on"
    ]
    query_lower = query.lower()
    return any(phrase in query_lower for phrase in detail_phrases)

# -----------------------------
# Search Ships (Fixed Tier Detection)
# -----------------------------
def search_ships(query, df, model):
    if df.empty:
        return pd.DataFrame()
    
    query_lower = query.lower()
    
    # Check for specific ship name query
    for ship_name in df['SHIP_NAME'].unique():
        if ship_name.lower() in query_lower and wants_details(query_lower):
            return df[df['SHIP_NAME'].str.lower() == ship_name.lower()].reset_index(drop=True)
    
    # Full database
    if any(phrase in query_lower for phrase in ["all ships", "full database", "entire database", "show all"]):
        return df.reset_index(drop=True)
    
    # **FIXED: Tier queries with exact matching**
    # Check for tier queries FIRST before other filters
    import re
    
    # Pattern 1: "tier X" or "tier-X" (with word boundaries)
    tier_match = re.search(r'\btier[\s-]*(\d+)\b', query_lower)
    if tier_match:
        tier_num = int(tier_match.group(1))
        result = df[df['TIER'] == tier_num].reset_index(drop=True)
        if not result.empty:
            return result
    
    # Pattern 2: "tX" format (e.g., "t1", "t10")
    tier_match_short = re.search(r'\bt(\d+)\b', query_lower)
    if tier_match_short:
        tier_num = int(tier_match_short.group(1))
        result = df[df['TIER'] == tier_num].reset_index(drop=True)
        if not result.empty:
            return result
    
    # Speed queries
    if "fastest" in query_lower or "quickest" in query_lower:
        sorted_df = df.sort_values('Speed', ascending=False).reset_index(drop=True)
        return sorted_df if "all" in query_lower else sorted_df.head(10)
    
    # DPS queries
    if "dps" in query_lower or "damage" in query_lower or "strongest" in query_lower:
        sorted_df = df.sort_values('Damage_Per_Second', ascending=False).reset_index(drop=True)
        return sorted_df if "all" in query_lower else sorted_df.head(10)
    
    # Tank queries
    if "tank" in query_lower or "durable" in query_lower or "defense" in query_lower:
        df_copy = df.copy()
        df_copy['total_defense'] = df_copy['Shield_Points'] + df_copy['Hull_Points']
        sorted_df = df_copy.sort_values('total_defense', ascending=False).reset_index(drop=True)
        return sorted_df if "all" in query_lower else sorted_df.head(10)
    
    # Beginner queries
    if "beginner" in query_lower or "starter" in query_lower:
        return df[df['TIER'] <= 2].reset_index(drop=True)
    
    # Faction queries
    factions = df['FACTION'].unique()
    for faction in factions:
        if faction.lower() in query_lower:
            return df[df['FACTION'].str.lower() == faction.lower()].reset_index(drop=True)
    
    # Ship size queries
    sizes = df['SHIP_SIZE'].unique()
    for size in sizes:
        if size.lower() in query_lower:
            return df[df['SHIP_SIZE'].str.lower() == size.lower()].reset_index(drop=True)
    
    # Semantic search fallback
    ship_texts = [
        f"{r['SHIP_NAME']} {r['FACTION']} tier {r['TIER']} {r['SHIP_SIZE']} "
        f"DPS {r['Damage_Per_Second']} speed {r['Speed']}"
        for _, r in df.iterrows()
    ]

    query_emb = model.encode([query])
    data_embs = model.encode(ship_texts)
    sims = np.dot(data_embs, query_emb.T).flatten()
    top_idx = np.argsort(sims)[::-1][:10]
    
    return df.iloc[top_idx].reset_index(drop=True)

# -----------------------------
# Generate Ship Details
# -----------------------------
def generate_ship_details(ship, query):
    context = f"""User asked: {query}

Ship: {ship['SHIP_NAME']}
Faction: {ship['FACTION']}
Tier: {ship['TIER']}
Size: {ship['SHIP_SIZE']}
DPS: {ship['Damage_Per_Second']}
Speed: {ship['Speed']}
Shields: {ship['Shield_Points']}
Hull: {ship['Hull_Points']}
Agility: {ship['Agility']}
Obtained: {ship['OBTAINED_BY']}

Provide 3-4 sentences covering strengths, role, and best use cases."""

    try:
        response = requests.post(
            API_URL,
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "openai/gpt-oss-20b:free",
                "messages": [
                    {"role": "system", "content": "You are a detailed ship analyst. Be specific and insightful."},
                    {"role": "user", "content": context}
                ],
                "max_tokens": 200,
                "temperature": 0.7
            },
            timeout=15
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
    except:
        pass
    
    return f"The {ship['SHIP_NAME']} is a tier {ship['TIER']} {ship['SHIP_SIZE']} from {ship['FACTION']}."

# -----------------------------
# Format Table for AI
# -----------------------------
def format_table_for_ai(df):
    if df.empty:
        return "No data."
    
    df_subset = df.head()
    table = "| " + " | ".join(df_subset.columns) + " |\n"
    table += "|" + "|".join(["---"] * len(df_subset.columns)) + "|\n"
    
    for _, row in df_subset.iterrows():
        table += "| " + " | ".join(str(v) for v in row.values) + " |\n"
    
    if len(df):
        table += f"\n...and {len(df)} more ships"
    
    return table

# -----------------------------
# AI Summary
# -----------------------------
def ask_ai(query, results_df, df):
    query_lower = query.lower()
    
    if any(word in query_lower for word in ["hello", "hi", "hey"]) and len(query.split()) <= 3:
        return "Hey! I'm your Starscape AI. Ask me about ships!"
    
    if "help" in query_lower:
        return "I can show you ship data, compare stats, or give details when you ask. Try 'show all ships' or 'tell me about Falcon'."
    
    if results_df.empty:
        return f"No ships matched that. Database has {len(df)} ships. Try asking about factions or tiers."
    
    ships_table = format_table_for_ai(results_df)
    context = f"User: {query}\n\nFound {len(results_df)} ships:\n{ships_table}\n\nSummarize in 2-3 sentences."

    try:
        response = requests.post(
            API_URL,
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "openai/gpt-oss-20b:free",
                "messages": [
                    {"role": "system", "content": "Brief analyst. 2-3 sentences max."},
                    {"role": "user", "content": context}
                ],
                "max_tokens": 200,
                "temperature": 0.7
            },
            timeout=20
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
    except:
        pass
    
    return f"Found {len(results_df)} ships matching your query."

# -----------------------------
# Chat Interface
# -----------------------------
st.title("🚀 Starscape AI Database Analyst(BETA)")
st.caption("AI-Powered Ship Database • Under-Development")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hey! Ask me about ships or say 'tell me about [ship name]' for details!"}
    ]

df = load_database()
model = get_embedding_model()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if "table_data" in message:
            with st.expander(f"📊 {len(message['table_data'])} Ships Found - Click to Expand"):
                st.dataframe(
                    message["table_data"].dropna(axis=1, how='all'),
                    use_container_width=True,
                    hide_index=True
                )

if prompt := st.chat_input("Ask about ships, show data, or request details..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🤖 Analyzing..."):
            results = search_ships(prompt, df, model)
            show_details = wants_details(prompt)
            
            ai_response = ask_ai(prompt, results, df)
            streamed_response = stream_text(ai_response, delay=0.015)
            
            if not results.empty:
                # Show collapsed data table
                with st.expander(f"📊 {len(results)} Ships Found - Click to Expand"):
                    st.dataframe(
                        results.dropna(axis=1, how='all'),
                        use_container_width=True,
                        hide_index=True
                    )
                
                # Only show details if explicitly requested
                if show_details:
                    st.markdown("---")
                    st.markdown("**🔍 Detailed Analysis:**")
                    
                    for _, ship in results.head(3).iterrows():
                        detail = generate_ship_details(ship, prompt)
                        st.markdown(f"### {ship['SHIP_NAME']}")
                        st.markdown(detail)
                        st.markdown("---")
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": streamed_response,
                    "table_data": results
                })
            else:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": streamed_response
                })

# Sidebar
with st.sidebar:
    st.success("✅ AI Connected")
    st.caption("Model: GPT-OSS-20B")
    
    st.markdown("---")
    
    st.header("Donate me here to fund my work")
    st.markdown("https://www.paypal.com/paypalme/savagestand")

    st.markdown("---")
    
    st.header("💡 Commands")
    st.markdown("""
    **View Data:**
    - "Show all ships"
    - "Kavani Mandate ships"
    - "Fastest 10 ships"
    
    **Get Details:**
    - "Tell me about Falcon"
    - "Details on Parrot"
    """)
    
    if st.button("🔄 Clear Chat"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Chat cleared!"}
        ]
        st.rerun()
    
    st.markdown("---")
    st.info(f"📊 {len(df)} ships total")