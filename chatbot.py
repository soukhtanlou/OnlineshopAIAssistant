import streamlit as st
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Load data and models with error handling
@st.cache_resource
def load_models():
    products = pd.read_csv("test_amazon_products.csv")
    products["search_text"] = (
        products["title"] + " " +
        products["description"].fillna("") + " " +
        products["categories"].fillna("")
    )
    
    # Try loading SentenceTransformer with fallback
    try:
        model_search = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", cache_folder="/tmp/model_cache")
    except Exception as e:
        st.error(f"Error loading SentenceTransformer: {e}. Using a lighter model.")
        model_search = SentenceTransformer("all-MiniLM-L6-v2", cache_folder="/tmp/model_cache")
    
    product_embeddings = model_search.encode(products["search_text"].tolist(), convert_to_tensor=True)
    
    # Try loading GPT-2 with fallback
    try:
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", cache_dir="/tmp/model_cache")
        model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2", cache_dir="/tmp/model_cache")
    except Exception as e:
        st.error(f"Error loading GPT-2: {e}. Using a smaller model.")
        tokenizer = AutoTokenizer.from_pretrained("distilgpt2", cache_dir="/tmp/model_cache")
        model = AutoModelForCausalLM.from_pretrained("distilgpt2", cache_dir="/tmp/model_cache")
    
    return products, model_search, product_embeddings, tokenizer, model

products, model_search, product_embeddings, tokenizer, model = load_models()

# Search function
def search_products(query, top_n=5):
    query_embedding = model_search.encode(query, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, product_embeddings)[0]
    top_results = cos_scores.argsort(descending=True)[:top_n]
    return products.iloc[top_results].to_dict("records")

# Generate text function
def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=40,
        temperature=0.6,
        do_sample=True,
        top_k=40,
        no_repeat_ngram_size=2
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Interactive chatbot logic with dynamic questions
def process_query_interactive(initial_query, user_response=None):
    if user_response is None:
        results = search_products(initial_query)
        product_list = "\n".join(
            [f"{r['title']} - ${r['final_price']} - Rating: {r['rating']} ({r['reviews_count']} reviews)"
             for r in results]
        )
        prompt = (
            f"User asked: '{initial_query}'. Here are some options:\n{product_list}\n"
            "What’s your maximum budget?"
        )
        return generate_text(prompt), {"step": "budget", "results": results}
    
    elif user_response["step"] == "budget":
        initial_results = user_response["results"]
        budget = float(user_response["budget"])
        filtered_results = [
            r for r in initial_results
            if float(str(r['final_price']).replace('"', '')) <= budget
        ]
        product_list = "\n".join(
            [f"{r['title']} - ${r['final_price']} - Rating: {r['rating']} ({r['reviews_count']} reviews)"
             for r in filtered_results]
        )
        if "headphones" in initial_query.lower() or "هدفون" in initial_query:
            feature_prompt = "Do you want wireless or wired headphones?"
        elif "shirt" in initial_query.lower() or "پیراهن" in initial_query or "تیشرت" in initial_query:
            feature_prompt = "What size do you want (e.g., S, M, L)?"
        elif "gift" in initial_query.lower() or "کادو" in initial_query or "تولد" in initial_query:
            feature_prompt = "Is it for a child or an adult?"
        else:
            feature_prompt = "Any specific preference (e.g., color, type)?"
        
        if not filtered_results:
            prompt = (
                f"User asked: '{initial_query}' with budget ${budget}. "
                "No items found under this budget. Try a higher budget?"
            )
        else:
            prompt = (
                f"User asked: '{initial_query}' with budget ${budget}. Options:\n{product_list}\n"
                f"{feature_prompt}"
            )
        return generate_text(prompt), {"step": "features", "results": filtered_results}
    
    elif user_response["step"] == "features":
        filtered_results = user_response["results"]
        feature_preference = user_response["feature"].lower()
        if "headphones" in initial_query.lower() or "هدفون" in initial_query:
            final_results = [
                r for r in filtered_results
                if (feature_preference in ["wireless", "wired"] and
                    ("headphones" in r["title"].lower() or "audio" in r["categories"].lower() or
                     feature_preference in r["description"].lower() or feature_preference in r["title"].lower()))
            ] if feature_preference in ["wireless", "wired"] else filtered_results
        elif "shirt" in initial_query.lower() or "پیراهن" in initial_query or "تیشرت" in initial_query:
            final_results = [
                r for r in filtered_results
                if (feature_preference in ["s", "m", "l", "xl"] and
                    (feature_preference in r["title"].lower() or feature_preference in r["description"].lower()))
            ] if feature_preference in ["s", "m", "l", "xl"] else filtered_results
        else:
            final_results = filtered_results
        
        if not final_results:
            prompt = (
                f"User asked: '{initial_query}' with budget ${user_response['budget']} and '{feature_preference}'. "
                "No exact matches. Here’s the closest option:\n"
                f"{filtered_results[0]['title']} - ${filtered_results[0]['final_price']} - Rating: {filtered_results[0]['rating']}"
            )
        else:
            best_item = max(final_results, key=lambda x: (x["rating"], x["reviews_count"]))
            prompt = (
                f"User asked: '{initial_query}' with budget ${user_response['budget']} and '{feature_preference}'.\n"
                f"Best option: {best_item['title']} - ${best_item['final_price']} - Rating: {best_item['rating']} "
                f"({best_item['reviews_count']} reviews) - it’s highly rated and popular!"
            )
        return generate_text(prompt), None

# Streamlit UI
st.title("Interactive Shopping Chatbot")
st.write("Enter your query below and follow the steps!")

# State management
if "state" not in st.session_state:
    st.session_state.state = None
    st.session_state.initial_query = ""
    st.session_state.response = ""

# Step 1: Initial query
initial_query = st.text_input("What do you want to buy?", value="")
if st.button("Submit Query") and initial_query:
    st.session_state.initial_query = initial_query
    response, state = process_query_interactive(initial_query)
    st.session_state.response = response
    st.session_state.state = state

# Display current response
if st.session_state.response:
    st.text(st.session_state.response)

# Step 2: Budget input
if st.session_state.state and st.session_state.state["step"] == "budget":
    budget = st.text_input("Enter your maximum budget (e.g., 100):", value="")
    if st.button("Submit Budget") and budget:
        st.session_state.state["budget"] = budget
        response, state = process_query_interactive(st.session_state.initial_query, st.session_state.state)
        st.session_state.response = response
        st.session_state.state = state

# Step 3: Feature input
if st.session_state.state and st.session_state.state["step"] == "features":
    feature = st.text_input("Enter your preference (e.g., wireless, S, etc.):", value="")
    if st.button("Submit Preference") and feature:
        st.session_state.state["feature"] = feature
        response, state = process_query_interactive(st.session_state.initial_query, st.session_state.state)
        st.session_state.response = response
        st.session_state.state = state
