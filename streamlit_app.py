import streamlit as st
from transformers import pipeline

# Set up the page configuration
st.set_page_config(
    page_title="NLP Transformer Demo",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Add a title and description for the app
st.title("🤗 NLP Transformer Demo")
st.write("""
    This app demonstrates three main NLP tasks using pre-trained transformer models from Hugging Face:
    
    - **Text Generation**: Automatically generate text based on a starting prompt.
    - **Sentiment Analysis**: Determine if a given text expresses positive or negative sentiment.
    - **Translation**: Translate text from English into multiple languages, including Albanian, German, Russian, Hindi, French, Indonesian, Dutch, Mandarin (Chinese), Cantonese (Traditional Chinese), Spanish, and Portuguese.
    
    Select a task from the dropdown menu to get started!
""")

# Sidebar for navigation between tasks
page = st.sidebar.selectbox("Choose a task", ["Text Generation", "Sentiment Analysis", "Translation"])

# Function to load models
@st.cache_resource
def load_model(task_name, target_language=None):
    if task_name == "Text Generation":
        return pipeline("text-generation", model="distilgpt2")  # Lighter model for text generation
    elif task_name == "Sentiment Analysis":
        return pipeline("sentiment-analysis")
    elif task_name == "Translation":
        if target_language == "Albanian":
            return pipeline("translation_en_to_sq", model="Helsinki-NLP/opus-mt-en-sq")
        elif target_language == "German":
            return pipeline("translation_en_to_de", model="Helsinki-NLP/opus-mt-en-de")
        elif target_language == "Russian":
            return pipeline("translation_en_to_ru", model="Helsinki-NLP/opus-mt-en-ru")
        elif target_language == "Hindi":
            return pipeline("translation_en_to_hi", model="Helsinki-NLP/opus-mt-en-hi")
        elif target_language == "French":
            return pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")
        elif target_language == "Indonesian":
            return pipeline("translation_en_to_id", model="Helsinki-NLP/opus-mt-en-id")
        elif target_language == "Dutch":
            return pipeline("translation_en_to_nl", model="Helsinki-NLP/opus-mt-en-nl")
        elif target_language == "Mandarin (Chinese)":
            return pipeline("translation_en_to_zh", model="Helsinki-NLP/opus-mt-en-zh")
        elif target_language == "Cantonese (Traditional Chinese)":
            return pipeline("translation_en_to_zh", model="Helsinki-NLP/opus-mt-en-zh")  # Shared model
        elif target_language == "Spanish":
            return pipeline("translation_en_to_es", model="Helsinki-NLP/opus-mt-en-es")
        elif target_language == "Portuguese":
            return pipeline("translation_en_to_pt", model="Helsinki-NLP/opus-mt-en-pt")

# Task 1: Text Generation
if page == "Text Generation":
    st.subheader("Text Generation")
    st.write("Enter a prompt below to generate text.")
    
    # Input for text generation
    user_input = st.text_area("Enter your text prompt here:", height=200)
    
    if st.button("Generate Text"):
        if user_input:
            # Load the model
            model = load_model("Text Generation")
            
            with st.spinner("Generating text..."):
                output = model(user_input, max_length=100, num_return_sequences=1)
                generated_text = output[0]["generated_text"]
                st.success("Generated Text:")
                st.write(generated_text)
        else:
            st.error("Please enter a text prompt.")

# Task 2: Sentiment Analysis
elif page == "Sentiment Analysis":
    st.subheader("Sentiment Analysis")
    st.write("Enter a sentence or paragraph below to analyze its sentiment.")
    
    # Input for sentiment analysis
    user_input = st.text_area("Enter your text here:", height=200)
    
    if st.button("Analyze Sentiment"):
        if user_input:
            # Load the model
            model = load_model("Sentiment Analysis")
            
            with st.spinner("Analyzing sentiment..."):
                result = model(user_input)[0]
                label = result["label"]
                score = result["score"]
                st.success("Sentiment Analysis Result:")
                st.write(f"**Label**: {label}, **Score**: {score:.4f}")
        else:
            st.error("Please enter some text.")

# Task 3: Translation
elif page == "Translation":
    st.subheader("Translation")
    st.write("Enter text below to translate it from English to a selected language.")
    
    # Sidebar for selecting translation language
    target_language = st.sidebar.selectbox("Select target language", [
        "Albanian", "German", "Russian", "Hindi", "French", "Indonesian", 
        "Dutch", "Mandarin (Chinese)", "Cantonese (Traditional Chinese)", 
        "Spanish", "Portuguese"
    ])
    
    # Input for translation
    user_input = st.text_area("Enter your text here:", height=200)
    
    if st.button("Translate Text"):
        if user_input:
            # Load the model
            model = load_model("Translation", target_language)
            
            with st.spinner(f"Translating text to {target_language}..."):
                translation = model(user_input)[0]["translation_text"]
                st.success(f"Translation (English to {target_language}):")
                st.write(translation)
        else:
            st.error("Please enter some text.")

