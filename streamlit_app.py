import streamlit as st
from transformers import pipeline

# Set up the page configuration
st.set_page_config(
    page_title="Live Transformer Demo",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Add an explanation of the app
st.title("🤗 AD577 Live Transformer Demo")
st.write("""
    Welcome to the Live Transformer Demo! This app demonstrates three main NLP tasks using transformer models from Hugging Face:
    
    - **Text Generation**: Automatically generate text based on a starting prompt.
    - **Sentiment Analysis**: Determine if a given text expresses positive or negative sentiment.
    - **Translation**: Translate text from English into multiple languages, including Albanian, German, Russian, Hindi, French, Indonesian, Dutch, Mandarin (Chinese), Cantonese (Traditional Chinese), Spanish, and Portuguese.
    
    Simply choose a task from the sidebar, enter your text, and click 'Run' to see the results!
""")

# Sidebar for selecting the task
task = st.sidebar.selectbox("Choose a task", [
    "Text Generation", 
    "Sentiment Analysis", 
    "Translation"
])

# Sidebar for selecting translation language if translation is chosen
target_language = None
if task == "Translation":
    target_language = st.sidebar.selectbox("Select target language", [
        "Albanian", "German", "Russian", "Hindi", "French", "Indonesian", 
        "Dutch", "Mandarin (Chinese)",  
        "Spanish", "Portuguese"
    ])

# Load the appropriate model based on the task and target language
@st.cache_resource
def load_model(task_name, target_language=None):
    if task_name == "Text Generation":
        return pipeline("text-generation", model="gpt2")
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
        elif target_language == "Spanish":
            return pipeline("translation_en_to_es", model="Helsinki-NLP/opus-mt-en-es")
        elif target_language == "Portuguese":
            return pipeline("translation_en_to_pt", model="Helsinki-NLP/opus-mt-en-pt")

model = load_model(task, target_language)

# Text input from the user
user_input = st.text_area("Enter your text here:", height=200)

# Perform the selected task
if st.button("Run"):
    if user_input:
        with st.spinner(f"Performing {task.lower()}..."):
            if task == "Text Generation":
                output = model(user_input, max_length=100, num_return_sequences=1)
                generated_text = output[0]["generated_text"]
                st.success("Generated Text:")
                st.write(generated_text)
            elif task == "Sentiment Analysis":
                result = model(user_input)[0]
                label = result["label"]
                score = result["score"]
                st.success("Sentiment Analysis Result:")
                st.write(f"**Label**: {label}, **Score**: {score:.4f}")
            elif task == "Translation":
                translation = model(user_input)[0]["translation_text"]
                st.success(f"Translation (English to {target_language}):")
                st.write(translation)
    else:
        st.error("Please enter some text.")

