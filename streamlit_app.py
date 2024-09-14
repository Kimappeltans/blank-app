import streamlit as st
from transformers import pipeline

# Set up the page configuration for a wide layout and better experience
st.set_page_config(
    page_title="Live Transformer Demo",
    layout="wide",  # This will allow a wider layout for your app
)

# Add an explanation of the app with title and markdown
st.title("🤗 Live Transformer Demo")
st.markdown("""
    Welcome to the Transformer NLP Demo! This app showcases **Text Generation**, **Sentiment Analysis**, and **Translation** tasks.
    
    - 🚀 **Text Generation** for creative text.
    - 🔍 **Sentiment Analysis** for understanding opinions.
    - 🌐 **Translation** across multiple languages, including Albanian, German, Russian, Hindi, French, Indonesian, Dutch, Mandarin (Chinese), Spanish, and Italian.

    Simply choose a task below, enter your text, and click 'Run' to see the results!
""")

# Two-column layout
col1, col2 = st.columns([2, 1])

# Left column for input and task selection
with col1:
    st.subheader("Start Exploring")
    
    # Task selection now in main body
    task = st.selectbox("Choose a task", ["Text Generation", "Sentiment Analysis", "Translation"])

    # Language selection shown only if "Translation" is chosen
    target_language = None
    if task == "Translation":
        target_language = st.selectbox("Select language", [
            "Albanian", "German", "Russian", "Hindi", "French", "Indonesian", 
            "Dutch", "Mandarin (Chinese)", "Spanish", "Italian"
        ])

    # Text input from the user
    user_input = st.text_area("Enter your text here:", height=150)

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
            elif target_language == "Italian":
                return pipeline("translation_en_to_it", model="Helsinki-NLP/opus-mt-en-it")

    model = load_model(task, target_language)

    # Cache the results of each task to avoid re-computation
    @st.cache_data
    def generate_text(input_text):
        return model(input_text, max_length=100, num_return_sequences=1)

    @st.cache_data
    def analyze_sentiment(input_text):
        return model(input_text)

    @st.cache_data
    def translate_text(input_text):
        return model(input_text)[0]["translation_text"]

    # Perform the selected task
    if st.button("Run"):
        if user_input:
            with st.spinner(f"Performing {task.lower()}..."):
                if task == "Text Generation":
                    output = generate_text(user_input)
                    generated_text = output[0]["generated_text"]
                    st.success("Generated Text:")
                    st.write(generated_text)
                elif task == "Sentiment Analysis":
                    result = analyze_sentiment(user_input)[0]
                    label = result["label"]
                    score = result["score"]
                    st.success("Sentiment Analysis Result:")
                    st.write(f"**Label**: {label}, **Score**: {score:.4f}")
                elif task == "Translation":
                    translation = translate_text(user_input)
                    st.success(f"Translation (English to {target_language}):")
                    st.write(translation)
        else:
            st.error("Please enter some text.")


