import streamlit as st
from transformers import pipeline

# Set up the page configuration
st.set_page_config(
    page_title="Live Transformer Demo",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Add an explanation of the app
st.title("🤗 Live Transformer Demo")
st.write("""
    Welcome to the Live Transformer Demo! This app demonstrates a variety of Natural Language Processing (NLP) tasks 
    using pre-trained transformer models from Hugging Face. Simply choose a task from the sidebar, enter your text, 
    and click 'Run' to see the results.
    
    **Here’s what you can do with this app:**
    - **Text Generation**: Automatically generate text based on a starting prompt.
    - **Sentiment Analysis**: Determine if a given text expresses positive or negative sentiment.
    - **Translation**: Translate text from English into multiple languages.
    - **Summarization**: Generate a concise summary of a longer text.
    - **Question Answering**: Provide a question and a context, and get an answer.
    - **Zero-Shot Classification**: Classify text into labels without training on them.
    - **Named Entity Recognition (NER)**: Detect people, places, and organizations in text.
    
    Explore each feature and see how powerful these NLP models are!
""")

# Sidebar for selecting the task
task = st.sidebar.selectbox("Choose a task", [
    "Text Generation", 
    "Sentiment Analysis", 
    "Translation", 
    "Summarization", 
    "Question Answering", 
    "Zero-Shot Classification", 
    "Named Entity Recognition (NER)"
])

# Sidebar for selecting translation language if translation is chosen
target_language = None
if task == "Translation":
    target_language = st.sidebar.selectbox("Select target language", [
        "Albanian", "German", "Kazakh", "Hindi", "French", "Indonesian", 
        "Dutch", "Mandarin (Chinese)", "Cantonese (Traditional Chinese)", 
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
        elif target_language == "Kazakh":
            return pipeline("translation_en_to_kk", model="Helsinki-NLP/opus-mt-en-kk")
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
            return pipeline("translation_en_to_zh", model="Helsinki-NLP/opus-mt-en-zh")  # Mandarin and Cantonese often share models
        elif target_language == "Spanish":
            return pipeline("translation_en_to_es", model="Helsinki-NLP/opus-mt-en-es")
        elif target_language == "Portuguese":
            return pipeline("translation_en_to_pt", model="Helsinki-NLP/opus-mt-en-pt")
    elif task_name == "Summarization":
        return pipeline("summarization", model="facebook/bart-large-cnn")
    elif task_name == "Question Answering":
        return pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    elif task_name == "Zero-Shot Classification":
        return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    elif task_name == "Named Entity Recognition (NER)":
        return pipeline("ner", grouped_entities=True)

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
            elif task == "Summarization":
                summary = model(user_input, max_length=130, min_length=30, do_sample=False)[0]["summary_text"]
                st.success("Summary:")
                st.write(summary)
            elif task == "Question Answering":
                context = st.text_area("Context", height=200)
                question = st.text_input("Enter your question")
                if st.button("Get Answer"):
                    answer = model(question=question, context=context)["answer"]
                    st.success("Answer:")
                    st.write(answer)
            elif task == "Zero-Shot Classification":
                labels = st.text_input("Enter labels separated by commas (e.g., 'sports, politics, technology')")
                candidate_labels = labels.split(",")
                result = model(user_input, candidate_labels=candidate_labels)
                st.success("Classification Result:")
                st.write(result["labels"])
                st.write(result["scores"])
            elif task == "Named Entity Recognition (NER)":
                entities = model(user_input)
                st.success("Named Entities:")
                for entity in entities:
                    st.write(f"{entity['word']} ({entity['entity_group']}): {entity['score']:.4f}")
    else:
        st.error("Please enter some text.")

