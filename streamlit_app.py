import streamlit as st
from transformers import pipeline

# Set up the page configuration
st.set_page_config(
    page_title="AD577 AI class Transformer Demo",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.title("🤗 AD577 AI class Transformer Demo")
st.write("This app uses a transformer model to perform NLP tasks.")

# Display your Streamlit Cloud app URL
app_url = "https://blank-app-3u33oita2ww.streamlit.app/"
st.markdown(f"**This app is also available at:** [{app_url}]({app_url})")

# Sidebar for selecting the task
task = st.sidebar.selectbox("Choose a task", ["Text Generation", "Sentiment Analysis", "Translation"])

# Initialize the transformer model
@st.cache_resource
def load_model(task_name):
    if task_name == "Text Generation":
        return pipeline("text-generation", model="gpt2")
    elif task_name == "Sentiment Analysis":
        return pipeline("sentiment-analysis")
    elif task_name == "Translation":
        return pipeline("translation_en_to_fr")
    else:
        return None

model = load_model(task)

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
                st.success("Translation (English to French):")
                st.write(translation)
    else:
        st.error("Please enter some text.")
