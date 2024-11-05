import streamlit as st
from transformers import TFAutoModelForCausalLM, AutoTokenizer
import tensorflow as tf

# Load the LLaMA model and tokenizer
@st.cache_resource
def load_model():
    model_name = "microsoft/Phi-3.5-mini-instruct"  # Replace with actual model name if different
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModelForCausalLM.from_pretrained(model_name, from_pt=True)
    return model, tokenizer

# Streamlit UI
st.title("LLaMA Model on Streamlit")
st.write("A simple interface to generate text using the LLaMA model.")

# Load the model and tokenizer
model, tokenizer = load_model()

# Text input
input_text = st.text_area("Enter text to generate continuation:", value="Once upon a time")

# Text generation settings
max_length = st.slider("Max Length", min_value=10, max_value=200, value=50, step=10)

# Generate text when the button is pressed
if st.button("Generate"):
    # Encode input text
    input_ids = tokenizer.encode(input_text, return_tensors="tf")
    
    # Generate text
    output = model.generate(input_ids, max_length=max_length, do_sample=True, top_p=0.9, temperature=0.7)
    
    # Decode generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Display output
    st.write("Generated Text:")
    st.write(generated_text)