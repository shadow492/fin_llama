import streamlit as st
from transformers import pipeline
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load pre-trained LLaMA model and tokenizer
model_name = "meta-llama/Llama-3.2-1B"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def generate_text(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=100)
    return tokenizer.decode(output[0], skip_special_tokens=True)

st.title("LLaMA Text Generation")

prompt = st.text_input("Enter a prompt:", "")

if st.button("Generate"):
    generated_text = generate_text(prompt)
    st.write(generated_text)
