import streamlit as st
from transformers import pipeline
import torch
from transformers import LLaMAForCausalLM, LLaMATokenizer

# Load pre-trained LLaMA model and tokenizer
model_name = "facebook/llama-3.2B"
model = LLaMAForCausalLM.from_pretrained(model_name)
tokenizer = LLaMATokenizer.from_pretrained(model_name)

def generate_text(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=100)
    return tokenizer.decode(output[0], skip_special_tokens=True)

st.title("LLaMA Text Generation")

prompt = st.text_input("Enter a prompt:", "")

if st.button("Generate"):
    generated_text = generate_text(prompt)
    st.write(generated_text)
