import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi, HfHubHTTPError

def verify_token(token):
    api = HfApi()
    try:
        user_info = api.whoami(token=token)
        st.success(f"Token is valid. Logged in as {user_info['name']}.")
        return True
    except HfHubHTTPError:
        st.error("Invalid token or access denied.")
        return False

# Your Streamlit app code
@st.cache
def load_model():
    model_name = "meta-llama/Llama-3.2-1B"  # Update with the desired model name
    token = "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"  # Replace with your token

    # Verify token
    if not verify_token(token):
        st.stop()  # Stop the app if the token is invalid

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=token)
    return model, tokenizer

# Streamlit UI
st.title("Phi-3.5 Mini Instruct Model on Streamlit")
st.write("A simple interface to generate text using the Phi-3.5 model.")

model, tokenizer = load_model()

# Text input
input_text = st.text_area("Enter text to generate continuation:", value="Once upon a time")

# Text generation settings
max_length = st.slider("Max Length", min_value=10, max_value=200, value=50, step=10)

# Generate text when the button is pressed
if st.button("Generate"):
    # Encode input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt")  # Change to PyTorch tensors

    # Generate text
    output = model.generate(input_ids, max_length=max_length, do_sample=True, top_p=0.9, temperature=0.7)
    
    # Decode generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Display output
    st.write("Generated Text:")
    st.write(generated_text)
