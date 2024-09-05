""" Dratos.com PandasAI Data Analysis Demo """
from pandasai.llm.local_llm import LocalLLM
from pandasai import SmartDataFrame
import streamlit as st
import pandas as pd

model = LocalLLM(
    api_base="http://localhost:8000/v1",
    model="NousResearch/Meta-Llama-3-8B-Instruct"
)

st.title = "Dratos Data Analysis"
uploaded_file = st.file_uploader("Upload a CSV file:")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data.head(3))

    df = SmartDataFrame(data, config={"llm": model})
    prompt = st.text_area("Enter your prompt:")

    if st.button("Generate"):
        if prompt:
            with st.spinner("Generating response..."):
                st.write(df.chat(prompt))

