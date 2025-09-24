import streamlit as st
import pandas as pd
from crew import run_enhanced_crew_analysis
import os
import json

st.set_page_config(layout="wide", page_title="AI Financial Crew")

st.title("👨‍💼 Your Personal AI Financial Crew")
st.markdown("""
Upload your UPI transaction history and let our crew of specialized AI agents perform an in-depth analysis. 
This isn't just a script; it's a collaborative team of AI experts built with **CrewAI**.
- The **Data Specialist** intelligently identifies the correct columns in any file.
- The **Transaction Analyst** crunches the numbers.
- The **Financial Storyteller** writes a personalized narrative just for you.
""")

uploaded_file = st.file_uploader("📂 Upload your transaction file (.csv or .xlsx)", type=["csv", "xlsx"])

if uploaded_file is not None:
    st.success(f"File '{uploaded_file.name}' received. Assembling the AI Crew...")

    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.write("### First 5 Rows of Your Data:")
        st.dataframe(df.head())

        if st.button("🚀 Unleash the AI Crew!", use_container_width=True):
            with st.spinner('The AI Crew is collaborating on your report... This may take a moment.'):
                try:
                    # This single function call runs the entire multi-agent workflow
                    result_json, pdf_path = run_enhanced_crew_analysis(df)
                    result_data = json.loads(result_json)
                    
                    st.success("The Crew has finished! Your report is ready. ✅")

                    st.write("### AI Financial Storyteller's Narrative:")
                    st.info(result_data['narrative'])

                    with open(pdf_path, "rb") as pdf_file:
                        PDFbyte = pdf_file.read()

                    st.download_button(
                        label="⬇️ Download Full PDF Report",
                        data=PDFbyte,
                        file_name="AI_Crew_Financial_Report.pdf",
                        mime='application/octet-stream',
                        use_container_width=True
                    )
                    
                    # Cleanup
                    for file in ["spending_chart.png", pdf_path]:
                        if os.path.exists(file):
                            os.remove(file)

                except Exception as e:
                    st.error(f"An error occurred while the crew was working: {e}")
                    st.error("Please ensure your file is a valid transaction sheet.")

    except Exception as e:
        st.error(f"Could not read the file: {e}")