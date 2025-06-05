import streamlit as st
from datetime import datetime
from PIL import Image
import os

from claim_agent import claim_agent
from utils.generate_pdf import generate_claim_pdf

st.set_page_config(page_title="🏚️ Insurance Claim Agent", layout="centered")
st.title("🏚️ House Insurance Claim - AI Workflow")

# Upload section
uploaded_files = st.file_uploader(
    "📂 Upload Claim Images/Documents (JPG, PNG)", 
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# Claim metadata input
st.subheader("📄 Claim Metadata")
policy_date = st.date_input("📅 Policy Inception Date")
dol = st.date_input("📅 Date of Loss (DOL)")
threshold = st.slider("📏 Allowed Days Between EXIF and DOL", 0, 10, 2)

# Text input for user explanation
st.subheader("🧾 Describe Your Claim")
user_claim_text = st.text_area("What happened? Why should this claim be approved?", height=200)

# Submit and process
if st.button("🚀 Submit Claim"):
    if not uploaded_files or not user_claim_text.strip():
        st.warning("⚠️ Please upload at least one image and provide a claim description.")
    else:
        os.makedirs("data/uploaded_images", exist_ok=True)

        for uploaded_file in uploaded_files:
            st.divider()
            st.subheader(f"🖼️ Processing: {uploaded_file.name}")

            file_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
            file_path = f"data/uploaded_images/{file_id}_{uploaded_file.name}"
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            state = {
                "file_path": file_path,
                "user_text": user_claim_text,
                "policy_data": {
                    "policy_date": policy_date.isoformat(),
                    "dol": dol.isoformat(),
                    "threshold": threshold
                }
            }

            with st.spinner("🤖 Running AI agent..."):
                result = claim_agent.invoke(state)

            # ✅ FINAL DECISION DISPLAY (with formatting)
            st.subheader("📋 Final Decision")

            verdict = result.get("final_verdict", "").strip().lower()
            if verdict == "approved":
                st.markdown(
                    "<div style='color:green; font-size:24px; font-weight:bold;'>✅ CLAIM APPROVED</div>",
                    unsafe_allow_html=True
                )
            elif verdict == "rejected":
                st.markdown(
                    "<div style='color:red; font-size:24px; font-weight:bold; font-family:Times New Roman;'>❌ CLAIM REJECTED</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    "<div style='color:orange; font-size:24px; font-weight:bold;'>⚠️ DECISION PENDING REVIEW</div>",
                    unsafe_allow_html=True
                )

            # 🧾 Explanation of Decision
            st.markdown("### 🧾 Reason:")
            st.write(result.get("final_decision", "No explanation available."))

            # 📑 Summary and Key Info
            st.markdown("### 📝 Summary:")
            st.write(result.get("summary", "Not available"))

            st.markdown("### 🔑 Key Information:")
            st.write(result.get("key_info", "Not extracted"))

            st.markdown("### 🚩 Misrepresentation Check:")
            st.write(result.get("misrep", "No output"))

            st.markdown("### 🖼️ Vision Labels:")
            st.write(result.get("labels", "No vision labels found."))

            # 📥 Generate downloadable PDF
            pdf_path = f"data/claim_report_{file_id}.pdf"
            generate_claim_pdf(
                output_path=pdf_path,
                summary=result.get("summary", ""),
                decision=result.get("final_decision", ""),
                labels=result.get("labels", ""),
                key_info=result.get("key_info", ""),
                misrep=result.get("misrep", "")
            )

            with open(pdf_path, "rb") as pdf_file:
                st.download_button(
                    label="📄 Download Claim Summary PDF",
                    data=pdf_file,
                    file_name=f"claim_report_{file_id}.pdf",
                    mime="application/pdf"
                )
