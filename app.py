import streamlit as st
from datetime import datetime
from PIL import Image
import os

from utils.exif_checker import extract_exif_data, get_datetime_original
from utils.ocr_extractor import extract_text_from_image
from utils.summarizer import summarize_text
from utils.key_info_extractor import extract_key_info

st.set_page_config(page_title="House Insurance Claim Detection", layout="centered")
st.title("🏚️ House Insurance Claim Detection")

# Allow multiple file uploads (images/documents/bills)
uploaded_files = st.file_uploader(
    "📷 Upload Claim Images and Documents (JPG, JPEG, PNG)", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True
)

# Policy details inputs
st.subheader("📅 Claim Information")
policy_date = st.date_input("📅 Policy Inception Date")
dol = st.date_input("📅 Date of Loss (DOL)")
threshold = st.slider("✏️ Allowed days difference between EXIF date and DOL", 0, 10, 2)

if st.button("🚀 Submit Claim"):
    if not uploaded_files:
        st.error("⚠️ Please upload at least one file.")
    else:
        for uploaded_file in uploaded_files:
            st.divider()
            st.subheader(f"🖼️ Processing: {uploaded_file.name}")

            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded File", use_container_width=True)

            # Save uploaded file properly
            os.makedirs("data/uploaded_images", exist_ok=True)
            file_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
            img_path = f"data/uploaded_images/{file_id}_{uploaded_file.name}"
            with open(img_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # EXIF metadata extraction and validation
            with st.spinner("🔍 Extracting EXIF metadata..."):
                exif = extract_exif_data(img_path)

            if not exif or "DateTimeOriginal" not in exif:
                st.error("⚠️ No reliable EXIF metadata found. Possible screenshot/downloaded/edited file.")
            else:
                st.markdown("**🧾 EXIF Data:**")
                exif_date = get_datetime_original(exif)
                camera_model = exif.get('Model', 'N/A')
                software_used = exif.get('Software', 'N/A')

                st.write(f"- 📅 **Date Taken**: `{exif_date if exif_date else 'Not Found'}`")
                st.write(f"- 📸 **Camera Model**: `{camera_model}`")
                st.write(f"- 🛠️ **Software Used**: `{software_used}`")

                issues = []

                if exif_date:
                    exif_only_date = exif_date.date()

                    if exif_only_date < policy_date:
                        issues.append("EXIF date is before policy inception date.")

                    if exif_only_date < dol:
                        issues.append("EXIF date is before Date of Loss.")

                    if abs((exif_only_date - dol).days) > threshold:
                        issues.append(f"EXIF date not close to Date of Loss ({threshold}+ days).")

                if issues:
                    st.error("🚩 **Suspicious Image Detected:**")
                    for issue in issues:
                        st.write(f"❌ {issue}")
                else:
                    st.success("✅ **Image EXIF seems valid and consistent.**")

            # OCR extraction
            with st.spinner("📝 Performing OCR extraction..."):
                ocr_text = extract_text_from_image(img_path)

            if ocr_text.strip():
                st.markdown("**📝 OCR Text:**")
                st.write(ocr_text)

                os.makedirs("data/extracted_texts", exist_ok=True)
                text_file_path = f"data/extracted_texts/{file_id}.txt"
                with open(text_file_path, "w") as text_file:
                    text_file.write(ocr_text)

                # Summarization using LLM
                with st.spinner("🧠 Summarizing extracted text..."):
                    summary = summarize_text(ocr_text)

                st.markdown("**📑 OCR Text Summary:**")
                st.write(summary)

                # After summarization step:
                with st.spinner("🔑 Extracting Key Information..."):
                    key_info = extract_key_info(summary)
                    st.subheader("🔑 Key Information Extracted:")
                    st.write(key_info)

            else:
                st.warning("⚠️ No text detected by OCR.")
