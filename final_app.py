import streamlit as st
import os
from datetime import datetime
from PIL import Image, ExifTags
from claim_agent import claim_agent
import pandas as pd

st.set_page_config(page_title="🏚️ Insurance Claim Agent", layout="centered")

st.title("🏚️ House Insurance Claim - AI Workflow")

uploaded_file = st.file_uploader("📂 Upload a Claim Image (JPG, PNG)", type=["jpg", "jpeg", "png"])

# ---- EXIF Debugging Section ----
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.subheader("🪪 EXIF Data (Important Fields)")
    exif_data = image._getexif()
    if exif_data:
        exif = {
            ExifTags.TAGS.get(k, k): v
            for k, v in exif_data.items()
            if k in ExifTags.TAGS
        }
        important_fields = ["DateTimeOriginal", "Make", "Model", "GPSInfo"]
        for field in important_fields:
            value = exif.get(field)
            if value:
                if field == "GPSInfo":
                    st.write("**GPSInfo:**", value)
                else:
                    st.write(f"**{field}:** {value}")
    else:
        st.warning("No EXIF metadata found in this image.")

st.subheader("📄 Claim Metadata")
policy_date = st.date_input("📅 Policy Inception Date")
dol = st.date_input("📅 Date of Loss (DOL)")
threshold = st.slider("📏 Allowed Days Between EXIF and DOL", 0, 10, 2)

st.subheader("📝 Describe Your Claim")
user_claim_text = st.text_area("What happened? Why should this claim be approved?", height=150)

if st.button("🚀 Submit Claim"):
    if not uploaded_file or not user_claim_text.strip():
        st.warning("⚠️ Please upload an image and provide claim description.")
    else:
        os.makedirs("data/uploaded_images", exist_ok=True)
        file_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
        file_path = f"data/uploaded_images/{file_id}_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.subheader(f"🖼️ Processing: {uploaded_file.name}")

        state = {
            "file_path": file_path,
            "user_text": user_claim_text,
            "policy_data": {
                "policy_date": policy_date.isoformat(),
                "dol": dol.isoformat(),
                "threshold": threshold,
            },
        }

        with st.spinner("🤖 Running AI agent..."):
            result = claim_agent.invoke(state)

        print(result)

        st.markdown("## 📊 Summary of Evaluation")

        summary_table = {
            "Field": [
                "📷 Image Relevance",
                "📅 EXIF vs Policy Start",
                "📆 EXIF vs DOL Match",
                "🚩 Misrepresentation Found",
                "📍 GPS Location Available"
            ],
            "Status": [
                "✅ Yes" if result.get("image_relevance") else "❌ No",
                "✅ Valid" if result.get("exif_vs_policy") == "valid" else "❌ Invalid",
                "✅ Close" if result.get("exif_vs_dol") in ["close", "approve"] else "❌ Too Far",
                "❌ Yes" if result.get("misrep_found") else "✅ No",  # RED if misrep, GREEN if not
                "✅ Yes" if result.get("gps_available") else "❌ No"
            ],
            "Explanation": [
                "Image matches claim context (e.g., laptop shown)." if result.get("image_relevance") else "Image is not related to the claim description.",
                "Photo taken after policy started." if result.get("exif_vs_policy") == "valid" else "Photo taken before policy started.",
                "EXIF date close to Date of Loss." if result.get("exif_vs_dol") == "close" else "EXIF date too far from Date of Loss.",
                "Conflicting details found." if result.get("misrep_found") else "No misrepresentation found.",
                "GPS metadata present." if result.get("gps_available") else "No GPS metadata found."
            ]
        }

        df = pd.DataFrame(summary_table)
        st.table(df)

        st.markdown("## 🧾 Final Verdict")

        final_decision = result.get("final_decision", "").strip().lower()
        approved_block = """
        <div style='
            font-size:36px;
            font-weight:bold;
            color:green;
            background: linear-gradient(90deg, #eaffd0 0%, #d0ffd6 100%);
            border-radius: 16px;
            padding: 32px;
            text-align:center;
            box-shadow: 0 4px 24px rgba(0,128,0,0.1);'>
            🎉✅ CLAIM APPROVED!<br>
            <span style='font-size:24px; color:#222; font-weight:normal;'>
                Congratulations! Your claim has been successfully approved.<br>
                We appreciate your patience and cooperation.<br>
                <span style='font-size:36px;'>🎊</span>
            </span>
        </div>
        """

        rejected_block = """
        <div style='
            font-size:36px;
            font-weight:bold;
            color:#b00020;
            background: linear-gradient(90deg, #ffe0e0 0%, #ffd6d6 100%);
            border-radius: 16px;
            padding: 32px;
            text-align:center;
            box-shadow: 0 4px 24px rgba(255,0,0,0.07);'>
            ❌ CLAIM REJECTED<br>
            <span style='font-size:24px; color:#222; font-weight:normal;'>
                We regret to inform you that your claim could not be approved.<br>
                Please check the reason below or contact support for further assistance.
            </span>
        </div>
        """

        under_review_block = """
        <div style='
            font-size:36px;
            font-weight:bold;
            color:#c77f00;
            background: linear-gradient(90deg, #fff6d0 0%, #ffeccd 100%);
            border-radius: 16px;
            padding: 32px;
            text-align:center;
            box-shadow: 0 4px 24px rgba(255,140,0,0.07);'>
            🕵️ UNDER REVIEW<br>
            <span style='font-size:24px; color:#222; font-weight:normal;'>
                Your claim is currently under review by our team.<br>
                We will notify you with an update soon.<br>
                Thank you for your patience.
            </span>
        </div>
        """

        if final_decision.startswith("approve"):
            st.markdown(approved_block, unsafe_allow_html=True)
            st.balloons()
        elif final_decision.startswith("reject"):
            st.markdown(rejected_block, unsafe_allow_html=True)
        elif final_decision.startswith("flag"):
            st.markdown(under_review_block, unsafe_allow_html=True)
        else:
            st.markdown(under_review_block, unsafe_allow_html=True)


        st.write("### Reason:")
        st.markdown(result.get("final_reason", "No detailed reason provided."))

        if result.get("summary"):
            st.write("### Claim Summary:")
            st.write(result["summary"])
