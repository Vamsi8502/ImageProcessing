import os
from typing import TypedDict, Optional
from datetime import datetime
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

from utils.exif_checker import extract_exif_data, get_datetime_original
from utils.vision_labels import get_image_labels
from utils.summarizer import summarize_text
from utils.key_info_extractor import extract_key_info
from utils.misrep_detector import detect_misrepresentation
from utils.similar_claims import retrieve_similar_claims

load_dotenv()

# llm = ChatOpenAI(model="gpt-4", api_key=os.getenv("OPENAI_API_KEY"))
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")



class ClaimState(TypedDict, total=False):
    file_path: str
    exif: dict
    exif_date: Optional[str]
    image_labels: Optional[str]
    user_text: Optional[str]
    summary: Optional[str]
    key_info: Optional[str]
    misrep: Optional[str]
    similar_claims: Optional[str]
    final_decision: Optional[str]
    policy_data: dict
    gps_available: bool
    image_relevance: bool
    exif_vs_policy: str
    exif_vs_dol: str
    misrep_found: bool

# EXIF node
from datetime import datetime

def process_exif(state: ClaimState) -> ClaimState:
    exif = extract_exif_data(state['file_path'])
    exif_date = get_datetime_original(exif) if exif else None
    state['exif'] = exif
    state['exif_date'] = str(exif_date) if exif_date else None

    # Check if GPS data is present
    state["gps_available"] = "GPSInfo" in exif if exif else False

    # Policy checks
    policy_start = state["policy_data"].get("policy_date")
    dol = state["policy_data"].get("dol")
    threshold = int(state["policy_data"].get("threshold", 0))

    # EXIF vs Policy
    if exif_date and policy_start:
        try:
            exif_dt = (
                exif_date if isinstance(exif_date, datetime)
                else datetime.fromisoformat(str(exif_date))
            )
            policy_dt = (
                policy_start if isinstance(policy_start, datetime)
                else datetime.fromisoformat(str(policy_start))
            )
            state["exif_vs_policy"] = "valid" if exif_dt >= policy_dt else "invalid"
        except Exception as e:
            print("EXIF vs Policy parse error:", e)
            state["exif_vs_policy"] = "unknown"
    else:
        state["exif_vs_policy"] = "unknown"

    # EXIF vs DOL (robust parsing and debug prints)
    def to_date(val):
        if isinstance(val, datetime):
            return val.date()
        if hasattr(val, "date"):
            return val.date()
        if isinstance(val, str):
            try:
                val = val.split("T")[0]
                return datetime.strptime(val, "%Y-%m-%d").date()
            except Exception:
                try:
                    return datetime.fromisoformat(val).date()
                except Exception:
                    return None
        return None

    exif_dt = to_date(exif_date)
    dol_dt = to_date(dol)

    print("EXIF_DATE (parsed):", exif_dt)
    print("DOL (parsed):", dol_dt)
    print("THRESHOLD:", threshold)

    if exif_dt and dol_dt:
        diff = abs((dol_dt - exif_dt).days)
        print("DATE DIFF:", diff)
        state["exif_vs_dol"] = "approve" if diff <= threshold else "too_far"
    else:
        state["exif_vs_dol"] = "unknown"

    return state


# Vision Labels
def process_vision_labels(state: ClaimState) -> ClaimState:
    labels = get_image_labels(state['file_path'])
    state['image_labels'] = ", ".join(labels) if labels else "No labels found"

    # Rough heuristic: If label includes any of the user-mentioned keywords, it's relevant
    user_text = state.get("user_text", "").lower()
    state["image_relevance"] = any(lbl.lower() in user_text for lbl in labels)
    return state

# Summary LLM
def summarize(state: ClaimState) -> ClaimState:
    user_text = state.get("user_text", "")
    labels = state.get("image_labels", "")

    summary_input = f"""
=== USER CLAIM TEXT ===
{user_text}

=== IMAGE CONTENT LABELS (Google Vision) ===
{labels}

=== TASK ===
1. Summarize the user's claim.
2. Comment whether image labels support the claim.

Return:
- 📝 Summary:
- 🔍 Visual Label Relevance:
"""
    summary = summarize_text(summary_input)
    state["summary"] = summary
    return state

# Key Info
def extract_keyinfo(state: ClaimState) -> ClaimState:
    key_info = extract_key_info(state["summary"])
    state["key_info"] = key_info
    return state

# Misrepresentation
def misrep_check(state: ClaimState) -> ClaimState:
    result = detect_misrepresentation(state["key_info"], state["policy_data"])
    state["misrep"] = result
    state["misrep_found"] = "yes" in result.get("verdict", "").lower()
    return state

# Similar Claims
def similar_claims(state: ClaimState) -> ClaimState:
    similar = retrieve_similar_claims(state["summary"])
    state["similar_claims"] = similar
    return state

# Final Verdict
def final_decision(state: ClaimState) -> ClaimState:
    prompt = PromptTemplate(
        input_variables=[
            "summary", "misrep", "similar_claims", "exif_date",
            "policy_start", "dol", "labels"
        ],
        template="""
        You are an expert insurance claim assessor.

        Here’s the claim:

        📋 Summary:
        {summary}

        🚩 Misrepresentation:
        {misrep}

        🧠 Similar Past Claims:
        {similar_claims}

        📸 EXIF Date: {exif_date}
        📅 Policy Start Date: {policy_start}
        📆 Reported DOL: {dol}
        🖼️ Vision Labels: {labels}

        Decision logic:
        - APPROVE only if ALL evidence and claim details are clear, correct, and matching.
        - REJECT if (any one of the following):
            1. EXIF data is missing.
            2. Policy start date is after or equal to EXIF date or DOL.
            3. The visual evidence (labels) does NOT match the claimed item or event.
            4. Clear evidence of fraud or deliberate misrepresentation.
        - FLAG (for human/manual review) if:
            1. The EXIF date and Date of Loss (DOL) are outside the allowed threshold, but other data is okay.
            2. There is any ambiguity, unclear detail, or the claim cannot be confidently approved or rejected automatically.
            3. The claim description or documentation is vague or incomplete, but not outright rejectable.

        On the FIRST LINE, output your FINAL DECISION in this format (in ALL CAPS):

        APPROVE: short reason  
        REJECT: short reason  
        FLAG: short reason  

        After the first line, add bullet points or explanation for your evaluation.
        Be precise—if any of the sure-shot reject conditions are present, use REJECT. If not, but something is ambiguous or threshold-related, use FLAG. Only use APPROVE for totally clean claims.
        """

    )

    result = llm.invoke(prompt.format(
        summary=state.get("summary", ""),
        misrep=state.get("misrep", ""),
        similar_claims=state.get("similar_claims", ""),
        exif_date=state.get("exif_date", "Not available"),
        policy_start=state["policy_data"].get("policy_date", "Not provided"),
        dol=state["policy_data"].get("dol", "Not provided"),
        labels=state.get("image_labels", "No labels")
    ))

    state["final_decision"] = result.content
    return state



# LangGraph setup
workflow = StateGraph(ClaimState)
workflow.add_node("EXIF", process_exif)
workflow.add_node("VISION_LABELS", process_vision_labels)
workflow.add_node("SUMMARIZE", summarize)
workflow.add_node("KEY_INFO", extract_keyinfo)
workflow.add_node("MISREP_CHECK", misrep_check)
# workflow.add_node("SIMILAR_CLAIMS", similar_claims)
workflow.add_node("FINAL_DECISION", final_decision)

workflow.set_entry_point("EXIF")
workflow.add_edge("EXIF", "VISION_LABELS")
workflow.add_edge("VISION_LABELS", "SUMMARIZE")
workflow.add_edge("SUMMARIZE", "KEY_INFO")
workflow.add_edge("KEY_INFO", "MISREP_CHECK")
workflow.add_edge("MISREP_CHECK", "FINAL_DECISION")
# workflow.add_edge("MISREP_CHECK", "SIMILAR_CLAIMS")
# workflow.add_edge("SIMILAR_CLAIMS", "FINAL_DECISION")
workflow.add_edge("FINAL_DECISION", END)

claim_agent = workflow.compile()
