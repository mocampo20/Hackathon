import json
import os
import re
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


CATEGORIES = [
    "Technical Issue",
    "Billing",
    "Account Access",
    "Refund",
    "Order Tracking",
    "Cancellation",
    "Complaint",
    "General Inquiry",
]


def load_kb():
    kb_path = Path(__file__).with_name("knowledge_base.json")
    return json.loads(kb_path.read_text(encoding="utf-8"))


def redact_sensitive_data(text):
    text = re.sub(r"[\w\.-]+@[\w\.-]+\.\w+", "[EMAIL]", text)
    text = re.sub(r"\b(?:\d[ -]*?){13,16}\b", "[CARD]", text)
    text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[ID]", text)
    return text


def keyword_demo_classifier(text):
    t = text.lower()

    if any(word in t for word in ["password", "login", "access", "account", "mfa"]):
        category = "Account Access"
    elif any(word in t for word in ["charged", "invoice", "billing", "payment", "refund"]):
        category = "Billing" if "refund" not in t else "Refund"
    elif any(word in t for word in ["order", "tracking", "shipment", "delivery"]):
        category = "Order Tracking"
    elif any(word in t for word in ["cancel", "subscription"]):
        category = "Cancellation"
    elif any(word in t for word in ["crash", "bug", "error", "upload", "broken"]):
        category = "Technical Issue"
    elif any(word in t for word in ["frustrating", "angry", "bad", "complaint"]):
        category = "Complaint"
    else:
        category = "General Inquiry"

    urgency = "High" if any(word in t for word in ["urgent", "now", "today", "immediately"]) else "Medium"
    sentiment = "Frustrated" if any(word in t for word in ["frustrating", "angry", "upset", "twice"]) else "Neutral"
    confidence = 0.86 if category != "General Inquiry" else 0.68

    missing = []
    if category in ["Billing", "Refund"] and "invoice" not in t:
        missing.append("invoice_id")
    if category == "Order Tracking" and "order" not in t:
        missing.append("order_id")
    if category in ["Account Access", "Cancellation"] and "email" not in t:
        missing.append("account_email")
    if category == "Technical Issue":
        missing.extend(["device", "app_version", "error_message"])

    return {
        "category": category,
        "urgency": urgency,
        "sentiment": sentiment,
        "intent": infer_intent(category),
        "missing_information": missing,
        "confidence": confidence,
    }


def infer_intent(category):
    return {
        "Technical Issue": "Resolve product or application problem",
        "Billing": "Clarify or correct a billing issue",
        "Account Access": "Recover account access",
        "Refund": "Request refund review",
        "Order Tracking": "Get shipment or delivery status",
        "Cancellation": "Cancel an active service or subscription",
        "Complaint": "Express dissatisfaction and request help",
        "General Inquiry": "Request general product or service information",
    }[category]


def retrieve_kb(category, kb):
    direct = [doc for doc in kb if doc["category"] == category]
    return direct[0] if direct else kb[-1]


def generate_demo_response(analysis, kb_doc):
    missing = analysis["missing_information"]
    missing_text = ""
    if missing:
        missing_text = " To continue, could you please confirm: " + ", ".join(missing) + "."

    return (
        f"Hi, thank you for contacting us. I understand this is related to "
        f"{analysis['category'].lower()} and we will help you with it. "
        f"{kb_doc['content']}{missing_text}"
    )


def analyze_with_openai(text, kb):
    if OpenAI is None or not os.getenv("OPENAI_API_KEY"):
        return None

    client = OpenAI()
    kb_text = "\n".join([f"- {doc['category']}: {doc['content']}" for doc in kb])
    prompt = f"""
Analyze this support ticket and return only valid JSON.

Categories: {", ".join(CATEGORIES)}

Knowledge base:
{kb_text}

Ticket:
{text}

Required JSON fields:
category, urgency, sentiment, intent, missing_information, confidence, suggested_response, sources
"""

    response = client.responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        input=prompt,
    )
    raw = response.output_text
    return json.loads(raw)


def analyze_ticket(text, kb, use_llm):
    clean_text = redact_sensitive_data(text.strip())

    if use_llm:
        try:
            llm_result = analyze_with_openai(clean_text, kb)
            if llm_result:
                return clean_text, llm_result, {"title": "LLM + Knowledge Base", "content": "OpenAI analysis"}
        except Exception as exc:
            st.warning(f"LLM unavailable, using local demo rules. Detail: {exc}")

    analysis = keyword_demo_classifier(clean_text)
    kb_doc = retrieve_kb(analysis["category"], kb)
    analysis["suggested_response"] = generate_demo_response(analysis, kb_doc)
    analysis["sources"] = [kb_doc["title"]]
    return clean_text, analysis, kb_doc


st.set_page_config(page_title="Support Ticket Copilot Demo", layout="wide")
st.title("Support Ticket Copilot Demo")

kb = load_kb()
sample_path = Path(__file__).with_name("sample_tickets.csv")
samples = pd.read_csv(sample_path)

with st.sidebar:
    st.header("Demo Controls")
    use_llm = st.toggle("Use OpenAI if API key exists", value=False)
    selected_ticket = st.selectbox(
        "Sample ticket",
        samples["ticket_id"].tolist(),
    )
    st.caption("Set OPENAI_API_KEY to enable real LLM analysis.")

default_text = samples.loc[samples["ticket_id"] == selected_ticket, "customer_text"].iloc[0]
ticket_text = st.text_area("Customer ticket", value=default_text, height=140)

if st.button("Analyze ticket", type="primary"):
    clean_text, analysis, kb_doc = analyze_ticket(ticket_text, kb, use_llm)

    left, middle, right = st.columns([1.1, 1, 1.2])

    with left:
        st.subheader("Preprocessing")
        st.write("Clean text")
        st.code(clean_text)
        st.metric("Confidence", f"{analysis['confidence']:.0%}" if isinstance(analysis["confidence"], float) else analysis["confidence"])

    with middle:
        st.subheader("Structured Extraction")
        st.json(
            {
                "category": analysis["category"],
                "urgency": analysis["urgency"],
                "sentiment": analysis["sentiment"],
                "intent": analysis["intent"],
                "missing_information": analysis["missing_information"],
            }
        )

    with right:
        st.subheader("Suggested Response")
        edited_response = st.text_area(
            "Agent editable response",
            value=analysis["suggested_response"],
            height=220,
        )
        st.write("Sources")
        st.write(", ".join(analysis.get("sources", [kb_doc.get("title", "Knowledge Base")])))

    st.divider()
    st.subheader("Human Validation")
    col1, col2, col3, col4 = st.columns(4)
    corrected_category = col1.selectbox("Final category", CATEGORIES, index=CATEGORIES.index(analysis["category"]))
    corrected_urgency = col2.selectbox("Final urgency", ["Low", "Medium", "High", "Critical"], index=["Low", "Medium", "High", "Critical"].index(analysis["urgency"]) if analysis["urgency"] in ["Low", "Medium", "High", "Critical"] else 1)
    accepted = col3.selectbox("Suggestion accepted", ["Yes", "Edited", "No"])
    escalated = col4.checkbox("Escalate")

    feedback = {
        "ticket_id": selected_ticket,
        "original_category": analysis["category"],
        "corrected_category": corrected_category,
        "original_urgency": analysis["urgency"],
        "corrected_urgency": corrected_urgency,
        "accepted": accepted,
        "escalated": escalated,
        "final_response": edited_response,
    }
    st.write("Feedback event")
    st.json(feedback)
else:
    st.info("Choose a sample ticket or paste a new one, then click Analyze ticket.")
