#frontend.py
"""
Streamlit frontend — updated to send active_documents (hash + filename)
to the backend so the intent node can resolve document-name references.
"""

import streamlit as st
import requests
import uuid
import json
import time

API_URL = "http://api:8000"

st.set_page_config(page_title="OpenMultiRAG", page_icon="📚", layout="wide")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
# Stores list of {"file_hash": ..., "filename": ...} dicts
if "active_documents" not in st.session_state:
    st.session_state.active_documents = []

st.title("📚 OpenMultiRAG Chat")
st.markdown(
    "Upload PDFs, let the worker index them (text + **tables** + images), "
    "then ask questions — even about **specific pages of specific PDFs**."
)

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("1. Upload Document(s)")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        if st.button("Process Document"):
            with st.spinner("Uploading..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                try:
                    res = requests.post(f"{API_URL}/upload", files=files, timeout=30)
                    res.raise_for_status()
                except requests.RequestException as e:
                    st.error(f"Upload failed: {e}")
                    st.stop()

            data = res.json()
            file_hash = data["file_hash"]
            status_msg = data["status"]
            st.success(f"Server: {status_msg}")

            if status_msg == "Processing started":
                status_placeholder = st.empty()
                status_placeholder.info("Connecting to live status stream...")
                final_status = "UNKNOWN"

                try:
                    with requests.get(
                        f"{API_URL}/document/{file_hash}/stream",
                        stream=True, timeout=1200,
                    ) as sse_response:
                        for raw_line in sse_response.iter_lines():
                            if not raw_line:
                                continue
                            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                            if line.startswith("data:"):
                                payload = json.loads(line[5:].strip())
                                current_status = payload.get("status", "UNKNOWN")
                                status_placeholder.info(f"Worker: {current_status}...")
                                if current_status in ("COMPLETED", "FAILED"):
                                    final_status = current_status
                                    break
                except requests.RequestException as e:
                    st.warning(f"SSE error — polling fallback. ({e})")
                    current_status = "PENDING"
                    while current_status in ("PENDING", "PROCESSING"):
                        time.sleep(3)
                        try:
                            r = requests.get(f"{API_URL}/document/{file_hash}/status", timeout=10)
                            current_status = r.json().get("status", "UNKNOWN")
                            status_placeholder.info(f"Worker: {current_status}...")
                        except Exception:
                            break
                    final_status = current_status

                if final_status == "COMPLETED":
                    status_placeholder.success("✅ Indexing complete!")
                    doc_entry = {"file_hash": file_hash, "filename": uploaded_file.name}
                    if file_hash not in [d["file_hash"] for d in st.session_state.active_documents]:
                        st.session_state.active_documents.append(doc_entry)
                else:
                    status_placeholder.error(f"❌ Worker status: {final_status}")

            elif status_msg == "Document already indexed":
                doc_entry = {"file_hash": file_hash, "filename": uploaded_file.name}
                if file_hash not in [d["file_hash"] for d in st.session_state.active_documents]:
                    st.session_state.active_documents.append(doc_entry)
                st.info("Already indexed. Added to workspace.")

    # ── Active workspace ──────────────────────────────────────────────────
    if st.session_state.active_documents:
        st.info("**Active Documents:**")
        for doc in st.session_state.active_documents:
            st.markdown(f"- 📄 `{doc['filename']}`")

        if st.button("🗑️ Clear Workspace"):
            st.session_state.active_documents = []
            st.session_state.messages = []
            st.session_state.thread_id = str(uuid.uuid4())
            st.rerun()

    st.divider()
    st.caption(
        "💡 **Tips:**\n"
        "- Ask about a specific page: *'What does page 5 of report.pdf say?'*\n"
        "- Ask about a specific doc: *'Summarise annual_report.pdf'*\n"
        "- Compare docs: *'Compare the revenue tables in both PDFs'*\n"
        "- Ask about tables: *'Show me the Q3 numbers from the finance doc'*"
    )

# ── CHAT ──────────────────────────────────────────────────────────────────────
st.header("2. Chat with your Documents")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            with st.expander("📎 Sources"):
                for cite in msg["citations"]:
                    icon = "🖼️" if cite.get("image_path") else "📄"
                    st.caption(f"{icon} Page {cite['page_number']} | {cite['source_file']}")
                    if cite.get("image_path"):
                        st.image(cite["image_path"], caption=f"Page {cite['page_number']}")

if prompt := st.chat_input("Ask anything about your documents..."):
    if not st.session_state.active_documents:
        st.warning("⚠️ Upload and process at least one document first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                payload = {
                    "query": prompt,
                    "thread_id": st.session_state.thread_id,
                    # Send rich active_documents list (hash + filename)
                    "active_documents": st.session_state.active_documents,
                    # Also send flat hashes for backward compat
                    "active_file_hashes": [
                        d["file_hash"] for d in st.session_state.active_documents
                    ],
                }

                try:
                    res = requests.post(f"{API_URL}/chat", json=payload, timeout=120)
                    res.raise_for_status()
                    data = res.json()

                    answer = data["answer"]
                    citations = data.get("citations", [])

                    st.markdown(answer)

                    if citations:
                        with st.expander("📎 Sources"):
                            for cite in citations:
                                icon = "🖼️" if cite.get("image_path") else "📄"
                                st.caption(
                                    f"{icon} Page {cite['page_number']} | {cite['source_file']}"
                                )
                                if cite.get("image_path"):
                                    st.image(cite["image_path"], caption=f"Page {cite['page_number']}")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "citations": citations,
                    })

                except requests.HTTPError as e:
                    if e.response is not None and e.response.status_code == 400:
                        st.error(f"⚠️ {e.response.json().get('detail', str(e))}")
                    else:
                        st.error(f"API Error: {e}")
                except Exception as e:
                    st.error(f"Error: {e}")
