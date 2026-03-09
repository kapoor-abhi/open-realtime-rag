import streamlit as st
import requests
import uuid
import time

API_URL = "http://api:8000"

st.set_page_config(page_title="OpenMultiRAG", page_icon="📚", layout="wide")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
# NEW: Use lists instead of single strings to hold multiple documents
if "active_file_hashes" not in st.session_state:
    st.session_state.active_file_hashes = []
if "active_filenames" not in st.session_state:
    st.session_state.active_filenames = []

st.title("📚 OpenMultiRAG Chat")
st.markdown("Upload documents, let the multimodal worker index them, and ask questions across all of them!")

with st.sidebar:
    st.header("1. Upload Document(s)")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        if st.button("Process Document"):
            with st.spinner("Uploading to backend..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                res = requests.post(f"{API_URL}/upload", files=files)
                
                if res.status_code == 200:
                    data = res.json()
                    file_hash = data["file_hash"]
                    status_msg = data["status"]
                    
                    st.success(f"Status: {status_msg}")
                    
                    if status_msg == "Processing started":
                        status_placeholder = st.empty()
                        current_status = "PENDING"
                        
                        while current_status in ["PENDING", "PROCESSING"]:
                            time.sleep(3)
                            status_res = requests.get(f"{API_URL}/document/{file_hash}/status")
                            if status_res.status_code == 200:
                                current_status = status_res.json()["status"]
                                status_placeholder.info(f"Worker Status: {current_status}...")
                        
                        if current_status == "COMPLETED":
                            status_placeholder.success("Indexing Complete! Ready to chat.")
                        else:
                            status_placeholder.error(f"Worker failed with status: {current_status}")
                    
                    # NEW: Append to the list instead of overwriting!
                    if file_hash not in st.session_state.active_file_hashes:
                        st.session_state.active_file_hashes.append(file_hash)
                        st.session_state.active_filenames.append(uploaded_file.name)
                else:
                    st.error(f"Upload failed: {res.text}")

    if st.session_state.active_filenames:
        st.info("**Active Documents in this Chat:**")
        for fname in st.session_state.active_filenames:
            st.markdown(f"- {fname}")
        if st.button("Clear Workspace"):
            st.session_state.active_file_hashes = []
            st.session_state.active_filenames = []
            st.session_state.messages = []
            st.rerun()

# --- MAIN CHAT INTERFACE ---
st.header("2. Chat with your Data")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "citations" in msg and msg["citations"]:
            with st.expander("View Sources & Citations"):
                for cite in msg["citations"]:
                    st.caption(f"📄 Page: {cite['page_number']} | File: {cite['source_file']}")
                    if cite.get("image_path"):
                        st.image(cite["image_path"], caption=f"Source Image from Page {cite['page_number']}")

if prompt := st.chat_input("Ask a question about any uploaded document..."):
    if not st.session_state.active_file_hashes:
        st.warning("Please upload and process at least one document first!")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # NEW: Pass the ARRAY of hashes to the backend!
                payload = {
                    "query": prompt,
                    "thread_id": st.session_state.thread_id,
                    "active_file_hashes": st.session_state.active_file_hashes
                }
                
                try:
                    res = requests.post(f"{API_URL}/chat", json=payload)
                    res.raise_for_status()
                    data = res.json()
                    
                    answer = data["answer"]
                    citations = data.get("citations", [])
                    
                    st.markdown(answer)
                    
                    if citations:
                        with st.expander("View Sources & Citations"):
                            for cite in citations:
                                st.caption(f"📄 Page: {cite['page_number']} | File: {cite['source_file']}")
                                if cite.get("image_path"):
                                    st.image(cite["image_path"], caption=f"Source Image from Page {cite['page_number']}")
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "citations": citations
                    })
                    
                except Exception as e:
                    st.error(f"API Error: {e}")