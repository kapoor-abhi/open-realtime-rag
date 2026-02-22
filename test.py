import requests
import time
import logging
import sys

logging.basicConfig(level=logging.INFO, format="\n%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8000"
FILE_PATH = "supply_chain_analysis.pdf"
THREAD_ID = "verbose_smart_test"

def run_tests():
    logger.info("Starting Smart Integration Test Suite...")

    # --- TEST 1: HEALTH ---
    res = requests.get(f"{BASE_URL}/health")
    if res.status_code != 200:
        logger.error("FAIL: Cannot reach API.")
        sys.exit(1)
    logger.info("SUCCESS: API Health Check")

    # --- TEST 2: UPLOAD ---
    with open(FILE_PATH, "rb") as f:
        res = requests.post(f"{BASE_URL}/upload", files={"file": (FILE_PATH, f, "application/pdf")})
    
    if res.status_code != 200:
        logger.error(f"FAIL: Upload failed. {res.text}")
        sys.exit(1)
        
    data = res.json()
    file_hash = data.get("file_hash")
    logger.info(f"SUCCESS: Upload accepted. Task ID: {data.get('task_id')}")

    # --- THE SMART POLLER ---
    logger.info("Polling Postgres every 5 seconds for worker completion...")
    max_retries = 120 # 10 minutes max
    attempts = 0
    status = "PENDING"
    
    while status in ["PENDING", "PROCESSING"] and attempts < max_retries:
        time.sleep(5)
        attempts += 1
        status_res = requests.get(f"{BASE_URL}/document/{file_hash}/status")
        
        if status_res.status_code == 200:
            status = status_res.json().get("status")
            sys.stdout.write(f"\rAttempt {attempts}: Status is [{status}]... ")
            sys.stdout.flush()
        else:
            logger.warning("\nStatus check failed, retrying...")
            
    print() # newline
    if status != "COMPLETED":
        logger.error(f"FAIL: Worker failed or timed out. Final status: {status}")
        sys.exit(1)
    
    logger.info("SUCCESS: Worker finished parsing, embedding, and uploading! Proceeding to chat...")

    # --- TEST 3: DEDUPLICATION ---
    with open(FILE_PATH, "rb") as f:
        res = requests.post(f"{BASE_URL}/upload", files={"file": (FILE_PATH, f, "application/pdf")})
    if res.json().get("status") == "Document already indexed":
        logger.info("SUCCESS: Deduplication blocked duplicate embedding.")

    # --- TEST 4: BASE RETRIEVAL ---
    query_1 = "What are the key technologies and methodologies used in the project?"
    logger.info(f"QUESTION: '{query_1}'")
    
    start_time = time.time()
    # UPDATED: Passing file_hash so the context-aware cache and retriever know where to look
    res = requests.post(f"{BASE_URL}/chat", json={"query": query_1, "thread_id": THREAD_ID, "file_hash": file_hash})
    req_time = time.time() - start_time
    
    if res.status_code == 200:
        logger.info(f"SUCCESS: Answer in {req_time:.2f}s -> {res.json()['answer']}")
    else:
        logger.error(f"FAIL: Chat 500 Error. Raw Response: {res.text}")

    # --- TEST 5: CACHING ---
    start_time = time.time()
    res = requests.post(f"{BASE_URL}/chat", json={"query": query_1, "thread_id": THREAD_ID, "file_hash": file_hash})
    cache_time = time.time() - start_time
    if res.status_code == 200:
        logger.info(f"SUCCESS: CACHED Answer in {cache_time:.2f}s (Cache faster? {'YES' if cache_time < req_time else 'NO'})")

    # --- TEST 6: MEMORY ---
    query_memory = "Can you list just the Python libraries from your previous answer?"
    res = requests.post(f"{BASE_URL}/chat", json={"query": query_memory, "thread_id": THREAD_ID, "file_hash": file_hash})
    if res.status_code == 200:
        logger.info(f"SUCCESS: Memory response -> {res.json()['answer']}")

    # --- TEST 7: INTENT ROUTING ---
    query_page = "Summarize the final model showdown on page 8."
    res = requests.post(f"{BASE_URL}/chat", json={"query": query_page, "thread_id": THREAD_ID, "file_hash": file_hash})
    if res.status_code == 200:
        logger.info(f"SUCCESS: Page routing -> {res.json()['answer']}")

    logger.info("\n=== INTEGRATION TEST SUITE COMPLETE ===")

if __name__ == "__main__":
    run_tests()