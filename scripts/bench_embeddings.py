#!/usr/bin/env python3
import os
import sys
import json
import time
import random
import threading
import logging
from dotenv import load_dotenv
from tqdm import tqdm

# Import the Ollama client (make sure the package is installed)
from ollama import Client

# Global counters, thread statuses, and a lock for thread-safe updates
success_calls = 0
success_time = 0.0

failure_calls = 0
faliure_time = 0.0

total_embeddings = 0
total_request_time = 0.0

thread_status = {}   # will map thread_id -> "running" | "crashed" | "completed"
lock = threading.Lock()

def worker(thread_id, batch, endpoint, duration, start_time):
    """
    Worker thread that instantiates its own Ollama client and repeatedly makes embedding requests.
    Checks the response code (or error field) and retries up to max_retries if the response is not successful.
    If retries are exhausted, the thread marks itself as "crashed" and raises an exception.
    """
    global success_calls, failure_calls, faliure_time, success_time, total_request_time, total_embeddings, thread_status
    with lock:
        thread_status[thread_id] = "running"
    client = Client(endpoint)
    try:
        while time.time() - start_time < duration:
            req_start = time.time()
            succ = True
            try:
                response = client.embed(model="bge-m3", input=batch)
                # Check for a status_code (if available) or an error field in the response.
                if (hasattr(response, "status_code") and response.status_code != 200) or \
                    (isinstance(response, dict) and "error" in response):
                    succ = False
            except Exception as e:
                succ = False
        
            elapsed = time.time() - req_start
            with lock:
                total_request_time += elapsed
                if succ:
                    success_calls += 1
                    total_embeddings += len(batch)
                    success_time += elapsed
                else:
                    failure_calls += 1
                    faliure_time += elapsed

                
    finally:
        # If exiting normally (duration expired) and not already marked crashed, mark as completed.
        with lock:
            if thread_status.get(thread_id) != "crashed":
                thread_status[thread_id] = "completed"

def progress_logger(duration, start_time):
    """
    Logs progress using a tqdm progress bar updated every second.
    """
    pbar = tqdm(total=duration, desc="Benchmark Time", unit="sec")
    while time.time() - start_time < duration:
        with lock:
            avg_time = total_request_time / success_calls if success_calls > 0 else 0
            pbar.set_postfix({
                "success": success_calls,
                "failures": failure_calls,
                "embeddings": total_embeddings,
                "avg_time": f"{avg_time:.4f}s"
            })
        elapsed = int(time.time() - start_time)
        pbar.n = elapsed
        pbar.refresh()
        time.sleep(1)
    pbar.close()

def main():
    # Set our logging level to INFO.
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # Suppress underlying client logs (from ollama and httpx) to avoid logging successful HTTP requests.
    logging.getLogger("ollama").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    if len(sys.argv) != 4:
        print("Usage: {} <path_to_json> <num_concurrent_requests> <batch_size>".format(sys.argv[0]))
        sys.exit(1)
        
    json_path = sys.argv[1]
    try:
        num_threads = int(sys.argv[2])
        batch_size = int(sys.argv[3])
    except ValueError:
        print("num_concurrent_requests and batch_size must be integers.")
        sys.exit(1)
    
    # Load the .env file and get the Ollama endpoint.
    load_dotenv()
    endpoint = os.getenv("OLLAMA_ENDPOINT")
    if not endpoint:
        print("OLLAMA_ENDPOINT not found in .env")
        sys.exit(1)
        
    # Load the dataset (an array of strings) from the JSON file.
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        sys.exit(1)
        
    if not isinstance(data, list):
        print("JSON file must contain an array of strings.")
        sys.exit(1)
        
    # Benchmark duration (in seconds)
    duration = 60
    start_time = time.time()

    threads = {}
    thread_counter = 0

    # Start initial worker threads.

    for _ in range(num_threads):
        thread_counter += 1
        if len(data) >= batch_size:
            batch = random.sample(data, batch_size)
        else:
            batch = random.choices(data, k=batch_size)
        t = threading.Thread(target=worker, args=(thread_counter, batch, endpoint, duration, start_time))
        t.start()
        threads[thread_counter] = t

    # Start progress logger thread.
    progress_thread = threading.Thread(target=progress_logger, args=(duration, start_time))
    progress_thread.start()

    # Supervisor loop: every second, check if any thread has crashed and, if so, spawn a replacement.
    while time.time() - start_time < duration:
        time.sleep(1)
        with lock:
            for tid in list(threads.keys()):
                # Only respawn if the thread has crashed.
                if not threads[tid].is_alive() and thread_status.get(tid) == "crashed":
                    logging.warning(f"Thread {tid} crashed. Spawning a new thread.")
                    thread_counter += 1
                    if len(data) >= batch_size:
                        new_batch = random.sample(data, batch_size)
                    else:
                        new_batch = random.choices(data, k=batch_size)
                    new_thread = threading.Thread(target=worker, args=(thread_counter, new_batch, endpoint, duration, start_time))
                    new_thread.start()
                    threads[thread_counter] = new_thread
                    del threads[tid]

    # Wait for all worker threads and progress logger to complete.
    for t in threads.values():
        t.join()
    progress_thread.join()

    avg_time = total_request_time / (success_calls + failure_calls) if (success_calls + failure_calls) > 0 else 0
    avg_succ_time = success_time / success_calls if success_calls > 0 else 0
    avg_fail_time = faliure_time / failure_calls if failure_calls > 0 else 0
    print("\nBenchmark Results:")
    print(f"🟡 Average request time: {avg_time:.4f} seconds")
    print(f"🟢 Average success time: {avg_succ_time:.4f} seconds")
    print(f"🔴 Average fail time: {avg_fail_time:.4f} seconds")
    print("")
    print(f"👍 Successful calls: {success_calls}")
    print(f"👎 Crashes/Failures: {failure_calls}")
    print(f"✅ Total embeddings made: {total_embeddings}")

if __name__ == "__main__":
    main()
