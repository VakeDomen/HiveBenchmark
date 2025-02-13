#!/usr/bin/env python3
import os
import sys
import json
import time
import random
import threading
import logging
import argparse
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

thread_status = {}   # maps thread_id -> "running" | "crashed" | "completed"
lock = threading.Lock()


def worker(thread_id, batch, endpoint, model, duration, start_time):
    """
    Worker thread that instantiates its own Ollama client and repeatedly makes embedding requests.
    Checks the response code (or error field) and records success or failure.
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
                response = client.embed(model=model, input=batch)
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
        # Mark thread as completed if not crashed.
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
            avg_time = total_request_time / (success_calls + failure_calls) if (success_calls + failure_calls) > 0 else 0
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
    parser = argparse.ArgumentParser(
        description="Benchmark Hive (Ollama proxy) embedding calls."
    )
    # Optional positional argument with default JSON file.
    parser.add_argument("json_file", nargs="?", default="../data/chunks.json",
                        help="Path to JSON file containing an array of strings (chunks). Default: ../data/chunks.json")
    parser.add_argument("--concurrent", "-c", type=int, default=100,
                        help="Number of concurrent requests (default: 100)")
    parser.add_argument("--batch", "-b", type=int, default=5,
                        help="Batch size (number of embeddings per request, default: 5)")
    parser.add_argument("--host", "-H", default=None,
                        help="Ollama host endpoint (default: env OLLAMA_ENDPOINT or http://localhost:11434)")
    parser.add_argument("--model", "-m", default="bge-m3",
                        help="Model to use for embedding (default: bge-m3)")
    parser.add_argument("--time", "-t", type=int, default=60,
                        help="Benchmark duration in seconds (default: 60)")
    args = parser.parse_args()

    # Load environment variables from .env if present.
    load_dotenv()
    # Determine host endpoint:
    endpoint = args.host or os.getenv("OLLAMA_ENDPOINT") or "http://localhost:11434"
    model = args.model
    duration = args.time
    num_threads = args.concurrent
    batch_size = args.batch


    print("Embedding benchmark configuration:")
    print(f"\t-> data:       {args.json_file}")
    print(f"\t-> model:      {model}")
    print(f"\t-> duration:   {duration} [s]")
    print(f"\t-> #threads:   {num_threads}")
    print(f"\t-> batch size: {batch_size}")

    # Set up logging.
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.getLogger("ollama").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Load dataset from JSON file.
    try:
        with open(args.json_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        sys.exit(1)

    if not isinstance(data, list):
        print("JSON file must contain an array of strings.")
        sys.exit(1)

    start_time = time.time()
    threads = {}
    thread_counter = 0

    for _ in range(num_threads):
        thread_counter += 1
        if len(data) >= batch_size:
            batch = random.sample(data, batch_size)
        else:
            batch = random.choices(data, k=batch_size)
        t = threading.Thread(target=worker, args=(thread_counter, batch, endpoint, model, duration, start_time))
        t.start()
        threads[thread_counter] = t

    # Start progress logger thread.
    progress_thread = threading.Thread(target=progress_logger, args=(duration, start_time))
    progress_thread.start()

    # Supervisor loop: every second, check if any thread has crashed and spawn a replacement.
    while time.time() - start_time < duration:
        time.sleep(1)
        with lock:
            for tid in list(threads.keys()):
                if not threads[tid].is_alive() and thread_status.get(tid) == "crashed":
                    logging.warning(f"Thread {tid} crashed. Spawning a new thread.")
                    thread_counter += 1
                    if len(data) >= batch_size:
                        new_batch = random.sample(data, batch_size)
                    else:
                        new_batch = random.choices(data, k=batch_size)
                    new_thread = threading.Thread(target=worker, args=(thread_counter, new_batch, endpoint, model, duration, start_time))
                    new_thread.start()
                    threads[thread_counter] = new_thread
                    del threads[tid]

    for t in threads.values():
        t.join()
    progress_thread.join()

    avg_time = total_request_time / (success_calls + failure_calls) if (success_calls + failure_calls) > 0 else 0
    avg_succ_time = success_time / success_calls if success_calls > 0 else 0
    avg_fail_time = faliure_time / failure_calls if failure_calls > 0 else 0
    print("\nBenchmark Results:")
    print(f"👍 Successful calls: {success_calls}")
    print(f"👎 Crashes/Failures: {failure_calls}")
    print(f"✅ Total embeddings made: {total_embeddings}")
    print("")
    print(f"🟡 Average request time: {avg_time:.4f} seconds")
    print(f"🟢 Average success time: {avg_succ_time:.4f} seconds")
    print(f"🔴 Average fail time: {avg_fail_time:.4f} seconds")
    
    

if __name__ == "__main__":
    main()
