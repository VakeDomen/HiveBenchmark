#!/usr/bin/env python3
import os
import sys
import time
import threading
import logging
import argparse
from dotenv import load_dotenv
from tqdm import tqdm
from ollama import Client

# Global counters, thread statuses, and a lock for thread-safe updates
success_calls = 0
success_time = 0.0

failure_calls = 0
faliure_time = 0.0

total_request_time = 0.0

thread_status = {}   # maps thread_id -> "running" | "crashed" | "completed"
lock = threading.Lock()


def worker(thread_id, question, endpoint, model, duration, start_time):
    """
    Worker thread that instantiates its own Ollama client and repeatedly asks a question.
    Checks the response (via status_code or error field) and records success or failure.
    """
    global success_calls, failure_calls, faliure_time, success_time, total_request_time, thread_status
    with lock:
        thread_status[thread_id] = "running"
    client = Client(endpoint)
    try:
        while time.time() - start_time < duration:
            req_start = time.time()
            succ = True
            try:
                response = client.generate(model=model, prompt=question)
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
                    success_time += elapsed
                else:
                    failure_calls += 1
                    faliure_time += elapsed
    finally:
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
            total_calls = success_calls + failure_calls
            avg_time = total_request_time / total_calls if total_calls > 0 else 0
            pbar.set_postfix({
                "success": success_calls,
                "failures": failure_calls,
                "avg_time": f"{avg_time:.4f}s"
            })
        elapsed = int(time.time() - start_time)
        pbar.n = elapsed
        pbar.refresh()
        time.sleep(1)
    pbar.close()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Hive (Ollama proxy) Q&A calls."
    )
    parser.add_argument("--question", "-q", default="What is the meaning of life?",
                        help="Question to ask the model (default: 'What is the meaning of life?')")
    parser.add_argument("--concurrent", "-c", type=int, default=20,
                        help="Number of concurrent requests (default: 20)")
    parser.add_argument("--host", "-H", default=None,
                        help="Ollama host endpoint (default: env OLLAMA_ENDPOINT or http://localhost:11434)")
    parser.add_argument("--model", "-m", default="mistral-nemo",
                        help="Model to use for Q&A (default: mistral-nemo)")
    parser.add_argument("--time", "-t", type=int, default=60,
                        help="Benchmark duration in seconds (default: 60)")
    args = parser.parse_args()

    load_dotenv()
    endpoint = args.host or os.getenv("OLLAMA_ENDPOINT") or "http://localhost:11434"
    model = args.model
    duration = args.time
    num_threads = args.concurrent
    question = args.question

    print("Q&A benchmark configuration:")
    print(f"\t-> Host:     {endpoint}")
    print(f"\t-> Model:    {model}")
    print(f"\t-> Duration: {duration} [s]")
    print(f"\t-> Threads:  {num_threads}")
    print(f"\t-> Question: {question}")

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.getLogger("ollama").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    start_time = time.time()
    threads = {}
    thread_counter = 0

    for _ in range(num_threads):
        thread_counter += 1
        t = threading.Thread(target=worker, args=(thread_counter, question, endpoint, model, duration, start_time))
        t.start()
        threads[thread_counter] = t

    progress_thread = threading.Thread(target=progress_logger, args=(duration, start_time))
    progress_thread.start()

    while time.time() - start_time < duration:
        time.sleep(1)
        with lock:
            for tid in list(threads.keys()):
                if not threads[tid].is_alive() and thread_status.get(tid) == "crashed":
                    logging.warning(f"Thread {tid} crashed. Spawning a new thread.")
                    thread_counter += 1
                    new_thread = threading.Thread(target=worker, args=(thread_counter, question, endpoint, model, duration, start_time))
                    new_thread.start()
                    threads[thread_counter] = new_thread
                    del threads[tid]

    for t in threads.values():
        t.join()
    progress_thread.join()

    total_calls = success_calls + failure_calls
    avg_time = total_request_time / total_calls if total_calls > 0 else 0
    avg_succ_time = success_time / success_calls if success_calls > 0 else 0
    avg_fail_time = faliure_time / failure_calls if failure_calls > 0 else 0
    print("\nBenchmark Results:")
    print(f"👍 Successful calls: {success_calls}")
    print(f"👎 Crashes/Failures: {failure_calls}")
    print("")
    print(f"🟡 Average request time: {avg_time:.4f} seconds")
    print(f"🟢 Average success time: {avg_succ_time:.4f} seconds")
    print(f"🔴 Average fail time: {avg_fail_time:.4f} seconds")


if __name__ == "__main__":
    main()
