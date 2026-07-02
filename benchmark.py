#!/usr/bin/env python3
"""
Rock Cluster Benchmark - Q4_K_M solo on rock0

Tests:
1. llama-bench for raw layer performance metrics (4 threads only)
2. llama-server with single and concurrent requests
"""

import subprocess
import json
import time
import sys
import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

MODEL_PATH = "/home/rock/LFM2.5-8B-A1B-Q4_K_M.gguf"
LLAMA_DIR = "/home/rock/llama.cpp/build/bin"
HOST = "127.0.0.1"
PORT = 8080
BASE_URL = f"http://{HOST}:{PORT}"

def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def run_bench():
    """Run llama-bench for raw performance metrics - 4 threads only"""
    log("=== llama-bench: raw inference performance (4 threads) ===")
    cmd = [
        f"{LLAMA_DIR}/llama-bench",
        "-m", MODEL_PATH,
        "-t", "4",
        "-p", "512",
        "-n", "128,512",
        "-r", "2",
        "-ngl", "99",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    print(result.stdout, flush=True)

def start_server(n_threads=4, cont_batching=True):
    """Start llama-server with given config"""
    cmd = [
        f"{LLAMA_DIR}/llama-server",
        "-m", MODEL_PATH,
        "-t", str(n_threads),
        "-c", "4096",
        "-ngl", "99",
        "--host", HOST,
        "--port", str(PORT),
        "-np", "4",
    ]
    if cont_batching:
        cmd.append("-cb")

    log(f"Starting server: threads={n_threads}, cont_batching={cont_batching}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for _ in range(120):
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=2)
            if r.status_code == 200:
                log("Server is ready")
                return proc
        except:
            pass
        time.sleep(1)

    log("ERROR: Server failed to start")
    out, err = proc.communicate(timeout=5)
    print(err.decode()[:2000])
    return None

def stop_server(proc):
    if proc:
        proc.terminate()
        proc.wait(timeout=10)
        log("Server stopped")

def send_completion(prompt, max_tokens=128, temperature=0.7, request_id=0):
    """Send a completion request and return timing + token stats"""
    start = time.time()
    try:
        r = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": "lfm2.5-8b",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
            },
            timeout=180,
        )
        elapsed = time.time() - start
        data = r.json()
        completion_tokens = data.get("usage", {}).get("completion_tokens", 0)
        prompt_tokens = data.get("usage", {}).get("prompt_tokens", 0)
        tps = completion_tokens / elapsed if elapsed > 0 and completion_tokens > 0 else 0
        return {
            "request_id": request_id,
            "status": "ok",
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "wall_time_s": round(elapsed, 2),
            "gen_tps": round(tps, 2),
        }
    except Exception as e:
        elapsed = time.time() - start
        return {
            "request_id": request_id,
            "status": "error",
            "error": str(e),
            "wall_time_s": round(elapsed, 2),
        }

PROMPTS = [
    "Explain how transformer attention works in simple terms.",
    "Write a Python function to merge two sorted lists.",
    "Describe the process of photosynthesis step by step.",
    "What are the main differences between TCP and UDP?",
    "Write a short poem about the ocean.",
    "Explain the concept of quantization in machine learning.",
    "What causes rainbows to form?",
    "List five best practices for writing clean code.",
]

def benchmark_single(n_tokens=128):
    """Single request benchmark"""
    log(f"=== Single Request ({n_tokens} tokens) ===")
    results = []
    for i, prompt in enumerate(PROMPTS[:3]):
        log(f"  Request {i+1}: '{prompt[:50]}...'")
        r = send_completion(prompt, max_tokens=n_tokens, request_id=i)
        results.append(r)
        if r["status"] == "ok":
            log(f"    -> {r['gen_tps']} tok/s, {r['completion_tokens']} tokens in {r['wall_time_s']}s")
        else:
            log(f"    -> FAILED: {r.get('error', 'unknown')}")
    ok = [r for r in results if r["status"] == "ok"]
    if ok:
        avg_tps = sum(r["gen_tps"] for r in ok) / len(ok)
        avg_time = sum(r["wall_time_s"] for r in ok) / len(ok)
        log(f"  AVERAGE: {avg_tps:.2f} tok/s, {avg_time:.2f}s per request")
    return results

def benchmark_concurrent(n_concurrent=2, n_tokens=128):
    """Concurrent request benchmark"""
    log(f"=== Concurrent ({n_concurrent} parallel, {n_tokens} tokens each) ===")
    prompts = PROMPTS[:n_concurrent]
    start = time.time()
    with ThreadPoolExecutor(max_workers=n_concurrent) as pool:
        futures = {
            pool.submit(send_completion, p, n_tokens, 0.7, i): i
            for i, p in enumerate(prompts)
        }
        results = []
        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            if r["status"] == "ok":
                log(f"    Req {r['request_id']}: {r['gen_tps']} tok/s, {r['wall_time_s']}s")
            else:
                log(f"    Req {r['request_id']}: FAILED")
    total_time = time.time() - start
    ok = [r for r in results if r["status"] == "ok"]
    if ok:
        total_tokens = sum(r["completion_tokens"] for r in ok)
        agg_tps = total_tokens / total_time if total_time > 0 else 0
        avg_per = sum(r["gen_tps"] for r in ok) / len(ok)
        log(f"  WALL: {total_time:.2f}s | AGGREGATE: {agg_tps:.2f} tok/s | AVG/REQ: {avg_per:.2f} tok/s")
    return results

def main():
    log("Rock Cluster Benchmark - Q4_K_M on rock0 (solo baseline)")
    log(f"Model: {MODEL_PATH}")
    log(f"Node: rock0 (RPi5, Cortex-A76, 4-core, 8GB)")
    log("")

    # Phase 1: Raw bench (4 threads only - the 2/1 thread tests are too slow)
    run_bench()

    # Phase 2: Server-based benchmarks
    proc = start_server(n_threads=4, cont_batching=True)
    if not proc:
        log("FATAL: Could not start server")
        sys.exit(1)

    try:
        log("\n--- Warmup ---")
        send_completion("Hello", max_tokens=8)

        log("")
        benchmark_single(n_tokens=128)
        log("")
        benchmark_single(n_tokens=256)

        log("")
        for n in [1, 2, 3, 4]:
            benchmark_concurrent(n_concurrent=n, n_tokens=128)
            log("")
    finally:
        stop_server(proc)

    log("=== Benchmark Complete ===")

if __name__ == "__main__":
    main()
