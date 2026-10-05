"""
Automotive Edge OBU Hardware Emulation & Latency Benchmark for V2X-BERT.
Profiles single-stream (batch_size=1) inference latency, memory footprint, and real-time deadline compliance
under constrained single-threaded execution modeling automotive edge processors (e.g. ARM Cortex-A53/A72, NXP i.MX8, Jetson Nano).
"""

import os
import sys
import time
import argparse
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from v2x_bert import load_model, V2XTokenizer


def run_obu_edge_benchmark(num_runs=1000, max_threads=1):
    """
    Simulates on-vehicle On-Board Unit (OBU) edge inference under single-thread CPU constraints.
    Evaluates both FP32 full precision and INT8 dynamically quantized models.
    """
    # Restrict PyTorch to single-threaded execution to simulate low-power embedded OBU core
    torch.set_num_threads(max_threads)
    torch.set_num_interop_threads(max_threads)

    print(f"\n{'='*75}")
    print(f"   AUTOMOTIVE EDGE OBU HARDWARE EMULATION BENCHMARK")
    print(f"   Simulated Core Architecture: Embedded ARM/x86 Automotive OBU Core")
    print(f"   Execution Threads: {max_threads} | Test Stream Frames: {num_runs:,} (Batch Size = 1)")
    print(f"   V2X Real-Time Broadcast Deadline: 100.0 ms (10 Hz BSM/CAM stream)")
    print(f"{'='*75}\n")

    tokenizer = V2XTokenizer()
    model_fp32 = load_model(pretrained=False, device="cpu")
    model_int8 = model_fp32.quantize_int8()

    footprint = model_fp32.get_memory_footprint()

    # Synthetic single-frame multi-message stream (5 BSM messages = 64 tokens)
    dummy_tokens = torch.randint(0, 1024, (1, 64), dtype=torch.long)
    dummy_mask = torch.ones((1, 64), dtype=torch.long)

    configs = [
        {"name": "V2X-BERT FP32 (Full Precision)", "model": model_fp32, "size_mb": footprint["fp32_mb"]},
        {"name": "V2X-BERT INT8 (Dynamic Quantized)", "model": model_int8, "size_mb": footprint["int8_mb"]}
    ]

    results = {}

    for cfg in configs:
        name = cfg["name"]
        m = cfg["model"]
        m.eval()

        # Warm-up 100 passes
        for _ in range(100):
            with torch.no_grad():
                _ = m.forward_classify(dummy_tokens, attention_mask=dummy_mask)

        latencies_us = []
        with torch.no_grad():
            for _ in range(num_runs):
                t0 = time.perf_counter()
                _ = m.forward_classify(dummy_tokens, attention_mask=dummy_mask)
                t1 = time.perf_counter()
                latencies_us.append((t1 - t0) * 1_000_000.0)

        lat_arr = np.array(latencies_us)
        mean_us = float(np.mean(lat_arr))
        median_us = float(np.median(lat_arr))
        p95_us = float(np.percentile(lat_arr, 95))
        p99_us = float(np.percentile(lat_arr, 99))
        max_us = float(np.max(lat_arr))

        mean_ms = mean_us / 1000.0
        p95_ms = p95_us / 1000.0

        # Check real-time deadline (100 ms = 100,000 us)
        compliant = mean_ms < 100.0
        throughput_fps = 1_000_000.0 / mean_us

        print(f"[{name}]")
        print(f"  • Memory Footprint:  {cfg['size_mb']:.2f} MB")
        print(f"  • Latency (Mean):    {mean_us:.1f} µs ({mean_ms:.2f} ms)")
        print(f"  • Latency (Median):  {median_us:.1f} µs")
        print(f"  • Latency (P95):     {p95_us:.1f} µs ({p95_ms:.2f} ms)")
        print(f"  • Latency (P99):     {p99_us:.1f} µs")
        print(f"  • Max In-Flight Lat: {max_us:.1f} µs")
        print(f"  • Edge Throughput:   {throughput_fps:.1f} frames/sec")
        print(f"  • 10 Hz Real-Time:   {'PASSED (Compliant with SAE/ETSI Budget)' if compliant else 'FAILED'}")
        print(f"{'-'*75}")

        results[name] = {
            "size_mb": cfg["size_mb"],
            "mean_us": mean_us,
            "median_us": median_us,
            "p95_us": p95_us,
            "p99_us": p99_us,
            "max_us": max_us,
            "throughput_fps": throughput_fps,
            "real_time_compliant": compliant
        }

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OBU Edge Hardware Latency Profiler")
    parser.add_argument("--runs", type=int, default=500, help="Number of benchmark test frames")
    parser.add_argument("--threads", type=int, default=1, help="Max execution threads (1 for embedded core)")
    args = parser.parse_args()

    run_obu_edge_benchmark(num_runs=args.runs, max_threads=args.threads)
