import torch
import time
import numpy as np
from philtorch.lti import lfilter, lfilter_zi
from functools import partial

def lowpass_biquad_coef(sample_rate, cutoff_freq, Q):
    w0 = 2 * torch.pi * cutoff_freq / sample_rate
    alpha = torch.sin(w0) / (2 * Q)
    b0 = (1 - torch.cos(w0)) / 2
    b1 = 1 - torch.cos(w0)
    b2 = b0
    a0 = 1 + alpha
    a1 = -2 * torch.cos(w0)
    a2 = 1 - alpha
    # Normalize by a0
    b = torch.stack([b0, b1, b2], dim=-1) / a0
    a = torch.stack([a1, a2], dim=-1) / a0
    return b, a

def run_benchmark():
    # --- Configuration ---
    batch_size = 8
    sr = 48000
    duration = 4
    num_samples = sr * duration
    cutoff = torch.tensor(4000.0)
    Q = torch.tensor(0.707)
    iterations = 50  # Number of runs to average
    
    print(f"Benchmarking: Batch={batch_size}, {duration}s @ {sr}Hz ({num_samples} samples)")
    print("-" * 50)

    results = {}

    for device in ["cpu", "cuda"]:
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, skipping...")
            continue
            
        # Move params to device
        dev_cutoff = cutoff.to(device)
        dev_Q = Q.to(device)
        
        # Get coefficients
        b, a = lowpass_biquad_coef(sr, dev_cutoff, dev_Q)
        zi_base = lfilter_zi(a, b)
        x = torch.randn(batch_size, num_samples, device=device)
        zi = zi_base.repeat(batch_size, 1) * x[:, :1]

        # --- Warm-up ---
        # Crucial for CUDA to initialize kernels/memory
        for _ in range(5):
            _ = lfilter(b, a, x, zi=zi)
        
        if device == "cuda":
            torch.cuda.synchronize()

        # --- Measured Loop ---
        start_time = time.perf_counter()
        for _ in range(iterations):
            x = torch.randn(batch_size, num_samples, device=device)
            y, _ = lfilter(b, a, x, zi=zi)
            
            if device == "cuda":
                # Ensure the GPU finished the work before stopping the clock
                torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        
        avg_time = (end_time - start_time) / iterations
        results[device] = avg_time
        print(f"{device.upper()} Avg Execution Time: {avg_time*1000:.3f} ms")

    # --- Summary ---
    if "cpu" in results and "cuda" in results:
        speedup = results["cpu"] / results["cuda"]
        print("-" * 50)
        print(f"CUDA Speedup: {speedup:.2f}x")

if __name__ == "__main__":
    run_benchmark()
