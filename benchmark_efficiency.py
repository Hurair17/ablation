# # benchmark_efficiency.py
# # -----------------------
# # Measures parameters, GFLOPs, peak GPU memory, and inference latency
# # for every fusion type + the camera-only YOLO11x baseline.
# #
# # Run from multimodal_training/ directory (same folder as multimodal_yolo11.py):
# #   pip install thop ultralytics
# #   python benchmark_efficiency.py
# #
# # Output: prints a Markdown table ready to paste into the paper.

# import sys, os, time, gc
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# import torch
# import torch.nn.functional as F

# DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
# IMG_SIZE   = 640
# BATCH_SIZE = 1           # latency is always measured at BS=1
# WARMUP     = 50          # GPU warmup iterations
# MEASURE    = 200         # timing iterations
# NC         = 10


# # ──────────────────────────────────────────────────────────────────────────────
# # Helpers
# # ──────────────────────────────────────────────────────────────────────────────

# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters())


# def count_flops(model, *dummy_inputs):
#     """Returns GFLOPs using thop; falls back to 'N/A' if thop not installed."""
#     try:
#         from thop import profile
#         macs, _ = profile(model, inputs=dummy_inputs, verbose=False)
#         return macs * 2 / 1e9   # MACs -> FLOPs (multiply + add = 2 ops)
#     except ImportError:
#         return None


# def peak_memory_mb(model, *dummy_inputs):
#     """Forward pass peak GPU memory in MB."""
#     if DEVICE == "cpu":
#         return None
#     torch.cuda.reset_peak_memory_stats(DEVICE)
#     torch.cuda.synchronize(DEVICE)
#     with torch.no_grad():
#         _ = model(*dummy_inputs)
#     torch.cuda.synchronize(DEVICE)
#     return torch.cuda.max_memory_allocated(DEVICE) / 1e6


# def latency_ms(model, *dummy_inputs):
#     """
#     Median single-image inference latency (ms) on GPU using CUDA events.
#     Falls back to time.perf_counter on CPU.
#     Model mode (train/eval) is set by the caller before invoking this function.
#     """
#     with torch.no_grad():
#         # Warmup
#         for _ in range(WARMUP):
#             _ = model(*dummy_inputs)

#         if DEVICE == "cuda":
#             torch.cuda.synchronize(DEVICE)
#             times = []
#             start_evt = torch.cuda.Event(enable_timing=True)
#             end_evt   = torch.cuda.Event(enable_timing=True)
#             for _ in range(MEASURE):
#                 start_evt.record()
#                 _ = model(*dummy_inputs)
#                 end_evt.record()
#                 torch.cuda.synchronize(DEVICE)
#                 times.append(start_evt.elapsed_time(end_evt))
#         else:
#             times = []
#             for _ in range(MEASURE):
#                 t0 = time.perf_counter()
#                 _ = model(*dummy_inputs)
#                 times.append((time.perf_counter() - t0) * 1000)

#     times_sorted = sorted(times)
#     n = len(times_sorted)
#     return times_sorted[n // 2]   # median


# # ──────────────────────────────────────────────────────────────────────────────
# # Benchmark multimodal fusion models
# # ──────────────────────────────────────────────────────────────────────────────

# def benchmark_multimodal(fusion_type):
#     from multimodal_yolo11 import MultiModalYOLO11

#     model = MultiModalYOLO11(nc=NC, cam_in_ch=3, lidar_in_ch=3,
#                               fusion_type=fusion_type).to(DEVICE)
#     model.eval()

#     cam_dummy   = torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
#     lidar_dummy = torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)

#     params   = count_parameters(model)
#     gflops   = count_flops(model, cam_dummy, lidar_dummy)
#     peak_mem = peak_memory_mb(model, cam_dummy, lidar_dummy)

#     # torch.jit.trace eliminates Python dispatch overhead on Windows (no Triton needed)
#     try:
#         with torch.no_grad():
#             traced = torch.jit.trace(model, (cam_dummy, lidar_dummy))
#         run_model = traced
#     except Exception:
#         run_model = model   # fallback: eager mode
#     lat      = latency_ms(run_model, cam_dummy, lidar_dummy)

#     del model; gc.collect()
#     if DEVICE == "cuda":
#         torch.cuda.empty_cache()

#     return params, gflops, peak_mem, lat


# # ──────────────────────────────────────────────────────────────────────────────
# # Benchmark camera-only YOLO11x baseline
# # ──────────────────────────────────────────────────────────────────────────────

# def benchmark_baseline():
#     from ultralytics import YOLO

#     model = YOLO("yolo11x.yaml")
#     net = model.model.to(DEVICE)

#     # eval() → BatchNorm uses fast running statistics (same as multimodal eval)
#     net.eval()
#     # Set only the final Detect head to training=True so it returns raw tensors
#     # and skips anchor-grid + sigmoid decoding — same execution path as the
#     # custom YOLO11HeadFused used in all multimodal variants.
#     net.model[-1].training = True

#     dummy = torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)

#     params   = count_parameters(net)
#     gflops   = count_flops(net, dummy)
#     peak_mem = peak_memory_mb(net, dummy)

#     # torch.jit.trace eliminates Python dispatch overhead on Windows (no Triton needed)
#     try:
#         with torch.no_grad():
#             traced = torch.jit.trace(net, dummy)
#         run_net = traced
#     except Exception:
#         run_net = net   # fallback: eager mode
#     lat      = latency_ms(run_net, dummy)

#     del net, model; gc.collect()
#     if DEVICE == "cuda":
#         torch.cuda.empty_cache()

#     return params, gflops, peak_mem, lat


# # ──────────────────────────────────────────────────────────────────────────────
# # Main
# # ──────────────────────────────────────────────────────────────────────────────

# def fmt_params(p):
#     return f"{p/1e6:.1f}M"

# def fmt_gflops(g):
#     return f"{g:.1f}" if g is not None else "N/A"

# def fmt_mem(m):
#     return f"{m:.0f}" if m is not None else "N/A"

# def fmt_lat(l):
#     return f"{l:.1f}"


# def main():
#     print(f"\nBenchmarking on: {DEVICE.upper()}")
#     if DEVICE == "cuda":
#         print(f"GPU: {torch.cuda.get_device_name(0)}")
#     print(f"Input: {BATCH_SIZE} x 3 x {IMG_SIZE} x {IMG_SIZE}  |  "
#           f"Warmup={WARMUP}  Measure={MEASURE}\n")

#     results = []

#     # Camera-only baseline
#     print("Measuring camera-only YOLO11x baseline ...")
#     p, g, m, l = benchmark_baseline()
#     results.append(("Camera-only (YOLO11x)", p, g, m, l))
#     print(f"  Done: {fmt_params(p)} params  {fmt_gflops(g)} GFLOPs  "
#           f"{fmt_mem(m)} MB  {fmt_lat(l)} ms")

#     # All fusion types
#     for ftype in ["gated", "adapt_gated", "gated_bilinear", "cross_attn"]:
#         label = {
#             "gated":          "Gated Fusion",
#             "adapt_gated":    "Adaptive Gated",
#             "gated_bilinear": "Gated Bilinear",
#             "cross_attn":     "Cross-Attention (ours)",
#         }[ftype]
#         print(f"Measuring {label} ...")
#         p, g, m, l = benchmark_multimodal(ftype)
#         results.append((label, p, g, m, l))
#         print(f"  Done: {fmt_params(p)} params  {fmt_gflops(g)} GFLOPs  "
#               f"{fmt_mem(m)} MB  {fmt_lat(l)} ms")

#     # ── Print Markdown table ──────────────────────────────────────────────────
#     print("\n\n" + "=" * 75)
#     print("EFFICIENCY TABLE  (paste into paper)")
#     print("=" * 75)
#     header = f"| {'Method':<26} | {'Params':>8} | {'GFLOPs':>8} | {'Peak Mem (MB)':>13} | {'Latency (ms)':>12} |"
#     sep    = "|" + "-"*28 + "|" + "-"*10 + "|" + "-"*10 + "|" + "-"*15 + "|" + "-"*14 + "|"
#     print(header)
#     print(sep)
#     for name, p, g, m, l in results:
#         print(f"| {name:<26} | {fmt_params(p):>8} | {fmt_gflops(g):>8} | "
#               f"{fmt_mem(m):>13} | {fmt_lat(l):>12} |")
#     print()
#     print(f"Hardware : {torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU'}")
#     print(f"Batch    : {BATCH_SIZE}   Image size: {IMG_SIZE}x{IMG_SIZE}")
#     print(f"Latency  : median of {MEASURE} runs after {WARMUP}-run GPU warmup (torch.jit.trace, eager fallback)")
#     print(f"FLOPs    : computed with thop (MACs x2)")
#     print(f"Peak Mem : torch.cuda.max_memory_allocated() per forward pass")

#     # ── Also save to CSV ──────────────────────────────────────────────────────
#     csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "efficiency_results.csv")
#     with open(csv_path, "w") as f:
#         f.write("method,params_M,GFLOPs,peak_mem_MB,latency_ms\n")
#         for name, p, g, m, l in results:
#             f.write(f"{name},{p/1e6:.2f},{fmt_gflops(g)},{fmt_mem(m)},{l:.2f}\n")
#     print(f"\nSaved: {csv_path}")


# if __name__ == "__main__":
#     main()





# benchmark_efficiency.py
# Consistent FP32 PyTorch eager-mode efficiency benchmark

import os
import gc
import csv
import random
import statistics
from pathlib import Path

import torch

from multimodal_yolo11 import MultiModalYOLO11
from ultralytics import YOLO

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 640
BATCH_SIZE = 1
NC = 10

WARMUP = 100
MEASURE = 500
REPEATS = 5

# Set to the trained camera-only checkpoint when available.
# Leave as None to benchmark an untrained YOLO11x architecture.
CAMERA_WEIGHTS = None

# Disable TF32 for more consistent FP32 computation.
if DEVICE.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def count_flops(model, inputs):
    try:
        from thop import profile
        with torch.no_grad():
            macs, _ = profile(
                model, inputs=inputs, verbose=False
            )
        return 2.0 * macs / 1e9
    except ImportError:
        print("WARNING: THOP not installed.")
        return None


def make_model(kind):
    if kind == "camera":
        if CAMERA_WEIGHTS:
            wrapper = YOLO(CAMERA_WEIGHTS)
        else:
            wrapper = YOLO("yolo11x.yaml")
        model = wrapper.model
    else:
        model = MultiModalYOLO11(
            nc=NC,
            cam_in_ch=3,
            lidar_in_ch=3,
            fusion_type=kind,
        )

    model = model.to(DEVICE).float().eval()
    return model


def make_inputs(kind):
    cam = torch.randn(
        BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE,
        device=DEVICE, dtype=torch.float32
    )
    if kind == "camera":
        return (cam,)

    lidar = torch.randn(
        BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE,
        device=DEVICE, dtype=torch.float32
    )
    return (cam, lidar)


def measure_memory(model, inputs):
    if DEVICE.type != "cuda":
        return None, None

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start_allocated = torch.cuda.memory_allocated()

    with torch.inference_mode():
        output = model(*inputs)
        torch.cuda.synchronize()

    peak = torch.cuda.max_memory_allocated()
    peak_total_mb = peak / 1e6
    forward_overhead_mb = max(
        0, peak - start_allocated
    ) / 1e6

    del output
    return peak_total_mb, forward_overhead_mb


def measure_latency(model, inputs):
    samples = []

    with torch.inference_mode():
        for _ in range(WARMUP):
            _ = model(*inputs)

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            for _ in range(MEASURE):
                start.record()
                _ = model(*inputs)
                end.record()
                end.synchronize()

                samples.append(start.elapsed_time(end))

        else:
            import time
            for _ in range(MEASURE):
                t0 = time.perf_counter()
                _ = model(*inputs)
                samples.append(
                    (time.perf_counter() - t0) * 1000
                )

    return statistics.median(samples)


METHODS = {
    "camera": "Camera-only (YOLO11x)",
    "gated": "Gated Fusion",
    "adapt_gated": "Adaptive Gated",
    "gated_bilinear": "Gated Bilinear",
    "cross_attn": "Cross-Attention (ours)",
}


def benchmark_method(kind):
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    model = make_model(kind)
    inputs = make_inputs(kind)

    # Same inference mode for every architecture.
    assert not model.training

    params = count_parameters(model)
    gflops = count_flops(model, inputs)

    peak_mb, overhead_mb = measure_memory(model, inputs)

    latencies = [
        measure_latency(model, inputs)
        for _ in range(REPEATS)
    ]

    result = {
        "method": METHODS[kind],
        "params_M": params / 1e6,
        "GFLOPs": gflops,
        "peak_mem_MB": peak_mb,
        "forward_overhead_MB": overhead_mb,
        "latency_median_ms": statistics.median(latencies),
        "latency_mean_ms": statistics.mean(latencies),
        "latency_sd_ms": statistics.stdev(latencies),
    }

    del model, inputs
    gc.collect()

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return result


def main():
    print("\n===== CONSISTENT EFFICIENCY BENCHMARK =====")
    print(f"Device: {DEVICE}")

    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("Execution: PyTorch eager")
    print("Precision: FP32")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Input resolution: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Warmup: {WARMUP}")
    print(f"Measurements: {MEASURE}")
    print(f"Repeats: {REPEATS}")

    results = []

    # Interleave methods to reduce systematic ordering effects.
    order = list(METHODS.keys())
    random.Random(42).shuffle(order)

    for kind in order:
        print(f"\nBenchmarking: {METHODS[kind]}")

        r = benchmark_method(kind)
        results.append(r)

        print(
            f"Params: {r['params_M']:.1f}M | "
            f"GFLOPs: "
            f"{r['GFLOPs']:.1f} "
            if r["GFLOPs"] is not None
            else f"Params: {r['params_M']:.1f}M | "
                 "GFLOPs: N/A"
        )

        print(
            f"Latency: {r['latency_mean_ms']:.2f} "
            f"+/- {r['latency_sd_ms']:.2f} ms"
        )

    # Restore desired display order.
    results.sort(
        key=lambda r: list(METHODS.values()).index(r["method"])
    )

    print("\n===== EFFICIENCY TABLE =====")
    print(
        "| Method | Params | GFLOPs | "
        "Peak Mem (MB) | Latency (ms) |"
    )
    print("|---|---:|---:|---:|---:|")

    for r in results:
        g = (
            f"{r['GFLOPs']:.1f}"
            if r["GFLOPs"] is not None
            else "N/A"
        )
        mem = (
            f"{r['peak_mem_MB']:.0f}"
            if r["peak_mem_MB"] is not None
            else "N/A"
        )

        print(
            f"| {r['method']} | "
            f"{r['params_M']:.1f}M | "
            f"{g} | {mem} | "
            f"{r['latency_mean_ms']:.2f} "
            f"+/- {r['latency_sd_ms']:.2f} |"
        )

    output_path = (
        Path(__file__).resolve().parent
        / "efficiency_results_corrected.csv"
    )

    with open(
        output_path, "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(results[0].keys())
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved: {output_path}")
    print("\nNOTE: THOP FLOPs require custom-operator validation.")
    print("NOTE: Verify equivalent detection-head output paths.")


if __name__ == "__main__":
    main()
