"""Shared helpers for the unifallmonitor trainer + experiments scripts.

Modules:
    labels        — single source of truth for canonical-label normalisation.
    smoothing     — per-window probability smoothing (rolling mean / HMM).
    run_registry  — content-addressable run ids + index.csv append.
    bootstrap     — paired/per-class bootstrap CIs for the writeup.
"""
