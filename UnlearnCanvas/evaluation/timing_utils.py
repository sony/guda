#!/usr/bin/env python3
"""
Detailed timing instrumentation for ELBO computation.
This module provides decorators and context managers for fine-grained profiling.
"""
import time
import functools
from contextlib import contextmanager
from collections import defaultdict
from typing import Dict, List

class TimingStats:
    """Collect and report timing statistics."""
    
    def __init__(self):
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self.counts: Dict[str, int] = defaultdict(int)
        
    def add(self, name: str, duration: float):
        """Add a timing measurement."""
        self.timings[name].append(duration)
        self.counts[name] += 1
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a timing category."""
        if name not in self.timings:
            return {"count": 0, "total": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0}
        
        times = self.timings[name]
        return {
            "count": len(times),
            "total": sum(times),
            "mean": sum(times) / len(times),
            "min": min(times),
            "max": max(times),
        }
    
    def report(self, title: str = "TIMING REPORT"):
        """Print a formatted timing report."""
        print("\n" + "="*80)
        print(title)
        print("="*80)
        
        # Sort by total time
        items = sorted(
            self.timings.items(),
            key=lambda x: sum(x[1]),
            reverse=True
        )
        
        print(f"{'Operation':<40} {'Count':>8} {'Total':>10} {'Mean':>10} {'Min':>10} {'Max':>10}")
        print("-"*80)
        
        for name, times in items:
            stats = self.get_stats(name)
            print(f"{name:<40} {stats['count']:>8} "
                  f"{stats['total']:>10.3f}s {stats['mean']:>10.3f}s "
                  f"{stats['min']:>10.3f}s {stats['max']:>10.3f}s")
        
        print("="*80 + "\n")

# Global timing stats instance
_timing_stats = TimingStats()

def get_timing_stats() -> TimingStats:
    """Get the global timing stats instance."""
    return _timing_stats

def reset_timing_stats():
    """Reset all timing statistics."""
    global _timing_stats
    _timing_stats = TimingStats()

@contextmanager
def time_block(name: str, enabled: bool = True):
    """Context manager for timing a code block."""
    if not enabled:
        yield
        return
    
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        _timing_stats.add(name, duration)

def time_function(name: str = None, enabled: bool = True):
    """Decorator for timing function calls."""
    def decorator(func):
        func_name = name or f"{func.__module__}.{func.__qualname__}"
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not enabled:
                return func(*args, **kwargs)
            
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.perf_counter() - start
                _timing_stats.add(func_name, duration)
        
        return wrapper
    return decorator

def print_timing_report(title: str = "TIMING REPORT"):
    """Print timing report from global stats."""
    _timing_stats.report(title)
