import time
from functools import wraps
from contextlib import contextmanager
from collections import defaultdict
import functools


# TIME MEASURMENTS
class TimingProfiler:
    """Simple timing profiler to track bottlenecks"""
    
    def __init__(self):
        self.timings = defaultdict(list)
        self.start_times = {}
        
    @contextmanager
    def timer(self, name):
        """Context manager for timing code blocks"""
        start_time = time.time()
        print(f"[⏱️] Starting: {name}")
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            self.timings[name].append(elapsed)
            print(f"[⏱️] Completed: {name} in {elapsed:.2f}s")
    
    def start_timer(self, name):
        """Start a named timer"""
        self.start_times[name] = time.time()
        print(f"[⏱️] Started: {name}")
    
    def end_timer(self, name):
        """End a named timer"""
        if name in self.start_times:
            elapsed = time.time() - self.start_times[name]
            self.timings[name].append(elapsed)
            print(f"[⏱️] Ended: {name} in {elapsed:.2f}s")
            del self.start_times[name]
            return elapsed
        else:
            print(f"[⚠️] Timer {name} was not started")
            return 0
    
    def get_summary(self):
        """Get timing summary"""
        print("\n" + "="*60)
        print("🕐 TIMING SUMMARY")
        print("="*60)
        
        total_time = 0
        for name, times in self.timings.items():
            avg_time = sum(times) / len(times)
            total_time_for_task = sum(times)
            total_time += total_time_for_task
            
            print(f"{name:<30}: {avg_time:>8.2f}s avg ({total_time_for_task:>8.2f}s total, {len(times)} calls)")
        
        print("-" * 60)
        print(f"{'TOTAL WORKFLOW TIME':<30}: {total_time:>8.2f}s ({total_time/60:.1f} min)")
        print("="*60)
        
        # Find bottlenecks
        bottlenecks = sorted([(sum(times), name) for name, times in self.timings.items()], reverse=True)
        print("\n🔍 TOP BOTTLENECKS:")
        for total_time, name in bottlenecks[:5]:
            percentage = (total_time / sum(sum(times) for times in self.timings.values())) * 100
            print(f"  {name:<30}: {total_time:>8.2f}s ({percentage:>5.1f}%)")

# Global profiler instance
profiler = TimingProfiler()

# Decorator for timing functions
def time_it(name=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = name or f"{func.__module__}.{func.__qualname__}"
            with profiler.timer(func_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator

# Async version
def time_it_async(name=None):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            func_name = name or f"{func.__module__}.{func.__qualname__}"
            with profiler.timer(func_name):
                return await func(*args, **kwargs)
        return wrapper
    return decorator