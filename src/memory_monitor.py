"""Peak-RSS sampling for algorithm runs."""
import threading
import time

import psutil


class MemoryMonitor:
    """Monitor memory usage during algorithm execution."""
    
    def __init__(self):
        self.peak_memory_mb = 0
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start memory monitoring in a separate thread."""
        self.peak_memory_mb = 0
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_memory)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop memory monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
            
    def _monitor_memory(self):
        """Monitor memory usage continuously."""
        process = psutil.Process()
        while self.monitoring:
            try:
                memory_info = process.memory_info()
                current_memory_mb = memory_info.rss / 1024 / 1024  # Convert to MB
                self.peak_memory_mb = max(self.peak_memory_mb, current_memory_mb)
                time.sleep(0.1)  # Check every 100ms
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
                
    def get_peak_memory_mb(self):
        """Get peak memory usage in MB."""
        return self.peak_memory_mb
