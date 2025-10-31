"""
Profiling helpers
-----------------
- Utilidades de profiling puntuales.
- Medir bloques específicos durante desarrollo.
"""
from time import perf_counter


class PerformanceProfiler:
    """Sistema de profiling para medir tiempos de cada componente"""
    def __init__(self, name="Profiler", enable=True):
        self.name = name
        self.enable = enable
        self.times = {}
        self.start_time = None
        self.current_operation = None
        self.frame_count = 0
        self.avg_times = {}
        self.max_times = {}
        self.min_times = {}

    def start(self, operation):
        if not self.enable:
            return
        self.current_operation = operation
        self.start_time = perf_counter()

    def mark(self, operation):
        if not self.enable or self.start_time is None:
            return
        if self.current_operation:
            elapsed = perf_counter() - self.start_time
            if self.current_operation not in self.times:
                self.times[self.current_operation] = []
            self.times[self.current_operation].append(elapsed)
            if self.current_operation not in self.avg_times:
                self.avg_times[self.current_operation] = 0
                self.max_times[self.current_operation] = 0
                self.min_times[self.current_operation] = float('inf')
            self.avg_times[self.current_operation] = 0.9 * self.avg_times[self.current_operation] + 0.1 * elapsed
            self.max_times[self.current_operation] = max(self.max_times[self.current_operation], elapsed)
            self.min_times[self.current_operation] = min(self.min_times[self.current_operation], elapsed)
        self.current_operation = operation
        self.start_time = perf_counter()

    def end(self):
        if not self.enable or self.start_time is None or not self.current_operation:
            return
        elapsed = perf_counter() - self.start_time
        if self.current_operation not in self.times:
            self.times[self.current_operation] = []
        self.times[self.current_operation].append(elapsed)
        if self.current_operation not in self.avg_times:
            self.avg_times[self.current_operation] = 0
            self.max_times[self.current_operation] = 0
            self.min_times[self.current_operation] = float('inf')
        self.avg_times[self.current_operation] = 0.9 * self.avg_times[self.current_operation] + 0.1 * elapsed
        self.max_times[self.current_operation] = max(self.max_times[self.current_operation], elapsed)
        self.min_times[self.current_operation] = min(self.min_times[self.current_operation], elapsed)
        self.current_operation = None
        self.start_time = None

    def get_report(self, show_details=True):
        if not self.enable or not self.avg_times:
            return "Profiling deshabilitado o sin datos"
        report = f"\n=== {self.name} - REPORTE DE RENDIMIENTO ===\n"
        sorted_ops = sorted(self.avg_times.items(), key=lambda x: x[1], reverse=True)
        total_time = sum(self.avg_times.values())
        for operation, avg_time in sorted_ops:
            percentage = (avg_time / total_time) * 100 if total_time > 0 else 0
            max_time = self.max_times.get(operation, 0)
            min_time = self.min_times.get(operation, float('inf'))
            min_time = min_time if min_time != float('inf') else 0
            report += f"{operation:20s}: {avg_time*1000:6.2f}ms avg | {max_time*1000:6.2f}ms max | {min_time*1000:6.2f}ms min | {percentage:5.1f}%\n"
        report += f"{'TOTAL':20s}: {total_time*1000:6.2f}ms\n"
        if show_details and self.frame_count > 0:
            fps = self.frame_count / total_time if total_time > 0 else 0
            report += f"FPS estimado: {fps:.1f}\n"
        return report

    def reset(self):
        self.times.clear()
        self.avg_times.clear()
        self.max_times.clear()
        self.min_times.clear()
        self.frame_count = 0


