"""
TimingsLogger (logger de tiempos con estadísticas)
---------------------------------------------------
- Helper para cronometrar etapas del pipeline y volcar métricas a CSV.
- Funcionalidades principales:
  * `start()`: inicia cronometraje de un frame u operación
  * `mark()`: marca un punto temporal (yolo, crop, clf, csv, images, etc.)
  * `write()`: escribe métricas a CSV con timestamps ISO y calcula latencias
  * Estadísticas acumulativas: min, max, avg, percentiles (P50, P95, P99)
  * Reportes periódicos y finales
- Mide tiempos de:
  - Pipeline de detección: yolo, parse, nms, validation, crop, clf, csv, images, total
  - Inicialización: device, optimizations, models, camera, ui
- Diagnosticar latencias en tiempo real para optimización.
- Llamado desde:
  * `model/detection/detection_service.py`: usa `TimingsLogger` para medir tiempos
    de cada etapa del pipeline de detección y clasificación
  * `app.py`: usa `TimingsLogger` para medir tiempos de inicialización
"""
from __future__ import annotations

import os
import csv
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import defaultdict

from core.logging import log_info, log_warning


class TimingsLogger:
    """Helper para cronometrar etapas y volcar CSV de tiempos con estadísticas acumulativas.

    Uso para pipeline de frames:
      t = TimingsLogger(log_dir)
      t.start()
      ...
      t.mark('yolo')
      ...
      t.mark('clf')
      ...
      t.mark('csv')
      ...
      t.mark('images')
      t.write(frame_id)
    
    Uso para operaciones únicas (inicialización):
      t = TimingsLogger(log_dir)
      t.start('init_device')
      # ... código ...
      t.end('init_device')
    """

    def __init__(self, log_dir: str, enable_stats: bool = True, report_interval: int = 100) -> None:
        self.log_dir = log_dir
        self.enable_stats = enable_stats
        self.report_interval = report_interval
        
        # Para medición de frames
        self.t0: float = 0.0
        self.marks: Dict[str, float] = {}
        self.csv_path = os.path.join(self.log_dir, "timings", "timings_log.csv")
        
        # Para estadísticas acumulativas
        self._stats: Dict[str, List[float]] = defaultdict(list)  # {operation: [tiempos_ms]}
        self._counts: Dict[str, int] = defaultdict(int)
        
        # Para operaciones únicas (inicialización)
        self._active_ops: Dict[str, float] = {}  # {operation: start_time}

    def start(self, operation: Optional[str] = None) -> None:
        """Inicia cronometraje de un frame (si operation=None) o de una operación única."""
        if operation is None:
            # Modo frame: resetear marcas
            self.t0 = time.time()
            self.marks.clear()
        else:
            # Modo operación única: guardar tiempo de inicio
            self._active_ops[operation] = time.time()

    def mark(self, name: str) -> None:
        """Marca un punto temporal en el pipeline de frames."""
        self.marks[name] = time.time()

    def end(self, operation: str) -> float:
        """Finaliza medición de una operación única y registra estadísticas.
        
        Returns:
            Tiempo transcurrido en milisegundos
        """
        if operation not in self._active_ops:
            log_warning(f"⚠️ TimingsLogger: end() llamado sin start() para '{operation}'")
            return 0.0
        
        start_time = self._active_ops.pop(operation)
        duration_ms = (time.time() - start_time) * 1000.0
        
        if self.enable_stats:
            self._stats[operation].append(duration_ms)
            self._counts[operation] += 1
            
            # Reporte periódico
            if self._counts[operation] % self.report_interval == 0:
                self._print_stats(operation)
        
        return duration_ms

    def write(self, frame_id: int) -> None:
        try:
            os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
            new_file = not os.path.exists(self.csv_path)
            with open(self.csv_path, "a", newline="") as tf:
                tw = csv.writer(tf)
                if new_file:
                    tw.writerow(["iso_ts","frame_id","yolo_ms","parse_ms","nms_ms","validation_ms","crop_ms","forward_ms","classify_ms","csv_ms","images_ms","total_ms"]) 
                now_iso = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                t_yolo = self.marks.get('yolo', self.t0)
                t_crop = self.marks.get('crop', t_yolo)
                t_clf = self.marks.get('clf', t_crop)
                t_csv = self.marks.get('csv', t_clf)
                t_img = self.marks.get('images', t_csv)
                # Calcular tiempos (soporta nuevas marcas: parse, nms, validation)
                t_parse = self.marks.get('parse', t_yolo)
                t_nms = self.marks.get('nms', t_parse)
                t_validation = self.marks.get('validation', t_nms)
                
                yolo_ms = (t_yolo - self.t0) * 1000.0
                parse_ms = (t_parse - t_yolo) * 1000.0
                nms_ms = (t_nms - t_parse) * 1000.0
                validation_ms = (t_validation - t_nms) * 1000.0
                crop_ms = (t_crop - t_validation) * 1000.0 if t_validation != t_nms else (t_crop - t_nms) * 1000.0
                forward_ms = (t_clf - t_crop) * 1000.0
                classify_ms = crop_ms + forward_ms
                csv_ms = (t_csv - t_clf) * 1000.0
                images_ms = (t_img - t_csv) * 1000.0
                total_ms = (t_img - self.t0) * 1000.0
                
                # Guardar en CSV (columnas extendidas)
                tw.writerow([
                    now_iso, frame_id,
                    f"{yolo_ms:.2f}", f"{parse_ms:.2f}", f"{nms_ms:.2f}", f"{validation_ms:.2f}",
                    f"{crop_ms:.2f}", f"{forward_ms:.2f}", f"{classify_ms:.2f}",
                    f"{csv_ms:.2f}", f"{images_ms:.2f}", f"{total_ms:.2f}"
                ])
                
                # Registrar estadísticas acumulativas
                if self.enable_stats:
                    self._stats['yolo'].append(yolo_ms)
                    self._stats['parse'].append(parse_ms)
                    self._stats['nms'].append(nms_ms)
                    self._stats['validation'].append(validation_ms)
                    self._stats['crop'].append(crop_ms)
                    self._stats['forward'].append(forward_ms)
                    self._stats['classify'].append(classify_ms)
                    self._stats['csv'].append(csv_ms)
                    self._stats['images'].append(images_ms)
                    self._stats['total'].append(total_ms)
                    
                    for op in ['yolo', 'parse', 'nms', 'validation', 'crop', 'forward', 'classify', 'csv', 'images', 'total']:
                        self._counts[op] += 1
                    
                    # Reporte periódico
                    if self._counts['total'] % self.report_interval == 0:
                        self._print_stats('total')
            
            log_info(
                f"[timings] frame={frame_id} yolo={yolo_ms:.1f}ms parse={parse_ms:.1f}ms nms={nms_ms:.1f}ms "
                f"validation={validation_ms:.1f}ms crop={crop_ms:.1f}ms forward={forward_ms:.1f}ms "
                f"classify={classify_ms:.1f}ms csv={csv_ms:.1f}ms images={images_ms:.1f}ms total={total_ms:.1f}ms",
                logger_name="timings",
            )
        except Exception:
            # No interrumpir la app por problemas de logging de tiempos
            pass

    def _print_stats(self, operation: str) -> None:
        """Imprime estadísticas de una operación."""
        if operation not in self._stats or not self._stats[operation]:
            return
        
        times = sorted(self._stats[operation])
        count = len(times)
        avg = sum(times) / count
        min_val = times[0]
        max_val = times[-1]
        p50 = times[count // 2]
        p95 = times[int(count * 0.95)] if count > 0 else 0.0
        p99 = times[int(count * 0.99)] if count > 0 else 0.0
        
        log_info(
            f"[timings_stats] {operation}: count={count} avg={avg:.2f}ms min={min_val:.2f}ms "
            f"max={max_val:.2f}ms p50={p50:.2f}ms p95={p95:.2f}ms p99={p99:.2f}ms",
            logger_name="timings"
        )

    def get_stats(self, operation: str) -> Optional[Dict[str, Any]]:
        """Obtiene estadísticas de una operación."""
        if operation not in self._stats or not self._stats[operation]:
            return None
        
        times = sorted(self._stats[operation])
        count = len(times)
        avg = sum(times) / count
        min_val = times[0]
        max_val = times[-1]
        p50 = times[count // 2]
        p95 = times[int(count * 0.95)] if count > 0 else 0.0
        p99 = times[int(count * 0.99)] if count > 0 else 0.0
        
        return {
            "count": count,
            "avg_ms": avg,
            "min_ms": min_val,
            "max_ms": max_val,
            "p50_ms": p50,
            "p95_ms": p95,
            "p99_ms": p99,
        }

    def print_report(self, sort_by: str = "avg_ms") -> None:
        """Imprime reporte completo de todas las operaciones."""
        if not self._stats:
            log_info("No hay estadísticas de timing disponibles", logger_name="timings")
            return
        
        all_stats = {}
        for op in self._stats.keys():
            stats = self.get_stats(op)
            if stats:
                all_stats[op] = stats
        
        if not all_stats:
            return
        
        # Ordenar
        sorted_ops = sorted(
            all_stats.items(),
            key=lambda x: x[1].get(sort_by, 0),
            reverse=True
        )
        
        log_info("=" * 100, logger_name="timings")
        log_info("REPORTE DE TIMING COMPLETO", logger_name="timings")
        log_info("=" * 100, logger_name="timings")
        log_info(
            f"{'Operación':<20} {'Count':<8} {'Avg (ms)':<12} {'Min (ms)':<12} "
            f"{'Max (ms)':<12} {'P50 (ms)':<12} {'P95 (ms)':<12} {'P99 (ms)':<12}",
            logger_name="timings"
        )
        log_info("-" * 100, logger_name="timings")
        
        for operation, stats in sorted_ops:
            log_info(
                f"{operation:<20} {stats['count']:<8} {stats['avg_ms']:>11.2f} "
                f"{stats['min_ms']:>11.2f} {stats['max_ms']:>11.2f} "
                f"{stats['p50_ms']:>11.2f} {stats['p95_ms']:>11.2f} {stats['p99_ms']:>11.2f}",
                logger_name="timings"
            )
        
        log_info("=" * 100, logger_name="timings")

    def save_report(self, filepath: Optional[str] = None) -> None:
        """Guarda reporte completo a archivo."""
        if filepath is None:
            filepath = os.path.join(self.log_dir, "timings", "timings_report.txt")
        
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            all_stats = {}
            for op in self._stats.keys():
                stats = self.get_stats(op)
                if stats:
                    all_stats[op] = stats
            
            if not all_stats:
                return
            
            with open(filepath, "w") as f:
                f.write("=" * 100 + "\n")
                f.write("REPORTE DE TIMING COMPLETO\n")
                f.write(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 100 + "\n\n")
                
                sorted_ops = sorted(
                    all_stats.items(),
                    key=lambda x: x[1].get("avg_ms", 0),
                    reverse=True
                )
                
                f.write(
                    f"{'Operación':<20} {'Count':<8} {'Avg (ms)':<12} {'Min (ms)':<12} "
                    f"{'Max (ms)':<12} {'P50 (ms)':<12} {'P95 (ms)':<12} {'P99 (ms)':<12}\n"
                )
                f.write("-" * 100 + "\n")
                
                for operation, stats in sorted_ops:
                    f.write(
                        f"{operation:<20} {stats['count']:<8} {stats['avg_ms']:>11.2f} "
                        f"{stats['min_ms']:>11.2f} {stats['max_ms']:>11.2f} "
                        f"{stats['p50_ms']:>11.2f} {stats['p95_ms']:>11.2f} {stats['p99_ms']:>11.2f}\n"
                    )
                
                f.write("=" * 100 + "\n")
            
            log_info(f"✅ Reporte de timing guardado en: {filepath}", logger_name="timings")
        except Exception as e:
            log_warning(f"⚠️ Error guardando reporte de timing: {e}", logger_name="timings")


