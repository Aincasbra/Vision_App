"""
TimingsLogger (logger de tiempos)
----------------------------------
- Helper para cronometrar etapas del pipeline y volcar métricas a CSV.
- Funcionalidades principales:
  * `start()`: inicia cronometraje de un frame
  * `mark()`: marca un punto temporal (yolo, crop, clf, csv, images, etc.)
  * `write()`: escribe métricas a CSV con timestamps ISO y calcula latencias:
    - yolo_ms: tiempo de inferencia YOLO
    - crop_ms: tiempo de recorte de ROI
    - forward_ms: tiempo de forward del clasificador
    - classify_ms: tiempo total de clasificación (crop + forward)
    - csv_ms: tiempo de escritura a CSV
    - images_ms: tiempo de guardado de imágenes
    - total_ms: tiempo total del pipeline
- Diagnosticar latencias en tiempo real para optimización.
- Llamado desde:
  * `model/detection/detection_service.py`: usa `TimingsLogger` para medir tiempos
    de cada etapa del pipeline de detección y clasificación
"""
from __future__ import annotations

import os
import csv
import time
from datetime import datetime
from typing import Dict

from core.logging import log_info


class TimingsLogger:
    """Pequeño helper para cronometrar etapas y volcar CSV de tiempos.

    Uso:
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
    """

    def __init__(self, log_dir: str) -> None:
        self.log_dir = log_dir
        self.t0: float = 0.0
        self.marks: Dict[str, float] = {}
        self.csv_path = os.path.join(self.log_dir, "timings", "timings_log.csv")

    def start(self) -> None:
        self.t0 = time.time()
        self.marks.clear()

    def mark(self, name: str) -> None:
        self.marks[name] = time.time()

    def write(self, frame_id: int) -> None:
        try:
            os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
            new_file = not os.path.exists(self.csv_path)
            with open(self.csv_path, "a", newline="") as tf:
                tw = csv.writer(tf)
                if new_file:
                    tw.writerow(["iso_ts","frame_id","yolo_ms","crop_ms","forward_ms","classify_ms","csv_ms","images_ms","total_ms"]) 
                now_iso = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                t_yolo = self.marks.get('yolo', self.t0)
                t_crop = self.marks.get('crop', t_yolo)
                t_clf = self.marks.get('clf', t_crop)
                t_csv = self.marks.get('csv', t_clf)
                t_img = self.marks.get('images', t_csv)
                yolo_ms = (t_yolo - self.t0) * 1000.0
                crop_ms = (t_crop - t_yolo) * 1000.0
                forward_ms = (t_clf - t_crop) * 1000.0
                classify_ms = crop_ms + forward_ms
                csv_ms = (t_csv - t_clf) * 1000.0
                images_ms = (t_img - t_csv) * 1000.0
                total_ms = (t_img - self.t0) * 1000.0
                tw.writerow([
                    now_iso, frame_id,
                    f"{yolo_ms:.2f}", f"{crop_ms:.2f}", f"{forward_ms:.2f}", f"{classify_ms:.2f}", f"{csv_ms:.2f}", f"{images_ms:.2f}", f"{total_ms:.2f}"
                ])
            log_info(
                f"[timings] frame={frame_id} yolo={yolo_ms:.1f}ms crop={crop_ms:.1f}ms forward={forward_ms:.1f}ms classify={classify_ms:.1f}ms csv={csv_ms:.1f}ms images={images_ms:.1f}ms total={total_ms:.1f}ms",
                logger_name="timings",
            )
        except Exception:
            # No interrumpir la app por problemas de logging de tiempos
            pass


