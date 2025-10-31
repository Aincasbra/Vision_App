"""
Servicio de inferencia YOLO desacoplado del bucle principal.

Objetivo: encapsular el hilo de inferencia y el manejo de colas sin
modificar la lógica existente. En una segunda fase, `app.py` usará este
servicio en lugar de su método interno.
"""
from __future__ import annotations

import time
import threading
import queue
"""
YoloService
-----------
- Hilo de inferencia YOLO + clasificación por lata.
- Publica resultados, registra CSV (`vision_log.csv`) y gestiona guardado
  de imágenes (bad/good) vía `core/recording.py`.
- Emite "heartbeat" y logs de política de clasificación.
"""

from typing import Optional, Any, List
import os
import csv
from datetime import datetime

import numpy as np

from core.logging import log_info, log_warning
from core.timings import TimingsLogger
from core.recording import ImagesManager
from vision.classifier import clf_predict_bgr
from vision.ops import merge_overlapping_detections
from vision.tracking import assign_stable_ids


class YoloService:
    """Gestiona el hilo de inferencia YOLO y publica resultados en una cola.

    No crea el modelo; recibe `yolo_model` ya cargado por el llamador.
    """

    def __init__(
        self,
        yolo_model: Any,
        infer_queue: "queue.Queue",
        conf_threshold: float = 0.4,
        process_every: int = 1,
        camera: Optional[Any] = None,
    ) -> None:
        self.yolo_model = yolo_model
        self.infer_queue = infer_queue
        self.conf_threshold = float(conf_threshold)
        self.process_every = int(process_every)
        self.camera = camera

        self._thread: Optional[threading.Thread] = None
        self._running: bool = False
        self._frame_count: int = 0
        # Heartbeat para logs en headless
        self._hb_start_ts: float = time.time()
        self._hb_last_ts: float = self._hb_start_ts
        self._hb_frames: int = 0
        self._hb_last_boxes: int = 0
        # CSV visión
        self._vision_csv_enabled: bool = str(os.environ.get("LOG_VISION_CSV", "1")).lower() in ("1", "true", "yes", "on")
        self._log_dir: str = os.environ.get("LOG_DIR", "/var/log/calippo")
        self._vision_csv_path: str = os.path.join(self._log_dir, "vision", "vision_log.csv")
        self._csv_writer = None
        self._csv_file = None
        # Imágenes
        self._images_enabled: bool = str(os.environ.get("LOG_IMAGES", "1")).lower() in ("1", "true", "yes", "on")
        self._images_mgr: Optional[ImagesManager] = None
        # Política de clasificación: prioriza buenas; sólo marca "bad" si >= umbral
        try:
            self._clf_bad_threshold: float = float(os.environ.get("CLF_BAD_THRESHOLD", 0.87))
        except Exception:
            self._clf_bad_threshold = 0.87
        self._bad_labels = {"malas", "defectuosas"}

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        log_info("🧵 YoloService iniciado")

    def stop(self) -> None:
        self._running = False
        t = self._thread
        if t is not None:
            t.join(timeout=0.5)
        self._thread = None
        try:
            if self._csv_file is not None:
                self._csv_file.flush()
                self._csv_file.close()
        except Exception:
            pass
        log_info("🧵 YoloService detenido")

    def _run(self) -> None:
        import builtins
        # Preparar CSV e imágenes si procede
        if self._vision_csv_enabled:
            try:
                os.makedirs(os.path.dirname(self._vision_csv_path), exist_ok=True)
                new_file = not os.path.exists(self._vision_csv_path)
                self._csv_file = open(self._vision_csv_path, mode="a", newline="")
                self._csv_writer = csv.writer(self._csv_file)
                if new_file:
                    self._csv_writer.writerow([
                        "ts","iso_ts","frame_id","num_boxes","classes","avg_conf","proc_ms",
                        "camera_exposure","camera_gain","width","height","yolo_threshold","bbox",
                        "verdict","clf_label","clf_conf","policy"
                    ])
            except Exception as e:
                log_warning(f"⚠️ No se pudo abrir vision CSV: {e}")
                self._csv_writer = None
                self._csv_file = None
        if self._images_enabled:
            try:
                self._images_mgr = ImagesManager()
            except Exception as e:
                log_warning(f"⚠️ No se pudo iniciar ImagesManager: {e}", logger_name="images")
        # Timings logger
        self._tlogger = TimingsLogger(self._log_dir)
        while self._running:
            try:
                if not hasattr(builtins, "latest_frame") or builtins.latest_frame is None:
                    time.sleep(0.01)
                    continue

                img_bgr = builtins.latest_frame
                frame_id = getattr(builtins, "latest_fid", 0)

                if self._frame_count % self.process_every != 0:
                    self._frame_count += 1
                    continue

                if self.yolo_model is None:
                    time.sleep(0.02)
                    continue

                t0 = time.time(); self._tlogger.start()
                try:
                    results = self.yolo_model.predict(img_bgr, conf_threshold=self.conf_threshold)
                    t_yolo_end = time.time(); self._tlogger.mark('yolo')

                    xyxy = np.array([])
                    confs = np.array([])
                    clss = np.array([])

                    if results and len(results) > 0:
                        raw_boxes, raw_confs, raw_clss = [], [], []
                        if isinstance(results[0], dict):
                            for det in results:
                                bbox = det.get("bbox") or det.get("xyxy") or det.get("box")
                                if not bbox or len(bbox) < 4:
                                    continue
                                raw_boxes.append([float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])])
                                raw_confs.append(float(det.get("confidence", 0.0)))
                                raw_clss.append(int(det.get("class_id", -1)))
                            raw_boxes = np.asarray(raw_boxes, dtype=np.float32)
                            raw_confs = np.asarray(raw_confs, dtype=np.float32)
                            raw_clss = np.asarray(raw_clss, dtype=np.int32)
                        else:
                            result = results[0]
                            if hasattr(result, "boxes") and result.boxes is not None:
                                raw_boxes = result.boxes.xyxy.cpu().numpy()
                                raw_confs = result.boxes.conf.cpu().numpy()
                                try:
                                    raw_clss = result.boxes.cls.cpu().numpy()
                                except Exception:
                                    raw_clss = np.full((len(raw_boxes),), -1, dtype=np.int32)
                            else:
                                raw_boxes = np.empty((0, 4), dtype=np.float32)
                                raw_confs = np.empty((0,), dtype=np.float32)
                                raw_clss = np.empty((0,), dtype=np.int32)

                        allowed_classes = {0, 1, 2}
                        mask = np.isin(raw_clss, list(allowed_classes))
                        xyxy = raw_boxes[mask]
                        confs = raw_confs[mask]
                        clss = raw_clss[mask]

                    if len(xyxy) > 0:
                        xyxy, confs, clss = merge_overlapping_detections(xyxy, confs, clss, iou_threshold=0.7)
                        if len(xyxy) > 0:
                            tids = assign_stable_ids(xyxy, confs, clss, self._frame_count)
                            infer_result = {
                                "frame_id": frame_id,
                                "xyxy": xyxy.tolist(),
                                "tids": tids,
                                "lids": tids,
                                "proc_ms": (time.time() - t0) * 1000.0,
                                "ts": time.time(),
                            }
                            try:
                                self.infer_queue.put_nowait(infer_result)
                            except queue.Full:
                                pass
                            # Log visión por evento
                            try:
                                classes_list: List[int] = list(map(int, clss.tolist())) if hasattr(clss, 'tolist') else []
                                avg_conf = float(np.mean(confs)) if len(confs) else 0.0
                                # Clasificación (tomamos la primera caja)
                                clf_label, clf_conf = "unknown", 0.0
                                try:
                                    if len(xyxy) > 0 and hasattr(builtins, "latest_frame") and builtins.latest_frame is not None:
                                        x1, y1, x2, y2 = [int(v) for v in xyxy[0].tolist()]
                                        crop = builtins.latest_frame[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
                                        if crop.size > 0:
                                            self._tlogger.mark('crop')
                                            clf_label, clf_conf, _ = clf_predict_bgr(crop)
                                    t_clf_end = time.time(); self._tlogger.mark('clf')
                                except Exception as e:
                                    log_warning(f"⚠️ Error en clasificación: {e}")
                                    t_clf_end = time.time(); self._tlogger.mark('clf')
                                # Veredicto: ok/bad (prioriza buenas)
                                verdict = "bad" if (clf_label in self._bad_labels and clf_conf >= self._clf_bad_threshold) else "ok"
                                policy_str = f"clasificador={clf_label} conf={clf_conf:.2f} < umbral={self._clf_bad_threshold:.2f} => buena" if verdict=="ok" else f"clasificador={clf_label} conf={clf_conf:.2f} >= umbral={self._clf_bad_threshold:.2f} => mala"
                                log_info(f"[VISION] frame={frame_id} boxes={len(xyxy)} classes={classes_list} avg_conf={avg_conf:.2f} {policy_str}", logger_name="vision")
                                if self._csv_writer is not None:
                                    # Obtener parámetros de cámara y dimensiones actuales
                                    try:
                                        from camera.camera_service import CameraService
                                        cam_svc = CameraService(self.camera) if self.camera is not None else None
                                        cam_exp = cam_svc.safe_get('ExposureTime', '') if cam_svc else ''
                                        cam_gain = cam_svc.safe_get('Gain', '') if cam_svc else ''
                                    except Exception:
                                        cam_exp, cam_gain = '', ''
                                    try:
                                        import builtins
                                        w = getattr(builtins, 'current_img_w', '')
                                        h = getattr(builtins, 'current_img_h', '')
                                    except Exception:
                                        w, h = '', ''
                                    try:
                                        first_bbox = xyxy[0].tolist() if len(xyxy) > 0 else []
                                    except Exception:
                                        first_bbox = []
                                    iso_ts = datetime.fromtimestamp(infer_result['ts']).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                                    self._csv_writer.writerow([
                                        f"{infer_result['ts']:.6f}",
                                        iso_ts,
                                        frame_id,
                                        len(xyxy),
                                        "|".join(map(str, classes_list)),
                                        f"{avg_conf:.4f}",
                                        f"{infer_result['proc_ms']:.2f}",
                                        cam_exp,
                                        cam_gain,
                                        w,
                                        h,
                                        f"{self.conf_threshold:.2f}",
                                        str(first_bbox),
                                        verdict,
                                        clf_label,
                                        f"{clf_conf:.4f}",
                                        policy_str,
                                    ])
                                    try:
                                        self._csv_file.flush()
                                    except Exception:
                                        pass
                                    t_csv_end = time.time(); self._tlogger.mark('csv')
                                # Guardado de imágenes: malas si hay boxes, buena periódica
                                if self._images_mgr is not None:
                                    try:
                                        if len(xyxy) > 0 and hasattr(builtins, "latest_frame") and builtins.latest_frame is not None:
                                            if verdict == "bad":
                                                self._images_mgr.save_bad(builtins.latest_frame, reason=f"clf:{clf_label}", avg_conf=clf_conf, cls=clf_label)
                                        if hasattr(builtins, "latest_frame") and builtins.latest_frame is not None:
                                            self._images_mgr.save_good_periodic(builtins.latest_frame)
                                    except Exception as e:
                                        log_warning(f"⚠️ Error guardando imagen: {e}", logger_name="images")
                                    finally:
                                        self._tlogger.mark('images')
                                        self._tlogger.write(frame_id)
                            except Exception as e:
                                log_warning(f"⚠️ Error registrando visión: {e}")
                except Exception as e:
                    log_warning(f"⚠️ Error en YoloService.predict: {e}")

                self._frame_count += 1
                # Heartbeat: cada ~1s, loguea fps y nº de boxes
                self._hb_frames += 1
                self._hb_last_boxes = int(len(xyxy)) if isinstance(xyxy, np.ndarray) else 0
                now = time.time()
                if (now - self._hb_last_ts) >= 1.0:
                    elapsed = max(1e-6, now - self._hb_last_ts)
                    fps = self._hb_frames / elapsed
                    log_info(f"[YOLO] fps={fps:.1f} boxes={self._hb_last_boxes}")
                    self._hb_last_ts = now
                    self._hb_frames = 0
            except Exception as e:
                log_warning(f"⚠️ Error en hilo YoloService: {e}")
                time.sleep(0.01)


