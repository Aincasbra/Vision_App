"""
Recording/Images manager
------------------------
- Sistema de grabación de video y gestión de imágenes (bad/good).
- Funcionalidades principales:
  * `Recorder`: clase para grabación de video:
    - `start()`: inicia grabación de N segundos, crea archivo MP4 y directorio de frames
    - `save_frame()`: guarda frame individual durante la grabación
    - `finalize()`: finaliza grabación y genera video MP4 desde frames
  * `ImagesManager`: clase para gestión de imágenes:
    - Guarda imágenes "bad" por detección y "good" periódicas
    - Mantiene CSV de trazabilidad (`images.csv`)
    - Archiva imágenes diariamente a carpeta `archive/`
    - Soporte para compresión gzip de imágenes antiguas
- Trazabilidad completa de imágenes y soporte a calidad.
- Llamado desde:
  * `model/detection/detection_service.py`: usa `ImagesManager` para guardar imágenes de detecciones
  * `developer_ui/handlers.py`: usa `Recorder` para grabación de video desde la UI
"""
import os
import time
import cv2
import subprocess
import threading
from core.logging import log_info, log_warning, log_error
from datetime import datetime, timedelta
import gzip
import shutil
import csv
from typing import Optional

class Recorder:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.active = False
        self.end_time = 0.0
        self.frame_count = 0
        self.frames_dir = None
        self.out_path = None
        self.last_log_sec = -1

    def start(self, seconds=60):
        os.makedirs(self.out_dir, exist_ok=True)
        base_ts = time.strftime("%Y%m%d_%H%M%S")
        ms = int((time.time() * 1000) % 1000)
        base_name = f"rec_{base_ts}_{ms:03d}"
        self.out_path = os.path.join(self.out_dir, f"{base_name}.mp4")
        self.frames_dir = os.path.join(self.out_dir, f"frames_{base_name}")
        os.makedirs(self.frames_dir, exist_ok=True)
        self.active = True
        self.end_time = time.time() + seconds
        self.frame_count = 0
        self.last_log_sec = -1
        log_info(f"🎥 Grabando {seconds}s en: {self.out_path}")
        log_info(f"📁 Frames temporales en: {self.frames_dir}")
        return self.out_path, self.frames_dir

    def save_frame(self, bgr):
        if not self.active:
            return False
        now_ts = time.time()
        secs_left = max(0, int(self.end_time - now_ts))
        if secs_left != self.last_log_sec:
            self.last_log_sec = secs_left
            log_info(f"🎬 Grabando... {secs_left}s restantes")
        frame_path = os.path.join(self.frames_dir, f"frame_{self.frame_count:06d}.jpg")
        ok, enc = cv2.imencode('.jpg', bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        if ok:
            enc.tofile(frame_path)
            self.frame_count += 1
            if now_ts >= self.end_time:
                self.active = False
                # Ensamblar video automáticamente al terminar
                self._assemble_video_async()
        return ok

    def draw_recording_overlay(self, img_bgr):
        """Dibuja el indicador de grabación en la imagen."""
        if not self.active:
            return img_bgr
        
        try:
            secs_left = max(0, int(self.end_time - time.time()))
            # Overlay rojo con cuenta atrás
            overlay = img_bgr.copy()
            cv2.rectangle(overlay, (10, 10), (210, 60), (0, 0, 255), -1)
            img_bgr = cv2.addWeighted(overlay, 0.3, img_bgr, 0.7, 0)
            cv2.circle(img_bgr, (35, 35), 8, (0, 0, 255), -1)
            cv2.putText(img_bgr, f"REC {secs_left:02d}s", (55, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        except Exception as e:
            log_warning(f"⚠️ Error dibujando REC: {e}")
        
        return img_bgr

    def stop(self):
        """Detiene grabación y ensambla video parcial."""
        if not self.active:
            return None, None, 0
        self.active = False
        log_info("⏹️ Finalizando grabación parcial...")
        if self.frames_dir and self.out_path and self.frame_count > 0:
            self._assemble_video_async()
        return self.out_path, self.frames_dir, self.frame_count

    def _assemble_video_async(self):
        """Ensambla video en hilo separado."""
        def _assemble():
            try:
                if not self.frames_dir or not self.out_path or self.frame_count == 0:
                    return
                log_info(f"🎬 Ensamblando video en background: {self.out_path}")
               
                cmd = [
                    "ffmpeg", "-y", "-framerate", "30",
                    "-i", os.path.join(self.frames_dir, "frame_%06d.jpg"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    self.out_path
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                if result.returncode == 0:
                    log_info(f"✅ Video creado: {self.out_path}")
                else:
                    log_warning(f"⚠️ FFmpeg falló: {result.stderr}")
            except FileNotFoundError:
                log_warning("⚠️ ffmpeg no encontrado. Dejo JPGs guardados.")
            except subprocess.CalledProcessError as e:
                log_error(f"❌ FFmpeg falló: {e}")
            except Exception as e:
                log_error(f"❌ Error ensamblando video: {e}")
        
        threading.Thread(target=_assemble, daemon=True).start()


class ImagesManager:
    """Gestión de guardado de imágenes (malas y buenas periódicas) + CSV y zip diario.

    - Guarda imágenes "malas" bajo criterio del llamador.
    - Guarda 1 imagen "buena" cada intervalo (ej. 30 min).
    - Registra CSV con metadata por imagen guardada.
    - Empaqueta cada día (cuando cambia la fecha) el directorio del día anterior en .zip.
    """

    def __init__(self, base_dir: str = None, good_interval_minutes: int = 30):
        self.base_dir = base_dir or os.path.join(os.environ.get("LOG_DIR", "/var/log/vision_app"), "images")
        self.good_interval = timedelta(minutes=good_interval_minutes)
        self._last_good_ts: float = 0.0
        self._last_date: str = ""
        self._csv_file = None
        self._csv_writer = None
        os.makedirs(self.base_dir, exist_ok=True)
        self._rollover_csv(datetime.now())

    def _rollover_csv(self, now: datetime):
        day_dir = os.path.join(self.base_dir, now.strftime("%Y-%m-%d"))
        os.makedirs(day_dir, exist_ok=True)
        csv_path = os.path.join(day_dir, "images.csv")
        new_file = not os.path.exists(csv_path)
        if self._csv_file:
            try:
                self._csv_file.flush(); self._csv_file.close()
            except Exception:
                pass
        self._csv_file = open(csv_path, mode="a", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        if new_file:
            self._csv_writer.writerow(["ts","type","path","reason","avg_conf","class","track_id"]) 
        self._last_date = now.strftime("%Y-%m-%d")

    def _maybe_zip_previous_day(self, now: datetime):
        try:
            today = now.strftime("%Y-%m-%d")
            if self._last_date and self._last_date != today:
                prev_dir = os.path.join(self.base_dir, self._last_date)
                if os.path.isdir(prev_dir):
                    zip_path = os.path.join(self.base_dir, f"{self._last_date}.zip")
                    shutil.make_archive(zip_path.replace('.zip',''), 'zip', prev_dir)
                    log_info(f"🗜️ Imagenes archivadas: {zip_path}", logger_name="images")
        except Exception as e:
            log_warning(f"⚠️ Error zipeando imágenes: {e}", logger_name="images")

    def _ensure_day(self, now: datetime):
        if now.strftime("%Y-%m-%d") != self._last_date:
            self._maybe_zip_previous_day(now)
            self._rollover_csv(now)

    def _save_image(self, image_bgr, img_type: str, reason: str = "", avg_conf: float = 0.0, cls: str = "", track_id: int = -1, subfolder: str = None) -> str:
        """
        Guarda una imagen en el directorio de imágenes.
        
        Args:
            image_bgr: Imagen BGR a guardar
            img_type: Tipo de imagen ("good", "bad", etc.)
            reason: Razón de la clasificación
            avg_conf: Confianza promedio
            cls: Clase detectada
            track_id: ID de tracking (legacy)
            subfolder: Subcarpeta opcional dentro del día (ej: "clasificado")
        
        Returns:
            Ruta del archivo guardado o cadena vacía si falla
        """
        now = datetime.now()
        self._ensure_day(now)
        day_dir = os.path.join(self.base_dir, now.strftime("%Y-%m-%d"))
        
        # Si se especifica subcarpeta, crear dentro del día
        if subfolder:
            day_dir = os.path.join(day_dir, subfolder)
            os.makedirs(day_dir, exist_ok=True)
        
        ts = now.strftime("%H%M%S")
        ms = int((time.time()*1000) % 1000)
        fname = f"{img_type}_{ts}_{ms:03d}.jpg"
        fpath = os.path.join(day_dir, fname)
        try:
            ok, enc = cv2.imencode('.jpg', image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if ok:
                enc.tofile(fpath)
                # CSV
                try:
                    self._csv_writer.writerow([
                        now.isoformat(), img_type, fpath, reason, f"{avg_conf:.4f}", cls, track_id
                    ])
                    self._csv_file.flush()
                except Exception:
                    pass
                log_info(f"🖼️ Imagen guardada: {fpath}", logger_name="images")
                return fpath
        except Exception as e:
            log_warning(f"⚠️ Error guardando imagen: {e}", logger_name="images")
        return ""

    def save_bad(self, image_bgr, reason: str = "", avg_conf: float = 0.0, cls: str = "", track_id: int = -1) -> str:
        """Guarda una imagen clasificada como 'bad' (mala/defectuosa).
        
        Args:
            image_bgr: Imagen BGR completa a guardar
            reason: Razón de la clasificación (ej: "clf:malas")
            avg_conf: Confianza promedio del clasificador
            cls: Clase detectada (ej: "malas", "defectuosas")
            track_id: ID de tracking (legacy, no usado)
            
        Returns:
            Ruta del archivo guardado o cadena vacía si falla
        """
        return self._save_image(image_bgr, "bad", reason, avg_conf, cls, track_id)

    def save_good(self, image_bgr, reason: str = "", avg_conf: float = 0.0, cls: str = "", track_id: int = -1) -> str:
        """Guarda una imagen clasificada como 'good' (buena).
        
        Args:
            image_bgr: Imagen BGR completa a guardar
            reason: Razón de la clasificación (ej: "clf:buenas")
            avg_conf: Confianza promedio del clasificador
            cls: Clase detectada (ej: "buenas")
            track_id: ID de tracking (legacy, no usado)
            
        Returns:
            Ruta del archivo guardado o cadena vacía si falla
        """
        return self._save_image(image_bgr, "good", reason, avg_conf, cls, track_id)

    def save_good_periodic(self, image_bgr) -> Optional[str]:
        """Guarda una imagen 'good' periódicamente (cada intervalo configurado).
        
        DEPRECATED: Usar save_good() directamente para guardar todas las buenas procesadas.
        Este método se mantiene para compatibilidad pero ya no se usa en el flujo principal.
        
        Args:
            image_bgr: Imagen BGR completa a guardar
            
        Returns:
            Ruta del archivo guardado o None si no se guardó (por intervalo)
        """
        now = time.time()
        if (now - self._last_good_ts) >= self.good_interval.total_seconds():
            self._last_good_ts = now
            return self._save_image(image_bgr, "good", "periodic")
        return None

    def save_classification_crop(self, crop_bgr, img_type: str, reason: str = "", avg_conf: float = 0.0, cls: str = "") -> str:
        """Guarda un crop usado para clasificación en la subcarpeta "clasificado".
        
        Este método guarda los crops (recortes) de los botes que se usan para clasificar.
        Se guardan en una subcarpeta "clasificado" dentro de la carpeta del día para facilitar
        la verificación y debugging.
        
        Args:
            crop_bgr: Crop BGR del bote usado para clasificación
            img_type: Tipo de imagen ("good" o "bad")
            reason: Razón de la clasificación (ej: "crop_clf:buenas")
            avg_conf: Confianza promedio del clasificador
            cls: Clase detectada (ej: "buenas", "malas")
            
        Returns:
            Ruta del archivo guardado o cadena vacía si falla
        """
        return self._save_image(crop_bgr, img_type, reason, avg_conf, cls, track_id=-1, subfolder="clasificado")
