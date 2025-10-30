import os
import time
import cv2
import subprocess
import threading
from core.logging import log_info, log_warning, log_error

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
