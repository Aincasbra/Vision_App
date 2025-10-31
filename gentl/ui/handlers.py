import time
import threading
import cv2
import os
import builtins

# Handler de acciones de UI. Mantiene la lógica fuera de app.py

# Estado de ventanas auxiliares
_info_open = False
_config_open = False
_info_thread = None
_config_thread = None

def handle_action(app, action):
    """Despacha acciones del panel. 'app' es instancia de App."""
    from core.logging import log_info, log_warning, log_error
    from camera.camera_service import CameraService

    if action == "RUN":
        # Configurar ROI para captura
        try:
            svc = CameraService(app.camera)
            def align(v, m=8):
                return int(v) // m * m
            h_max = int(svc.safe_get('HeightMax', svc.safe_get('Height', 1240)))
            w_max = int(svc.safe_get('WidthMax', svc.safe_get('Width', 1624)))
            y1 = align(h_max * 0.50, 8)
            y2 = align(h_max * 0.9, 8)
            new_h = max(align(y2 - y1, 8), 64)
            new_w = align(w_max, 8)
            svc.set_roi(0, y1, new_w, new_h)
            builtins.offsetY = int(y1)
            builtins.roi = (0, int(y1), int(new_w), int(new_h))
        except Exception:
            pass
        try:
            app.camera.start()
            # Activar YOLO
            try:
                app.yolo_running = True
            except Exception:
                pass
            log_info("🔄 Backend iniciado, acquisition_running = True")
            return {"acquisition_running": True}
        except Exception as e:
            log_error(f"❌ Error iniciando backend: {e}")
            return {}

    if action == "STOP":
        try:
            # Restaurar ROI original antes de parar
            try:
                # Restaurar tamaño completo de la cámara usando servicio
                from camera.camera_service import CameraService
                svc = CameraService(app.camera)
                w_max, h_max = svc.restore_full_frame()
                log_info(f"🔄 ROI restaurado a tamaño completo: {w_max}x{h_max}")
            except Exception as e:
                log_warning(f"⚠️ Error restaurando ROI: {e}")
            
            app.camera.stop()
        except Exception:
            pass
        # Detener grabación si estaba activa
        try:
            if hasattr(app, 'recorder') and app.recorder.active:
                app.recorder.stop()
        except Exception:
            pass
        # Detener YOLO
        try:
            app.yolo_running = False
        except Exception:
            pass
        log_info("⏹️ Backend detenido")
        # limpiar colas si las tienes en contexto
        try:
            with app.context.cap_queue.mutex:
                app.context.cap_queue.queue.clear()
            with app.context.infer_queue.mutex:
                app.context.infer_queue.queue.clear()
        except Exception:
            pass
        return {"acquisition_running": False}

    if action == "EXIT":
        # Parar backend y salir del bucle principal
        try:
            app.camera.stop()
        except Exception:
            pass
        try:
            if hasattr(app, 'recorder') and app.recorder.active:
                app.recorder.stop()
        except Exception:
            pass
        app.running = False
        return {"acquisition_running": False, "exit": True}

    if action == "INFO":
        try:
            _show_info_window(app.camera)
        except Exception as e:
            log_warning(f"⚠️ Error abriendo INFO: {e}")
        return {}

    if action == "CONFIG":
        try:
            _show_config_window(app.camera)
        except Exception as e:
            log_warning(f"⚠️ Error abriendo CONFIG: {e}")
        return {}

    if action == "AWB_ONCE":
        def _awb_once():
            try:
                # Mostrar indicador visual temporal
                try:
                    app.awb_indicator_active = True
                    app.awb_indicator_time = time.time() + 2.0  # 2 segundos
                except Exception:
                    pass
                
                did = (app.camera.set_node_value('BalanceWhiteAuto', 'Once') or app.camera.set_node_value('WhiteBalanceAuto', 'Once'))
                if did:
                    time.sleep(0.6)
                    app.camera.set_node_value('BalanceWhiteAuto', 'Off'); app.camera.set_node_value('WhiteBalanceAuto', 'Off')
                    log_info("✅ AWB Once completado")
                else:
                    log_warning("⚠️ No se encontró nodo AWB (BalanceWhiteAuto/WhiteBalanceAuto)")
            except Exception as e:
                log_error(f"❌ Error en AWB: {e}")
            finally:
                try:
                    app.awb_indicator_active = False
                except Exception:
                    pass
        threading.Thread(target=_awb_once, daemon=True).start()
        return {}

    if action == "AUTO_CAL":
        # Implementación simplificada de Auto Cal estándar: activar autos, esperar y desactivar.
        def _auto_cal():
            try:
                # Mostrar indicador visual temporal
                try:
                    app.auto_cal_indicator_active = True
                    app.auto_cal_indicator_time = time.time() + 3.0  # 3 segundos
                except Exception:
                    pass
                
                app.camera.set_node_value('ExposureAuto', 'Continuous')
                app.camera.set_node_value('GainAuto', 'Continuous')
                app.camera.set_node_value('BalanceWhiteAuto', 'Continuous')
                app.camera.set_node_value('WhiteBalanceAuto', 'Continuous')
                log_info("⏳ AutoCal calibrando 0.5s…")
                time.sleep(0.5)
                app.camera.set_node_value('ExposureAuto', 'Off')
                app.camera.set_node_value('GainAuto', 'Off')
                app.camera.set_node_value('BalanceWhiteAuto', 'Off')
                app.camera.set_node_value('WhiteBalanceAuto', 'Off')
                log_info("✅ AutoCal completado")
            except Exception as e:
                log_warning(f"⚠️ Error en AUTO_CAL: {e}")
            finally:
                try:
                    app.auto_cal_indicator_active = False
                except Exception:
                    pass
        threading.Thread(target=_auto_cal, daemon=True).start()
        return {}

    if action.startswith("GAMMA_"):
        try:
            gamma_val = float(action.split("_")[1])
            app.gamma_actual = gamma_val
            log_info(f"📊 Gamma ajustado a: {gamma_val}")
        except Exception:
            pass
        return {}

    if action.startswith("BAYER_"):
        patron = action.split("_")[1]
        try:
            if patron == "BG":
                cv_code = cv2.COLOR_BayerBG2BGR
            elif patron == "RG":
                cv_code = cv2.COLOR_BayerRG2BGR
            elif patron == "GR":
                cv_code = cv2.COLOR_BayerGR2BGR
            elif patron == "GB":
                cv_code = cv2.COLOR_BayerGB2BGR
            else:
                cv_code = cv2.COLOR_BayerBG2BGR
            try:
                app.camera.bayer_code = cv_code
            except Exception:
                pass
            app.patron_actual = patron
        except Exception:
            pass
        return {}

    if action == "RECORD_60S":
        return {"record_start": 60}

    # Botón no reconocido
    return {}


def _show_info_window(camera):
    """Ventana INFO modal (main thread) para compatibilidad GTK/X en Jetson."""
    import cv2
    import numpy as np
    from core.logging import log_info, log_warning
    from camera.camera_service import CameraService

    global _info_open
    if _info_open:
        return
    _info_open = True

    svc = CameraService(camera)
    def safe_get(cam, name, default=None):
        try:
            return svc.safe_get(name, default)
        except Exception:
            return default

    win = "Info"
    try:
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        font = cv2.FONT_HERSHEY_SIMPLEX
        log_info("ℹ️ Ventana INFO abierta")
        t0 = time.time()
        ever_visible = False
        invisible_count = 0

        img0 = np.full((520, 760, 3), 16, dtype=np.uint8)
        cv2.putText(img0, "Cargando informacion...", (16, 40), font, 0.8, (0,255,255), 2)
        cv2.imshow(win, img0)
        cv2.waitKey(10)

        while True:
            img = np.full((520, 760, 3), 16, dtype=np.uint8)
            lines = ["INFORMACION (solo lectura)"]
            def add_line(label, getter):
                try:
                    val = getter()
                    if val is None:
                        val = 'N/A'
                    lines.append(f"{label}: {val}")
                except Exception as e:
                    lines.append(f"{label}: N/A")
                    try:
                        from core.logging import log_warning as _lw
                        _lw(f"INFO lector {label}: {e}")
                    except Exception:
                        pass

            add_line("Vendor", lambda: safe_get(camera, 'DeviceVendorName', 'N/A'))
            add_line("Model", lambda: safe_get(camera, 'DeviceModelName', 'N/A'))
            add_line("Serial", lambda: safe_get(camera, 'DeviceSerialNumber', 'N/A'))
            add_line("PixelFormat", lambda: safe_get(camera, 'PixelFormat', 'N/A'))
            add_line("TriggerMode", lambda: safe_get(camera, 'TriggerMode', 'N/A'))
            add_line("ExposureMode", lambda: (safe_get(camera, 'ExposureMode', None) or 'Timed' if safe_get(camera,'ExposureTime',None) is not None else 'N/A'))
            add_line("ExposureTime", lambda: safe_get(camera, 'ExposureTime', 'N/A'))
            add_line("Gain", lambda: safe_get(camera, 'Gain', 'N/A'))
            add_line("AcqFrameRate", lambda: safe_get(camera, 'AcquisitionFrameRate', 'N/A'))
            add_line("ExposureAuto", lambda: (safe_get(camera, 'ExposureAuto', None) or 'Off'))
            add_line("GainAuto", lambda: (safe_get(camera, 'GainAuto', None) or 'Off'))

            def roi_read():
                ox = safe_get(camera,'OffsetX', None); oy = safe_get(camera,'OffsetY', None)
                w = safe_get(camera,'Width', None); h = safe_get(camera,'Height', None)
                if ox is None or oy is None or w is None or h is None:
                    try:
                        import builtins
                        rx, ry, rw, rh = getattr(builtins, 'roi', (None,None,None,None))
                        if rx is not None:
                            return f"X:{rx} Y:{ry} W:{rw} H:{rh}"
                    except Exception:
                        pass
                    return 'N/A'
                return f"X:{ox} Y:{oy} W:{w} H:{h}"

            add_line("ROI", roi_read)
            lines.append(f"t={int((time.time()-t0)*1000)}ms")

            y = 36
            cv2.putText(img, lines[0], (16, y), font, 0.9, (0,255,0), 2); y += 28
            for ln in lines[1:]:
                cv2.putText(img, str(ln), (16, y), font, 0.6, (255,255,255), 2); y += 24
            cv2.imshow(win, img)

            k = cv2.waitKey(60) & 0xFF
            if k in (27, ord('q')):
                break
            # Cierre por X robusto: sólo si alguna vez fue visible y se ve invisible 3 veces seguidas
            try:
                vis = cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE)
                if vis >= 1:
                    ever_visible = True
                    invisible_count = 0
                else:
                    invisible_count += 1
                if ever_visible and invisible_count >= 3:
                    break
            except Exception:
                # No cerrar por errores del backend; permitir cierre con ESC
                pass
        cv2.destroyWindow(win)
    except Exception as e:
        log_warning(f"⚠️ Error en INFO: {e}")
    finally:
        _info_open = False


def _show_config_window(camera):
    """CONFIG modal (hilo principal) con sliders y teclas, robusto para GTK/X."""
    import cv2
    import numpy as np
    from core.logging import log_info, log_warning
    from camera.camera_service import CameraService
    global _config_open
    if _config_open:
        return
    _config_open = True
    try:
            svc = CameraService(camera)
            def safe_get(cam, name, default=None):
                try:
                    return svc.safe_get(name, default)
                except Exception:
                    return default

            def safe_set(cam, name, value):
                try:
                    return svc.safe_set(name, value)
                except Exception:
                    return False

            # Setters robustos para nodos con variantes entre cámaras
            def set_exposure_us(cam, exposure_us):
                # Apagar auto y asegurar modo
                safe_set(cam, 'ExposureAuto', 'Off')
                safe_set(cam, 'ExposureMode', 'Timed')
                # Intentar nodos más comunes
                if safe_set(cam, 'ExposureTime', float(exposure_us)):
                    val = safe_get(cam, 'ExposureTime', exposure_us)
                elif safe_set(cam, 'ExposureTimeAbs', float(exposure_us)):
                    val = safe_get(cam, 'ExposureTimeAbs', exposure_us)
                elif safe_set(cam, 'ExposureTimeRaw', int(exposure_us)):
                    val = safe_get(cam, 'ExposureTimeRaw', exposure_us)
                else:
                    val = safe_get(cam, 'ExposureTime', exposure_us)
                # Si no cambia, forzar con stop/start si está disponible
                try:
                    if abs(float(val) - float(exposure_us)) > 1.0:
                        if hasattr(cam, 'stop') and hasattr(cam, 'start'):
                            cam.stop(); time.sleep(0.05)
                            safe_set(cam, 'ExposureAuto', 'Off')
                            safe_set(cam, 'ExposureMode', 'Timed')
                            safe_set(cam, 'ExposureTime', float(exposure_us))
                            cam.start()
                            val = safe_get(cam, 'ExposureTime', exposure_us)
                except Exception:
                    pass
                return val

            def set_fps(cam, fps_val):
                try:
                    safe_set(cam, 'AcquisitionFrameRateEnable', True)
                except Exception:
                    pass
                if not safe_set(cam, 'AcquisitionFrameRate', float(fps_val)):
                    # Algunas cámaras usan periodo (s)
                    period = 1.0/max(0.1, float(fps_val))
                    safe_set(cam, 'AcquisitionFrameRate', float(period))
                val = safe_get(cam, 'AcquisitionFrameRate', fps_val)
                try:
                    if abs(float(val) - float(fps_val)) > 0.5:
                        if hasattr(cam, 'stop') and hasattr(cam, 'start'):
                            cam.stop(); time.sleep(0.05)
                            safe_set(cam, 'AcquisitionFrameRateEnable', True)
                            safe_set(cam, 'AcquisitionFrameRate', float(fps_val))
                            cam.start()
                            val = safe_get(cam, 'AcquisitionFrameRate', fps_val)
                except Exception:
                    pass
                return val

            # Preparación inicial
            safe_set(camera, 'ExposureAuto', 'Off')
            safe_set(camera, 'GainAuto', 'Off')
            safe_set(camera, 'ExposureMode', 'Timed')
            try:
                safe_set(camera, 'AcquisitionFrameRateEnable', True)
            except Exception:
                pass

            # Rangos (intenta leer límites; usa valores razonables si no existen)
            et_min = float(safe_get(camera, 'ExposureTimeMin', 100) or 100)
            et_max = float(safe_get(camera, 'ExposureTimeMax', 20000) or 20000)
            if et_max < et_min:
                et_min, et_max = et_max, et_min
            expo_us = float(safe_get(camera, 'ExposureTime', (et_min+et_max)/2) or (et_min+et_max)/2)

            g_min = float(safe_get(camera, 'GainMin', 0.0) or 0.0)
            g_max = float(safe_get(camera, 'GainMax', 36.0) or 36.0)
            if g_max < g_min:
                g_min, g_max = g_max, g_min
            gain_db = float(safe_get(camera, 'Gain', (g_min+g_max)/2) or (g_min+g_max)/2)

            f_min = float(safe_get(camera, 'AcquisitionFrameRateMin', 1.0) or 1.0)
            f_max = float(safe_get(camera, 'AcquisitionFrameRateMax', 60.0) or 60.0)
            if f_max < f_min:
                f_min, f_max = f_max, f_min
            fps = float(safe_get(camera, 'AcquisitionFrameRate', (f_min+f_max)/2) or (f_min+f_max)/2)

            # Guardar originales para ESC
            orig_expo = expo_us
            orig_gain = gain_db
            orig_fps = fps

            win = "Config"
            # Limpieza defensiva por si quedó colgada
            try:
                cv2.destroyWindow(win)
                cv2.waitKey(1)
            except Exception:
                pass
            cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
            # Trackbars normalizados 0..1000 para mapear a rangos reales
            def to_pos(val, lo, hi):
                try:
                    return int(round(1000.0*(float(val)-lo)/max(1e-6, (hi-lo))))
                except Exception:
                    return 500
            def from_pos(pos, lo, hi):
                return float(lo + (hi-lo)*(pos/1000.0))

            cv2.createTrackbar('Exposure', win, to_pos(expo_us, et_min, et_max), 1000, lambda v: None)
            cv2.createTrackbar('Gain', win, to_pos(gain_db, g_min, g_max), 1000, lambda v: None)
            cv2.createTrackbar('FPS', win, to_pos(fps, f_min, f_max), 1000, lambda v: None)

            font = cv2.FONT_HERSHEY_SIMPLEX
            log_info("⚙️ Ventana CONFIG abierta")
            t0 = time.time()
            accepted = False
            ever_visible = False
            invisible_count = 0
            while True:
                pos_expo = cv2.getTrackbarPos('Exposure', win)
                pos_gain = cv2.getTrackbarPos('Gain', win)
                pos_fps = cv2.getTrackbarPos('FPS', win)

                new_expo = max(et_min, min(et_max, from_pos(pos_expo, et_min, et_max)))
                new_gain = max(g_min, min(g_max, from_pos(pos_gain, g_min, g_max)))
                new_fps = max(f_min, min(f_max, from_pos(pos_fps, f_min, f_max)))

                # Aplicar con garantías (desactivar autos por si el SDK los re-activa)
                read_expo = set_exposure_us(camera, float(new_expo))

                safe_set(camera, 'GainAuto', 'Off')
                # Selector de ganancia (si existe)
                safe_set(camera, 'GainSelector', 'AnalogAll')
                safe_set(camera, 'Gain', float(new_gain))

                read_fps = set_fps(camera, float(new_fps))

                img = np.zeros((420, 760, 3), dtype=np.uint8)
                y = 30
                cv2.putText(img, "CONFIGURACION (editable)", (16, y), font, 0.9, (0,255,0), 2); y += 28
                cv2.putText(img, f"Exposure range(us): {int(et_min)}..{int(et_max)}", (16, y), font, 0.5, (160,160,160), 1); y += 18
                cv2.putText(img, f"PixelFormat: {safe_get(camera,'PixelFormat','N/A')}", (16, y), font, 0.6, (255,255,255), 2); y += 22
                cv2.putText(img, f"TriggerMode: {safe_get(camera,'TriggerMode','N/A')} (T)", (16, y), font, 0.6, (255,255,255), 2); y += 22
                cv2.putText(img, f"ExposureAuto: {safe_get(camera,'ExposureAuto','Off')} (A)", (16, y), font, 0.6, (255,255,255), 2); y += 22
                cv2.putText(img, f"GainAuto: {safe_get(camera,'GainAuto','Off')} (G)", (16, y), font, 0.6, (255,255,255), 2); y += 26
                cv2.putText(img, f"ET(us): {int(read_expo) if read_expo is not None else int(new_expo)}", (16, y), font, 0.6, (200,200,0), 2); y += 22
                cv2.putText(img, f"Gain(dB): {new_gain:.1f}", (16, y), font, 0.6, (200,200,0), 2); y += 22
                try:
                    disp_fps = float(read_fps)
                except Exception:
                    disp_fps = float(new_fps)
                cv2.putText(img, f"FPS: {disp_fps:.1f}", (16, y), font, 0.6, (200,200,0), 2); y += 30
                cv2.putText(img, "ENTER=Guardar  |  ESC=Restaurar y cerrar", (16, 400), font, 0.6, (180,180,180), 1)

                cv2.imshow(win, img)
                k = cv2.waitKey(40) & 0xFF
                if k in (27, ord('q')):
                    accepted = False
                    break
                if k in (13, 10):
                    accepted = True
                    break
                if k in (ord('t'), ord('T')):
                    cur = str(safe_get(camera, 'TriggerMode', 'Off')).lower()
                    safe_set(camera, 'TriggerMode', 'Off' if cur == 'on' else 'On')
                if k in (ord('a'), ord('A')):
                    cur = str(safe_get(camera, 'ExposureAuto', 'Off')).lower()
                    safe_set(camera, 'ExposureAuto', 'Off' if cur in ('continuous','once') else 'Continuous')
                if k in (ord('g'), ord('G')):
                    cur = str(safe_get(camera, 'GainAuto', 'Off')).lower()
                    safe_set(camera, 'GainAuto', 'Off' if cur in ('continuous','once') else 'Continuous')
                # Cierre por X robusto (como INFO): solo tras haber sido visible y 3 lecturas invisibles seguidas
                try:
                    vis = cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE)
                    if vis >= 1:
                        ever_visible = True
                        invisible_count = 0
                    else:
                        invisible_count += 1
                    if ever_visible and invisible_count >= 3:
                        break
                except Exception:
                    pass

            try:
                cv2.destroyWindow(win)
                cv2.waitKey(1)
            except Exception:
                pass

            # Restaurar valores si se canceló (ESC)
            if not accepted:
                try:
                    set_exposure_us(camera, float(orig_expo))
                    safe_set(camera, 'GainAuto', 'Off')
                    safe_set(camera, 'Gain', float(orig_gain))
                    set_fps(camera, float(orig_fps))
                except Exception:
                    pass
    except Exception as e:
        log_warning(f"⚠️ Error en CONFIG: {e}")
    finally:
        _config_open = False