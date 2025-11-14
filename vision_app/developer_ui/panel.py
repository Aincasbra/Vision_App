"""
Panel de control (UI desarrollador)
-----------------------------------
- Composición del panel lateral y detección de clics (zonas RUN/STOP/CONFIG/INFO, etc.).
- Es usado por `developer_ui/window.py` para componer la interfaz y por
  `developer_ui/app_controller.py` para mapear clics a acciones.
"""
import cv2
import numpy as np


def crear_panel_control_stviewer(img_width, img_height):
    """Crea el panel de control estilo StViewer (copiado del original)"""
    # Asegurar altura mínima para dibujar todos los controles (incluyendo clasificador + tracks activos)
    # El panel SIEMPRE tiene 350px de ancho (fijo)
    PANEL_WIDTH = 350
    min_h = 900
    ph = max(int(img_height), min_h)
    panel = np.full((ph, PANEL_WIDTH, 3), (40, 40, 40), dtype=np.uint8)
    
    # Título con gradiente
    cv2.rectangle(panel, (0, 0), (350, 60), (64, 64, 64), -1)
    cv2.putText(panel, "YOLO + GenTL", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Estado de la cámara
    y_offset = 80
    cv2.rectangle(panel, (20, y_offset+10), (330, y_offset+40), (64, 64, 64), -1)
    cv2.rectangle(panel, (20, y_offset+10), (330, y_offset+40), (255, 0, 0), 2)
    cv2.putText(panel, "DETENIDA", (30, y_offset+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # Botones RUN/STOP
    y_offset = 150
    cv2.rectangle(panel, (20, y_offset+10), (160, y_offset+40), (0, 150, 0), -1)
    cv2.rectangle(panel, (20, y_offset+10), (160, y_offset+40), (255, 255, 255), 2)
    cv2.putText(panel, "RUN", (60, y_offset+28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.rectangle(panel, (170, y_offset+10), (310, y_offset+40), (0, 0, 150), -1)
    cv2.rectangle(panel, (170, y_offset+10), (310, y_offset+40), (255, 255, 255), 2)
    cv2.putText(panel, "STOP", (200, y_offset+28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Botón GRABAR (debajo de RUN/STOP)
    y_offset = 190
    cv2.rectangle(panel, (20, y_offset+10), (310, y_offset+40), (0, 0, 255), -1)
    cv2.rectangle(panel, (20, y_offset+10), (310, y_offset+40), (255, 255, 255), 2)
    cv2.putText(panel, "GRABAR 60s", (90, y_offset+28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Mini-checklist de configuración optimizada
    y_offset = 200
    cv2.putText(panel, "CONFIG OPTIMIZADA:", (20, y_offset+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    cv2.putText(panel, "12 FPS | 5ms Expo | 24dB Gain", (20, y_offset+20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    cv2.putText(panel, "Conf: 0.30 | IOU: 0.45 | Track: 90", (20, y_offset+35), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    
    # Botones AWB/AUTO CAL
    y_offset = 220
    cv2.rectangle(panel, (20, y_offset+10), (160, y_offset+40), (100, 100, 0), -1)
    cv2.rectangle(panel, (20, y_offset+10), (160, y_offset+40), (255, 255, 255), 2)
    cv2.putText(panel, "AWB ONCE", (30, y_offset+28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    cv2.rectangle(panel, (170, y_offset+10), (310, y_offset+40), (0, 100, 100), -1)
    cv2.rectangle(panel, (170, y_offset+10), (310, y_offset+40), (255, 255, 255), 2)
    cv2.putText(panel, "AUTO CAL", (180, y_offset+28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Slider Gamma
    y_offset = 280
    cv2.putText(panel, "Gamma:", (20, y_offset+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.rectangle(panel, (20, y_offset+10), (310, y_offset+35), (64, 64, 64), -1)
    cv2.rectangle(panel, (20, y_offset+10), (310, y_offset+35), (128, 128, 128), 2)
    
    # Patrones Bayer
    y_offset = 340
    cv2.putText(panel, "Bayer:", (20, y_offset+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    patrones = [("BG", 20), ("RG", 90), ("GR", 160), ("GB", 230)]
    for patron, x in patrones:
        cv2.rectangle(panel, (x, y_offset+10), (x+60, y_offset+35), (80, 80, 80), -1)
        cv2.rectangle(panel, (x, y_offset+10), (x+60, y_offset+35), (255, 255, 255), 2)
        cv2.putText(panel, patron, (x+20, y_offset+28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Información de dispositivos (se actualizará en actualizar_panel_control)
    # Mover más abajo para dar más espacio
    y_offset = 400
    cv2.putText(panel, "Devices:", (20, y_offset+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    # Fondo más oscuro para mejor contraste - más alto para dos líneas
    cv2.rectangle(panel, (18, y_offset+18), (332, y_offset+60), (20, 20, 20), -1)
    device_info_pytorch = "PyTorch: ?"
    device_info_opencv = "OpenCV: CPU"
    cv2.putText(panel, device_info_pytorch, (20, y_offset+35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
    cv2.putText(panel, device_info_opencv, (20, y_offset+55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
    
    # Métricas YOLO (mover más abajo)
    y_offset = 470
    cv2.putText(panel, "YOLO Stats:", (20, y_offset+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    # Fondo para textos de métricas YOLO para mejor visibilidad
    cv2.rectangle(panel, (18, y_offset+18), (332, y_offset+100), (25, 25, 25), -1)
    cv2.putText(panel, "FPS: 0.0", (20, y_offset+35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(panel, "Tracks: 0", (20, y_offset+55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(panel, "Detections: 0", (20, y_offset+75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(panel, "Confidence: 0.00", (20, y_offset+95), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Información del Clasificador (solo visualización, ajuste desde CONFIG)
    y_offset = 570
    cv2.putText(panel, "CLASIFICADOR:", (20, y_offset+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
    
    # Barra de confianza para clasificación (valor inicial placeholder, se actualizará desde config)
    # NOTA: El valor real debe venir desde config_model.yaml, este es solo un placeholder visual
    classifier_conf_placeholder = 0.70  # Solo para inicialización visual, se sobrescribe en actualizar_panel_control
    cv2.putText(panel, f"Confianza: {classifier_conf_placeholder:.1f}", (20, y_offset+25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.rectangle(panel, (20, y_offset+30), (310, y_offset+50), (64, 64, 64), -1)
    cv2.rectangle(panel, (20, y_offset+30), (310, y_offset+50), (128, 128, 128), 2)
    # Barra de progreso de confianza
    conf_width = int((classifier_conf_placeholder - 0.5) / 0.5 * 290)  # 0.5-1.0 -> 0-290px
    conf_width = max(0, min(290, conf_width))
    cv2.rectangle(panel, (20, y_offset+30), (20+conf_width, y_offset+50), (255, 165, 0), -1)
    
    # Nota: Ajustar desde CONFIG
    cv2.putText(panel, "Ajustar desde CONFIG", (20, y_offset+60), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
    
    # Estadísticas del clasificador
    y_offset = 650
    cv2.putText(panel, "Clasificadas: 0", (20, y_offset+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(panel, "Buenas: 0", (20, y_offset+25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.putText(panel, "Malas: 0", (20, y_offset+45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    # Botón Config
    y_offset = 710
    cv2.rectangle(panel, (20, y_offset+10), (160, y_offset+40), (100, 100, 100), -1)
    cv2.rectangle(panel, (20, y_offset+10), (160, y_offset+40), (255, 255, 255), 2)
    cv2.putText(panel, "CONFIG", (30, y_offset+28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Botón Info (derecha)
    cv2.rectangle(panel, (170, y_offset+10), (310, y_offset+40), (0, 100, 150), -1)
    cv2.rectangle(panel, (170, y_offset+10), (310, y_offset+40), (255, 255, 255), 2)
    cv2.putText(panel, "INFO", (200, y_offset+28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Botón Salir (derecha)
    y_offset = 780
    cv2.rectangle(panel, (170, y_offset+10), (310, y_offset+40), (100, 0, 0), -1)
    cv2.rectangle(panel, (170, y_offset+10), (310, y_offset+40), (255, 255, 255), 2)
    cv2.putText(panel, "SALIR", (200, y_offset+28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return panel


def actualizar_panel_control(panel, metricas, estado_camara, img_width, img_height, gamma_actual=0.8, patron_actual="BG", yolo_stats=None, cam=None, context=None, yolo_conf_threshold=None, classifier_conf_threshold=None, classifier_stats=None):
    """Actualiza el panel de control con información en tiempo real"""
    # Crear una copia del panel base
    panel_actualizado = panel.copy()
    
    # Actualizar estado de la cámara
    y_offset = 80
    color_estado = (0, 255, 0) if estado_camara else (255, 0, 0)
    texto_estado = "GRABANDO" if estado_camara else "DETENIDA"
    cv2.rectangle(panel_actualizado, (20, y_offset+10), (330, y_offset+40), (64, 64, 64), -1)
    cv2.rectangle(panel_actualizado, (20, y_offset+10), (330, y_offset+40), color_estado, 2)
    cv2.putText(panel_actualizado, texto_estado, (30, y_offset+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_estado, 2)
    
    # Actualizar información de dispositivos (PyTorch y OpenCV) - MEJORADO
    y_offset = 400
    try:
        import torch
        # Detectar dispositivo PyTorch
        if torch.cuda.is_available():
            # Acortar nombre del dispositivo si es muy largo
            device_name = torch.cuda.get_device_name(0)
            if len(device_name) > 15:
                device_name = device_name[:12] + "..."
            pytorch_device = f"CUDA ({device_name})"
            device_color = (0, 255, 0)  # Verde para GPU
        else:
            pytorch_device = "CPU"
            device_color = (255, 255, 0)  # Amarillo para CPU
        
        # Detectar dispositivo OpenCV
        try:
            if hasattr(cv2, 'cuda') and hasattr(cv2.cuda, 'getCudaEnabledDeviceCount'):
                cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
                opencv_device = "CUDA" if cuda_count > 0 else "CPU"
            else:
                opencv_device = "CPU"
        except Exception:
            opencv_device = "CPU"
        
        device_info_pytorch = f"PyTorch: {pytorch_device}"
        device_info_opencv = f"OpenCV: {opencv_device}"
    except Exception:
        device_info_pytorch = "PyTorch: ?"
        device_info_opencv = "OpenCV: CPU"
        device_color = (255, 255, 0)
    
    # Dibujar fondo más oscuro para mejor contraste del texto de dispositivos - más alto para dos líneas
    cv2.rectangle(panel_actualizado, (18, y_offset+18), (332, y_offset+60), (30, 30, 30), -1)
    # Texto en dos líneas para evitar que salga fuera
    cv2.putText(panel_actualizado, device_info_pytorch, (20, y_offset+35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, device_color, 2)
    cv2.putText(panel_actualizado, device_info_opencv, (20, y_offset+55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, device_color, 2)
    
    # Actualizar métricas YOLO (tracks, detecciones, y confianza desde config) - mover más abajo
    # yolo_conf_threshold es OBLIGATORIO (debe venir de config_model.yaml)
    if yolo_conf_threshold is None:
        raise ValueError("yolo_conf_threshold es obligatorio. Debe venir de config_model.yaml")
    yolo_conf = float(yolo_conf_threshold)
    
    y_offset = 470
    # Fondo para textos de métricas YOLO para mejor visibilidad
    cv2.rectangle(panel_actualizado, (18, y_offset+18), (332, y_offset+100), (25, 25, 25), -1)
    
    if yolo_stats:
        cv2.putText(panel_actualizado, f"FPS: {yolo_stats.get('fps', 0.0):.1f}", (20, y_offset+35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(panel_actualizado, f"Tracks: {yolo_stats.get('tracks', 0)}", (20, y_offset+55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(panel_actualizado, f"Detections: {yolo_stats.get('detections', 0)}", (20, y_offset+75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(panel_actualizado, f"Confidence: {yolo_conf:.2f}", (20, y_offset+95), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Mostrar estadísticas de procesamiento adaptativo
        if 'skip_ratio' in yolo_stats:
            skip_pct = yolo_stats['skip_ratio'] * 100
            cv2.putText(panel_actualizado, f"Skip: {skip_pct:.0f}%", (20, y_offset+115), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    else:
        # Mostrar valores por defecto si no hay stats
        cv2.putText(panel_actualizado, "FPS: 0.0", (20, y_offset+35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(panel_actualizado, "Tracks: 0", (20, y_offset+55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(panel_actualizado, "Detections: 0", (20, y_offset+75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(panel_actualizado, f"Confidence: {yolo_conf:.2f}", (20, y_offset+95), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Actualizar confianza del clasificador desde config
    # classifier_conf_threshold es OBLIGATORIO (debe venir de config_model.yaml)
    if classifier_conf_threshold is None:
        raise ValueError("classifier_conf_threshold es obligatorio. Debe venir de config_model.yaml")
    classifier_conf = float(classifier_conf_threshold)
    
    y_offset = 570
    # Fondo para mejor visibilidad del texto
    cv2.rectangle(panel_actualizado, (18, y_offset+20), (332, y_offset+65), (25, 25, 25), -1)
    cv2.putText(panel_actualizado, f"Confianza: {classifier_conf:.1f}", (20, y_offset+25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.rectangle(panel_actualizado, (20, y_offset+30), (310, y_offset+50), (64, 64, 64), -1)
    cv2.rectangle(panel_actualizado, (20, y_offset+30), (310, y_offset+50), (128, 128, 128), 2)
    # Barra de progreso de confianza
    conf_width = int((classifier_conf - 0.5) / 0.5 * 290)  # 0.5-1.0 -> 0-290px
    conf_width = max(0, min(290, conf_width))
    cv2.rectangle(panel_actualizado, (20, y_offset+30), (20+conf_width, y_offset+50), (255, 165, 0), -1)
    cv2.putText(panel_actualizado, "Ajustar desde CONFIG", (20, y_offset+60), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
    
    # Actualizar estadísticas del clasificador (buenas/malas)
    y_offset = 650
    # Fondo para mejor visibilidad
    cv2.rectangle(panel_actualizado, (18, y_offset+0), (332, y_offset+50), (25, 25, 25), -1)
    
    if classifier_stats:
        total = classifier_stats.get('total', 0)
        buenas = classifier_stats.get('buenas', 0)
        malas = classifier_stats.get('malas', 0)
    else:
        # Intentar obtener desde context si está disponible
        total = 0
        buenas = 0
        malas = 0
        if context and hasattr(context, 'classifier_stats'):
            stats = context.classifier_stats
            total = stats.get('total', 0)
            buenas = stats.get('buenas', 0)
            malas = stats.get('malas', 0)
    
    cv2.putText(panel_actualizado, f"Clasificadas: {total}", (20, y_offset+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(panel_actualizado, f"Buenas: {buenas}", (20, y_offset+40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.putText(panel_actualizado, f"Malas: {malas}", (20, y_offset+60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    # Actualizar indicador de gamma
    y_offset = 280
    # Mapear 0.3–1.5 al slider de 20–310
    gamma_pos = int(20 + (gamma_actual - 0.3) * 290 / (1.5 - 0.3))
    cv2.rectangle(panel_actualizado, (20, y_offset+10), (310, y_offset+35), (64, 64, 64), -1)
    cv2.rectangle(panel_actualizado, (20, y_offset+10), (310, y_offset+35), (128, 128, 128), 2)
    cv2.rectangle(panel_actualizado, (gamma_pos-8, y_offset+5), (gamma_pos+8, y_offset+40), (0, 255, 255), -1)
    cv2.putText(panel_actualizado, f"{gamma_actual:.2f}", (gamma_pos-15, y_offset+55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    # Actualizar patrón Bayer seleccionado
    y_offset = 340
    patrones = [("BG", 20), ("RG", 90), ("GR", 160), ("GB", 230)]
    for i, (patron, x) in enumerate(patrones):
        if patron == patron_actual:
            color = (150, 150, 150)  # Seleccionado
            borde = (0, 255, 255)  # Borde cyan
        else:
            color = (80, 80, 80)  # No seleccionado
            borde = (255, 255, 255)  # Borde blanco
        cv2.rectangle(panel_actualizado, (x, y_offset+10), (x+60, y_offset+35), color, -1)
        cv2.rectangle(panel_actualizado, (x, y_offset+10), (x+60, y_offset+35), borde, 2)
        cv2.putText(panel_actualizado, patron, (x+20, y_offset+28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return panel_actualizado


def detectar_clic_panel_control(x, y, panel_offset_x, img_height):
    """Detecta clics en el panel de control lateral usando offset explícito"""
    panel_x_click = x - panel_offset_x
    panel_y_click = y
    
    # Debug: Log de todos los clics en el panel
    print(f" Clic en panel: x={panel_x_click}, y={panel_y_click}")
    
    PANEL_WIDTH = 350
    if panel_x_click < 0 or panel_x_click >= PANEL_WIDTH or panel_y_click < 0 or panel_y_click >= img_height:
        return None
    
    # Botón RUN
    if 20 <= panel_x_click <= 160 and 160 <= panel_y_click <= 195:
        return "RUN"
    # Botón STOP
    elif 170 <= panel_x_click <= 310 and 160 <= panel_y_click <= 195:
        return "STOP"
    # Botón GRABAR 60s (debajo de RUN/STOP)
    elif 20 <= panel_x_click <= 310 and 200 <= panel_y_click <= 240:
        print(" Botón GRABAR 60s detectado!")
        return "RECORD_60S"
    # Botón AWB ONCE
    elif 20 <= panel_x_click <= 160 and 230 <= panel_y_click <= 260:
        return "AWB_ONCE"
    # Botón AUTO CAL
    elif 170 <= panel_x_click <= 310 and 230 <= panel_y_click <= 260:
        return "AUTO_CAL"
    # Slider Gamma - área más amplia para facilitar el clic
    elif 20 <= panel_x_click <= 310 and 290 <= panel_y_click <= 325:
        # Calcular valor de gamma basado en la posición X (rango más amplio 0.3–1.5)
        gamma_value = 0.3 + (panel_x_click - 20) * (1.5 - 0.3) / 290
        # Redondear a 2 decimales para evitar valores muy específicos
        gamma_value = round(gamma_value, 2)
        gamma_value = max(0.3, min(1.5, gamma_value))  # Limitar entre 0.3 y 1.5
        return f"GAMMA_{gamma_value:.2f}"
    # Patrones Bayer - área más amplia para facilitar el clic
    elif 20 <= panel_x_click <= 80 and 350 <= panel_y_click <= 385:
        return "BAYER_BG"
    elif 90 <= panel_x_click <= 150 and 350 <= panel_y_click <= 385:
        return "BAYER_RG"
    elif 160 <= panel_x_click <= 220 and 350 <= panel_y_click <= 385:
        return "BAYER_GR"
    elif 230 <= panel_x_click <= 290 and 350 <= panel_y_click <= 385:
        return "BAYER_GB"
    # Botón CONFIG
    elif 20 <= panel_x_click <= 160 and 710 <= panel_y_click <= 740:
        return "CONFIG"
    # Botón INFO (derecha)
    elif 170 <= panel_x_click <= 310 and 710 <= panel_y_click <= 740:
        return "INFO"
    # Botón SALIR (derecha)
    elif 170 <= panel_x_click <= 310 and 780 <= panel_y_click <= 810:
        return "EXIT"
    
    return None
