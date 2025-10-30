import cv2
import numpy as np


def crear_panel_control_stviewer(img_width, img_height):
    """Crea el panel de control estilo StViewer (copiado del original)"""
    # Asegurar altura mínima para dibujar todos los controles (incluyendo clasificador + tracks activos)
    min_h = 900
    ph = max(int(img_height), min_h)
    panel = np.full((ph, 350, 3), (40, 40, 40), dtype=np.uint8)
    
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
    
    # Métricas YOLO
    y_offset = 400
    cv2.putText(panel, "YOLO Stats:", (20, y_offset+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    # Mostrar información de dispositivos
    device_info = "PyTorch: CPU | OpenCV: CPU"
    cv2.putText(panel, f"Devices: {device_info}", (20, y_offset+25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.putText(panel, "FPS: 0.0", (20, y_offset+45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(panel, "Tracks: 0", (20, y_offset+65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(panel, "Detections: 0", (20, y_offset+85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Controles del Clasificador
    y_offset = 500
    cv2.putText(panel, "CLASIFICADOR:", (20, y_offset+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
    
    # Barra de confianza para clasificación
    cv2.putText(panel, f"Confianza: {0.7:.1f}", (20, y_offset+25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.rectangle(panel, (20, y_offset+30), (310, y_offset+50), (64, 64, 64), -1)
    cv2.rectangle(panel, (20, y_offset+30), (310, y_offset+50), (128, 128, 128), 2)
    # Barra de progreso de confianza
    conf_width = int((0.7 - 0.5) / 0.5 * 290)  # 0.5-1.0 -> 0-290px
    conf_width = max(0, min(290, conf_width))
    cv2.rectangle(panel, (20, y_offset+30), (20+conf_width, y_offset+50), (255, 165, 0), -1)
    
    # Botones de ajuste de confianza
    cv2.rectangle(panel, (20, y_offset+55), (80, y_offset+75), (100, 100, 100), -1)
    cv2.rectangle(panel, (20, y_offset+55), (80, y_offset+75), (255, 255, 255), 2)
    cv2.putText(panel, "-0.1", (25, y_offset+70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    cv2.rectangle(panel, (90, y_offset+55), (150, y_offset+75), (100, 100, 100), -1)
    cv2.rectangle(panel, (90, y_offset+55), (150, y_offset+75), (255, 255, 255), 2)
    cv2.putText(panel, "+0.1", (95, y_offset+70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Modo conservador
    mode_color = (0, 255, 0) if False else (255, 0, 0)
    cv2.rectangle(panel, (160, y_offset+55), (310, y_offset+75), mode_color, -1)
    cv2.rectangle(panel, (160, y_offset+55), (310, y_offset+75), (255, 255, 255), 2)
    mode_text = "CONSERVADOR" if False else "NORMAL"
    cv2.putText(panel, mode_text, (170, y_offset+70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Estadísticas del clasificador
    y_offset = 580
    cv2.putText(panel, "Clasificadas: 0", (20, y_offset+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(panel, "Buenas: 0", (20, y_offset+25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.putText(panel, "Malas: 0", (20, y_offset+45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    # Botón Config
    y_offset = 640
    cv2.rectangle(panel, (20, y_offset+10), (160, y_offset+40), (100, 100, 100), -1)
    cv2.rectangle(panel, (20, y_offset+10), (160, y_offset+40), (255, 255, 255), 2)
    cv2.putText(panel, "CONFIG", (30, y_offset+28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Botón Info (derecha)
    cv2.rectangle(panel, (170, y_offset+10), (310, y_offset+40), (0, 100, 150), -1)
    cv2.rectangle(panel, (170, y_offset+10), (310, y_offset+40), (255, 255, 255), 2)
    cv2.putText(panel, "INFO", (200, y_offset+28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Botón Salir (derecha)
    y_offset = 710
    cv2.rectangle(panel, (170, y_offset+10), (310, y_offset+40), (100, 0, 0), -1)
    cv2.rectangle(panel, (170, y_offset+10), (310, y_offset+40), (255, 255, 255), 2)
    cv2.putText(panel, "SALIR", (200, y_offset+28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return panel


def actualizar_panel_control(panel, metricas, estado_camara, img_width, img_height, gamma_actual=0.8, patron_actual="BG", yolo_stats=None, cam=None):
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
    
    # Actualizar métricas YOLO (solo tracks y detecciones, sin FPS)
    y_offset = 400
    if yolo_stats:
        cv2.putText(panel_actualizado, f"Tracks: {yolo_stats.get('tracks', 0)}", (20, y_offset+45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(panel_actualizado, f"Detections: {yolo_stats.get('detections', 0)}", (20, y_offset+65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(panel_actualizado, f"Confidence: {0.3:.2f}", (20, y_offset+85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Mostrar estadísticas de procesamiento adaptativo
        if 'skip_ratio' in yolo_stats:
            skip_pct = yolo_stats['skip_ratio'] * 100
            cv2.putText(panel_actualizado, f"Skip: {skip_pct:.0f}%", (20, y_offset+105), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
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
    # Controles del Clasificador
    # Botón -0.1 confianza
    elif 20 <= panel_x_click <= 80 and 555 <= panel_y_click <= 575:
        return "CLF_CONF_DOWN"
    # Botón +0.1 confianza
    elif 90 <= panel_x_click <= 150 and 555 <= panel_y_click <= 575:
        return "CLF_CONF_UP"
    # Botón modo conservador/normal
    elif 160 <= panel_x_click <= 310 and 555 <= panel_y_click <= 575:
        return "CLF_MODE_TOGGLE"
    
    # Botón CONFIG
    elif 20 <= panel_x_click <= 160 and 650 <= panel_y_click <= 680:
        return "CONFIG"
    # Botón INFO (derecha)
    elif 170 <= panel_x_click <= 310 and 650 <= panel_y_click <= 680:
        return "INFO"
    # Botón SALIR (derecha)
    elif 170 <= panel_x_click <= 310 and 720 <= panel_y_click <= 750:
        return "EXIT"
    
    return None
