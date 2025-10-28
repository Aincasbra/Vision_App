#!/usr/bin/env python3
"""
Sistema de Logging para Producción Continua - Calippo Jetson
Basado en syslog con configuración robusta para entorno industrial

ESTRUCTURA DEL SISTEMA:
- 4 subsistemas de logging separados por carpetas:
  1. SYSTEM: funcionamiento (arranque/parada/errores/metricas)
  2. DIGITAL: salidas digitales/PLC (IO)
  3. PHOTOS: snapshots periódicos + imágenes de defectos
  4. VISION: registro detallado por lata (CSV/JSONL + imagen opcional)

- Niveles soportados: debug, info, warning, error, critical
- Rotación automática diaria a las 00:00 con compresión gzip
- Integración con syslog (facility LOCAL0) + archivos locales
"""

# Imports estándar de Python para logging
import logging                    # Sistema de logging estándar de Python
import logging.handlers          # Handlers avanzados (rotativos, syslog)
import syslog                    # Para integración con syslog del sistema
import os                        # Operaciones del sistema operativo
import json                      # Serialización JSON para logs estructurados
import time                      # Timestamps y control temporal
from datetime import datetime    # Manejo de fechas y horas
from typing import Optional, Dict, Any, List  # Tipos para mejor documentación
import threading                 # Locks para acceso thread-safe
from pathlib import Path         # Manejo moderno de rutas de archivos
import csv                       # Escritura de archivos CSV para visión

# Import opcional de OpenCV para guardar imágenes
try:
    import cv2  # Para guardar imágenes si está disponible
except Exception:
    cv2 = None  # Si no está disponible, se desactiva funcionalidad de imágenes

class ProductionLogger:
    """
    Sistema de logging robusto para producción continua en fábrica
    Integra syslog, archivos rotativos y métricas de rendimiento
    
    FUNCIONALIDADES PRINCIPALES:
    - 4 subsistemas de logging separados por dominio
    - Rotación automática de archivos con compresión
    - Integración con syslog del sistema operativo
    - Métricas de rendimiento en tiempo real
    - Guardado de imágenes (snapshots y defectos)
    - Registro estructurado por lata (CSV/JSONL)
    """
    
    def __init__(self, 
                 app_name: str = "calippo_jetson",           # Nombre de la aplicación
                 log_dir: str = "/var/log/calippo",         # Directorio base de logs
                 facility: int = syslog.LOG_LOCAL0,         # Facility de syslog (LOCAL0-7)
                 enable_syslog: bool = True,                # Habilitar syslog
                 enable_file_logging: bool = True,          # Habilitar archivos locales
                 enable_console: bool = False):             # Habilitar consola (solo desarrollo)
        
        # ===== CONFIGURACIÓN BÁSICA =====
        self.app_name = app_name                            # Nombre para identificar logs
        self.log_dir = Path(log_dir)                        # Directorio base como objeto Path
        self.facility = facility                            # Facility de syslog (LOCAL0-7)
        self.enable_syslog = enable_syslog                  # Control de syslog
        self.enable_file_logging = enable_file_logging      # Control de archivos
        self.enable_console = enable_console                # Control de consola
        
        # Crear directorio de logs si no existe
        if self.enable_file_logging:
            self.log_dir.mkdir(parents=True, exist_ok=True)  # Crear con permisos por defecto
        
        # ===== CONFIGURACIÓN DEL LOGGER PRINCIPAL =====
        self.logger = logging.getLogger(self.app_name)      # Logger principal con nombre único
        self.logger.setLevel(logging.DEBUG)                 # Nivel más bajo para capturar todo
        
        # Evitar duplicación de logs (limpiar handlers existentes)
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # ===== ESTRUCTURA DE DIRECTORIOS ESPECÍFICOS =====
        # Separación clara de los 4 subsistemas de logging
        self.system_dir = self.log_dir / "system"          # Funcionamiento: arranque/parada/errores/metricas
        self.digital_dir = self.log_dir / "digital"        # Salidas digitales / comunicación PLC
        self.photos_dir = self.log_dir / "photos"          # Fotos: snapshots y defectos
        self.images_dir = self.photos_dir                   # Alias interno para compatibilidad
        self.snapshots_dir = self.photos_dir / "snapshots"  # Snapshots periódicos cada 30min
        self.defects_dir = self.photos_dir / "defects"      # Imágenes de latas defectuosas
        self.vision_dir = self.log_dir / "vision"          # Registro detallado por lata
        
        # Crear todos los directorios necesarios
        if self.enable_file_logging:
            for d in [self.system_dir, self.digital_dir, self.photos_dir, 
                     self.snapshots_dir, self.defects_dir, self.vision_dir]:
                d.mkdir(parents=True, exist_ok=True)        # Crear con permisos por defecto

        # ===== CONTROL DE SNAPSHOTS PERIÓDICOS =====
        self.snapshot_interval_seconds = 30 * 60           # Intervalo: 30 minutos
        self._last_snapshot_ts: Optional[float] = None     # Timestamp del último snapshot

        # ===== CONFIGURACIÓN DE HANDLERS =====
        self._setup_handlers()                             # Configurar todos los handlers
        
        # ===== MÉTRICAS DE PRODUCCIÓN =====
        # Diccionario thread-safe para métricas en tiempo real
        self.metrics = {
            'start_time': datetime.now(),                  # Momento de inicio del sistema
            'frames_processed': 0,                         # Contador de frames procesados
            'detections_count': 0,                         # Contador de detecciones YOLO
            'errors_count': 0,                            # Contador de errores
            'warnings_count': 0,                          # Contador de warnings
            'last_frame_time': None,                       # Timestamp del último frame
            'avg_fps': 0.0,                               # FPS promedio actual
            'memory_usage': 0.0                           # Uso de memoria en MB
        }
        
        # Lock para acceso thread-safe a métricas
        self.metrics_lock = threading.Lock()
        
        # ===== LOG INICIAL DEL SISTEMA =====
        # Registrar que el sistema se ha inicializado correctamente
        self.logger.info("🚀 Sistema de logging de producción inicializado", extra={
            'component': 'logging_system',                  # Componente que genera el log
            'facility': facility,                          # Facility de syslog usado
            'log_dir': str(self.log_dir),                 # Directorio de logs configurado
            'syslog_enabled': enable_syslog,               # Estado de syslog
            'file_logging_enabled': enable_file_logging    # Estado de archivos
        })

    def _setup_handlers(self):
        """
        Configura todos los handlers de logging
        
        HANDLERS CONFIGURADOS:
        1. SyslogHandler: Envía logs al sistema syslog (LOCAL0)
        2. RotatingFileHandler: Archivos rotativos con límite de tamaño
        3. ConsoleHandler: Salida a consola (solo desarrollo)
        
        ESTRUCTURA DE ARCHIVOS:
        - system/: logs generales, errores y métricas
        - digital/: logs de salidas digitales/PLC
        - photos/: snapshots y defectos (imágenes)
        - vision/: CSV/JSONL por lata + imágenes asociadas
        """
        
        # ===== 1. HANDLER PARA SYSLOG (PRODUCCIÓN) =====
        if self.enable_syslog:
            # Crear handler que envía logs al sistema syslog
            syslog_handler = logging.handlers.SysLogHandler(
                address='/dev/log',                          # Socket de syslog del sistema
                facility=self.facility                       # Facility LOCAL0-7
            )
            # Formatter específico para syslog (formato industrial)
            syslog_formatter = ProductionSyslogFormatter()
            syslog_handler.setFormatter(syslog_formatter)
            syslog_handler.setLevel(logging.INFO)            # Solo INFO y superiores a syslog
            self.logger.addHandler(syslog_handler)
        
        # ===== 2. HANDLERS PARA ARCHIVOS ROTATIVOS =====
        if self.enable_file_logging:
            
            # --- LOG GENERAL (system/) ---
            # Archivo principal con todos los logs de nivel DEBUG y superior
            general_handler = logging.handlers.RotatingFileHandler(
                self.system_dir / f"{self.app_name}.log",    # Archivo principal
                maxBytes=50*1024*1024,                       # Límite: 50MB
                backupCount=10                              # Mantener 10 backups
            )
            general_formatter = ProductionFileFormatter()    # Formatter detallado
            general_handler.setFormatter(general_formatter)
            general_handler.setLevel(logging.DEBUG)          # Capturar todo desde DEBUG
            self.logger.addHandler(general_handler)
            
            # --- LOG DE ERRORES CRÍTICOS (system/) ---
            # Archivo separado solo para ERROR y CRITICAL
            error_handler = logging.handlers.RotatingFileHandler(
                self.system_dir / f"{self.app_name}_errors.log",  # Archivo de errores
                maxBytes=10*1024*1024,                       # Límite: 10MB
                backupCount=5                               # Mantener 5 backups
            )
            error_handler.setFormatter(general_formatter)
            error_handler.setLevel(logging.ERROR)            # Solo ERROR y CRITICAL
            self.logger.addHandler(error_handler)
            
            # --- LOG DE MÉTRICAS (system/) ---
            # Archivo específico para métricas de rendimiento en formato JSON
            metrics_handler = logging.handlers.RotatingFileHandler(
                self.system_dir / f"{self.app_name}_metrics.log",  # Archivo de métricas
                maxBytes=20*1024*1024,                       # Límite: 20MB
                backupCount=7                               # Mantener 7 backups
            )
            metrics_formatter = MetricsFormatter()           # Formatter JSON
            metrics_handler.setFormatter(metrics_formatter)
            metrics_handler.setLevel(logging.INFO)           # Solo INFO y superiores
            self.logger.addHandler(metrics_handler)

            # --- LOG DE SALIDAS DIGITALES (digital/) ---
            # Archivo específico para acciones de IO digital/PLC
            digital_handler = logging.handlers.RotatingFileHandler(
                self.digital_dir / f"{self.app_name}_digital.log",  # Archivo de IO
                maxBytes=10*1024*1024,                       # Límite: 10MB
                backupCount=5                               # Mantener 5 backups
            )
            digital_formatter = ProductionFileFormatter()
            digital_handler.setFormatter(digital_formatter)
            digital_handler.setLevel(logging.INFO)           # Solo INFO y superiores
            self.logger.addHandler(digital_handler)
        
        # ===== 3. HANDLER PARA CONSOLA (DESARROLLO) =====
        if self.enable_console:
            # Handler para mostrar logs en consola con colores
            console_handler = logging.StreamHandler()
            console_formatter = ConsoleFormatter()           # Formatter con colores
            console_handler.setFormatter(console_formatter)
            console_handler.setLevel(logging.INFO)           # Solo INFO y superiores
            self.logger.addHandler(console_handler)

    def log_camera_event(self, event_type: str, message: str, **kwargs):
        """
        Log específico para eventos de cámara
        
        FUNCIÓN: Registra eventos relacionados con la cámara (conexión, configuración, errores)
        DESTINO: system/calippo_jetson.log + syslog (si está habilitado)
        NIVELES: info (por defecto), warning, error, critical según event_type
        
        PARÁMETROS:
        - event_type: tipo de evento ('info', 'warning', 'error', 'critical')
        - message: mensaje descriptivo del evento
        - **kwargs: datos adicionales (ip, parámetros, etc.)
        """
        # Preparar datos adicionales para el log
        extra = {
            'component': 'camera',                           # Identificar componente
            'event_type': event_type,                        # Tipo de evento
            **kwargs                                         # Datos adicionales
        }
        
        # Seleccionar nivel de log según el tipo de evento
        if event_type in ['error', 'critical']:
            self.logger.error(f"📷 {message}", extra=extra)  # ERROR/CRITICAL
        elif event_type == 'warning':
            self.logger.warning(f"📷 {message}", extra=extra)  # WARNING
        else:
            self.logger.info(f"📷 {message}", extra=extra)   # INFO por defecto

    def log_detection_event(self, detection_data: Dict[str, Any]):
        """
        Log específico para detecciones YOLO
        
        FUNCIÓN: Registra cada detección de YOLO con sus datos completos
        DESTINO: system/calippo_jetson.log + syslog
        MÉTRICAS: Incrementa contador de detecciones (thread-safe)
        
        PARÁMETROS:
        - detection_data: diccionario con datos de la detección
          - class: clase detectada ('lata_ok', 'lata_defecto', etc.)
          - confidence: confianza de la detección (0.0-1.0)
          - bbox: bounding box [x, y, w, h]
          - otros campos personalizados
        """
        # Incrementar contador de detecciones de forma thread-safe
        with self.metrics_lock:
            self.metrics['detections_count'] += 1
        
        # Preparar datos para el log
        extra = {
            'component': 'yolo_detection',                   # Identificar componente
            'detection_data': detection_data                 # Datos completos de detección
        }
        
        # Log con información resumida de la detección
        self.logger.info(f"🎯 Detección: {detection_data.get('class', 'unknown')} "
                        f"(conf: {detection_data.get('confidence', 0):.2f})", extra=extra)

    def log_performance_metrics(self):
        """
        Log periódico de métricas de rendimiento
        
        FUNCIÓN: Registra métricas del sistema en formato JSON
        DESTINO: system/calippo_jetson_metrics.log (formato JSON)
        FRECUENCIA: Llamar periódicamente (ej: cada minuto)
        
        MÉTRICAS INCLUIDAS:
        - uptime_seconds: tiempo de funcionamiento
        - frames_processed: frames procesados
        - detections_count: detecciones realizadas
        - errors_count: errores acumulados
        - warnings_count: warnings acumulados
        - avg_fps: FPS promedio actual
        - memory_usage_mb: uso de memoria en MB
        """
        # Obtener métricas de forma thread-safe
        with self.metrics_lock:
            uptime = datetime.now() - self.metrics['start_time']  # Calcular uptime
            metrics_data = {
                'uptime_seconds': uptime.total_seconds(),        # Tiempo total en segundos
                'frames_processed': self.metrics['frames_processed'],  # Frames procesados
                'detections_count': self.metrics['detections_count'],  # Detecciones realizadas
                'errors_count': self.metrics['errors_count'],     # Errores acumulados
                'warnings_count': self.metrics['warnings_count'], # Warnings acumulados
                'avg_fps': self.metrics['avg_fps'],              # FPS promedio
                'memory_usage_mb': self.metrics['memory_usage']  # Memoria en MB
            }
        
        # Preparar datos para el log
        extra = {
            'component': 'performance_metrics',              # Identificar componente
            'metrics': metrics_data                         # Datos de métricas
        }
        
        # Log en formato JSON (usado por MetricsFormatter)
        self.logger.info("📊 Métricas de rendimiento", extra=extra)

    def log_system_event(self, event_type: str, message: str, **kwargs):
        """
        Log para eventos del sistema
        
        FUNCIÓN: Registra eventos de funcionamiento del sistema (arranque, parada, errores)
        DESTINO: system/calippo_jetson.log + syslog
        MÉTRICAS: Actualiza contadores de errores/warnings (thread-safe)
        
        EVENTOS SOPORTADOS:
        - startup/shutdown: inicio y cierre del sistema
        - acquisition_start/acquisition_stop: inicio/parada de adquisición
        - line_start/line_stop: inicio/parada de línea de producción
        - config_change: cambios de configuración
        - error: errores del sistema
        - warning: advertencias del sistema
        - debug: información de depuración
        
        PARÁMETROS:
        - event_type: tipo de evento del sistema
        - message: mensaje descriptivo
        - **kwargs: datos adicionales del evento
        """
        # Preparar datos adicionales para el log
        extra = {
            'component': 'system',                           # Identificar componente
            'event_type': event_type,                        # Tipo de evento
            **kwargs                                         # Datos adicionales
        }
        
        # Seleccionar nivel y acción según el tipo de evento
        if event_type in ['startup', 'shutdown', 'config_change', 'acquisition_start', 'acquisition_stop', 'line_start', 'line_stop']:
            self.logger.info(f"🔧 {message}", extra=extra)   # INFO para eventos normales
        elif event_type == 'error':
            # Incrementar contador de errores de forma thread-safe
            with self.metrics_lock:
                self.metrics['errors_count'] += 1
            self.logger.error(f"❌ {message}", extra=extra)  # ERROR
        elif event_type == 'warning':
            # Incrementar contador de warnings de forma thread-safe
            with self.metrics_lock:
                self.metrics['warnings_count'] += 1
            self.logger.warning(f"⚠️ {message}", extra=extra)  # WARNING
        elif event_type == 'debug':
            self.logger.debug(f"🧪 {message}", extra=extra)  # DEBUG

    def update_frame_metrics(self, fps: float, memory_usage: float):
        """
        Actualiza métricas de frames
        
        FUNCIÓN: Actualiza métricas de rendimiento por frame procesado
        MÉTRICAS: Incrementa contador de frames y actualiza FPS/memoria
        THREAD-SAFE: Usa lock para acceso concurrente seguro
        
        PARÁMETROS:
        - fps: frames por segundo actual
        - memory_usage: uso de memoria en MB
        """
        # Actualizar métricas de forma thread-safe
        with self.metrics_lock:
            self.metrics['frames_processed'] += 1           # Incrementar contador
            self.metrics['avg_fps'] = fps                   # Actualizar FPS
            self.metrics['memory_usage'] = memory_usage     # Actualizar memoria
            self.metrics['last_frame_time'] = datetime.now() # Timestamp actual

    def log_configuration_change(self, parameter: str, old_value: Any, new_value: Any):
        """
        Log para cambios de configuración
        
        FUNCIÓN: Registra cambios en parámetros de configuración
        DESTINO: system/calippo_jetson.log + syslog
        AUDITORÍA: Permite rastrear cambios de configuración
        
        PARÁMETROS:
        - parameter: nombre del parámetro cambiado
        - old_value: valor anterior
        - new_value: valor nuevo
        """
        # Preparar datos para el log
        extra = {
            'component': 'configuration',                   # Identificar componente
            'parameter': parameter,                         # Parámetro cambiado
            'old_value': str(old_value),                    # Valor anterior (como string)
            'new_value': str(new_value)                     # Valor nuevo (como string)
        }
        
        # Log del cambio de configuración
        self.logger.info(f"⚙️ Configuración cambiada: {parameter} = {new_value}", extra=extra)

    def log_production_event(self, event_type: str, product_id: str = None, **kwargs):
        """
        Log específico para eventos de producción
        
        FUNCIÓN: Registra eventos relacionados con la producción (calidad, línea)
        DESTINO: system/calippo_jetson.log + syslog
        NIVELES: info, warning, critical según importancia
        
        EVENTOS SOPORTADOS:
        - quality_pass: producto aprobado
        - quality_fail: producto rechazado
        - line_stop: línea de producción detenida (CRITICAL)
        - line_start: línea de producción iniciada
        
        PARÁMETROS:
        - event_type: tipo de evento de producción
        - product_id: ID del producto (opcional)
        - **kwargs: datos adicionales del evento
        """
        # Preparar datos adicionales para el log
        extra = {
            'component': 'production',                      # Identificar componente
            'event_type': event_type,                       # Tipo de evento
            'product_id': product_id,                       # ID del producto
            **kwargs                                         # Datos adicionales
        }
        
        # Seleccionar nivel según el tipo de evento
        if event_type == 'quality_pass':
            self.logger.info(f"✅ Producto aprobado: {product_id}", extra=extra)
        elif event_type == 'quality_fail':
            self.logger.warning(f"❌ Producto rechazado: {product_id}", extra=extra)
        elif event_type == 'line_stop':
            self.logger.critical(f"🛑 Línea de producción detenida", extra=extra)  # CRITICAL
        elif event_type == 'line_start':
            self.logger.info(f"▶️ Línea de producción iniciada", extra=extra)

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Obtiene resumen de métricas actuales"""
        with self.metrics_lock:
            uptime = datetime.now() - self.metrics['start_time']
            return {
                'uptime_hours': uptime.total_seconds() / 3600,
                'frames_processed': self.metrics['frames_processed'],
                'detections_count': self.metrics['detections_count'],
                'errors_count': self.metrics['errors_count'],
                'warnings_count': self.metrics['warnings_count'],
                'avg_fps': self.metrics['avg_fps'],
                'memory_usage_mb': self.metrics['memory_usage']
            }

    # =====================
    # SUBSISTEMA 2: SALIDAS DIGITALES (PLC/IO)
    # =====================
    def log_digital_output(self, channel: int, value: int, message: str = "", level: str = "info", **kwargs):
        """
        Log de acción sobre salidas digitales (IO) con nivel configurable
        
        FUNCIÓN: Registra acciones en puertos digitales/relés/PLC
        DESTINO: digital/calippo_jetson_digital.log + syslog
        NIVELES: debug, info, warning, error, critical (configurable)
        
        USO TÍPICO:
        - Activar/desactivar relés
        - Controlar salidas digitales
        - Comunicación con PLC
        - Triggers de hardware
        
        PARÁMETROS:
        - channel: número de canal/puerto digital
        - value: valor enviado (0/1, o valor numérico)
        - message: mensaje descriptivo (opcional)
        - level: nivel de log (debug/info/warning/error/critical)
        - **kwargs: datos adicionales (motivo, causa, etc.)
        """
        # Preparar datos adicionales para el log
        extra = {
            'component': 'digital_io',                      # Identificar componente
            'channel': channel,                             # Canal utilizado
            'value': value,                                 # Valor enviado
            **kwargs                                         # Datos adicionales
        }
        # Mensaje por defecto si no se proporciona
        text = message or f"Salida digital ch={channel} -> {value}"
        
        # Seleccionar nivel de log según parámetro
        level_l = (level or "info").lower()
        if level_l == 'debug':
            self.logger.debug(f"🔌 {text}", extra=extra)
        elif level_l == 'warning':
            self.logger.warning(f"🔌 {text}", extra=extra)
        elif level_l == 'error':
            self.logger.error(f"🔌 {text}", extra=extra)
        elif level_l == 'critical' or level_l == 'fatal':
            self.logger.critical(f"🔌 {text}", extra=extra)
        else:
            self.logger.info(f"🔌 {text}", extra=extra)    # INFO por defecto

    # =====================
    # SUBSISTEMA 3: GESTIÓN DE FOTOS
    # =====================
    def _write_image(self, image: Any, out_path: Path) -> bool:
        """
        Intenta guardar una imagen en disco
        
        FUNCIÓN: Guarda imagen en diferentes formatos (ndarray, bytes, archivo)
        SOPORTE: OpenCV ndarray, bytes, rutas de archivos existentes
        ERRORES: Captura excepciones y las registra en logs
        
        PARÁMETROS:
        - image: imagen a guardar (ndarray, bytes, o ruta de archivo)
        - out_path: ruta de destino donde guardar
        
        RETORNA: True si se guardó correctamente, False si hubo error
        """
        try:
            # Crear directorio padre si no existe
            out_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Caso 1: ndarray de OpenCV (imagen procesada)
            if cv2 is not None and hasattr(image, 'shape'):
                return bool(cv2.imwrite(str(out_path), image))  # Guardar con OpenCV
            
            # Caso 2: datos binarios (bytes)
            if isinstance(image, (bytes, bytearray)):
                with open(out_path, 'wb') as f:
                    f.write(image)
                return True
            
            # Caso 3: ruta de archivo existente (copiar)
            if isinstance(image, (str, Path)) and Path(image).exists():
                data = Path(image).read_bytes()              # Leer archivo fuente
                with open(out_path, 'wb') as f:
                    f.write(data)                           # Escribir en destino
                return True
        except Exception as e:
            # Registrar error en logs si falla el guardado
            self.logger.error(f"❌ Error guardando imagen: {e}", extra={'component': 'photos'})
        return False

    def save_periodic_snapshot(self, image: Any, now_ts: Optional[float] = None, level: str = "info") -> Optional[Path]:
        """
        Guarda un snapshot cada 30 min como máximo y lo registra en logs con nivel configurable
        
        FUNCIÓN: Guarda snapshots periódicos para monitoreo continuo
        FRECUENCIA: Máximo cada 30 minutos (configurable)
        DESTINO: photos/snapshots/snapshot_YYYYmmdd_HHMMSS.jpg
        LOG: Registra el evento en logs con nivel configurable
        
        PARÁMETROS:
        - image: imagen a guardar (ndarray, bytes, o ruta)
        - now_ts: timestamp actual (opcional, usa time.time() si no se proporciona)
        - level: nivel de log (debug/info/warning/error/critical)
        
        RETORNA: Path del archivo guardado, o None si no se guardó
        """
        now_ts = now_ts or time.time()                      # Timestamp actual
        
        # Verificar si ha pasado suficiente tiempo desde el último snapshot
        if self._last_snapshot_ts is not None and (now_ts - self._last_snapshot_ts) < self.snapshot_interval_seconds:
            return None                                     # No guardar si no ha pasado el intervalo
        
        # Generar nombre de archivo con timestamp
        ts_str = datetime.fromtimestamp(now_ts).strftime('%Y%m%d_%H%M%S')
        out_path = self.snapshots_dir / f"snapshot_{ts_str}.jpg"
        
        # Intentar guardar la imagen
        if self._write_image(image, out_path):
            self._last_snapshot_ts = now_ts                 # Actualizar timestamp del último snapshot
            
            # Preparar datos para el log
            extra = {
                'component': 'photos',                      # Identificar componente
                'path': str(out_path),                      # Ruta del archivo guardado
                'event_type': 'snapshot'                    # Tipo de evento
            }
            
            # Registrar evento con nivel configurable
            level_l = (level or "info").lower()
            if level_l == 'debug':
                self.logger.debug("🖼️ Snapshot periódico guardado", extra=extra)
            elif level_l == 'warning':
                self.logger.warning("🖼️ Snapshot periódico guardado", extra=extra)
            elif level_l == 'error':
                self.logger.error("🖼️ Snapshot periódico guardado", extra=extra)
            elif level_l == 'critical' or level_l == 'fatal':
                self.logger.critical("🖼️ Snapshot periódico guardado", extra=extra)
            else:
                self.logger.info("🖼️ Snapshot periódico guardado", extra=extra)
            return out_path
        return None

    def save_defect_image(self, image: Any, can_id: str, classification: str, confidence: float, extra_info: Optional[Dict[str, Any]] = None, level: str = "warning") -> Optional[Path]:
        """
        Guarda imagen de lata defectuosa y la registra en logs con nivel configurable
        
        FUNCIÓN: Guarda imágenes de latas rechazadas para análisis posterior
        ORGANIZACIÓN: Por fecha (photos/defects/YYYYmmdd/)
        NOMENCLATURA: defect_{can_id}_{classification}_{confidence%}_{timestamp}.jpg
        LOG: Registra el evento con nivel configurable (warning por defecto)
        
        PARÁMETROS:
        - image: imagen de la lata defectuosa
        - can_id: ID único de la lata
        - classification: tipo de defecto detectado
        - confidence: confianza de la detección (0.0-1.0)
        - extra_info: información adicional (opcional)
        - level: nivel de log (debug/info/warning/error/critical)
        
        RETORNA: Path del archivo guardado, o None si falló
        """
        # Crear directorio por fecha
        date_dir = self.defects_dir / datetime.now().strftime('%Y%m%d')
        
        # Generar nombre de archivo descriptivo
        ts_str = datetime.now().strftime('%H%M%S_%f')[:-3]  # Timestamp con milisegundos
        filename = f"defect_{can_id}_{classification}_{int(confidence*100)}_{ts_str}.jpg"
        out_path = date_dir / filename
        
        # Intentar guardar la imagen
        if self._write_image(image, out_path):
            # Preparar información del log
            info = {
                'component': 'photos',                      # Identificar componente
                'event_type': 'defect_image',               # Tipo de evento
                'path': str(out_path),                      # Ruta del archivo
                'can_id': can_id,                           # ID de la lata
                'class': classification,                    # Clasificación del defecto
                'confidence': confidence                    # Confianza de la detección
            }
            if extra_info:
                info.update(extra_info)                     # Agregar información adicional
            
            # Registrar evento con nivel configurable
            level_l = (level or "warning").lower()
            if level_l == 'debug':
                self.logger.debug("🖼️ Imagen de defecto guardada", extra=info)
            elif level_l == 'info':
                self.logger.info("🖼️ Imagen de defecto guardada", extra=info)
            elif level_l == 'error':
                self.logger.error("🖼️ Imagen de defecto guardada", extra=info)
            elif level_l == 'critical' or level_l == 'fatal':
                self.logger.critical("🖼️ Imagen de defecto guardada", extra=info)
            else:
                self.logger.warning("🖼️ Imagen de defecto guardada", extra=info)  # WARNING por defecto
            return out_path
        return None

    # =====================
    # SUBSISTEMA 4: LOGGING DE VISIÓN POR LATA
    # =====================
    def log_vision_can(self,
                       can_id: str,
                       verdict: str,
                       camera_params: Dict[str, Any],
                       yolo_params: Dict[str, Any],
                       bbox: Optional[List[int]] = None,
                       image: Any = None,
                       level: str = "info") -> None:
        """
        Registra un evento de visión completo por lata en CSV/JSONL y opcionalmente guarda imagen
        
        FUNCIÓN: Registro completo de inspección por lata con todos los parámetros
        DESTINOS MÚLTIPLES:
        1. Log estructurado: system/calippo_jetson.log + syslog
        2. CSV: vision/vision_log.csv (para análisis en Excel/otros)
        3. JSONL: vision/vision_log.jsonl (para análisis programático)
        4. Imagen: vision/images/vision_{can_id}_{ok|bad}_{timestamp}.jpg (opcional)
        
        DATOS REGISTRADOS:
        - Timestamp ISO de la inspección
        - ID único de la lata
        - Verdicto (ok/bad)
        - Parámetros de cámara (exposición, gain, gamma, formato, dimensiones)
        - Parámetros de YOLO (umbral, clase, confianza)
        - Bounding box de la detección
        - Imagen asociada (opcional)
        
        PARÁMETROS:
        - can_id: ID único de la lata inspeccionada
        - verdict: resultado ('ok' o 'bad')
        - camera_params: diccionario con parámetros de cámara
        - yolo_params: diccionario con parámetros de YOLO
        - bbox: bounding box [x, y, w, h] (opcional)
        - image: imagen de la lata (opcional)
        - level: nivel de log (debug/info/warning/error/critical)
        """
        # Generar timestamp ISO para el registro
        timestamp_iso = datetime.now().isoformat()

        # Construir registro completo con todos los datos
        record = {
            'timestamp': timestamp_iso,                      # Timestamp ISO
            'can_id': can_id,                               # ID de la lata
            'verdict': verdict,                             # Resultado: 'ok' | 'bad'
            'camera_exposure': camera_params.get('ExposureTime'),      # Tiempo de exposición
            'camera_gain': camera_params.get('Gain'),               # Ganancia
            'camera_gamma': camera_params.get('Gamma'),             # Gamma
            'pixel_format': camera_params.get('PixelFormat'),       # Formato de píxel
            'width': camera_params.get('Width'),                    # Ancho de imagen
            'height': camera_params.get('Height'),                  # Alto de imagen
            'yolo_threshold': yolo_params.get('threshold'),         # Umbral de YOLO
            'yolo_class': yolo_params.get('class'),                 # Clase detectada
            'yolo_confidence': yolo_params.get('confidence'),       # Confianza de YOLO
            'bbox': bbox                                          # Bounding box
        }

        # ===== 1. LOG ESTRUCTURADO AL ARCHIVO GENERAL Y SYSLOG =====
        extra_first = {
            'component': 'vision',                          # Identificar componente
            'event_type': 'can_inspection',                 # Tipo de evento
            'can_id': can_id,                               # ID de la lata
            'verdict': verdict,                             # Verdicto
            **{k: v for k, v in record.items() if k not in ['timestamp']}  # Todos los datos excepto timestamp
        }
        
        # Registrar con nivel configurable
        level_l = (level or "info").lower()
        if level_l == 'debug':
            self.logger.debug("👁️ Registro de visión por lata", extra=extra_first)
        elif level_l == 'warning':
            self.logger.warning("👁️ Registro de visión por lata", extra=extra_first)
        elif level_l == 'error':
            self.logger.error("👁️ Registro de visión por lata", extra=extra_first)
        elif level_l == 'critical' or level_l == 'fatal':
            self.logger.critical("👁️ Registro de visión por lata", extra=extra_first)
        else:
            self.logger.info("👁️ Registro de visión por lata", extra=extra_first)

        # ===== 2. GUARDAR CSV (PARA ANÁLISIS EN EXCEL/OTROS) =====
        csv_path = self.vision_dir / 'vision_log.csv'
        write_header = not csv_path.exists()                # Escribir cabecera solo si el archivo no existe
        
        try:
            with open(csv_path, 'a', newline='') as f:
                # Definir columnas del CSV
                writer = csv.DictWriter(f, fieldnames=[
                    'timestamp','can_id','verdict','camera_exposure','camera_gain','camera_gamma',
                    'pixel_format','width','height','yolo_threshold','yolo_class','yolo_confidence','bbox'
                ])
                if write_header:
                    writer.writeheader()                    # Escribir cabecera si es nuevo archivo
                
                # Preparar fila para CSV (convertir bbox a JSON string)
                row = record.copy()
                row['bbox'] = json.dumps(bbox) if bbox is not None else None
                writer.writerow(row)                       # Escribir fila
        except Exception as e:
            # Registrar error si falla la escritura CSV
            self.logger.error(f"❌ Error escribiendo CSV de visión: {e}", extra={'component': 'vision'})

        # ===== 3. GUARDAR JSONL (PARA ANÁLISIS PROGRAMÁTICO) =====
        jsonl_path = self.vision_dir / 'vision_log.jsonl'
        try:
            with open(jsonl_path, 'a') as f:
                f.write(json.dumps(record) + "\n")         # Una línea JSON por registro
        except Exception as e:
            # Registrar error si falla la escritura JSONL
            self.logger.error(f"❌ Error escribiendo JSONL de visión: {e}", extra={'component': 'vision'})

        # ===== 4. GUARDAR IMAGEN ASOCIADA (OPCIONAL) =====
        if image is not None:
            # Generar nombre de archivo descriptivo
            img_filename = f"vision_{can_id}_{'ok' if verdict=='ok' else 'bad'}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}.jpg"
            img_path = self.vision_dir / 'images'
            img_path.mkdir(parents=True, exist_ok=True)    # Crear directorio si no existe
            out_path = img_path / img_filename
            
            # Intentar guardar la imagen
            if self._write_image(image, out_path):
                # Registrar evento de imagen guardada con nivel configurable
                extra_img = {'component': 'vision', 'path': str(out_path), 'can_id': can_id}
                if level_l == 'debug':
                    self.logger.debug("🖼️ Imagen de visión guardada", extra=extra_img)
                elif level_l == 'warning':
                    self.logger.warning("🖼️ Imagen de visión guardada", extra=extra_img)
                elif level_l == 'error':
                    self.logger.error("🖼️ Imagen de visión guardada", extra=extra_img)
                elif level_l == 'critical' or level_l == 'fatal':
                    self.logger.critical("🖼️ Imagen de visión guardada", extra=extra_img)
                else:
                    self.logger.info("🖼️ Imagen de visión guardada", extra=extra_img)


# =====================
# CLASES FORMATTER PARA DIFERENTES DESTINOS
# =====================

class ProductionSyslogFormatter(logging.Formatter):
    """
    Formatter específico para syslog en producción
    
    FUNCIÓN: Formatea logs para envío al sistema syslog
    FORMATO: [YYYY-MM-DD HH:MM:SS] COMPONENTE:EVENTO - mensaje | METRICS: {...}
    USO: Handler SysLogHandler usa este formatter
    """
    
    def format(self, record):
        # Formato optimizado para syslog industrial
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
        
        # Extraer información del componente
        component = getattr(record, 'component', 'unknown')      # Componente que genera el log
        event_type = getattr(record, 'event_type', '')          # Tipo de evento
        
        # Crear mensaje estructurado
        message = f"[{timestamp}] {component.upper()}"         # Timestamp + componente
        if event_type:
            message += f":{event_type.upper()}"                 # Agregar tipo de evento si existe
        
        message += f" - {record.getMessage()}"                  # Agregar mensaje principal
        
        # Agregar datos adicionales si existen (métricas, etc.)
        if hasattr(record, 'metrics'):
            message += f" | METRICS: {json.dumps(record.metrics)}"
        
        return message


class ProductionFileFormatter(logging.Formatter):
    """
    Formatter detallado para archivos de log
    
    FUNCIÓN: Formatea logs para archivos locales con información completa
    FORMATO: [YYYY-MM-DD HH:MM:SS.mmm] LEVEL | COMPONENTE | mensaje | EVENT: tipo | METRICS: {...}
    USO: Handlers RotatingFileHandler usan este formatter
    """
    
    def format(self, record):
        # Timestamp con milisegundos para precisión
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        # Información básica del log
        level = record.levelname                               # Nivel (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        component = getattr(record, 'component', 'unknown')    # Componente que genera el log
        message = record.getMessage()                          # Mensaje principal
        
        # Construir línea de log base
        log_line = f"[{timestamp}] {level:8} | {component:15} | {message}"
        
        # Agregar información adicional si existe
        if hasattr(record, 'event_type'):
            log_line += f" | EVENT: {record.event_type}"      # Tipo de evento
        
        if hasattr(record, 'metrics'):
            log_line += f" | METRICS: {json.dumps(record.metrics, indent=None)}"  # Métricas en JSON
        
        if hasattr(record, 'detection_data'):
            log_line += f" | DETECTION: {json.dumps(record.detection_data, indent=None)}"  # Datos de detección
        
        return log_line


class MetricsFormatter(logging.Formatter):
    """
    Formatter específico para métricas (formato JSON)
    
    FUNCIÓN: Formatea logs de métricas en formato JSON puro
    FORMATO: {"timestamp": "...", "level": "...", "component": "...", "message": "...", "metrics": {...}}
    USO: Handler de métricas usa este formatter para análisis programático
    """
    
    def format(self, record):
        # Timestamp en formato ISO
        timestamp = datetime.fromtimestamp(record.created).isoformat()
        
        # Estructura base del log JSON
        log_entry = {
            'timestamp': timestamp,                            # Timestamp ISO
            'level': record.levelname,                         # Nivel del log
            'component': getattr(record, 'component', 'unknown'),  # Componente
            'message': record.getMessage()                     # Mensaje principal
        }
        
        # Agregar métricas si existen
        if hasattr(record, 'metrics'):
            log_entry['metrics'] = record.metrics              # Datos de métricas
        
        return json.dumps(log_entry)                           # Retornar JSON string


class ConsoleFormatter(logging.Formatter):
    """
    Formatter para consola con colores
    
    FUNCIÓN: Formatea logs para consola con colores ANSI
    FORMATO: [HH:MM:SS] LEVEL | COMPONENTE | mensaje (con colores)
    USO: Handler StreamHandler usa este formatter para desarrollo
    """
    
    # Códigos de color ANSI para diferentes niveles
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset color
    }
    
    def format(self, record):
        # Timestamp simple para consola
        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S')
        level = record.levelname                               # Nivel del log
        component = getattr(record, 'component', 'unknown')    # Componente
        message = record.getMessage()                          # Mensaje
        
        # Seleccionar color según nivel
        color = self.COLORS.get(level, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # Retornar línea con colores
        return f"{color}[{timestamp}] {level:8}{reset} | {component:15} | {message}"


# =====================
# INSTANCIA GLOBAL Y FUNCIONES DE CONVENIENCIA
# =====================

# Instancia global del logger de producción
production_logger = None

def initialize_production_logging(app_name: str = "calippo_jetson", 
                                log_dir: str = "/var/log/calippo",
                                enable_console: bool = False) -> ProductionLogger:
    """
    Inicializa el sistema de logging de producción
    
    FUNCIÓN: Crea y configura la instancia global del logger
    USO: Llamar una vez al inicio de la aplicación
    
    PARÁMETROS:
    - app_name: nombre de la aplicación (por defecto: "calippo_jetson")
    - log_dir: directorio base de logs (por defecto: "/var/log/calippo")
    - enable_console: mostrar logs en consola (por defecto: False, solo desarrollo)
    
    RETORNA: Instancia de ProductionLogger configurada
    """
    global production_logger
    
    # Crear instancia del logger
    production_logger = ProductionLogger(
        app_name=app_name,
        log_dir=log_dir,
        enable_console=enable_console
    )
    
    return production_logger

def get_production_logger() -> Optional[ProductionLogger]:
    """
    Obtiene la instancia del logger de producción
    
    FUNCIÓN: Acceso a la instancia global del logger
    USO: Para acceso directo a métodos del logger
    
    RETORNA: Instancia de ProductionLogger o None si no está inicializado
    """
    return production_logger

# =====================
# FUNCIONES DE CONVENIENCIA PARA LOGGING RÁPIDO
# =====================

def log_camera_event(event_type: str, message: str, **kwargs):
    """
    Log rápido para eventos de cámara
    
    FUNCIÓN: Wrapper para log_camera_event del logger global
    USO: log_camera_event("info", "Cámara conectada", ip="192.168.1.100")
    """
    if production_logger:
        production_logger.log_camera_event(event_type, message, **kwargs)

def log_detection(detection_data: Dict[str, Any]):
    """
    Log rápido para detecciones
    
    FUNCIÓN: Wrapper para log_detection_event del logger global
    USO: log_detection({"class": "lata_ok", "confidence": 0.95, "bbox": [100,100,200,200]})
    """
    if production_logger:
        production_logger.log_detection_event(detection_data)

def log_system_event(event_type: str, message: str, **kwargs):
    """
    Log rápido para eventos del sistema
    
    FUNCIÓN: Wrapper para log_system_event del logger global
    USO: log_system_event("startup", "Sistema iniciado")
    """
    if production_logger:
        production_logger.log_system_event(event_type, message, **kwargs)

def log_production_event(event_type: str, product_id: str = None, **kwargs):
    """
    Log rápido para eventos de producción
    
    FUNCIÓN: Wrapper para log_production_event del logger global
    USO: log_production_event("quality_pass", "LATA_001")
    """
    if production_logger:
        production_logger.log_production_event(event_type, product_id, **kwargs)

def update_performance_metrics(fps: float, memory_usage: float):
    """
    Actualiza métricas de rendimiento
    
    FUNCIÓN: Wrapper para update_frame_metrics del logger global
    USO: update_performance_metrics(15.5, 1024.5)  # FPS y memoria en MB
    """
    if production_logger:
        production_logger.update_frame_metrics(fps, memory_usage)

def log_performance():
    """
    Log periódico de métricas
    
    FUNCIÓN: Wrapper para log_performance_metrics del logger global
    USO: Llamar periódicamente (ej: cada minuto)
    """
    if production_logger:
        production_logger.log_performance_metrics()

# =====================
# NUEVAS FUNCIONES DE CONVENIENCIA PARA LOS 4 SUBSISTEMAS
# =====================

def log_digital(channel: int, value: int, message: str = "", level: str = "info", **kwargs):
    """
    Log rápido de salida digital
    
    FUNCIÓN: Wrapper para log_digital_output del logger global
    USO: log_digital(1, 1, "Activa expulsor", level="info", motivo="defecto")
    """
    if production_logger:
        production_logger.log_digital_output(channel, value, message, level, **kwargs)

def save_snapshot(image: Any, now_ts: Optional[float] = None, level: str = "info"):
    """
    Guardar snapshot periódico si corresponde
    
    FUNCIÓN: Wrapper para save_periodic_snapshot del logger global
    USO: save_snapshot(frame_bgr)  # Guarda cada 30 min máximo
    """
    if production_logger:
        return production_logger.save_periodic_snapshot(image, now_ts, level)
    return None

def save_defect(image: Any, can_id: str, classification: str, confidence: float, extra_info: Optional[Dict[str, Any]] = None, level: str = "warning"):
    """
    Guardar imagen de defecto
    
    FUNCIÓN: Wrapper para save_defect_image del logger global
    USO: save_defect(frame_bgr, "LATA_123", "dent", 0.78, level="error")
    """
    if production_logger:
        return production_logger.save_defect_image(image, can_id, classification, confidence, extra_info, level)
    return None

def log_vision_event(can_id: str,
                     verdict: str,
                     camera_params: Dict[str, Any],
                     yolo_params: Dict[str, Any],
                     bbox: Optional[List[int]] = None,
                     image: Any = None,
                     level: str = "info"):
    """
    Log completo de visión por lata
    
    FUNCIÓN: Wrapper para log_vision_can del logger global
    USO: log_vision_event("LATA_001", "ok", cam_params, yolo_params, bbox=[100,120,200,220], image=frame)
    """
    if production_logger:
        production_logger.log_vision_can(can_id, verdict, camera_params, yolo_params, bbox, image, level)

if __name__ == "__main__":
    # Ejemplo de uso
    logger = initialize_production_logging(enable_console=True)
    
    # Ejemplos de logging
    logger.log_system_event("startup", "Sistema iniciado")
    logger.log_camera_event("info", "Cámara conectada", ip="172.20.2.151")
    logger.log_detection_event({
        'class': 'lata_ok',
        'confidence': 0.95,
        'bbox': [100, 100, 200, 200]
    })
    logger.log_production_event("quality_pass", "LATA_001")
    logger.update_frame_metrics(15.5, 1024.5)
    logger.log_performance_metrics()
    
    print("Sistema de logging configurado correctamente")
