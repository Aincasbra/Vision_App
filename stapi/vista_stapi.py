#!/usr/bin/env python3
"""
Vista en Directo Ventana - Cámara Omron usando StApi
Versión con VENTANA GRÁFICA para ver la imagen en tiempo real

DESCRIPCIÓN GENERAL:
Este script conecta a una cámara Omron GigE Vision usando la API StApi de Sentech
y muestra la imagen en tiempo real en una ventana OpenCV. Incluye:
- Detección automática del patrón Bayer para colores correctos
- Balance de blancos automático
- Configuración optimizada de GigE Vision
- Procesamiento de imagen en tiempo real
- Interfaz gráfica con información del frame

FLUJO DE FUNCIONAMIENTO:
1. Cargar librerías StApi (.NET assemblies)
2. Inicializar sistema StApi
3. Conectar a la cámara y configurar parámetros
4. Iniciar adquisición continua de frames
5. Procesar cada frame y mostrarlo en ventana OpenCV
6. Limpiar recursos al terminar

NOTA: Este script NO graba video, solo muestra en tiempo real.
Para grabar, usar save_video_python_stapi_real.py
"""

import os
import time
import cv2
import numpy as np
from pythonnet import load
load("netfx")  # Cargar .NET Framework para Python
import clr

def load_stapi_globally():
    """
    CARGAR LIBRERÍAS STAPI AL NIVEL DEL MÓDULO
    
    Esta función carga las DLLs de StApi y GenApi de Sentech que son necesarias
    para comunicarse con las cámaras Omron. Las DLLs se cargan una sola vez
    al nivel del módulo para evitar recargas múltiples.
    
    RUTA DE LAS DLLs:
    - StApiDotNet40_v1_1.dll: API principal para control de cámaras
    - GenApiDotNet40_v3_2.dll: API para acceso a parámetros de cámara
    
    RETORNA:
    - True: Si las librerías se cargaron correctamente
    - False: Si hubo algún error (DLLs no encontradas, etc.)
    """
    try:
        # Ruta donde están instaladas las DLLs de StApi
        stapi_path = r"C:\Program Files\Common Files\OMRON_SENTECH\StApi\v1_1"
        stapi_dll = os.path.join(stapi_path, "StApiDotNet40_v1_1.dll")
        genapi_dll = os.path.join(stapi_path, "GenApiDotNet40_v3_2.dll")
        
        # Verificar que las DLLs existan
        if os.path.exists(stapi_dll) and os.path.exists(genapi_dll):
            # Cargar las DLLs en el runtime de .NET
            clr.AddReference(stapi_dll)
            clr.AddReference(genapi_dll)
            
            # Importar los módulos de Python.NET
            import importlib
            global StApi, GenApi
            StApi = importlib.import_module("Sentech.StApiDotNET")
            GenApi = importlib.import_module("Sentech.GenApiDotNET")
            print("✅ StApi y GenApi cargados globalmente")
            return True
        else:
            print("❌ DLLs de StApi no encontradas")
            return False
    except Exception as e:
        print(f"❌ Error al cargar StApi globalmente: {e}")
        return False

class VistaDirectoVentana:
    """
    CLASE PRINCIPAL PARA VISTA EN DIRECTO
    
    Esta clase maneja toda la funcionalidad de conexión a la cámara,
    adquisición de frames y visualización en tiempo real.
    
    ARQUITECTURA:
    - load_stapi_assemblies(): Carga las librerías StApi
    - create_system(): Inicializa el sistema StApi
    - connect_camera(): Conecta y configura la cámara
    - start_acquisition(): Inicia la captura de frames
    - show_live_view_window(): Muestra frames en ventana OpenCV
    - stop_acquisition(): Detiene la captura
    - cleanup_resources(): Libera recursos de memoria
    """
    
    def __init__(self):
        """
        INICIALIZACIÓN DE LA CLASE
        
        Configura los parámetros iniciales y crea la ventana OpenCV
        donde se mostrará la imagen en tiempo real.
        """
        # Configuración para vista en directo CONTINUA (sin límite de frames)
        self.nCountOfImagesToGrab = 1000000  # Número muy alto para evitar límite
        
        # Referencias a objetos StApi (como en C#)
        self.api = None          # CStApiAutoInit - inicialización del sistema
        self.system = None       # CStSystem - sistema principal de StApi
        self.device = None       # IStDevice - dispositivo cámara
        self.dataStream = None   # IStDataStream - flujo de datos de la cámara
        self.running = False     # Flag para controlar el bucle de captura
        self.frame_count = 0     # Contador de frames procesados
        
        # Estado de la interfaz gráfica
        self.acquisition_active = False
        self.button_clicked = None
        
        # Configuración de ventana OpenCV para mostrar la imagen
        self.window_name = "Vista en Directo - Cámara Omron"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)  # Ventana redimensionable
        cv2.resizeWindow(self.window_name, 1280, 720)         # Tamaño para toolbar
        
        # Configurar callback del mouse para botones
        cv2.setMouseCallback(self.window_name, self.handle_mouse_click)
        
        # Crear elementos de la interfaz
        self.toolbar = self.create_toolbar()
        self.stopped_view = self.create_stopped_view()
    
    def create_toolbar(self):
        """Crea la barra de herramientas superior con botones"""
        # Crear barra gris superior (1280x60)
        toolbar = np.full((60, 1280, 3), (64, 64, 64), dtype=np.uint8)
        
        # Título de la aplicación
        cv2.putText(toolbar, "OMRON STC-MCS163POE - StApi", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Botones en la parte derecha de la barra
        # Botón RUN (verde)
        cv2.rectangle(toolbar, (800, 15), (900, 45), (0, 255, 0), -1)
        cv2.rectangle(toolbar, (800, 15), (900, 45), (255, 255, 255), 2)
        cv2.putText(toolbar, "RUN", (830, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        
        # Botón STOP (rojo)
        cv2.rectangle(toolbar, (920, 15), (1020, 45), (0, 0, 255), -1)
        cv2.rectangle(toolbar, (920, 15), (1020, 45), (255, 255, 255), 2)
        cv2.putText(toolbar, "STOP", (950, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Botón EXIT (gris oscuro)
        cv2.rectangle(toolbar, (1040, 15), (1140, 45), (48, 48, 48), -1)
        cv2.rectangle(toolbar, (1040, 15), (1140, 45), (255, 255, 255), 2)
        cv2.putText(toolbar, "EXIT", (1070, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        
        return toolbar
    
    def create_stopped_view(self):
        """Crea la vista cuando la cámara está detenida"""
        # Crear imagen gris para área principal (1280x660, debajo de la toolbar)
        img = np.full((660, 1280, 3), (128, 128, 128), dtype=np.uint8)
        
        # Título centrado
        cv2.putText(img, "CAMARA DETENIDA", (500, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(img, "StApi + Hardware Color", (450, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2, cv2.LINE_AA)
        
        # Instrucciones
        cv2.putText(img, "Haz clic en RUN para iniciar la camara", (400, 400),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
        cv2.putText(img, "STOP para pausar, EXIT para salir", (400, 430),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
        
        return img
    
    def combine_toolbar_and_image(self, toolbar, image):
        """Combina la barra de herramientas con la imagen principal"""
        # Combinar toolbar (60x1280) + imagen (660x1280) = (720x1280)
        combined = np.vstack([toolbar, image])
        return combined
    
    def handle_mouse_click(self, event, x, y, flags, param):
        """Maneja los clics del mouse en los botones de la toolbar"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Botón RUN (verde) - coordenadas ajustadas para la toolbar
            if 800 <= x <= 900 and 15 <= y <= 45:
                if not self.acquisition_active:
                    print("🚀 Botón RUN presionado - Iniciando cámara...")
                    self.button_clicked = "RUN"
            
            # Botón STOP (rojo) - coordenadas ajustadas para la toolbar
            elif 920 <= x <= 1020 and 15 <= y <= 45:
                if self.acquisition_active:
                    print("⏹️ Botón STOP presionado - Pausando cámara...")
                    self.button_clicked = "STOP"
            
            # Botón EXIT (gris) - coordenadas ajustadas para la toolbar
            elif 1040 <= x <= 1140 and 15 <= y <= 45:
                print("🚪 Botón EXIT presionado - Cerrando aplicación...")
                self.button_clicked = "EXIT"
        
    def load_stapi_assemblies(self):
        """
        CARGAR LAS LIBRERÍAS STAPI REALES
        
        Verifica que las librerías StApi estén disponibles y cargadas
        antes de proceder con la inicialización del sistema.
        
        RETORNA:
        - True: Si las librerías están cargadas correctamente
        - False: Si hay algún problema con la carga
        """
        try:
            print("Cargando librerías StApi reales...")
            
            # Llamar a la función global primero
            if not load_stapi_globally():
                return False
            
            # Verificar que StApi esté cargado
            if StApi is None:
                print("ERROR: StApi no se cargó correctamente")
                return False
            
            print("✅ Librerías StApi cargadas")
            return True
            
        except Exception as e:
            print(f"ERROR al cargar librerías StApi: {e}")
            return False
    
    def create_system(self):
        """
        CREAR SISTEMA STAPI REAL
        
        Inicializa el sistema StApi creando las instancias necesarias:
        1. CStApiAutoInit: Inicializa el runtime de StApi (OBLIGATORIO)
        2. CStSystem: Crea el sistema principal para manejar dispositivos
        
        Esta función debe ejecutarse ANTES de intentar conectar a cualquier cámara.
        
        RETORNA:
        - True: Si el sistema se creó correctamente
        - False: Si hubo algún error en la inicialización
        """
        try:
            print("Creando sistema StApi real...")
            
            # Inicializar StApi primero (OBLIGATORIO - sin esto falla todo)
            self.api = StApi.CStApiAutoInit()
            print("✅ CStApiAutoInit creado")
            
            # Crear instancia de CStSystem (como en C#)
            self.system = StApi.CStSystem()
            print("✅ CStSystem creado")
            
            return True
            
        except Exception as e:
            print(f"ERROR al crear sistema: {e}")
            return False
    
    def connect_camera(self):
        """
        CONECTAR A LA CÁMARA OMRON REAL
        
        Esta función realiza la conexión física y lógica a la cámara:
        1. Busca la primera cámara disponible en el sistema
        2. Obtiene información del dispositivo
        3. Configura parámetros de GigE Vision para optimizar la transmisión
        4. Detecta automáticamente el patrón Bayer para colores correctos
        5. Activa el balance de blancos automático
        6. Crea el DataStream para recibir frames
        
        CONFIGURACIÓN GIGE VISION:
        - TriggerMode: Off (captura continua)
        - AcquisitionMode: Continuous (modo continuo)
        - FPS: 15 (moderado para evitar timeouts)
        - PacketSize: 1440 (optimizado para MTU 1500)
        - InterPacketDelay: 100 (evita picos de red)
        - ThroughputLimit: Off (sin límite de ancho de banda)
        
        RETORNA:
        - True: Si la cámara se conectó y configuró correctamente
        - False: Si hubo algún error en la conexión
        """
        try:
            print("Conectando a la cámara Omron real...")

            # Crear dispositivo - busca la primera cámara disponible
            self.device = self.system.CreateFirstStDevice()
            if self.device is None:
                print("ERROR: No se pudo crear el dispositivo")
                return False

            # Obtener información del dispositivo para confirmar conexión
            info = self.device.GetIStDeviceInfo()
            print(f"✅ Cámara conectada: {info.DisplayName}")

            # Obtener NodeMap remoto para configurar parámetros
            remote = self.device.GetRemoteIStPort()
            nm = remote.GetINodeMap()

            # DETECTAR PATRÓN BAYER AUTOMÁTICAMENTE
            # El patrón Bayer determina cómo se organizan los píxeles de color
            # en el sensor. Conocerlo es CRÍTICO para colores correctos.
            def get_bayer_pattern(nm):
                # Algunos modelos exponen "BayerPattern", otros "PixelColorFilter"
                names = ["BayerPattern", "PixelColorFilter"]
                for n in names:
                    try:
                        enum = nm.GetNode[GenApi.IEnumeration](n)
                        return enum.GetCurrentEntry().Symbolic  # "RG", "BG", "GR" o "GB"
                    except:
                        pass
                return None

            bayer = get_bayer_pattern(nm)
            print("BayerPattern:", bayer)
            self.bayer_pattern = bayer  # Guardar para uso posterior

            # ACTIVAR BALANCE DE BLANCOS AUTOMÁTICO
            # Esto corrige automáticamente el tinte de color de la imagen
            try:
                nm.GetNode[GenApi.IEnumeration]("BalanceWhiteAuto").FromString("Once")
            except:
                pass  # Algunas cámaras no tienen esta función

            # HELPERS SEGUROS PARA CONFIGURAR NODOS
            # Estas funciones manejan casos donde algunos nodos no existen
            def set_enum(name, value):
                try: nm.GetNode[GenApi.IEnumeration](name).FromString(value)
                except: pass

            def set_bool(name, value):
                try: nm.GetNode[GenApi.IBoolean](name).Value = value
                except: pass

            def set_float(name, value):
                try: nm.GetNode[GenApi.IFloat](name).Value = float(value)
                except: pass

            def set_int(name, value):
                try: nm.GetNode[GenApi.IInteger](name).Value = int(value)
                except: pass

            # CONFIGURACIÓN BÁSICA DE LA CÁMARA
            set_enum("TriggerMode", "Off")           # Sin trigger - captura continua
            set_enum("AcquisitionMode", "Continuous") # Modo continuo
            set_bool("AcquisitionFrameRateEnable", True)  # Habilitar control de FPS
            set_float("AcquisitionFrameRate", 15.0)      # 15 FPS (moderado)
            set_int("GevSCPSPacketSize", 1440)           # Tamaño de paquete optimizado
            set_int("GevSCPD", 100)                      # Delay entre paquetes
            set_enum("DeviceLinkThroughputLimitMode", "Off") # Sin límite de throughput

            # Reenvío de paquetes si existe (mejora la robustez de GigE)
            try: set_enum("GevGVSPPacketResendMode", "Enabled")
            except: pass

            print("✅ Configuración GigE Vision aplicada")

            # Crear DataStream - canal de comunicación para recibir frames
            self.dataStream = self.device.CreateStDataStream(0)
            if self.dataStream is None:
                print("ERROR: No se pudo crear DataStream")
                return False

            print("✅ DataStream creado")
            return True

        except Exception as e:
            print(f"ERROR al conectar cámara: {e}")
            return False
    
    def start_acquisition(self):
        """
        INICIAR ADQUISICIÓN REAL
        
        Inicia el proceso de captura de frames desde la cámara:
        1. Inicia el DataStream para recibir frames
        2. Activa la adquisición en la cámara física
        
        IMPORTANTE: Esta función debe ejecutarse ANTES de intentar
        obtener frames con RetrieveBuffer().
        
        RETORNA:
        - True: Si la adquisición se inició correctamente
        - False: Si hubo algún error al iniciar
        """
        try:
            print("Iniciando adquisición real...")
            
            # Iniciar DataStream - comienza a recibir frames
            self.dataStream.StartAcquisition(self.nCountOfImagesToGrab)
            print("✅ Adquisición iniciada")

            # Iniciar adquisición en cámara (equivalente a device.AcquisitionStart())
            self.device.AcquisitionStart()
            
            return True
            
        except Exception as e:
            print(f"ERROR al inicializar StApi: {e}")
            return False
    
    def start_camera_acquisition(self):
        """Inicia todo el proceso de la cámara (conexión + adquisición)"""
        try:
            # Si ya hay una cámara activa, limpiarla completamente primero
            if self.acquisition_active:
                self.cleanup_camera_resources()
            
            # 1. Cargar librerías StApi
            if not self.load_stapi_assemblies():
                return False
            
            # 2. Crear sistema
            if not self.create_system():
                return False
            
            # 3. Conectar cámara
            if not self.connect_camera():
                return False
            
            # 4. Iniciar adquisición
            if not self.start_acquisition():
                return False
            
            # Marcar como activa ANTES de lanzar el hilo para evitar carreras
            self.acquisition_active = True
            
            # 5. Iniciar bucle de captura en hilo separado
            import threading
            self.capture_thread = threading.Thread(target=self.capture_loop)
            self.capture_thread.daemon = True
            self.capture_thread.start()
            
            return True
            
        except Exception as e:
            print(f"❌ Error iniciando cámara: {e}")
            self.acquisition_active = False
            return False
    
    def cleanup_camera_resources(self):
        """Limpia completamente los recursos de la cámara para poder reiniciar"""
        try:
            print("🧹 Limpiando recursos de cámara para reinicio...")
            
            # Parar el hilo de captura
            if hasattr(self, 'capture_thread') and self.capture_thread:
                self.acquisition_active = False
                if self.capture_thread.is_alive():
                    self.capture_thread.join(timeout=1.0)
            
            # Parar adquisición
            if self.dataStream:
                try:
                    self.dataStream.StopAcquisition()
                except:
                    pass
            
            if self.device:
                try:
                    self.device.AcquisitionStop()
                except:
                    pass
            
            # Liberar recursos
            if self.dataStream:
                try:
                    self.dataStream.Dispose()
                    self.dataStream = None
                except:
                    pass
            
            if self.device:
                try:
                    self.device.Dispose()
                    self.device = None
                except:
                    pass
            
            if self.system:
                try:
                    self.system.Dispose()
                    self.system = None
                except:
                    pass
            
            print("✅ Recursos de cámara limpiados")
            
        except Exception as e:
            print(f"⚠️ Error limpiando recursos: {e}")
    
    def capture_loop(self):
        """Bucle de captura que se ejecuta mientras acquisition_active es True"""
        try:
            print("🔄 Iniciando bucle de captura...")
            
            while self.acquisition_active and self.running:
                try:
                    # Obtener buffer con timeout corto para baja latencia
                    buffer = self.dataStream.RetrieveBuffer(60)
                    
                    if buffer is None:
                        continue
                    
                    # Verificar si hay imagen presente
                    if buffer.GetIStStreamBufferInfo().IsImagePresent:
                        # Obtener información de la imagen
                        image = buffer.GetIStImage()
                        width = image.ImageWidth
                        height = image.ImageHeight
                        
                        # Procesar imagen para mostrar
                        frame = self.process_frame_for_display(image)
                        
                        if frame is not None:
                            # Mostrar frame con toolbar
                            self.display_frame_with_toolbar(frame, width, height)
                        
                        self.frame_count += 1
                    
                    # Liberar buffer
                    try:
                        buffer.Dispose()
                    except:
                        pass
                        
                except Exception as e:
                    # Silenciar errores comunes tras detener
                    if not self.acquisition_active:
                        break
                    continue
            
            print("✅ Bucle de captura terminado")
            
        except Exception as e:
            print(f"❌ Error en bucle de captura: {e}")
    
    def process_frame_for_display(self, image):
        """Procesa frame para mostrar en la interfaz"""
        try:
            # Obtener datos de imagen
            img_bytes = image.GetByteArray()
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            
            width = image.ImageWidth
            height = image.ImageHeight
            pixel_format = str(image.ImagePixelFormat)
            
            # Procesar según formato
            if "BayerRG8" in pixel_format:
                # Debayering
                frame = img_array.reshape((height, width))
                code_map = {
                    "RG": cv2.COLOR_BayerRG2BGR,
                    "BG": cv2.COLOR_BayerBG2BGR,
                    "GR": cv2.COLOR_BayerGR2BGR,
                    "GB": cv2.COLOR_BayerGB2BGR,
                }
                cv_code = code_map.get(self.bayer_pattern or "BG", cv2.COLOR_BayerBG2BGR)
                frame = cv2.cvtColor(frame, cv_code)
            elif "Mono8" in pixel_format:
                frame = img_array.reshape((height, width))
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                cv2.putText(frame, f"Formato no soportado: {pixel_format}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            return frame
            
        except Exception as e:
            print(f"❌ Error procesando frame: {e}")
            return None
    
    def display_frame_with_toolbar(self, frame, width, height):
        """Muestra frame con toolbar en la interfaz"""
        try:
            # Redimensionar frame para que quepa debajo de la toolbar
            frame_resized = cv2.resize(frame, (1280, 660))
            
            # Información en pantalla (ligera para no penalizar)
            cv2.putText(frame_resized, f"{width}x{height}", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Combinar toolbar + imagen de la cámara
            camera_view = self.combine_toolbar_and_image(self.toolbar, frame_resized)
            cv2.imshow(self.window_name, camera_view)
            
        except Exception as e:
            pass
    
    def show_live_view_window(self):
        """
        MOSTRAR VISTA EN DIRECTO EN VENTANA OPENCV
        
        Este es el método principal que ejecuta el bucle de captura:
        1. Obtiene frames de la cámara usando RetrieveBuffer()
        2. Procesa cada frame según su formato de píxel
        3. Aplica debayering si es necesario para colores correctos
        4. Muestra la imagen en la ventana OpenCV
        5. Maneja la salida del usuario (tecla 'q')
        
        PROCESAMIENTO DE IMAGEN:
        - BayerRG8: Aplica debayering con OpenCV usando el patrón detectado
        - Mono8: Convierte a escala de grises
        - RGB8/BGR8: Muestra directamente en color
        - Otros: Intenta conversión a escala de grises
        
        DEBAYERING:
        El debayering convierte la imagen Bayer (patrón de píxeles de color)
        en una imagen RGB completa. El patrón correcto es CRÍTICO para
        colores naturales (sin tinte azulado).
        
        BUCLE PRINCIPAL:
        - Captura frames continuamente hasta que el usuario presione 'q'
        - Maneja timeouts de red (5 segundos por frame)
        - Libera buffers correctamente para evitar memory leaks
        - Muestra información del frame en tiempo real
        """
        try:
            print("=== Iniciando Vista en Directo VENTANA ===")
            print("Presiona 'q' para salir")
            print("Esperando frames de la cámara...")
            
            self.running = True
            timeout_count = 0  # Contador de timeouts consecutivos
            
            # BUCLE PRINCIPAL DE CAPTURA
            while self.running:
                try:
                    # Obtener buffer de imagen con timeout de 5 segundos
                    buffer = self.dataStream.RetrieveBuffer(5000)
                    
                    # Manejar caso de timeout (no hay frame disponible)
                    if buffer is None:
                        print("No se pudo obtener buffer")
                        timeout_count += 1
                        if timeout_count > 10:  # Máximo 10 timeouts consecutivos
                            print("Demasiados timeouts, saliendo...")
                            break
                        continue
                    
                    timeout_count = 0  # Reset timeout counter
                    
                    # Verificar si el buffer contiene una imagen válida
                    if buffer.GetIStStreamBufferInfo().IsImagePresent:
                        # Obtener información de la imagen
                        image = buffer.GetIStImage()
                        width = image.ImageWidth
                        height = image.ImageHeight
                        
                        # PROCESAR IMAGEN SEGÚN SU FORMATO
                        try:
                            # Obtener datos de la imagen como array de bytes
                            if hasattr(image, 'GetByteArray'):
                                image_data = image.GetByteArray()
                                if image_data is not None:
                                    # Convertir bytes a numpy array para OpenCV
                                    img_array = np.frombuffer(image_data, dtype=np.uint8)
                                    pixel_format = str(image.ImagePixelFormat)
                                    
                                    # PROCESAR SEGÚN FORMATO DE PÍXEL
                                    if "BayerRG8" in pixel_format:
                                        # FORMATO BAYER: APLICAR DEBAYERING
                                        # Reshape a matriz 2D (height x width)
                                        frame = img_array.reshape((height, width))
                                        
                                        # MAPEO DEL PATRÓN BAYER A OPENCV
                                        # Cada patrón requiere un código de conversión diferente
                                        code_map = {
                                            "RG": cv2.COLOR_BayerRG2BGR,  # Red-Green
                                            "BG": cv2.COLOR_BayerBG2BGR,  # Blue-Green
                                            "GR": cv2.COLOR_BayerGR2BGR,  # Green-Red
                                            "GB": cv2.COLOR_BayerGB2BGR,  # Green-Blue
                                        }
                                        
                                        # Usar el patrón detectado o BG por defecto
                                        cv_code = code_map.get(self.bayer_pattern or "BG", cv2.COLOR_BayerBG2BGR)
                                        frame = cv2.cvtColor(frame, cv_code)
                                        print(f"✅ Debayering OpenCV con patrón {self.bayer_pattern or 'BG'}")
                                        
                                    elif "Mono8" in pixel_format:
                                        # FORMATO MONOCROMÁTICO: ESCALA DE GRISES
                                        frame = img_array.reshape((height, width))
                                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                                        print("✅ Imagen Mono8 mostrada en escala de grises")
                                        
                                    elif "RGB8" in pixel_format or "BGR8" in pixel_format:
                                        # FORMATO COLOR: MOSTRAR DIRECTAMENTE
                                        frame = img_array.reshape((height, width, 3))
                                        if "RGB8" in pixel_format:
                                            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                                        print("✅ Imagen RGB/BGR mostrada en color")
                                        
                                    else:
                                        # FORMATO DESCONOCIDO: INTENTAR ESCALA DE GRISES
                                        if len(img_array) == width * height:
                                            frame = img_array.reshape((height, width))
                                            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                                        else:
                                            frame = np.zeros((height, width, 3), dtype=np.uint8)
                                        print(f"⚠️ Formato {pixel_format} mostrado en escala de grises")
                                    
                                    # AÑADIR INFORMACIÓN AL FRAME
                                    # Texto verde sobre la imagen con información en tiempo real
                                    cv2.putText(frame, f"Frame: {self.frame_count}", (10, 30), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                    cv2.putText(frame, f"Size: {width}x{height}", (10, 60), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                    cv2.putText(frame, f"Format: {pixel_format}", (10, 90), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                    cv2.putText(frame, f"Time: {time.strftime('%H:%M:%S')}", (10, 120), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                    
                                    # Mostrar frame en la ventana OpenCV
                                    cv2.imshow(self.window_name, frame)
                                else:
                                    print("❌ Datos de imagen son None")
                            else:
                                print("❌ No se encontró método GetByteArray")
                        except Exception as e:
                            print(f"❌ Error al mostrar imagen: {e}")
                            # Crear imagen de error para mostrar en ventana
                            frame = np.zeros((height, width, 3), dtype=np.uint8)
                            cv2.putText(frame, f"Error: {e}", (10, 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            cv2.imshow(self.window_name, frame)
                        
                        # VERIFICAR TECLA DE SALIDA
                        # OpenCV debe procesar eventos de ventana
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                        
                        self.frame_count += 1
                        
                        # Mostrar progreso cada 50 frames
                        if self.frame_count % 50 == 0:
                            print(f"Progreso: {self.frame_count} frames")
                    
                    # LIBERAR BUFFER CORRECTAMENTE
                    # Usar Dispose() como en el script que funciona
                    # Esto es CRÍTICO para evitar memory leaks y timeouts
                    try:
                        buffer.Dispose()
                    except:
                        pass
                    
                except Exception as e:
                    print(f"Error al procesar frame: {e}")
                    continue
            
            print("✅ Vista en directo terminada")
            
        except Exception as e:
            print(f"ERROR en vista en directo: {e}")
        finally:
            cv2.destroyAllWindows()  # Cerrar todas las ventanas OpenCV
    
    def stop_acquisition(self):
        """Detiene la adquisición y limpia recursos para poder reiniciar"""
        try:
            print("⏹️ Deteniendo adquisición...")
            
            # Marcar como inactiva para que el hilo se detenga
            self.acquisition_active = False
            
            # Esperar a que el hilo termine
            if hasattr(self, 'capture_thread') and self.capture_thread:
                if self.capture_thread.is_alive():
                    self.capture_thread.join(timeout=2.0)
            
            # Limpiar recursos de cámara
            self.cleanup_camera_resources()
            
            print("✅ Adquisición detenida y recursos limpiados")
            return True
            
        except Exception as e:
            print(f"❌ Error deteniendo adquisición: {e}")
            return False
    
    def cleanup_resources(self):
        """
        LIMPIAR RECURSOS REALES
        
        Libera todos los recursos de memoria y objetos StApi:
        1. DataStream: Canal de comunicación con la cámara
        2. Device: Objeto de la cámara
        3. System: Sistema principal de StApi
        
        IMPORTANTE: Siempre llamar esta función al terminar para
        evitar memory leaks y liberar la cámara para otros usos.
        """
        try:
            print("Limpiando recursos reales...")
            
            # Usar el método de limpieza completo
            self.cleanup_camera_resources()
            
            # Limpiar también el API si existe
            if self.api:
                try:
                    self.api.Dispose()
                except:
                    pass
                self.api = None
            
            print("✅ Recursos liberados")
            
        except Exception as e:
            print(f"Advertencia al limpiar recursos: {e}")
    
    def run(self):
        """
        MÉTODO PRINCIPAL QUE EJECUTA LA VISTA EN DIRECTO REAL
        
        Este método coordina todo el flujo de trabajo:
        1. Carga las librerías StApi
        2. Inicializa el sistema
        3. Conecta a la cámara
        4. Inicia la adquisición
        5. Muestra la vista en directo
        6. Limpia recursos al terminar
        
        FLUJO COMPLETO:
        - load_stapi_assemblies(): Carga DLLs de StApi
        - create_system(): Inicializa runtime de StApi
        - connect_camera(): Conecta y configura cámara
        - start_acquisition(): Inicia captura de frames
        - show_live_view_window(): Bucle principal de visualización
        - stop_acquisition(): Para la captura
        - cleanup_resources(): Libera memoria
        
        RETORNA:
        - True: Si todo el proceso se ejecutó correctamente
        - False: Si hubo algún error en cualquier paso
        
        NOTA: Los recursos se limpian automáticamente en el bloque finally
        independientemente de si hubo éxito o error.
        """
        try:
            print("=== Vista en Directo Python StApi REAL - INTERFAZ GRÁFICA ===")
            print("Controles:")
            print("- RUN: Inicia la cámara")
            print("- STOP: Para la cámara")
            print("- EXIT: Cierra la aplicación")
            print("================================================")
            
            # Mostrar GUI inicial (no recrear en cada iteración)
            gui_img = self.combine_toolbar_and_image(self.toolbar, self.stopped_view)
            cv2.imshow(self.window_name, gui_img)
            
            self.running = True
            
            # Bucle principal de la interfaz
            while self.running:
                if not self.acquisition_active:
                    # Mantener GUI estática mientras esté detenido
                    cv2.imshow(self.window_name, gui_img)
                
                # Procesar botones y teclas (latencia mínima)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('1') and not self.acquisition_active:  # RUN
                    self.button_clicked = "RUN"
                elif key == ord('2') and self.acquisition_active:      # STOP
                    self.button_clicked = "STOP"
                elif key == ord('3'):
                    break
                
                # Manejar clics de botones
                if self.button_clicked == "RUN" and not self.acquisition_active:
                    if self.start_camera_acquisition():
                        # Nada más; el hilo de captura actualiza la ventana
                        pass
                    self.button_clicked = None
                elif self.button_clicked == "STOP" and self.acquisition_active:
                    self.stop_acquisition()
                    self.acquisition_active = False
                    self.button_clicked = None
                elif self.button_clicked == "EXIT":
                    print("🚪 Cerrando aplicación desde botón EXIT...")
                    self.running = False
                    # Cerrar ventana OpenCV explícitamente
                    cv2.destroyAllWindows()
                    break
            
            return True
        
        except Exception as e:
            print(f"ERROR general: {e}")
            return False
        
        finally:
            # LIMPIEZA AUTOMÁTICA DE RECURSOS
            # Este bloque se ejecuta SIEMPRE, independientemente de errores
            self.stop_acquisition()
            self.cleanup_resources()
            # Cerrar ventana OpenCV
            cv2.destroyAllWindows()

def main():
    """
    FUNCIÓN PRINCIPAL
    
    Punto de entrada del programa que:
    1. Crea la instancia de VistaDirectoVentana
    2. Ejecuta la vista en directo
    3. Maneja interrupciones del usuario (Ctrl+C)
    4. Muestra mensajes de éxito/error
    5. Espera input del usuario antes de cerrar
    
    MANEJO DE ERRORES:
    - KeyboardInterrupt: Usuario presionó Ctrl+C
    - Exception: Cualquier otro error inesperado
    - En ambos casos se limpian los recursos automáticamente
    """
    print("=== Vista en Directo - Cámara Omron ===")
    print("Versión con ventana gráfica")
    print("=====================================")
    
    vista = VistaDirectoVentana()
    
    try:
        # Ejecutar la vista en directo
        success = vista.run()
        
        if success:
            print("\nPresiona Enter para salir...")
            input()
        else:
            print("\nLa vista en directo falló. Presiona Enter para salir...")
            input()
            
    except KeyboardInterrupt:
        print("\nVista en directo interrumpida por el usuario")
        vista.cleanup_resources()
    except Exception as e:
        print(f"\nError inesperado: {e}")
        vista.cleanup_resources()

if __name__ == "__main__":
    main()
