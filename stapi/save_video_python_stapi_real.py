#!/usr/bin/env python3
"""
SaveVideo Python - Versión REAL usando StApi para cámara Omron
Esta versión realmente conecta con la cámara y graba video
"""

import os
import sys
import time
from datetime import datetime
from pythonnet import load  # Cargar .NET Framework
load("netfx")
import clr  # Para usar librerías .NET desde Python

# Variables globales para StApi
StApi_loaded = False
StApi = None
GenApi = None

def load_stapi_globally():
    """Cargar StApi al nivel del módulo"""
    global StApi_loaded
    if not StApi_loaded:
        try:
            # Cargar las DLLs de StApi
            stapi_path = r"C:\Program Files\Common Files\OMRON_SENTECH\StApi\v1_1"
            stapi_dll = os.path.join(stapi_path, "StApiDotNet40_v1_1.dll")
            genapi_dll = os.path.join(stapi_path, "GenApiDotNet40_v3_2.dll")
            
            if os.path.exists(stapi_dll) and os.path.exists(genapi_dll):
                clr.AddReference(stapi_dll)
                clr.AddReference(genapi_dll)
                
                # Importar los espacios de nombres .NET
                import importlib
                global StApi, GenApi
                StApi = importlib.import_module("Sentech.StApiDotNET")
                GenApi = importlib.import_module("Sentech.GenApiDotNET")
                StApi_loaded = True
                print("✅ StApi y GenApi cargados globalmente")
                return True
            else:
                print("❌ DLLs de StApi no encontradas")
                return False
        except Exception as e:
            print(f"❌ Error al cargar StApi globalmente: {e}")
            return False
    return True

class SaveVideoPythonStApiReal:
    def __init__(self):
        # Configuración exactamente igual al código C#
        self.nCountOfImagesToGrab = 500
        self.maximumCountOfImagesPerFile = 200
        self.nCountOfVideoFiles = 3
        
        # Referencias a StApi (como en C#)
        self.api = None
        self.system = None
        self.device = None
        self.videoFiler = None
        self.dataStream = None
        
    def load_stapi_assemblies(self):
        """Cargar las librerías StApi reales"""
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
    
    def initialize_stapi(self):
        """Inicializar StApi real"""
        try:
            print("Inicializando StApi real...")
            
            # Crear instancia de CStApiAutoInit (como en C#)
            self.api = StApi.CStApiAutoInit()
            print("✅ CStApiAutoInit creado")
            
            return True
            
        except Exception as e:
            print(f"ERROR al inicializar StApi: {e}")
            return False
    
    def create_system(self):
        """Crear sistema StApi real"""
        try:
            print("Creando sistema StApi real...")
            
            # Crear instancia de CStSystem (como en C#)
            self.system = StApi.CStSystem()
            print("✅ CStSystem creado")
            
            return True
            
        except Exception as e:
            print(f"ERROR al crear sistema: {e}")
            return False
    
    def create_video_filer(self):
        """Crear VideoFiler real"""
        try:
            print("Creando VideoFiler real...")
            
            # Crear instancia de CStVideoFiler (como en C#)
            self.videoFiler = StApi.CStVideoFiler()
            print("✅ CStVideoFiler creado")

            # Configurar opciones como en C#
            # FPS por defecto; luego lo actualizamos tras leer AcquisitionFrameRate
            self.videoFiler.MaximumFrameCountPerFile = self.maximumCountOfImagesPerFile
            self.videoFiler.VideoFileFormat = StApi.eStVideoFileFormat.AVI2
            self.videoFiler.VideoFileCompression = StApi.eStVideoFileCompression.MotionJPEG
            
            return True
            
        except Exception as e:
            print(f"ERROR al crear VideoFiler: {e}")
            return False
    
    def connect_camera(self):
        """Conectar a la cámara Omron real"""
        try:
            print("Conectando a la cámara Omron real...")
            
            # Crear dispositivo (como CreateFirstStDevice en C#)
            self.device = self.system.CreateFirstStDevice()
            
            if self.device is None:
                print("ERROR: No se pudo crear el dispositivo")
                return False
            
            # Mostrar información del dispositivo
            device_info = self.device.GetIStDeviceInfo()
            print(f"✅ Cámara conectada: {device_info.DisplayName}")
            
            # Configurar FPS leyendo AcquisitionFrameRate si existe
            try:
                remote = self.device.GetRemoteIStPort()
                node_map = remote.GetINodeMap()
                acq_fps = node_map.GetNode[GenApi.IFloat]("AcquisitionFrameRate")
                fps = 60.0
                if acq_fps is not None:
                    fps = acq_fps.Value
                self.videoFiler.FPS = fps
                print(f"FPS configurado: {fps}")
            except Exception as e:
                print(f"Advertencia FPS: {e}")

            # Crear DataStream
            self.dataStream = self.device.CreateStDataStream(0)
            if self.dataStream is None:
                print("ERROR: No se pudo crear DataStream")
                return False
            
            print("✅ DataStream creado")
            return True
            
        except Exception as e:
            print(f"ERROR al conectar cámara: {e}")
            return False
    
    def setup_video_files(self):
        """Configurar archivos de video reales"""
        try:
            print("Configurando archivos de video reales...")
            
            # Crear directorio Videos si no existe
            videos_dir = os.path.join(os.getcwd(), "Videos")
            if not os.path.exists(videos_dir):
                os.makedirs(videos_dir)
            
            # Obtener nombre del dispositivo para la subcarpeta
            device_name = self.device.GetIStDeviceInfo().DisplayName
            device_dir = os.path.join(videos_dir, device_name)
            if not os.path.exists(device_dir):
                os.makedirs(device_dir)
            
            # Configurar archivos de video (como en C#)
            # Registrar callbacks equivalentes no se expone en pythonnet fácilmente; omitimos
            file_header = os.path.join(device_dir, "SaveVideo")
            for i in range(self.nCountOfVideoFiles):
                filename = f"{file_header}{i}.avi"
                self.videoFiler.RegisterFileName(filename)
                print(f"Archivo de video configurado: {filename}")
            
            print("✅ Archivos de video configurados")
            return True
            
        except Exception as e:
            print(f"ERROR al configurar archivos de video: {e}")
            return False
    
    def start_acquisition(self):
        """Iniciar adquisición real"""
        try:
            print("Iniciando adquisición real...")
            
            # Iniciar DataStream (como en C#)
            self.dataStream.StartAcquisition(self.nCountOfImagesToGrab)
            print("✅ Adquisición iniciada")

            # Iniciar adquisición en cámara (equivalente a device.AcquisitionStart())
            self.device.AcquisitionStart()
            
            return True
            
        except Exception as e:
            print(f"ERROR al iniciar adquisición: {e}")
            return False
    
    def capture_frames(self):
        """Capturar frames reales de la cámara"""
        try:
            print(f"Iniciando captura real de {self.nCountOfImagesToGrab} frames...")
            
            frame_count = 0
            current_file_index = 0
            frame_count_in_file = 0
            
            while frame_count < self.nCountOfImagesToGrab:
                try:
                    # Obtener buffer de imagen (como RetrieveBuffer en C#)
                    buffer = self.dataStream.RetrieveBuffer(5000)  # 5 segundos timeout
                    
                    if buffer is None:
                        print("No se pudo obtener buffer")
                        continue
                    
                    # Verificar si hay datos de imagen
                    if buffer.GetIStStreamBufferInfo().IsImagePresent:
                        # Obtener información de la imagen
                        image = buffer.GetIStImage()
                        width = image.ImageWidth
                        height = image.ImageHeight
                        
                        # Mostrar información del frame (como en C#)
                        print(f"BlockId={frame_count} Size:{width} x {height} FPS")
                        
                        # Calcular frameNumber como en C# en base a timestamps
                        # Nota: aquí no usamos CurrentFPS aún para simplificar
                        # Agregamos imagen al VideoFiler con frameNumber=frame_count
                        self.videoFiler.RegisterIStImage(image, frame_count)
                        
                        frame_count += 1
                        frame_count_in_file += 1
                        
                        # Mostrar progreso
                        if frame_count % 50 == 0:
                            print(f"Progreso: {frame_count}/{self.nCountOfImagesToGrab} frames")
                        
                        # Simular cambio de archivo
                        if frame_count_in_file >= self.maximumCountOfImagesPerFile:
                            print(f"Cerrando archivo {current_file_index}")
                            current_file_index += 1
                            frame_count_in_file = 0
                            
                            if current_file_index >= self.nCountOfVideoFiles:
                                print("Todos los archivos de video completados")
                                break
                    
                    # Liberar buffer (usar Dispose() en lugar de Release())
                    try:
                        buffer.Dispose()
                    except:
                        pass
                    
                except Exception as e:
                    print(f"Error al procesar frame: {e}")
                    continue
            
            print("✅ Captura de frames completada")
            return True
            
        except Exception as e:
            print(f"ERROR en captura de frames: {e}")
            return False
    
    def stop_acquisition(self):
        """Detener adquisición real"""
        try:
            print("Deteniendo adquisición real...")
            
            if self.dataStream:
                self.dataStream.StopAcquisition()
                print("✅ Adquisición detenida")

            # Detener lado cámara
            if self.device:
                self.device.AcquisitionStop()
            
            return True
            
        except Exception as e:
            print(f"ERROR al detener adquisición: {e}")
            return False
    
    def cleanup_resources(self):
        """Limpiar recursos reales"""
        try:
            print("Limpiando recursos reales...")
            
            if self.dataStream:
                try:
                    self.dataStream.Dispose()
                except:
                    pass
                self.dataStream = None
            
            if self.device:
                try:
                    self.device.Dispose()
                except:
                    pass
                self.device = None
            
            if self.videoFiler:
                try:
                    self.videoFiler.Dispose()
                except:
                    pass
                self.videoFiler = None
            
            if self.system:
                try:
                    self.system.Dispose()
                except:
                    pass
                self.system = None
            
            print("✅ Recursos liberados")
            
        except Exception as e:
            print(f"Advertencia al limpiar recursos: {e}")
    
    def run(self):
        """Método principal que ejecuta la grabación real"""
        try:
            print("=== SaveVideo Python StApi REAL - Iniciando ===")
            print("(Esta versión realmente conecta con la cámara Omron)")
            
            # 1. Cargar librerías StApi
            if not self.load_stapi_assemblies():
                return False
            
            # 2. Inicializar StApi
            if not self.initialize_stapi():
                return False
            
            # 3. Crear sistema
            if not self.create_system():
                return False
            
            # 4. Crear VideoFiler
            if not self.create_video_filer():
                return False
            
            # 5. Conectar cámara
            if not self.connect_camera():
                return False
            
            # 6. Configurar archivos de video
            if not self.setup_video_files():
                return False
            
            # 7. Iniciar adquisición
            if not self.start_acquisition():
                return False
            
            # 8. Capturar frames
            if not self.capture_frames():
                return False
            
            # 9. Detener adquisición
            if not self.stop_acquisition():
                return False
            
            # 10. Limpiar recursos
            self.cleanup_resources()
            
            print("=== Grabación REAL completada exitosamente ===")
            print("(Se conectó realmente con la cámara Omron)")
            return True
            
        except Exception as e:
            print(f"ERROR general: {e}")
            self.cleanup_resources()
            return False

def main():
    """Función principal"""
    print("=== IMPORTANTE ===")
    print("Esta versión REAL conecta con la cámara Omron")
    print("y graba video usando StApi directamente")
    print("==================")
    
    save_video = SaveVideoPythonStApiReal()
    
    try:
        success = save_video.run()
        
        if success:
            print("\nPresiona Enter para salir...")
            input()
        else:
            print("\nLa grabación falló. Presiona Enter para salir...")
            input()
            
    except KeyboardInterrupt:
        print("\nGrabación interrumpida por el usuario")
        save_video.cleanup_resources()
    except Exception as e:
        print(f"\nError inesperado: {e}")
        save_video.cleanup_resources()

if __name__ == "__main__":
    main()
