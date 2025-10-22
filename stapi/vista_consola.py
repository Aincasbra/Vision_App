#!/usr/bin/env python3
"""
Vista en Directo Consola - Cámara Omron usando StApi
Versión CONSOLA para verificar funcionamiento
"""

import os
import sys
import time
from datetime import datetime
from pythonnet import load
load("netfx")
import clr

# Variables globales para StApi
StApi_loaded = False
StApi = None
GenApi = None

def load_stapi_globally():
    """Cargar StApi al nivel del módulo"""
    global StApi_loaded
    if not StApi_loaded:
        try:
            stapi_path = r"C:\Program Files\Common Files\OMRON_SENTECH\StApi\v1_1"
            stapi_dll = os.path.join(stapi_path, "StApiDotNet40_v1_1.dll")
            genapi_dll = os.path.join(stapi_path, "GenApiDotNet40_v3_2.dll")
            
            if os.path.exists(stapi_dll) and os.path.exists(genapi_dll):
                clr.AddReference(stapi_dll)
                clr.AddReference(genapi_dll)
                
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

class VistaDirectoConsola:
    def __init__(self):
        # Configuración para vista en directo CONTINUA
        self.nCountOfImagesToGrab = 1000000  # Número muy alto para evitar límite
        
        # Referencias a StApi (como en C#)
        self.api = None
        self.system = None
        self.device = None
        self.dataStream = None
        self.running = False
        self.frame_count = 0
        
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
    
    def connect_camera(self):
        """Conectar a la cámara Omron real"""
        try:
            print("Conectando a la cámara Omron real...")

            # Crear dispositivo
            self.device = self.system.CreateFirstStDevice()
            if self.device is None:
                print("ERROR: No se pudo crear el dispositivo")
                return False

            # Info
            info = self.device.GetIStDeviceInfo()
            print(f"✅ Cámara conectada: {info.DisplayName}")

            # NodeMap remoto
            remote = self.device.GetRemoteIStPort()
            nm = remote.GetINodeMap()   # <-- ESTA LÍNEA FALTABA

            # Helpers seguros para tocar nodos (algunas cámaras no tienen todos)
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

            # Configuración básica: free-run continuo
            set_enum("TriggerMode", "Off")
            set_enum("AcquisitionMode", "Continuous")

            # FPS moderado para evitar timeouts
            set_bool("AcquisitionFrameRateEnable", True)
            set_float("AcquisitionFrameRate", 15.0)

            # Paquetes GigE (tamaño/espaciado)
            set_int("GevSCPSPacketSize", 1440)  # MTU 1500 → payload 1440
            set_int("GevSCPD", 100)             # inter-packet delay

            # Throughput (desactivado por defecto)
            set_enum("DeviceLinkThroughputLimitMode", "Off")
            # Si lo prefieres activado:
            # set_enum("DeviceLinkThroughputLimitMode", "On")
            # set_int("DeviceLinkThroughputLimit", 800_000_000)

            # Reenvío de paquetes si existe
            try: set_enum("GevGVSPPacketResendMode", "Enabled")
            except: pass

            print("✅ Configuración GigE Vision aplicada")

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
            print(f"ERROR al inicializar StApi: {e}")
            return False
    
    def show_live_view_console(self):
        """Mostrar vista en directo en consola"""
        try:
            print("=== Iniciando Vista en Directo CONSOLA ===")
            print("Presiona Ctrl+C para salir")
            print("Esperando frames de la cámara...")
            
            # --- antes del bucle, INTENTA poner free-run (opcional)
            try:
                remote = self.device.GetRemoteIStPort()
                node_map = remote.GetINodeMap()
                node_map.GetNode[GenApi.IEnumeration]("TriggerMode").FromString("Off")
                node_map.GetNode[GenApi.IEnumeration]("AcquisitionMode").FromString("Continuous")
                print("✅ Free-run configurado en bucle")
            except Exception:
                pass  # si tu modelo no tiene esos nodos, no pasa nada
            
            # --- bucle continuo INFINITO
            empty_streak = 0
            while True:  # BUCLE INFINITO - se ejecuta hasta Ctrl+C
                try:
                    # Recupera buffer (2 s de timeout; ajusta si quieres)
                    buffer = self.dataStream.RetrieveBuffer(2000)

                    try:
                        # ¿hay imagen dentro del buffer?
                        if buffer.GetIStStreamBufferInfo().IsImagePresent:
                            empty_streak = 0
                            img = buffer.GetIStImage()
                            w = img.ImageWidth
                            h = img.ImageHeight
                            # aquí haces lo que quieras (solo mostrar por consola en este script)
                            print(f"✅ Frame {self.frame_count}: {w}x{h} - {time.strftime('%H:%M:%S')}")
                            self.frame_count += 1
                            
                            # Mostrar progreso
                            if self.frame_count % 50 == 0:
                                print(f"Progreso: {self.frame_count} frames")
                        else:
                            empty_streak += 1
                            print(f"⚠️ Buffer sin imagen (incomplete/descartado) - Streak: {empty_streak}")
                    finally:
                        # DEVOLVER el buffer SIEMPRE
                        try:
                            buffer.Dispose()   # <-- USA DISPOSE() como en el que funciona
                        except Exception:
                            pass

                    # si hay demasiados seguidos "vacíos", reinicia adquisición
                    if empty_streak >= 10:
                        print("⏳ Muchos vacíos seguidos → reinicio de adquisición...")
                        try:
                            self.device.GetRemoteIStPort().GetINodeMap().GetNode[GenApi.ICommand]("AcquisitionStop").Execute()
                        except Exception:
                            pass
                        self.dataStream.StopAcquisition()
                        self.dataStream.StartAcquisition(self.nCountOfImagesToGrab)
                        try:
                            self.device.GetRemoteIStPort().GetINodeMap().GetNode[GenApi.ICommand]("AcquisitionStart").Execute()
                        except Exception:
                            pass
                        empty_streak = 0

                except KeyboardInterrupt:
                    print("Interrumpido por el usuario")
                    break
                except Exception as e:
                    print(f"❌ RetrieveBuffer/processing: {e}")
                    empty_streak += 1
            
            print("✅ Vista en directo terminada")
            
        except Exception as e:
            print(f"ERROR en vista en directo: {e}")
    
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
        """Método principal que ejecuta la vista en directo real"""
        try:
            print("=== Vista en Directo Python StApi REAL - Iniciando ===")
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
            
            # 4. Conectar cámara
            if not self.connect_camera():
                return False
            
            # 5. Iniciar adquisición
            if not self.start_acquisition():
                return False
            
            # 6. Mostrar vista en directo en consola
            self.show_live_view_console()
            
            return True
            
        except Exception as e:
            print(f"ERROR general: {e}")
            return False
        
        finally:
            self.stop_acquisition()
            self.cleanup_resources()

def main():
    """Función principal"""
    print("=== IMPORTANTE ===")
    print("Esta versión REAL conecta con la cámara Omron")
    print("y muestra vista en directo usando StApi directamente")
    print("==================")
    
    vista = VistaDirectoConsola()
    
    try:
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
