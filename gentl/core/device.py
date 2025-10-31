"""
Device types
------------
- Tipos/abstracciones de dispositivo si se requieren.
- Mantiene tipado y modelos de datos separados de lógica.
"""
"""
Módulo para optimizaciones del sistema y configuración del dispositivo.
"""
import os
import subprocess
import psutil
from core.logging import log_info, log_warning, log_error


def apply_system_optimizations():
    """
    Aplica optimizaciones del sistema para mejorar el rendimiento.
    """
    try:
        log_info("🔧 Aplicando optimizaciones del sistema...")
        
        # Optimizaciones de CPU
        _optimize_cpu()
        
        # Optimizaciones de memoria
        _optimize_memory()
        
        # Optimizaciones de GPU (si está disponible)
        _optimize_gpu()
        
        # Optimizaciones de red
        _optimize_network()
        
        log_info("✅ Optimizaciones del sistema aplicadas")
        
    except Exception as e:
        log_error(f"❌ Error aplicando optimizaciones: {e}")


def _optimize_cpu():
    """Optimizaciones específicas de CPU."""
    try:
        # Establecer governor de CPU para máximo rendimiento
        cpu_count = psutil.cpu_count()
        log_info(f"🖥️ Optimizando {cpu_count} CPUs...")
        
        # Intentar establecer governor a performance
        success_count = 0
        for cpu_id in range(cpu_count):
            try:
                governor_path = f"/sys/devices/system/cpu/cpu{cpu_id}/cpufreq/scaling_governor"
                if os.path.exists(governor_path):
                    with open(governor_path, 'w') as f:
                        f.write('performance')
                    success_count += 1
            except PermissionError:
                log_warning("⚠️ Permisos insuficientes para cambiar CPU governor")
                break
            except Exception:
                pass  # Ignorar otros errores
        
        if success_count > 0:
            log_info(f"✅ CPU governor establecido a 'performance' en {success_count} cores")
        else:
            log_info("ℹ️ CPU governor no modificado (requiere permisos de root)")
                
    except Exception as e:
        log_warning(f"⚠️ Error optimizando CPU: {e}")


def _optimize_memory():
    """Optimizaciones específicas de memoria."""
    try:
        # Limpiar caché de memoria si es posible
        try:
            subprocess.run(['sync'], check=False)
            subprocess.run(['echo', '3'], stdout=open('/proc/sys/vm/drop_caches', 'w'), check=False)
        except Exception:
            pass  # Ignorar si no se puede limpiar caché
            
        log_info("🧠 Optimizaciones de memoria aplicadas")
        
    except Exception as e:
        log_warning(f"⚠️ Error optimizando memoria: {e}")


def _optimize_gpu():
    """Optimizaciones específicas de GPU."""
    try:
        # Verificar si estamos en Jetson
        if os.path.exists('/etc/nv_tegra_release'):
            log_info("🚀 Detectado Jetson - aplicando optimizaciones GPU...")
            
            # Establecer modo máximo de rendimiento para Jetson
            try:
                result1 = subprocess.run(['sudo', 'nvpmodel', '-m', '0'], 
                                       capture_output=True, text=True, check=False)
                result2 = subprocess.run(['sudo', 'jetson_clocks'], 
                                       capture_output=True, text=True, check=False)
                
                if result1.returncode == 0 and result2.returncode == 0:
                    log_info("✅ Jetson optimizado: nvpmodel + jetson_clocks")
                else:
                    log_warning("⚠️ Optimizaciones Jetson fallaron (requieren permisos de root)")
                    log_info("💡 Para máximo rendimiento, ejecuta manualmente:")
                    log_info("   sudo nvpmodel -m 0 && sudo jetson_clocks")
            except Exception as e:
                log_warning(f"⚠️ Error ejecutando optimizaciones Jetson: {e}")
                
        else:
            log_info("🖥️ Sistema no-Jetson detectado")
            
    except Exception as e:
        log_warning(f"⚠️ Error optimizando GPU: {e}")


def _optimize_network():
    """Optimizaciones específicas de red."""
    try:
        # Optimizar buffers de red
        network_params = {
            '/proc/sys/net/core/rmem_max': '16777216',
            '/proc/sys/net/core/wmem_max': '16777216',
            '/proc/sys/net/core/rmem_default': '262144',
            '/proc/sys/net/core/wmem_default': '262144',
        }
        
        for param_path, value in network_params.items():
            try:
                if os.path.exists(param_path):
                    with open(param_path, 'w') as f:
                        f.write(value)
            except Exception:
                pass  # Ignorar si no se puede cambiar
                
        log_info("🌐 Optimizaciones de red aplicadas")
        
    except Exception as e:
        log_warning(f"⚠️ Error optimizando red: {e}")


def get_system_info():
    """
    Obtiene información del sistema.
    
    Returns:
        dict: Información del sistema
    """
    try:
        info = {
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            'memory': psutil.virtual_memory()._asdict(),
            'disk': psutil.disk_usage('/')._asdict(),
            'is_jetson': os.path.exists('/etc/nv_tegra_release'),
        }
        
        return info
        
    except Exception as e:
        log_warning(f"⚠️ Error obteniendo info del sistema: {e}")
        return {}


def set_process_priority(pid=None, priority='high'):
    """
    Establece la prioridad del proceso.
    
    Args:
        pid: ID del proceso (None para proceso actual)
        priority: 'low', 'normal', 'high', 'realtime'
    """
    try:
        if pid is None:
            pid = os.getpid()
            
        priority_map = {
            'low': psutil.BELOW_NORMAL_PRIORITY_CLASS,
            'normal': psutil.NORMAL_PRIORITY_CLASS,
            'high': psutil.HIGH_PRIORITY_CLASS,
            'realtime': psutil.REALTIME_PRIORITY_CLASS,
        }
        
        if priority in priority_map:
            process = psutil.Process(pid)
            process.nice(priority_map[priority])
            log_info(f"📈 Prioridad del proceso establecida a: {priority}")
            
    except Exception as e:
        log_warning(f"⚠️ Error estableciendo prioridad: {e}")
