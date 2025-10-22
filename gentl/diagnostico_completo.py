#!/usr/bin/env python3
"""
DIAGNÓSTICO COMPLETO UNIFICADO
Combina toda la información del sistema, JetPack, CUDA, PyTorch, YOLO y librerías ML
"""

import json
import os
import sys
import subprocess
import platform
import psutil
import time
from datetime import datetime
from pathlib import Path

# Configuración
REPORT_FILE = "/home/nvidia/Desktop/Calippo_jetson-aravis-yolo/gentl/diagnostico_completo.json"
TIMESTAMP = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def run_command(cmd, shell=True):
    """Ejecuta comando y retorna resultado"""
    try:
        result = subprocess.run(cmd, shell=shell, capture_output=True, text=True, timeout=30)
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except subprocess.TimeoutExpired:
        return "", "Timeout", 1
    except Exception as e:
        return "", str(e), 1

def get_system_info():
    """Información completa del sistema"""
    # Información de Ubuntu/Debian
    ubuntu_info = {}
    try:
        with open('/etc/os-release', 'r') as f:
            for line in f:
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    ubuntu_info[key] = value.strip('"')
    except:
        ubuntu_info = {"error": "No se pudo leer /etc/os-release"}
    
    # Información de hardware
    hardware_info = {}
    try:
        # CPU info
        with open('/proc/cpuinfo', 'r') as f:
            cpu_info = f.read()
            if 'model name' in cpu_info:
                for line in cpu_info.split('\n'):
                    if line.startswith('model name'):
                        hardware_info['cpu_model'] = line.split(':', 1)[1].strip()
                        break
        
        # GPU info
        stdout, stderr, returncode = run_command("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits")
        if returncode == 0 and stdout.strip():
            gpu_info = stdout.strip().split(', ')
            if len(gpu_info) >= 3:
                hardware_info['gpu_name'] = gpu_info[0]
                hardware_info['gpu_driver'] = gpu_info[1]
                hardware_info['gpu_memory_mb'] = gpu_info[2]
    except:
        hardware_info = {"error": "No se pudo obtener información de hardware"}
    
    return {
        "hostname": platform.node(),
        "user": os.getenv('USER', 'unknown'),
        "system": platform.platform(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "kernel": platform.release(),
        "python_version": platform.python_version(),
        "ubuntu": ubuntu_info,
        "hardware": hardware_info,
        "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "memory_used_gb": round(psutil.virtual_memory().used / (1024**3), 2),
        "memory_free_gb": round(psutil.virtual_memory().available / (1024**3), 2),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_total_gb": round(psutil.disk_usage('/').total / (1024**3), 2),
        "disk_used_gb": round(psutil.disk_usage('/').used / (1024**3), 2),
        "disk_free_gb": round(psutil.disk_usage('/').free / (1024**3), 2),
        "disk_percent": round(psutil.disk_usage('/').used / psutil.disk_usage('/').total * 100, 2),
        "uptime_seconds": time.time() - psutil.boot_time(),
        "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0],
        "cpu_count": psutil.cpu_count(),
        "cpu_percent": psutil.cpu_percent(interval=1)
    }

def get_jetpack_info():
    """Información de JetPack"""
    stdout, stderr, returncode = run_command("jetson_release")
    if returncode == 0:
        lines = stdout.split('\n')
        jetpack_info = {}
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                jetpack_info[key.strip()] = value.strip()
        return jetpack_info
    else:
        return {"error": "jetson_release no encontrado", "stderr": stderr}

def get_cuda_info():
    """Información completa de CUDA"""
    cuda_info = {}
    
    # NVCC
    stdout, stderr, returncode = run_command("nvcc --version")
    if returncode == 0:
        for line in stdout.split('\n'):
            if 'release' in line.lower():
                cuda_info['nvcc_version'] = line.split('release')[1].split(',')[0].strip()
                break
        cuda_info['nvcc_available'] = True
    else:
        cuda_info['nvcc_available'] = False
        cuda_info['nvcc_version'] = "No disponible"
    
    # CUDA runtime
    cuda_version_file = "/usr/local/cuda/version.txt"
    if os.path.exists(cuda_version_file):
        with open(cuda_version_file, 'r') as f:
            cuda_info['cuda_runtime'] = f.read().strip()
    else:
        cuda_info['cuda_runtime'] = "No disponible"
    
    # nvidia-smi
    stdout, stderr, returncode = run_command("nvidia-smi --version")
    if returncode == 0:
        cuda_info['nvidia_smi_available'] = True
        cuda_info['nvidia_smi_version'] = stdout.split('\n')[0]
    else:
        cuda_info['nvidia_smi_available'] = False
        cuda_info['nvidia_smi_version'] = "No disponible"
    
    # Verificar herramientas específicas de Jetson
    cuda_info['tegrastats_available'] = run_command("tegrastats --help")[2] == 0
    cuda_info['jetson_clocks_available'] = run_command("jetson_clocks --help")[2] == 0
    
    # Variables de entorno
    cuda_info['cuda_home'] = os.getenv('CUDA_HOME', 'No definido')
    cuda_info['cuda_path'] = os.getenv('PATH', '').split(':')
    cuda_info['cuda_path'] = [p for p in cuda_info['cuda_path'] if 'cuda' in p]
    
    return cuda_info

def get_pytorch_info():
    """Información completa de PyTorch"""
    try:
        import torch
        pytorch_info = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "location": torch.__file__,
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "device_capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
            "total_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if torch.cuda.is_available() else None,
            "memory_reserved_mb": round(torch.cuda.memory_reserved(0) / (1024**2), 2) if torch.cuda.is_available() else None,
            "memory_allocated_mb": round(torch.cuda.memory_allocated(0) / (1024**2), 2) if torch.cuda.is_available() else None,
            "backend_info": {
                "cudnn_enabled": torch.backends.cudnn.enabled,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None
            },
            "tensor_operations": {
                "float16_supported": torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 5,
                "float32_supported": True,
                "float64_supported": True,
                "int8_supported": torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 6
            }
        }
        
        # Test de rendimiento básico
        if pytorch_info['cuda_available']:
            try:
                # Test CPU
                start_time = time.time()
                x_cpu = torch.randn(1000, 1000)
                y_cpu = torch.randn(1000, 1000)
                z_cpu = torch.mm(x_cpu, y_cpu)
                cpu_time = time.time() - start_time
                
                # Test CUDA
                start_time = time.time()
                x_cuda = torch.randn(1000, 1000).cuda()
                y_cuda = torch.randn(1000, 1000).cuda()
                z_cuda = torch.mm(x_cuda, y_cuda)
                cuda_time = time.time() - start_time
                
                pytorch_info['performance_test'] = {
                    "cpu_time_seconds": round(cpu_time, 3),
                    "cuda_time_seconds": round(cuda_time, 3),
                    "speedup": round(cpu_time / cuda_time, 1)
                }
            except Exception as e:
                pytorch_info['performance_test'] = {"error": str(e)}
        
        return pytorch_info
    except ImportError:
        return {"error": "PyTorch no instalado"}

def get_ml_libraries_info():
    """Información detallada de librerías ML"""
    ml_libraries = {
        "pytorch": {"name": "PyTorch", "import_name": "torch"},
        "torchvision": {"name": "Torchvision", "import_name": "torchvision"},
        "ultralytics": {"name": "Ultralytics YOLO", "import_name": "ultralytics"},
        "onnxruntime": {"name": "ONNX Runtime", "import_name": "onnxruntime"},
        "tensorrt": {"name": "TensorRT", "import_name": "tensorrt"},
        "opencv": {"name": "OpenCV", "import_name": "cv2"}
    }
    
    detailed_info = {}
    
    for lib_key, lib_config in ml_libraries.items():
        try:
            module = __import__(lib_config["import_name"])
            version = getattr(module, "__version__", "Desconocida")
            location = getattr(module, "__file__", "Desconocida")
            
            # Verificar CUDA
            cuda_available = False
            if lib_key == "pytorch":
                cuda_available = torch.cuda.is_available()
            elif lib_key == "opencv":
                cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
            elif lib_key == "onnxruntime":
                providers = module.get_available_providers()
                cuda_available = "CUDAExecutionProvider" in providers
            
            detailed_info[lib_key] = {
                "available": True,
                "version": version,
                "location": location,
                "cuda_available": cuda_available
            }
            
        except ImportError as e:
            detailed_info[lib_key] = {
                "available": False,
                "error": str(e),
                "cuda_available": False
            }
        except Exception as e:
            detailed_info[lib_key] = {
                "available": True,
                "error": str(e),
                "cuda_available": False
            }
    
    return detailed_info

def get_yolo_info():
    """Información específica de YOLO"""
    yolo_info = {
        "models_available": [],
        "ultralytics_available": False,
        "model_files": []
    }
    
    # Verificar Ultralytics
    try:
        from ultralytics import YOLO
        yolo_info["ultralytics_available"] = True
        yolo_info["ultralytics_version"] = ultralytics.__version__
    except ImportError:
        yolo_info["ultralytics_available"] = False
    
    # Buscar archivos de modelo
    model_extensions = ['.pt', '.onnx', '.engine']
    current_dir = Path.cwd()
    for ext in model_extensions:
        model_files = list(current_dir.glob(f"*{ext}"))
        for model_file in model_files:
            yolo_info["model_files"].append({
                "name": model_file.name,
                "path": str(model_file),
                "size_mb": round(model_file.stat().st_size / (1024*1024), 2),
                "type": ext[1:]
            })
    
    return yolo_info

def generate_complete_report():
    """Genera el reporte completo unificado"""
    print(f"🔍 Generando diagnóstico completo unificado - {TIMESTAMP}")
    print("=" * 60)
    
    report = {
        "timestamp": TIMESTAMP,
        "system": get_system_info(),
        "jetpack": get_jetpack_info(),
        "cuda": get_cuda_info(),
        "pytorch": get_pytorch_info(),
        "ml_libraries": get_ml_libraries_info(),
        "yolo": get_yolo_info()
    }
    
    # Guardar reporte
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Reporte completo guardado en: {REPORT_FILE}")
    
    # Mostrar resumen
    print("\n📊 RESUMEN COMPLETO:")
    print("-" * 40)
    print(f"Sistema: {report['system']['architecture']} {report['system']['kernel']}")
    print(f"JetPack: {report['jetpack'].get('L4T', 'No disponible')}")
    print(f"CUDA: {report['cuda'].get('nvcc_version', 'No disponible')}")
    print(f"PyTorch: {report['pytorch'].get('version', 'No disponible')}")
    print(f"CUDA PyTorch: {'Sí' if report['pytorch'].get('cuda_available') else 'No'}")
    
    print("\n🤖 LIBRERÍAS ML:")
    ml_libs = report.get('ml_libraries', {})
    for lib_name, lib_info in ml_libs.items():
        if lib_info.get('available'):
            cuda_status = "✅ CUDA" if lib_info.get('cuda_available') else "❌ CPU"
            print(f"   {lib_name}: {lib_info.get('version', 'N/A')} ({cuda_status})")
        else:
            print(f"   {lib_name}: ❌ No disponible")
    
    print(f"\n🎯 YOLO:")
    print(f"   Ultralytics: {'Sí' if report['yolo'].get('ultralytics_available') else 'No'}")
    print(f"   Modelos encontrados: {len(report['yolo'].get('model_files', []))}")
    
    print(f"\n💾 RECURSOS:")
    print(f"   Memoria: {report['system']['memory_used_gb']:.1f}GB / {report['system']['memory_total_gb']:.1f}GB")
    print(f"   Disco: {report['system']['disk_used_gb']:.1f}GB / {report['system']['disk_total_gb']:.1f}GB")
    
    return report

if __name__ == "__main__":
    try:
        report = generate_complete_report()
        print(f"\n🎉 Diagnóstico completo finalizado")
        print(f"📁 Archivo JSON: {REPORT_FILE}")
    except Exception as e:
        print(f"❌ Error durante el diagnóstico: {e}")
        sys.exit(1)
