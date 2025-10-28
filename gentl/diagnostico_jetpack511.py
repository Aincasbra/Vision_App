#!/usr/bin/env python3
"""
DIAGNÓSTICO COMPLETO JETPACK 5.1.1
Genera reporte JSON actualizable con todas las versiones del sistema y entorno virtual
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

REPORT_FILE = str(Path.cwd() / "diagnostico_jetpack511.json")

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

def in_venv():
    """Detecta si estamos en un entorno virtual"""
    return (hasattr(sys, "real_prefix") or 
            getattr(sys, "base_prefix", sys.prefix) != sys.prefix or 
            bool(os.environ.get("VIRTUAL_ENV")))

def safe_import(mod):
    """Importa un módulo de forma segura"""
    try:
        m = __import__(mod)
        return m, None
    except Exception as e:
        return None, str(e)

def pytorch_smoke_test(torch):
    """Prueba real de CUDA con smoke test"""
    try:
        if torch.cuda.is_available():
            x = torch.randn(2, 2, device="cuda")
            y = x @ x.t()
            _ = y.sum().item()  # suma y luego convierte a scalar (fuerza sync)
            torch.cuda.synchronize()  # sync adicional
            return True
        return False
    except Exception as e:
        return False  # Silencioso, no imprimir en producción

def torchvision_nms_test():
    """Prueba NMS en GPU si está disponible"""
    try:
        from torchvision import ops
        import torch as T
        if T.cuda.is_available():
            boxes = T.tensor([[0.,0.,10.,10.],[1.,1.,10.,10.],[100.,100.,110.,110.]], device="cuda")
            scores = T.tensor([0.9,0.8,0.7], device="cuda")
            keep = ops.nms(boxes, scores, 0.5)
            return bool(len(keep) >= 1)
    except:
        pass
    return False

def get_system_info():
    """Información del sistema"""
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
    
    # Información de LSB
    lsb_info = {}
    try:
        stdout, stderr, returncode = run_command("lsb_release -a")
        if returncode == 0:
            for line in stdout.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    lsb_info[key.strip()] = value.strip()
    except:
        lsb_info = {"error": "lsb_release no disponible"}
    
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
        "lsb": lsb_info,
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
    """Información de CUDA"""
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
    
    # nvidia-smi (puede no estar disponible en Jetson)
    stdout, stderr, returncode = run_command("nvidia-smi --version")
    if returncode == 0:
        cuda_info['nvidia_smi_available'] = True
        cuda_info['nvidia_smi_version'] = stdout.split('\n')[0]
    else:
        cuda_info['nvidia_smi_available'] = False
        cuda_info['nvidia_smi_version'] = "No disponible"
    
    # Verificar alternativas para Jetson
    # tegrastats (específico de Jetson)
    stdout, stderr, returncode = run_command("tegrastats --help")
    if returncode == 0:
        cuda_info['tegrastats_available'] = True
    else:
        cuda_info['tegrastats_available'] = False
    
    # jetson_clocks (específico de Jetson)
    stdout, stderr, returncode = run_command("jetson_clocks --help")
    if returncode == 0:
        cuda_info['jetson_clocks_available'] = True
    else:
        cuda_info['jetson_clocks_available'] = False
    
    # Verificar paquetes NVIDIA instalados
    stdout, stderr, returncode = run_command("dpkg -l | grep -E '(nvidia|cuda)' | wc -l")
    if returncode == 0:
        cuda_info['nvidia_packages_count'] = int(stdout.strip())
    else:
        cuda_info['nvidia_packages_count'] = 0
    
    # Verificar si hay drivers NVIDIA
    stdout, stderr, returncode = run_command("dpkg -l | grep nvidia-driver")
    if returncode == 0 and stdout.strip():
        cuda_info['nvidia_driver_installed'] = True
        cuda_info['nvidia_driver_packages'] = stdout.strip().split('\n')
    else:
        cuda_info['nvidia_driver_installed'] = False
        cuda_info['nvidia_driver_packages'] = []
    
    # Variables de entorno
    cuda_info['cuda_home'] = os.getenv('CUDA_HOME', 'No definido')
    cuda_info['cuda_path'] = os.getenv('PATH', '').split(':')
    cuda_info['cuda_path'] = [p for p in cuda_info['cuda_path'] if 'cuda' in p]
    
    # Librerías CUDA
    cuda_lib_path = "/usr/local/cuda/lib64"
    if os.path.exists(cuda_lib_path):
        cuda_libs = [f for f in os.listdir(cuda_lib_path) if f.endswith('.so')]
        cuda_info['cuda_libraries_count'] = len(cuda_libs)
        cuda_info['cuda_libraries_path'] = cuda_lib_path
        cuda_info['cuda_libraries'] = cuda_libs
    else:
        cuda_info['cuda_libraries_count'] = 0
        cuda_info['cuda_libraries'] = []
    
    # Verificar librerías CUDA importantes
    important_cuda_libs = ['libcudart.so', 'libcublas.so', 'libcurand.so', 'libcufft.so', 'libcudnn.so']
    cuda_info['important_libraries'] = {}
    for lib in important_cuda_libs:
        cuda_info['important_libraries'][lib] = any(lib in f for f in cuda_info['cuda_libraries'])
    
    # Verificar cuDNN específicamente
    cuda_info['cudnn_info'] = {}
    
    # Buscar cuDNN en múltiples ubicaciones
    cudnn_paths = [
        "/usr/lib/aarch64-linux-gnu",      # Para ARM64 (Jetson)
        "/usr/lib/x86_64-linux-gnu",
        "/usr/local/cuda/lib64",
        "/usr/lib"
    ]
    
    cudnn_libs = []
    for lib_path in cudnn_paths:
        if os.path.exists(lib_path):
            try:
                libs = [f for f in os.listdir(lib_path) if 'cudnn' in f.lower() and f.endswith('.so')]
                cudnn_libs.extend([f"{lib_path}/{lib}" for lib in libs])
            except:
                pass
    
    cuda_info['cudnn_info']['libraries'] = cudnn_libs
    cuda_info['cudnn_info']['libraries_count'] = len(cudnn_libs)
    
    # Verificar headers cuDNN
    cudnn_header_paths = [
        "/usr/include/aarch64-linux-gnu/cudnn.h",  # Para ARM64 (Jetson)
        "/usr/include/x86_64-linux-gnu/cudnn.h",
        "/usr/include/cudnn.h",
        "/usr/local/cuda/include/cudnn.h"
    ]
    
    cuda_info['cudnn_info']['headers'] = {}
    for header_path in cudnn_header_paths:
        cuda_info['cudnn_info']['headers'][header_path] = os.path.exists(header_path)
    
    # Verificar con pkg-config
    stdout, stderr, returncode = run_command("pkg-config --modversion cudnn")
    if returncode == 0:
        cuda_info['cudnn_info']['pkg_config_version'] = stdout.strip()
    else:
        cuda_info['cudnn_info']['pkg_config_version'] = "No disponible"
    
    # Verificar con dpkg
    stdout, stderr, returncode = run_command("dpkg -l | grep cudnn")
    if returncode == 0 and stdout.strip():
        cuda_info['cudnn_info']['dpkg_packages'] = stdout.strip().split('\n')
    else:
        cuda_info['cudnn_info']['dpkg_packages'] = []
    
    return cuda_info

def get_tensorrt_info():
    """Información de TensorRT"""
    tensorrt_info = {}
    
    # Verificar TensorRT en el sistema (rutas específicas para Jetson)
    tensorrt_paths = [
        "/usr/include/x86_64-linux-gnu",
        "/usr/include/aarch64-linux-gnu",  # Para ARM64
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib/aarch64-linux-gnu",      # Para ARM64
        "/usr/local/tensorrt",
        "/opt/tensorrt",
        "/usr/include/NvInfer.h",
        "/usr/include/NvInferRuntime.h"
    ]
    
    tensorrt_info['system_paths'] = {}
    for path in tensorrt_paths:
        tensorrt_info['system_paths'][path] = os.path.exists(path)
    
    # Verificar librerías TensorRT en múltiples ubicaciones
    tensorrt_lib_paths = [
        "/usr/lib/aarch64-linux-gnu",      # Para ARM64 (Jetson)
        "/usr/lib/x86_64-linux-gnu",
        "/usr/local/tensorrt/lib",
        "/opt/tensorrt/lib",
        "/usr/lib"
    ]
    
    tensorrt_libs = []
    for lib_path in tensorrt_lib_paths:
        if os.path.exists(lib_path):
            try:
                libs = [f for f in os.listdir(lib_path) if 'tensorrt' in f.lower() and f.endswith('.so')]
                tensorrt_libs.extend([f"{lib_path}/{lib}" for lib in libs])
            except:
                pass
    
    tensorrt_info['libraries'] = tensorrt_libs
    tensorrt_info['libraries_count'] = len(tensorrt_libs)
    
    # Verificar headers TensorRT
    tensorrt_header_paths = [
        "/usr/include/aarch64-linux-gnu/NvInfer.h",  # Para ARM64 (Jetson)
        "/usr/include/x86_64-linux-gnu/NvInfer.h",
        "/usr/include/NvInfer.h",
        "/usr/local/tensorrt/include/NvInfer.h",
        "/opt/tensorrt/include/NvInfer.h"
    ]
    
    tensorrt_info['headers'] = {}
    for header_path in tensorrt_header_paths:
        tensorrt_info['headers'][header_path] = os.path.exists(header_path)
    
    # Verificar si TensorRT está disponible en Python
    try:
        import tensorrt
        tensorrt_info['python_available'] = True
        tensorrt_info['python_version'] = tensorrt.__version__
        tensorrt_info['python_location'] = tensorrt.__file__
    except ImportError:
        tensorrt_info['python_available'] = False
        tensorrt_info['python_version'] = "No disponible"
        tensorrt_info['python_location'] = "No disponible"
    
    # Verificar en el entorno virtual
    local_venv = Path.cwd() / '.venv'
    if local_venv.exists():
        venv_python = local_venv / 'bin' / 'python'
        if venv_python.exists():
            try:
                result = run_command(f"{venv_python} -c 'import tensorrt; print(tensorrt.__version__); print(tensorrt.__file__)'")
                if result[2] == 0:
                    version, location = result[0].strip().split('\n')
                    tensorrt_info['venv_available'] = True
                    tensorrt_info['venv_version'] = version
                    tensorrt_info['venv_location'] = location
                else:
                    tensorrt_info['venv_available'] = False
                    tensorrt_info['venv_version'] = "No disponible"
                    tensorrt_info['venv_location'] = "No disponible"
                    tensorrt_info['venv_error'] = result[1]
            except Exception as e:
                tensorrt_info['venv_available'] = False
                tensorrt_info['venv_error'] = str(e)
        else:
            tensorrt_info['venv_available'] = False
            tensorrt_info['venv_error'] = "Python del venv no disponible"
    else:
        tensorrt_info['venv_available'] = False
        tensorrt_info['venv_error'] = "Entorno virtual no encontrado"
    
    # Verificar con pkg-config
    stdout, stderr, returncode = run_command("pkg-config --modversion tensorrt")
    if returncode == 0:
        tensorrt_info['pkg_config_version'] = stdout.strip()
    else:
        tensorrt_info['pkg_config_version'] = "No disponible"
    
    # Verificar con dpkg (paquetes instalados)
    stdout, stderr, returncode = run_command("dpkg -l | grep tensorrt")
    if returncode == 0 and stdout.strip():
        tensorrt_info['dpkg_packages'] = stdout.strip().split('\n')
    else:
        tensorrt_info['dpkg_packages'] = []
    
    return tensorrt_info

def get_python_info():
    """Información de Python"""
    # Detectar entorno virtual local si existe
    local_venv = Path.cwd() / '.venv'
    if local_venv.exists() and not os.getenv('VIRTUAL_ENV'):
        # Activar entorno virtual local temporalmente
        venv_python = local_venv / 'bin' / 'python'
        venv_pip = local_venv / 'bin' / 'pip'
        if venv_python.exists():
            # Usar el Python del venv para obtener información
            venv_info = run_command(f"{venv_python} -c 'import sys; print(sys.version)'")[0]
            venv_pip_version = run_command(f"{venv_pip} --version")[0]
            virtual_env = str(local_venv)
        else:
            virtual_env = 'No activo'
            venv_info = 'No disponible'
            venv_pip_version = 'No disponible'
    else:
        virtual_env = os.getenv('VIRTUAL_ENV', 'No activo')
        venv_info = 'No disponible'
        venv_pip_version = 'No disponible'
    
    python_info = {
        "system": {
            "python3": run_command("python3 --version")[0],
            "pip3": run_command("pip3 --version")[0],
            "python3_location": run_command("which python3")[0],
            "pip3_location": run_command("which pip3")[0]
        },
        "virtual_env": {
            "active": virtual_env,
            "python_version": venv_info,
            "pip_version": venv_pip_version,
            "python_path": sys.executable,
            "pip_path": run_command("which pip")[0],
            "python_implementation": platform.python_implementation(),
            "python_compiler": platform.python_compiler(),
            "python_build": platform.python_build()
        }
    }
    
    # Información del entorno virtual
    if python_info['virtual_env']['active'] != 'No activo':
        venv_path = Path(python_info['virtual_env']['active'])
        python_info['virtual_env']['venv_python'] = str(venv_path / 'bin' / 'python')
        python_info['virtual_env']['venv_pip'] = str(venv_path / 'bin' / 'pip')
        python_info['virtual_env']['venv_site_packages'] = str(venv_path / 'lib' / f'python{sys.version_info.major}.{sys.version_info.minor}' / 'site-packages')
        
        # Verificar si el venv tiene paquetes instalados
        if venv_path.exists():
            site_packages = venv_path / 'lib' / f'python{sys.version_info.major}.{sys.version_info.minor}' / 'site-packages'
            if site_packages.exists():
                installed_packages = list(site_packages.glob('*'))
                python_info['virtual_env']['packages_count'] = len([p for p in installed_packages if p.is_dir() and not p.name.startswith('.')])
            else:
                python_info['virtual_env']['packages_count'] = 0
        else:
            python_info['virtual_env']['packages_count'] = 0
    else:
        python_info['virtual_env']['packages_count'] = 0
    
    return python_info

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
            "torch_compile_available": hasattr(torch, 'compile'),
            "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
            "xpu_available": hasattr(torch.backends, 'xpu') and torch.backends.xpu.is_available(),
            "gpu_smoke_test": pytorch_smoke_test(torch),  # NUEVO: Smoke test real
            "venv_active": in_venv(),  # NUEVO: Detección de venv
            # Información adicional específica para Jetson
            "backend_info": {
                "cudnn_enabled": torch.backends.cudnn.enabled,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
                "cudnn_cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None
            },
            "tensor_operations": {
                "float16_supported": torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 5,
                "float32_supported": True,
                "float64_supported": True,
                "int8_supported": torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 6
            },
            "optimization_features": {
                "torch_compile": hasattr(torch, 'compile'),
                "torch_jit": hasattr(torch, 'jit'),
                "torch_script": hasattr(torch, 'script'),
                "torch_autograd": hasattr(torch, 'autograd'),
                "torch_nn": hasattr(torch, 'nn'),
                "torch_optim": hasattr(torch, 'optim')
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

def get_torchvision_info():
    """Información de Torchvision"""
    try:
        import torchvision
        torchvision_info = {
            "version": torchvision.__version__,
            "location": torchvision.__file__,
            "_C_available": False,
            "_C_location": None,
            "ops_available": False,
            "cuda_nms_smoke": False  # NUEVO: Test NMS en GPU
        }
        
        try:
            import torchvision._C as _C
            torchvision_info['_C_available'] = True
            torchvision_info['_C_location'] = _C.__file__
        except ImportError as e:
            torchvision_info['_C_error'] = str(e)
        
        try:
            from torchvision.ops import nms
            torchvision_info['ops_available'] = True
            torchvision_info['cuda_nms_smoke'] = torchvision_nms_test()  # NUEVO: Test NMS GPU
        except ImportError as e:
            torchvision_info['ops_error'] = str(e)
        
        return torchvision_info
    except ImportError:
        return {"error": "Torchvision no instalado"}

def get_libraries_info():
    """Información completa de librerías Python"""
    libraries = [
        'numpy', 'opencv-python', 'pillow', 'matplotlib', 'scipy', 
        'pandas', 'yaml', 'tqdm', 'requests', 'scikit-learn',
        'tensorflow', 'keras', 'transformers', 'ultralytics', 'onnxruntime',
        'torch', 'torchvision', 'torchaudio', 'tensorrt', 'psutil'
    ]
    
    lib_info = {
        "system": {},
        "virtual_env": {}
    }
    
    # Verificar librerías en el sistema
    for lib in libraries:
        try:
            if lib == 'opencv-python':
                import cv2
                lib_info["system"][lib] = {
                    "available": True,
                    "version": cv2.__version__,
                    "location": cv2.__file__
                }
            elif lib == 'pillow':
                import PIL
                lib_info["system"][lib] = {
                    "available": True,
                    "version": PIL.__version__,
                    "location": PIL.__file__
                }
            else:
                module_name = lib.replace('-', '_')
                module = __import__(module_name)
                lib_info["system"][lib] = {
                    "available": True,
                    "version": getattr(module, '__version__', 'Desconocida'),
                    "location": getattr(module, '__file__', 'Desconocida')
                }
        except ImportError as e:
            lib_info["system"][lib] = {
                "available": False,
                "error": str(e)
            }
    
    # Verificar librerías en el entorno virtual
    local_venv = Path.cwd() / '.venv'
    if local_venv.exists():
        venv_python = local_venv / 'bin' / 'python'
        if venv_python.exists():
            for lib in libraries:
                try:
                    if lib == 'opencv-python':
                        result = run_command(f"{venv_python} -c 'import cv2; print(cv2.__version__); print(cv2.__file__)'")
                        if result[2] == 0:
                            version, location = result[0].strip().split('\n')
                            lib_info["virtual_env"][lib] = {
                                "available": True,
                                "version": version,
                                "location": location
                            }
                        else:
                            lib_info["virtual_env"][lib] = {
                                "available": False,
                                "error": result[1]
                            }
                    elif lib == 'pillow':
                        result = run_command(f"{venv_python} -c 'import PIL; print(PIL.__version__); print(PIL.__file__)'")
                        if result[2] == 0:
                            version, location = result[0].strip().split('\n')
                            lib_info["virtual_env"][lib] = {
                                "available": True,
                                "version": version,
                                "location": location
                            }
                        else:
                            lib_info["virtual_env"][lib] = {
                                "available": False,
                                "error": result[1]
                            }
                    else:
                        module_name = lib.replace('-', '_')
                        result = run_command(f"{venv_python} -c 'import {module_name}; print(getattr({module_name}, \"__version__\", \"Desconocida\")); print(getattr({module_name}, \"__file__\", \"Desconocida\"))'")
                        if result[2] == 0:
                            version, location = result[0].strip().split('\n')
                            lib_info["virtual_env"][lib] = {
                                "available": True,
                                "version": version,
                                "location": location
                            }
                        else:
                            lib_info["virtual_env"][lib] = {
                                "available": False,
                                "error": result[1]
                            }
                except Exception as e:
                    lib_info["virtual_env"][lib] = {
                        "available": False,
                        "error": str(e)
                    }
        else:
            for lib in libraries:
                lib_info["virtual_env"][lib] = {
                    "available": False,
                    "error": "Python del venv no disponible"
                }
    else:
        for lib in libraries:
            lib_info["virtual_env"][lib] = {
                "available": False,
                "error": "Entorno virtual no encontrado"
            }
    
    return lib_info


def get_pip_packages():
    """Listado de paquetes pip instalados en sistema y en el .venv local."""
    pkgs = {"system": {}, "virtual_env": {}}
    # System
    out, err, rc = run_command("pip3 freeze")
    pkgs["system"]["freeze"] = out.split("\n") if rc == 0 else []
    pkgs["system"]["error"] = None if rc == 0 else err
    # Venv
    venv = Path.cwd() / '.venv'
    vpy = venv / 'bin' / 'python'
    if vpy.exists():
        out, err, rc = run_command(f"{vpy} -m pip freeze")
        pkgs["virtual_env"]["freeze"] = out.split("\n") if rc == 0 else []
        pkgs["virtual_env"]["error"] = None if rc == 0 else err
        # Paths y binarios efectivos dentro del venv
        # Usar comillas adecuadas para evitar conflicto con "\n"
        pycode = "import sys,os;print(sys.executable);print(os.getcwd());print('|||'.join(sys.path))"
        out_py, _, rc_py = run_command(f"{vpy} -c \"{pycode}\"")
        if rc_py == 0:
            lines = out_py.split('\n')
            pkgs["virtual_env"]["python_executable"] = lines[0] if lines else str(vpy)
            pkgs["virtual_env"]["cwd"] = lines[1] if len(lines) > 1 else str(Path.cwd())
            # sys.path viene separado por '|||' para no romper las comillas del shell
            pkgs["virtual_env"]["sys_path"] = (lines[2].split('|||') if len(lines) > 2 else [])
    else:
        pkgs["virtual_env"]["error"] = "Entorno virtual no encontrado"
    return pkgs


def get_opencv_build_info():
    """Información detallada de compilación de OpenCV (sistema y venv)."""
    info = {"system": {}, "virtual_env": {}}
    try:
        import cv2
        info["system"]["version"] = cv2.__version__
        try:
            info["system"]["build"] = cv2.getBuildInformation()
        except Exception as e:
            info["system"]["build_error"] = str(e)
    except Exception as e:
        info["system"]["error"] = str(e)
    venv = Path.cwd() / '.venv'
    vpy = venv / 'bin' / 'python'
    if vpy.exists():
        out, err, rc = run_command(f"{vpy} -c 'import cv2; print(cv2.__version__); print(cv2.getBuildInformation())'")
        if rc == 0:
            sp = out.split('\n', 1)
            info["virtual_env"]["version"] = sp[0].strip()
            info["virtual_env"]["build"] = sp[1] if len(sp) > 1 else ""
        else:
            info["virtual_env"]["error"] = err
    else:
        info["virtual_env"]["error"] = "Entorno virtual no encontrado"
    return info


def get_ultralytics_info():
    """Detalles de Ultralytics YOLO en sistema y venv (versión, providers, device)."""
    info = {"system": {}, "virtual_env": {}}
    # System
    try:
        import ultralytics
        info["system"]["version"] = getattr(ultralytics, "__version__", "Desconocida")
        try:
            from ultralytics.utils.torch_utils import select_device
            dev = str(select_device('0' if (hasattr(__import__('torch'), 'cuda') and __import__('torch').cuda.is_available()) else 'cpu'))
            info["system"]["selected_device"] = dev
        except Exception as e:
            info["system"]["device_error"] = str(e)
    except Exception as e:
        info["system"]["error"] = str(e)
    # Venv
    venv = Path.cwd() / '.venv'
    vpy = venv / 'bin' / 'python'
    if vpy.exists():
        code = (
            "import ultralytics, torch;"
            "print(getattr(ultralytics,'__version__','Desconocida'));"
            "print(torch.cuda.is_available());"
            "print(torch.version.cuda if torch.cuda.is_available() else 'None');"
            "print(torch.backends.cudnn.version() if torch.cuda.is_available() else 'None')"
        )
        out, err, rc = run_command(f"{vpy} -c \"{code}\"")
        if rc == 0:
            lines = out.strip().split('\n')
            if len(lines) >= 4:
                info["virtual_env"]["version"] = lines[0]
                info["virtual_env"]["cuda_available"] = (lines[1].strip() == 'True')
                info["virtual_env"]["cuda_version"] = lines[2]
                info["virtual_env"]["cudnn_version"] = lines[3]
        else:
            info["virtual_env"]["error"] = err
    else:
        info["virtual_env"]["error"] = "Entorno virtual no encontrado"
    return info

def check_pytorch_status():
    """Check PyTorch status con check CUDA fiable"""
    info = {"available": False, "version": "No disponible", "cuda": False}
    try:
        import torch
        info["available"] = True
        info["version"] = getattr(torch, "__version__", "desconocida")
        # ¡Esta es la verdad canónica!
        info["cuda"] = bool(torch.cuda.is_available())
        # Opcional: fuerza una alloc para evitar falsos positivos
        if info["cuda"]:
            _ = torch.zeros(1, device="cuda")
    except Exception as e:
        info["error"] = str(e)
    return info

def check_torchvision_status():
    """Check Torchvision status con NMS en GPU"""
    info = {"available": False, "version": "No disponible", "ops": False, "cuda_nms": False}
    try:
        import torchvision, torch
        info["available"] = True
        info["version"] = getattr(torchvision, "__version__", "desconocida")
        from torchvision import ops
        info["ops"] = hasattr(ops, "nms")
        # NMS en GPU si Torch tiene CUDA
        if torch.cuda.is_available() and info["ops"]:
            b = torch.tensor([[0.,0.,10.,10.],[1.,1.,10.,10.],[100.,100.,110.,110.]], device="cuda")
            s = torch.tensor([0.9,0.8,0.7], device="cuda")
            _ = ops.nms(b, s, 0.5)
            info["cuda_nms"] = True
    except Exception as e:
        info["error"] = str(e)
    return info

def check_opencv_status():
    """Check OpenCV status separando error OpenGL/TLS"""
    info = {"available": False, "version": "No disponible", "cuda": False, "error_gl": None}
    try:
        import cv2
        info["available"] = True
        info["version"] = cv2.__version__
        info["cuda"] = cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception as e:
        error_str = str(e)
        if "libGLdispatch" in error_str or "TLS" in error_str:
            info["error_gl"] = error_str
        else:
            info["error"] = error_str
    return info

def check_ultralytics_status():
    """Check Ultralytics YOLO status"""
    info = {"available": False, "version": "No disponible", "cuda": False}
    try:
        import ultralytics
        import torch
        info["available"] = True
        info["version"] = getattr(ultralytics, "__version__", "desconocida")
        info["cuda"] = torch.cuda.is_available()
    except Exception as e:
        info["error"] = str(e)
    return info

def get_ml_libraries_detailed():
    """Información detallada de librerías de machine learning - USANDO VENV ACTIVO"""
    # Prioriza siempre el venv activo
    USE_VENV = in_venv()
    
    return {
        "pytorch": check_pytorch_status(),
        "torchvision": check_torchvision_status(),
        "opencv": check_opencv_status(),
        "ultralytics": check_ultralytics_status(),
        # ONNX Runtime y TensorRT mantienen la misma lógica
    }

def get_gpiod_info():
    """Información de GPIOD"""
    gpiod_info = {}
    
    # Verificar comandos
    gpiod_info['gpioset_available'] = run_command("which gpioset")[2] == 0
    gpiod_info['gpioget_available'] = run_command("which gpioget")[2] == 0
    
    if gpiod_info['gpioset_available']:
        gpiod_info['gpioset_version'] = run_command("gpioset --version")[0]
    if gpiod_info['gpioget_available']:
        gpiod_info['gpioget_version'] = run_command("gpioget --version")[0]
    
    # Verificar acceso a GPIO
    gpiod_info['gpiochip0_accessible'] = os.access("/dev/gpiochip0", os.R_OK)
    gpiod_info['gpiochip0_exists'] = os.path.exists("/dev/gpiochip0")
    
    return gpiod_info

def get_aravis_info():
    """Información de Aravis 0.6"""
    aravis_info = {
        "system": {},
        "virtual_env": {},
        "packages": [],
        "python_import": {}
    }
    
    # Paquetes instalados
    try:
        stdout, _, _ = run_command("dpkg -l | grep aravis")
        aravis_info["packages"] = [line for line in stdout.split('\n') if line.strip()]
    except:
        aravis_info["packages"] = []
    
    # Verificar Aravis en el sistema
    try:
        import gi
        gi.require_version('Aravis', '0.6')
        from gi.repository import Aravis
        
        try:
            version = Aravis.get_version()
            camera_count = Aravis.get_n_devices()
        except:
            version = "0.6.x"
            camera_count = 0
        
        aravis_info["system"] = {
            "available": True,
            "version": version,
            "camera_count": camera_count,
            "location": gi.__file__,
            "gi_version": gi.__version__
        }
        
        aravis_info["python_import"] = {
            "success": True,
            "version_checked": "0.6"
        }
    except ImportError as e:
        aravis_info["system"] = {
            "available": False,
            "error": str(e)
        }
        aravis_info["python_import"] = {
            "success": False,
            "error": str(e)
        }
    except Exception as e:
        aravis_info["system"] = {
            "available": False,
            "error": str(e)
        }
        aravis_info["python_import"] = {
            "success": False,
            "error": str(e)
        }
    
    # Verificar Aravis en el entorno virtual
    local_venv = Path.cwd() / '.venv'
    if local_venv.exists():
        venv_python = local_venv / 'bin' / 'python'
        if venv_python.exists():
            try:
                result = run_command(f"{venv_python} -c 'import gi; gi.require_version(\"Aravis\", \"0.8\"); from gi.repository import Aravis; print(Aravis.get_version()); print(Aravis.get_n_devices()); import gi; print(gi.__version__)'")
                if result[2] == 0:
                    version, camera_count, gi_version = result[0].strip().split('\n')
                    aravis_info["virtual_env"] = {
                        "available": True,
                        "version": version,
                        "camera_count": int(camera_count),
                        "gi_version": gi_version
                    }
                else:
                    aravis_info["virtual_env"] = {
                        "available": False,
                        "error": result[1]
                    }
            except Exception as e:
                aravis_info["virtual_env"] = {
                    "available": False,
                    "error": str(e)
                }
        else:
            aravis_info["virtual_env"] = {
                "available": False,
                "error": "Python del venv no disponible"
            }
    else:
        aravis_info["virtual_env"] = {
            "available": False,
            "error": "Entorno virtual no encontrado"
        }
    
    return aravis_info

def get_environment_variables():
    """Variables de entorno importantes"""
    important_vars = [
        'PATH', 'PYTHONPATH', 'LD_LIBRARY_PATH', 'CUDA_HOME', 
        'VIRTUAL_ENV', 'CONDA_PREFIX', 'JETPACK_VERSION',
        'CUDA_PATH', 'CUDA_ROOT', 'CUDA_BIN_PATH', 'CUDA_LIB_PATH'
    ]
    
    env_vars = {}
    for var in important_vars:
        value = os.getenv(var)
        if value:
            if var == 'PATH':
                env_vars[var] = value.split(':')
            else:
                env_vars[var] = value
        else:
            env_vars[var] = None
    
    return env_vars

def get_library_paths():
    """Información detallada de rutas de librerías"""
    lib_paths = {
        "system": {
            "python_paths": [],
            "site_packages": [],
            "dist_packages": [],
            "local_packages": []
        },
        "virtual_env": {
            "python_paths": [],
            "site_packages": [],
            "dist_packages": [],
            "local_packages": []
        }
    }
    
    # Rutas del sistema
    try:
        import sys
        lib_paths["system"]["python_paths"] = sys.path
        
        # Site-packages del sistema
        for path in sys.path:
            if 'site-packages' in path and 'dist-packages' not in path:
                lib_paths["system"]["site_packages"].append(path)
            elif 'dist-packages' in path:
                lib_paths["system"]["dist_packages"].append(path)
            elif 'local' in path:
                lib_paths["system"]["local_packages"].append(path)
    except Exception as e:
        lib_paths["system"]["error"] = str(e)
    
    # Rutas del entorno virtual
    local_venv = Path.cwd() / '.venv'
    if local_venv.exists():
        venv_python = local_venv / 'bin' / 'python'
        if venv_python.exists():
            try:
                result = run_command(f"{venv_python} -c 'import sys; print(\"\\n\".join(sys.path))'")
                if result[2] == 0:
                    venv_paths = result[0].strip().split('\n')
                    lib_paths["virtual_env"]["python_paths"] = venv_paths
                    
                    for path in venv_paths:
                        if 'site-packages' in path and 'dist-packages' not in path:
                            lib_paths["virtual_env"]["site_packages"].append(path)
                        elif 'dist-packages' in path:
                            lib_paths["virtual_env"]["dist_packages"].append(path)
                        elif 'local' in path:
                            lib_paths["virtual_env"]["local_packages"].append(path)
                else:
                    lib_paths["virtual_env"]["error"] = result[1]
            except Exception as e:
                lib_paths["virtual_env"]["error"] = str(e)
        else:
            lib_paths["virtual_env"]["error"] = "Python del venv no disponible"
    else:
        lib_paths["virtual_env"]["error"] = "Entorno virtual no encontrado"
    
    return lib_paths

def get_export_commands():
    """Genera comandos de export para replicar el entorno"""
    export_commands = {
        "system": [],
        "virtual_env": [],
        "cuda": [],
        "python": [],
        "complete_setup": []
    }
    
    # Variables de entorno actuales
    env_vars = get_environment_variables()
    
    # Comandos para el sistema
    if env_vars.get('PATH'):
        export_commands["system"].append(f"export PATH=\"{':'.join(env_vars['PATH'])}\"")
    
    if env_vars.get('LD_LIBRARY_PATH'):
        export_commands["system"].append(f"export LD_LIBRARY_PATH=\"{env_vars['LD_LIBRARY_PATH']}\"")
    
    if env_vars.get('CUDA_HOME'):
        export_commands["system"].append(f"export CUDA_HOME=\"{env_vars['CUDA_HOME']}\"")
    
    # Comandos para CUDA
    export_commands["cuda"].append("export CUDA_HOME=/usr/local/cuda")
    export_commands["cuda"].append("export PATH=$CUDA_HOME/bin:$PATH")
    export_commands["cuda"].append("export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH")
    
    # Comandos para Python
    export_commands["python"].append("export PYTHONPATH=\"/home/nvidia/Desktop/Calippo_jetson/gentl/.venv/lib/python3.8/site-packages\"")
    
    # Comandos para entorno virtual
    venv_path = "/home/nvidia/Desktop/Calippo_jetson/gentl/.venv"
    export_commands["virtual_env"].append(f"export VIRTUAL_ENV=\"{venv_path}\"")
    export_commands["virtual_env"].append(f"export PATH=\"{venv_path}/bin:$PATH\"")
    export_commands["virtual_env"].append("export PYTHONPATH=\"$VIRTUAL_ENV/lib/python3.8/site-packages:$PYTHONPATH\"")
    
    # Setup completo
    export_commands["complete_setup"] = [
        "# Setup completo para replicar el entorno",
        "export CUDA_HOME=/usr/local/cuda",
        "export PATH=$CUDA_HOME/bin:$PATH",
        "export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH",
        f"export VIRTUAL_ENV=\"{venv_path}\"",
        f"export PATH=\"{venv_path}/bin:$PATH\"",
        "export PYTHONPATH=\"$VIRTUAL_ENV/lib/python3.8/site-packages:$PYTHONPATH\"",
        "source $VIRTUAL_ENV/bin/activate"
    ]
    
    return export_commands

def get_system_configuration():
    """Configuración completa del sistema para replicación"""
    config = {
        "jetson_clocks": {},
        "power_mode": {},
        "fan_mode": {},
        "nvidia_settings": {},
        "system_services": {},
        "installed_packages": {},
        "system_files": {}
    }
    
    # Jetson Clocks
    stdout, stderr, returncode = run_command("jetson_clocks --show")
    if returncode == 0:
        config["jetson_clocks"]["current"] = stdout.strip()
    else:
        config["jetson_clocks"]["error"] = stderr
    
    # Power mode
    stdout, stderr, returncode = run_command("sudo nvpmodel -q")
    if returncode == 0:
        config["power_mode"]["current"] = stdout.strip()
    else:
        config["power_mode"]["error"] = stderr
    
    # Fan mode
    stdout, stderr, returncode = run_command("sudo tegrastats --interval 1000 --logfile /tmp/tegrastats.log & sleep 2; kill %1; cat /tmp/tegrastats.log | head -1")
    if returncode == 0:
        config["fan_mode"]["sample"] = stdout.strip()
    else:
        config["fan_mode"]["error"] = stderr
    
    # Paquetes instalados importantes
    important_packages = [
        "nvidia-jetpack", "cuda-toolkit", "libcudnn8", "libcudnn8-dev",
        "tensorrt", "libnvinfer8", "libnvinfer-dev", "libnvinfer-plugin8",
        "libnvinfer-plugin-dev", "python3-opencv", "python3-pip",
        "python3-venv", "python3-dev", "build-essential", "cmake",
        "git", "wget", "curl", "unzip", "libjpeg-dev", "libpng-dev",
        "libtiff-dev", "libavcodec-dev", "libavformat-dev", "libswscale-dev",
        "libgtk2.0-dev", "libcanberra-gtk-module", "libcanberra-gtk3-module"
    ]
    
    for pkg in important_packages:
        stdout, stderr, returncode = run_command(f"dpkg -l | grep {pkg}")
        if returncode == 0:
            config["installed_packages"][pkg] = stdout.strip()
        else:
            config["installed_packages"][pkg] = "No instalado"
    
    # Archivos de configuración importantes
    config_files = [
        "/etc/environment",
        "/etc/profile",
        "/home/nvidia/.bashrc",
        "/home/nvidia/.profile",
        "/etc/ld.so.conf.d/cuda.conf",
        "/etc/ld.so.conf.d/nvidia-tegra.conf"
    ]
    
    for config_file in config_files:
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config["system_files"][config_file] = f.read()
            except Exception as e:
                config["system_files"][config_file] = f"Error leyendo: {e}"
        else:
            config["system_files"][config_file] = "No existe"
    
    return config

def get_replication_scripts():
    """Genera scripts para replicar el sistema completo"""
    scripts = {
        "setup_system.sh": [],
        "setup_venv.sh": [],
        "install_packages.sh": [],
        "configure_environment.sh": [],
        "restore_configs.sh": []
    }
    
    # Script de configuración del sistema
    scripts["setup_system.sh"] = [
        "#!/bin/bash",
        "# Script para configurar Jetson Orin con JetPack 5.1.1",
        "set -e",
        "",
        "echo '🚀 Configurando Jetson Orin...'",
        "",
        "# Actualizar sistema",
        "sudo apt update && sudo apt upgrade -y",
        "",
        "# Configurar Jetson Clocks",
        "sudo jetson_clocks",
        "",
        "# Configurar modo de potencia",
        "sudo nvpmodel -m 0",
        "",
        "# Configurar fan",
        "sudo sh -c 'echo 255 > /sys/devices/pwm-fan/target_pwm'",
        "",
        "echo '✅ Sistema configurado'"
    ]
    
    # Script de instalación de paquetes
    scripts["install_packages.sh"] = [
        "#!/bin/bash",
        "# Script para instalar paquetes necesarios",
        "set -e",
        "",
        "echo '📦 Instalando paquetes...'",
        "",
        "# Paquetes base",
        "sudo apt install -y python3-pip python3-venv python3-dev",
        "sudo apt install -y build-essential cmake git wget curl unzip",
        "sudo apt install -y libjpeg-dev libpng-dev libtiff-dev",
        "sudo apt install -y libavcodec-dev libavformat-dev libswscale-dev",
        "sudo apt install -y libgtk2.0-dev libcanberra-gtk-module libcanberra-gtk3-module",
        "",
        "# OpenCV dependencies",
        "sudo apt install -y libopencv-dev python3-opencv",
        "",
        "# GPIOD",
        "sudo apt install -y gpiod",
        "",
        "echo '✅ Paquetes instalados'"
    ]
    
    # Script de configuración del entorno virtual
    scripts["setup_venv.sh"] = [
        "#!/bin/bash",
        "# Script para crear y configurar entorno virtual",
        "set -e",
        "",
        "echo '🐍 Configurando entorno virtual...'",
        "",
        "# Crear directorio del proyecto",
        "mkdir -p /home/nvidia/Desktop/Calippo_jetson/gentl",
        "cd /home/nvidia/Desktop/Calippo_jetson/gentl",
        "",
        "# Crear entorno virtual",
        "python3 -m venv .venv",
        "source .venv/bin/activate",
        "",
        "# Actualizar pip",
        "pip install --upgrade pip setuptools wheel",
        "",
        "echo '✅ Entorno virtual creado'"
    ]
    
    # Script de configuración de variables de entorno
    scripts["configure_environment.sh"] = [
        "#!/bin/bash",
        "# Script para configurar variables de entorno",
        "set -e",
        "",
        "echo '🔧 Configurando variables de entorno...'",
        "",
        "# Configurar CUDA",
        "echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc",
        "echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc",
        "echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc",
        "",
        "# Configurar Python",
        "echo 'export PYTHONPATH=$VIRTUAL_ENV/lib/python3.8/site-packages:$PYTHONPATH' >> ~/.bashrc",
        "",
        "# Recargar configuración",
        "source ~/.bashrc",
        "",
        "echo '✅ Variables de entorno configuradas'"
    ]
    
    return scripts

def get_venv_replication_data():
    """Datos completos para replicar el entorno virtual"""
    venv_data = {
        "requirements": [],
        "pip_freeze": [],
        "python_version": "",
        "pip_version": "",
        "site_packages": [],
        "installed_packages": {},
        "environment_variables": {}
    }
    
    # Obtener requirements del venv
    venv = Path.cwd() / '.venv'
    vpy = venv / 'bin' / 'python'
    vpip = venv / 'bin' / 'pip'
    
    if vpy.exists() and vpip.exists():
        # Python version
        out, _, rc = run_command(f"{vpy} --version")
        if rc == 0:
            venv_data["python_version"] = out.strip()
        
        # Pip version
        out, _, rc = run_command(f"{vpip} --version")
        if rc == 0:
            venv_data["pip_version"] = out.strip()
        
        # Pip freeze
        out, _, rc = run_command(f"{vpip} freeze")
        if rc == 0:
            venv_data["pip_freeze"] = out.strip().split('\n')
        
        # Generar requirements.txt
        venv_data["requirements"] = [
            "# Requirements generados automáticamente",
            "# Para replicar el entorno virtual exacto",
            ""
        ] + venv_data["pip_freeze"]
        
        # Variables de entorno del venv
        out, _, rc = run_command(f"{vpy} -c 'import os; print(dict(os.environ))'")
        if rc == 0:
            try:
                import ast
                venv_data["environment_variables"] = ast.literal_eval(out.strip())
            except:
                venv_data["environment_variables"] = {"error": "No se pudo parsear"}
    
    return venv_data

def get_tool_versions():
    """Información de versiones de herramientas importantes"""
    tools = {
        "system": {},
        "virtual_env": {}
    }
    
    # Herramientas a verificar
    tool_commands = {
        "gcc": "gcc --version",
        "g++": "g++ --version", 
        "make": "make --version",
        "cmake": "cmake --version",
        "git": "git --version",
        "pip": "pip --version",
        "pip3": "pip3 --version",
        "python": "python --version",
        "python3": "python3 --version",
        "nvcc": "nvcc --version",
        "nvidia-smi": "nvidia-smi --version",
        "gpioset": "gpioset --version",
        "gpioget": "gpioget --version"
    }
    
    # Verificar herramientas del sistema
    for tool, cmd in tool_commands.items():
        stdout, stderr, returncode = run_command(cmd)
        if returncode == 0:
            tools["system"][tool] = {
                "available": True,
                "version": stdout.strip().split('\n')[0],
                "full_output": stdout.strip()
            }
        else:
            tools["system"][tool] = {
                "available": False,
                "error": stderr.strip() if stderr else "No disponible"
            }
    
    # Verificar herramientas del entorno virtual
    local_venv = Path.cwd() / '.venv'
    if local_venv.exists():
        venv_python = local_venv / 'bin' / 'python'
        venv_pip = local_venv / 'bin' / 'pip'
        
        if venv_python.exists():
            # Python del venv
            stdout, stderr, returncode = run_command(f"{venv_python} --version")
            if returncode == 0:
                tools["virtual_env"]["python"] = {
                    "available": True,
                    "version": stdout.strip(),
                    "location": str(venv_python)
                }
            else:
                tools["virtual_env"]["python"] = {
                    "available": False,
                    "error": stderr.strip()
                }
            
            # Pip del venv
            stdout, stderr, returncode = run_command(f"{venv_pip} --version")
            if returncode == 0:
                tools["virtual_env"]["pip"] = {
                    "available": True,
                    "version": stdout.strip(),
                    "location": str(venv_pip)
                }
            else:
                tools["virtual_env"]["pip"] = {
                    "available": False,
                    "error": stderr.strip()
                }
        else:
            tools["virtual_env"]["python"] = {"available": False, "error": "Python del venv no disponible"}
            tools["virtual_env"]["pip"] = {"available": False, "error": "Pip del venv no disponible"}
    else:
        tools["virtual_env"]["python"] = {"available": False, "error": "Entorno virtual no encontrado"}
        tools["virtual_env"]["pip"] = {"available": False, "error": "Entorno virtual no encontrado"}
    
    return tools

def generate_report():
    """Genera el reporte completo"""
    print(f"🔍 Generando diagnóstico JetPack 5.1.1 - {TIMESTAMP}")
    print("=" * 60)
    
    report = {
        "timestamp": TIMESTAMP,
        "jetpack_version": "5.1.1",
        "system": get_system_info(),
        "jetpack": get_jetpack_info(),
        "cuda": get_cuda_info(),
        "tensorrt": get_tensorrt_info(),
        "python": get_python_info(),
        "pytorch": get_pytorch_info(),
        "torchvision": get_torchvision_info(),
        "libraries": get_libraries_info(),
        "ml_libraries_detailed": get_ml_libraries_detailed(),
        "gpiod": get_gpiod_info(),
        "aravis": get_aravis_info(),
        "environment_variables": get_environment_variables(),
        "library_paths": get_library_paths(),
        "export_commands": get_export_commands(),
        "tool_versions": get_tool_versions(),
        "pip_packages": get_pip_packages(),
        "opencv_build": get_opencv_build_info(),
        "ultralytics": get_ultralytics_info(),
        "system_configuration": get_system_configuration(),
        "replication_scripts": get_replication_scripts(),
        "venv_replication_data": get_venv_replication_data()
    }
    
    # Guardar reporte
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Reporte guardado en: {REPORT_FILE}")
    
    # Mostrar resumen
    print("\n📊 RESUMEN:")
    print("-" * 30)
    print(f"Sistema: {report['system']['architecture']} {report['system']['kernel']}")
    print(f"JetPack: {report['jetpack_version']}")
    print(f"CUDA: {report['cuda'].get('nvcc_version', 'No disponible')}")
    # Verificar herramientas NVIDIA disponibles
    nvidia_tools = []
    if report['cuda'].get('nvidia_smi_available'):
        nvidia_tools.append("nvidia-smi")
    if report['cuda'].get('tegrastats_available'):
        nvidia_tools.append("tegrastats")
    if report['cuda'].get('jetson_clocks_available'):
        nvidia_tools.append("jetson_clocks")
    
    nvidia_tools_str = ", ".join(nvidia_tools) if nvidia_tools else "No disponible"
    print(f"Herramientas NVIDIA: {nvidia_tools_str}")
    print(f"Paquetes NVIDIA: {report['cuda'].get('nvidia_packages_count', 0)}")
    # Verificar TensorRT (headers o librerías)
    tensorrt_available = (
        report['tensorrt'].get('python_available') or 
        report['tensorrt'].get('libraries_count', 0) > 0 or
        any(report['tensorrt'].get('headers', {}).values())
    )
    print(f"TensorRT: {'Sí' if tensorrt_available else 'No'}")
    print(f"Python Sistema: {report['python']['system']['python3']}")
    print(f"Python Venv: {report['python']['virtual_env'].get('python_version', 'No disponible')}")
    venv_active = report['pytorch'].get('venv_active', False)
    print(f"Venv activo: {'✅ Sí' if venv_active else '❌ No'}")
    print(f"PyTorch: {report['pytorch'].get('version', 'No disponible')}")
    print(f"Torchvision: {report['torchvision'].get('version', 'No disponible')}")
    cuda_avail = report['pytorch'].get('cuda_available', False)
    smoke_test = report['pytorch'].get('gpu_smoke_test', False)
    print(f"CUDA PyTorch: {'✅ Sí' if cuda_avail else '❌ No'}  | Smoke Test: {'✅ OK' if smoke_test else '❌ Falló'}")
    
    # Información detallada de librerías ML - USANDO VENV ACTIVO
    print("\n🤖 LIBRERÍAS DE MACHINE LEARNING:")
    ml_libs = report.get('ml_libraries_detailed', {})
    for lib_name, lib_info in ml_libs.items():
        if lib_info.get('available'):
            # Usar 'cuda' en vez de 'cuda_available'
            cuda_status = "✅ CUDA" if lib_info.get('cuda', False) else "❌ CPU"
            version = lib_info.get('version', 'N/A')
            
            # Casos especiales
            if lib_name == "torchvision" and lib_info.get('cuda_nms', False):
                print(f"   {lib_name}: {version} ({cuda_status} + NMS GPU)")
            elif lib_name == "opencv" and lib_info.get('error_gl'):
                print(f"   {lib_name}: {version} (⚠️ Error OpenGL/TLS)")
            else:
                print(f"   {lib_name}: {version} ({cuda_status})")
        else:
            error = lib_info.get('error', '')
            if 'libGLdispatch' in error or 'TLS' in error:
                print(f"   {lib_name}: ⚠️ Error OpenGL/TLS")
            else:
                print(f"   {lib_name}: ❌ No disponible")
    
    print(f"GPIOD: {'Sí' if report['gpiod'].get('gpioset_available') else 'No'}")
    print(f"Memoria: {report['system']['memory_used_gb']:.1f}GB / {report['system']['memory_total_gb']:.1f}GB")
    print(f"Disco: {report['system']['disk_used_gb']:.1f}GB / {report['system']['disk_total_gb']:.1f}GB")
    
    return report

def generate_replication_files(report):
    """Genera archivos para replicar el sistema completo"""
    replication_dir = Path.cwd() / "replication_files"
    replication_dir.mkdir(exist_ok=True)
    
    print(f"\n📁 Generando archivos de replicación en: {replication_dir}")
    
    # 1. Generar requirements.txt
    requirements_file = replication_dir / "requirements.txt"
    with open(requirements_file, 'w') as f:
        for line in report.get('venv_replication_data', {}).get('requirements', []):
            f.write(line + '\n')
    print(f"✅ {requirements_file}")
    
    # 2. Generar scripts de instalación
    scripts = report.get('replication_scripts', {})
    for script_name, script_content in scripts.items():
        script_file = replication_dir / script_name
        with open(script_file, 'w') as f:
            for line in script_content:
                f.write(line + '\n')
        # Hacer ejecutable
        script_file.chmod(0o755)
        print(f"✅ {script_file}")
    
    # 3. Generar script maestro de replicación
    master_script = replication_dir / "replicate_system.sh"
    with open(master_script, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Script maestro para replicar sistema completo\n")
        f.write("set -e\n\n")
        f.write("echo '🚀 Iniciando replicación del sistema...'\n\n")
        f.write("# Ejecutar scripts en orden\n")
        f.write("chmod +x *.sh\n")
        f.write("./setup_system.sh\n")
        f.write("./install_packages.sh\n")
        f.write("./setup_venv.sh\n")
        f.write("./configure_environment.sh\n\n")
        f.write("# Instalar paquetes Python\n")
        f.write("cd /home/nvidia/Desktop/Calippo_jetson/gentl\n")
        f.write("source .venv/bin/activate\n")
        f.write("pip install -r requirements.txt\n\n")
        f.write("echo '✅ Replicación completada'\n")
    master_script.chmod(0o755)
    print(f"✅ {master_script}")
    
    # 4. Generar archivo de configuración del sistema
    config_file = replication_dir / "system_config.json"
    with open(config_file, 'w') as f:
        json.dump(report.get('system_configuration', {}), f, indent=2)
    print(f"✅ {config_file}")
    
    # 5. Generar README de replicación
    readme_file = replication_dir / "README_REPLICATION.md"
    with open(readme_file, 'w') as f:
        f.write("# Replicación de Sistema Jetson Orin JetPack 5.1.1\n\n")
        f.write("## Archivos incluidos:\n")
        f.write("- `requirements.txt`: Paquetes Python exactos\n")
        f.write("- `setup_system.sh`: Configuración del sistema\n")
        f.write("- `install_packages.sh`: Instalación de paquetes\n")
        f.write("- `setup_venv.sh`: Creación del entorno virtual\n")
        f.write("- `configure_environment.sh`: Variables de entorno\n")
        f.write("- `replicate_system.sh`: Script maestro\n")
        f.write("- `system_config.json`: Configuración del sistema\n\n")
        f.write("## Instrucciones:\n")
        f.write("1. Copiar todos los archivos al nuevo Jetson\n")
        f.write("2. Ejecutar: `chmod +x *.sh`\n")
        f.write("3. Ejecutar: `./replicate_system.sh`\n")
        f.write("4. Reiniciar el sistema\n\n")
        f.write("## Verificación:\n")
        f.write("```bash\n")
        f.write("cd /home/nvidia/Desktop/Calippo_jetson/gentl\n")
        f.write("source .venv/bin/activate\n")
        f.write("python3 diagnostico_jetpack511.py\n")
        f.write("```\n")
    print(f"✅ {readme_file}")
    
    print(f"\n🎉 Archivos de replicación generados en: {replication_dir}")
    return replication_dir

if __name__ == "__main__":
    try:
        import argparse
        parser = argparse.ArgumentParser(description="Diagnóstico JetPack 5.1.1")
        parser.add_argument("--replicate", action="store_true", help="Generar carpeta replication_files")
        args = parser.parse_args()

        report = generate_report()

        if args.replicate:
            replication_dir = generate_replication_files(report)
            print(f"\n🎉 Diagnóstico completado exitosamente")
            print(f"📁 Archivo JSON: {REPORT_FILE}")
            print(f"📁 Archivos de replicación: {replication_dir}")
            print(f"🔄 Para actualizar, ejecuta: python3 {__file__}")
            print(f"\n📋 Para replicar en otro Jetson:")
            print(f"   1. Copiar carpeta: {replication_dir}")
            print(f"   2. Ejecutar: ./replicate_system.sh")
        else:
            print(f"\n🎉 Diagnóstico completado exitosamente")
            print(f"📁 Archivo JSON: {REPORT_FILE}")
            print(f"💡 Usa '--replicate' para generar carpeta de replicación")
    except Exception as e:
        print(f"❌ Error durante el diagnóstico: {e}")
        sys.exit(1)
