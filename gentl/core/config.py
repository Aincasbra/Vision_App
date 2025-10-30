import yaml


def load_yaml_config(config_path="config_yolo.yaml"):
    """Carga configuración desde archivo YAML con fallback a defaults vacíos."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"✅ Configuración cargada desde {config_path}")
        return config or {}
    except FileNotFoundError:
        print(f"⚠️ Archivo {config_path} no encontrado, usando configuración por defecto")
        return {}
    except Exception as e:
        print(f"❌ Error cargando {config_path}: {e}")
        return {}


