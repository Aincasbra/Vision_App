#!/usr/bin/env python3
"""
Entry point (main)
------------------
- Lanza la aplicación importando `vision_app.app.main()`.
- Usado por el servicio systemd `vision-app.service` y para ejecución manual.
"""
import sys

def main():
    """Función principal que ejecuta la aplicación."""
    try:
        from vision_app.app import main as app_main
        sys.exit(app_main())
    except Exception as e:
        # Fallback para errores tempranos
        print(f"Error al ejecutar la aplicación: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()


