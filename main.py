#!/usr/bin/env python3
"""
Punto de entrada principal de la aplicación de visión industrial.
"""
import sys

def main():
    """Función principal que ejecuta la aplicación."""
    try:
        from gentl.app import main as app_main
        sys.exit(app_main())
    except Exception as e:
        # Fallback para errores tempranos
        print(f"Error al ejecutar la aplicación: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()


