"""
AppController (UI)
------------------
- Coordina eventos de la UI (clics, botones) y los traduce a acciones
  del `App` usando la cola de eventos.
- Se llama desde `app.py` en el bucle principal y utiliza `handlers.py`.
"""
"""
AppController: capa fina para despachar acciones de UI desde la cola
sin mezclar la lógica en el bucle principal de App.
"""
from __future__ import annotations

from typing import Dict, Any
import cv2
import builtins

from core.logging import log_info, log_warning
from ui.handlers import handle_action


class AppController:
    """Controlador de acciones del panel: lee de la cola y delega en handlers.
    Mantiene a `App` como orquestador sin lógica de negocio de UI.
    """
    def process_pending(self, app) -> Dict[str, Any]:
        """Procesa una acción pendiente en la cola si existe.
        Devuelve un dict con banderas como 'acquisition_running' o 'record_start'."""
        try:
            accion = app.context.evt_queue.get_nowait()
        except Exception:
            return {}

        if not accion:
            return {}

        log_info(f"🎯 Procesando acción: {accion}")
        try:
            resp = handle_action(app, accion)
            log_info(f"✅ Respuesta de acción: {resp}")
            return resp or {}
        except Exception as e:
            log_warning(f"⚠️ Error procesando acción {accion}: {e}")
            return {}

    def handle_mouse_click(self, event, x, y, flags, app):
        """Callback de ratón para la ventana principal."""
        if event == cv2.EVENT_LBUTTONDOWN:
            try:
                # Debug de clics
                panel_offset_x = getattr(builtins, 'panel_offset_x', 0)
                current_img_h = getattr(builtins, 'current_img_h', 720)
                from core.logging import log_info
                log_info(f"[click] x={x}, y={y}, panel_x={x - panel_offset_x}, panel_y={y}")

                from ui.panel import detectar_clic_panel_control
                panel_h_effective = max(current_img_h, 900)
                accion = detectar_clic_panel_control(x, y, panel_offset_x, panel_h_effective)
                log_info(f"🔍 Detección de clic: panel_offset_x={panel_offset_x}, panel_h_effective={panel_h_effective}, accion={accion}")

                if accion:
                    log_info(f"📤 Enviando acción a cola: {accion}")
                    app.context.evt_queue.put(accion)
            except Exception:
                from core.logging import log_warning
                log_warning("⚠️ Error procesando clic")


