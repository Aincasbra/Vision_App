import builtins


def set_ui_dimensions(width: int, height: int) -> None:
    """Expone dimensiones actuales de la imagen a builtins.
    Esto permite que el manejador de clics calcule correctamente offsets.
    """
    try:
        builtins.current_img_w = int(width)
        builtins.current_img_h = int(height)
        builtins.panel_offset_x = int(width)
    except Exception:
        pass


