import os
import subprocess
import sys

import cv2
from PIL import Image, ImageTk


def bgr_para_tk(imagem_bgr, largura_max, altura_max, ampliar=False):
    """Converte imagem OpenCV para Tk, cabendo em largura_max x altura_max."""
    alt, larg = imagem_bgr.shape[:2]
    escala = min(largura_max / larg, altura_max / alt)
    if not ampliar:
        escala = min(1.0, escala)
    nova = (max(1, int(larg * escala)), max(1, int(alt * escala)))
    if nova != (larg, alt):
        interp = cv2.INTER_AREA if escala < 1 else cv2.INTER_CUBIC
        imagem_bgr = cv2.resize(imagem_bgr, nova, interpolation=interp)
    rgb = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)
    return ImageTk.PhotoImage(Image.fromarray(rgb)), escala


def abrir_no_sistema(caminho):
    """Abre um arquivo ou pasta com o programa padrão do sistema."""
    try:
        if sys.platform.startswith("win"):
            os.startfile(caminho)  # noqa: S606
        elif sys.platform == "darwin":
            subprocess.Popen(["open", caminho])
        else:
            subprocess.Popen(["xdg-open", caminho])
    except Exception:
        pass
