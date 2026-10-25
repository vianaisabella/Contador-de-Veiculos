"""
Mede a velocidade do detector NA MÁQUINA onde o programa vai rodar e
estima quanto tempo leva um vídeo de 1 h e de 2 h.

    python scripts/medir_velocidade.py caminho/do/video.mp4
    python scripts/medir_velocidade.py video.mp4 --fps-analise 10 --frames 150
"""

import argparse
import os
import sys
import time

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

import cv2  # noqa: E402

from processamento.config import encontrar_modelo, formatar_tempo  # noqa: E402
from processamento.detector import Detector  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--modelo", default=None, help="padrão: o primeiro em modelos/")
    parser.add_argument("--fps-analise", type=float, default=10.0)
    parser.add_argument("--frames", type=int, default=100, help="quantos frames medir")
    args = parser.parse_args()

    caminho = args.modelo or encontrar_modelo()
    print("Carregando", caminho)
    det = Detector(caminho)
    print("Modelo:", det.descricao(), "| classes:", det.classes)

    video = cv2.VideoCapture(args.video)
    fps = video.get(cv2.CAP_PROP_FPS) or 30
    passo = max(1, round(fps / args.fps_analise))

    t_leitura = t_det = 0.0
    n = 0
    for _ in range(5):  # aquecimento
        ok, frame = video.read()
        if ok:
            det.detectar(frame)

    while n < args.frames:
        t0 = time.perf_counter()
        for _ in range(passo - 1):
            video.grab()
        ok, frame = video.read()
        if not ok:
            break
        t1 = time.perf_counter()
        det.detectar(frame)
        t2 = time.perf_counter()
        t_leitura += t1 - t0
        t_det += t2 - t1
        n += 1

    if n == 0:
        sys.exit("Não foi possível ler o vídeo.")

    ms_leitura = 1000 * t_leitura / n
    ms_det = 1000 * t_det / n
    por_seg_video = args.fps_analise * (ms_leitura + ms_det) / 1000

    print(f"\nFrame inteiro ({frame.shape[1]}x{frame.shape[0]}):")
    print(f"  leitura do vídeo: {ms_leitura:6.1f} ms por frame analisado")
    print(f"  detector:         {ms_det:6.1f} ms por frame")
    print(f"\nAnalisando {args.fps_analise:g} frames por segundo de vídeo (pior caso, detector sempre ligado):")
    print(f"  velocidade: {1 / por_seg_video:.1f}x o tempo real")
    print(f"  vídeo de 1 h: ~{formatar_tempo(3600 * por_seg_video)}")
    print(f"  vídeo de 2 h: ~{formatar_tempo(7200 * por_seg_video)}")
    print("\nNo programa costuma ser mais rápido: o filtro de movimento pula trechos sem veículos.")


if __name__ == "__main__":
    main()
