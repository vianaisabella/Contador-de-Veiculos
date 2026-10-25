"""
Extrai frames dos vídeos para anotar e treinar um modelo próprio
(carro / moto / caminhão / ônibus nas câmeras de vocês).

    python scripts/extrair_frames.py video1.mp4 video2.mp4 --a-cada 4 --saida dataset/imagens

Dica: pegue vídeos de horários e câmeras diferentes (dia, noite, chuva).
Umas 300 a 800 imagens já fazem bastante diferença.
"""

import argparse
import os

import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", nargs="+")
    parser.add_argument("--a-cada", type=float, default=4.0, help="segundos entre frames")
    parser.add_argument("--saida", default="dataset/imagens")
    parser.add_argument("--maximo", type=int, default=400, help="máximo por vídeo")
    args = parser.parse_args()

    os.makedirs(args.saida, exist_ok=True)

    for caminho in args.videos:
        video = cv2.VideoCapture(caminho)
        fps = video.get(cv2.CAP_PROP_FPS) or 30
        total = int(video.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        passo = max(1, int(fps * args.a_cada))
        nome = os.path.splitext(os.path.basename(caminho))[0]
        salvos = 0

        for n in range(0, total, passo):
            video.set(cv2.CAP_PROP_POS_FRAMES, n)
            ok, frame = video.read()
            if not ok:
                break
            # imencode + tofile funciona com acentos no caminho (cv2.imwrite não, no Windows)
            ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if ok:
                jpg.tofile(os.path.join(args.saida, f"{nome}_{n:07d}.jpg"))
            salvos += 1
            if salvos >= args.maximo:
                break

        video.release()
        print(f"{caminho}: {salvos} imagens")


if __name__ == "__main__":
    main()
