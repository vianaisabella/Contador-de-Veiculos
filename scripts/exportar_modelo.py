"""
Converte o modelo YOLO (.pt) para OpenVINO e ONNX, que rodam sem PyTorch.

Rode UMA vez no ambiente de desenvolvimento (precisa do ultralytics):

    python scripts/exportar_modelo.py
    python scripts/exportar_modelo.py --pesos runs/detect/train/weights/best.pt   (modelo treinado)

O resultado vai para a pasta modelos/. O programa usa OpenVINO se existir,
senão ONNX.
"""

import argparse
import os
import shutil
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="Exporta o modelo YOLO para OpenVINO/ONNX")
    parser.add_argument("--pesos", default=os.path.join(RAIZ, "yolo11n.pt"), help="arquivo .pt")
    parser.add_argument("--imgsz", type=int, default=640, help="tamanho de entrada (padrão 640)")
    parser.add_argument("--formato", choices=["openvino", "onnx", "ambos"], default="ambos")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("Instale as dependências de desenvolvimento: pip install -r requirements-dev.txt")

    nome = os.path.basename(args.pesos)
    oficial = nome.startswith(("yolo11", "yolov8", "yolo12")) and nome.endswith(".pt")
    if not os.path.isfile(args.pesos) and not oficial:
        sys.exit(f"Arquivo não encontrado: {args.pesos}")
    if not os.path.isfile(args.pesos):
        args.pesos = nome  # modelo oficial: o ultralytics baixa sozinho

    destino = os.path.join(RAIZ, "modelos")
    os.makedirs(destino, exist_ok=True)

    modelo = YOLO(args.pesos)
    print("Classes do modelo:", modelo.names)

    if args.formato in ("openvino", "ambos"):
        caminho = modelo.export(format="openvino", imgsz=args.imgsz, half=False, dynamic=False)
        alvo = os.path.join(destino, os.path.basename(str(caminho).rstrip("/\\")))
        if os.path.exists(alvo):
            shutil.rmtree(alvo)
        shutil.move(str(caminho), alvo)
        print("OpenVINO ->", alvo)

    if args.formato in ("onnx", "ambos"):
        caminho = modelo.export(format="onnx", imgsz=args.imgsz, dynamic=False, simplify=True)
        alvo = os.path.join(destino, os.path.basename(str(caminho)))
        shutil.move(str(caminho), alvo)
        print("ONNX ->", alvo)

    print("\nPronto. Se houver modelos antigos em modelos/, apague-os: o programa usa o primeiro que encontrar.")


if __name__ == "__main__":
    main()
