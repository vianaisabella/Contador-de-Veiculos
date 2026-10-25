"""
Detector de veículos sem PyTorch.

Roda um modelo YOLO exportado para OpenVINO (preferido em CPU Intel) ou
ONNX (via onnxruntime). O pré e pós-processamento são feitos aqui com
numpy/OpenCV, então o .exe não precisa de torch nem ultralytics.

Para gerar o modelo, rode uma vez:  python scripts/exportar_modelo.py
"""

import ast
import os
import re
from dataclasses import dataclass

import cv2
import numpy as np

from .config import mapear_nome_classe, ordenar_classes


# Nomes do COCO, usados se o modelo não trouxer metadados.
NOMES_COCO = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


@dataclass
class Deteccao:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float          # confiança da melhor classe
    classe: str          # melhor classe (nome interno)
    pontuacoes: np.ndarray  # pontuação de cada classe em Detector.classes

    @property
    def centro(self):
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def caixa(self):
        return (self.x1, self.y1, self.x2, self.y2)


# =================================================================
# LEITURA DOS NOMES DE CLASSE
# =================================================================

def _nomes_de_metadata_yaml(caminho_yaml):
    """Lê a seção 'names:' do metadata.yaml do Ultralytics sem precisar de PyYAML."""
    nomes = {}
    if not os.path.isfile(caminho_yaml):
        return None
    dentro = False
    with open(caminho_yaml, encoding="utf-8") as f:
        for linha in f:
            if linha.strip().startswith("names:"):
                dentro = True
                continue
            if dentro:
                m = re.match(r"^\s+(\d+):\s*(.+?)\s*$", linha)
                if m:
                    nomes[int(m.group(1))] = m.group(2).strip("'\"")
                elif linha.strip() and not linha.startswith((" ", "\t")):
                    break
    if not nomes:
        return None
    return [nomes[i] for i in sorted(nomes)]


def _nomes_de_string(texto):
    try:
        dados = ast.literal_eval(texto)
        if isinstance(dados, dict):
            return [dados[k] for k in sorted(dados)]
        if isinstance(dados, (list, tuple)):
            return list(dados)
    except Exception:
        pass
    return None


# =================================================================
# DETECTOR
# =================================================================

class Detector:

    def __init__(self, caminho_modelo, conf_minima=0.15, iou_nms=0.5, threads=None):

        if not caminho_modelo or not os.path.exists(caminho_modelo):
            raise FileNotFoundError(
                "Modelo não encontrado. Rode 'python scripts/exportar_modelo.py' "
                "para gerar a pasta modelos/."
            )

        self.caminho_modelo = caminho_modelo
        self.conf_minima = conf_minima
        self.iou_nms = iou_nms
        self.backend = None
        self.nomes_modelo = None
        self.tamanho_entrada = (640, 640)  # (altura, largura)

        if os.path.isdir(caminho_modelo) or caminho_modelo.lower().endswith(".xml"):
            self._carregar_openvino(caminho_modelo, threads)
        else:
            self._carregar_onnx(caminho_modelo, threads)

        if not self.nomes_modelo:
            self.nomes_modelo = NOMES_COCO

        # Índice do modelo -> índice em self.classes
        mapa = {}
        for indice, nome in enumerate(self.nomes_modelo):
            interno = mapear_nome_classe(nome)
            if interno:
                mapa[indice] = interno

        if not mapa:
            raise ValueError(
                "O modelo não tem nenhuma classe de veículo reconhecida. "
                f"Classes do modelo: {self.nomes_modelo}"
            )

        self.classes = ordenar_classes(mapa.values())
        self._indices_modelo = np.array(sorted(mapa), dtype=np.int64)
        self._destino = np.array(
            [self.classes.index(mapa[i]) for i in self._indices_modelo],
            dtype=np.int64,
        )

    # -------------------------------------------------------------
    # CARREGAMENTO
    # -------------------------------------------------------------

    def _carregar_openvino(self, caminho, threads):
        import openvino as ov

        if os.path.isdir(caminho):
            xmls = [f for f in os.listdir(caminho) if f.endswith(".xml")]
            if not xmls:
                raise FileNotFoundError(f"Nenhum .xml em {caminho}")
            pasta = caminho
            caminho_xml = os.path.join(caminho, xmls[0])
        else:
            pasta = os.path.dirname(caminho)
            caminho_xml = caminho

        core = ov.Core()
        modelo = core.read_model(caminho_xml)

        self.nomes_modelo = _nomes_de_metadata_yaml(os.path.join(pasta, "metadata.yaml"))

        if not self.nomes_modelo:
            try:
                rotulos = modelo.get_rt_info(["model_info", "labels"]).astype(str)
                self.nomes_modelo = [n.replace("_", " ") for n in rotulos.split()]
            except Exception:
                pass

        forma = modelo.input(0).get_partial_shape()
        if forma.rank.get_length() == 4 and forma[2].is_static and forma[3].is_static:
            self.tamanho_entrada = (forma[2].get_length(), forma[3].get_length())

        config = {"PERFORMANCE_HINT": "LATENCY"}
        if threads:
            config["INFERENCE_NUM_THREADS"] = int(threads)

        self._compilado = core.compile_model(modelo, "CPU", config)
        self._requisicao = self._compilado.create_infer_request()
        self.backend = "OpenVINO"

    def _carregar_onnx(self, caminho, threads):
        try:
            import onnxruntime as ort
        except ImportError:
            # OpenVINO também lê ONNX diretamente
            self._carregar_openvino(caminho, threads)
            self.backend = "OpenVINO (ONNX)"
            return

        opcoes = ort.SessionOptions()
        if threads:
            opcoes.intra_op_num_threads = int(threads)

        self._sessao = ort.InferenceSession(
            caminho, sess_options=opcoes, providers=["CPUExecutionProvider"]
        )
        self._nome_entrada = self._sessao.get_inputs()[0].name

        forma = self._sessao.get_inputs()[0].shape
        if len(forma) == 4 and isinstance(forma[2], int) and isinstance(forma[3], int):
            self.tamanho_entrada = (forma[2], forma[3])

        meta = self._sessao.get_modelmeta().custom_metadata_map
        if "names" in meta:
            self.nomes_modelo = _nomes_de_string(meta["names"])

        self.backend = "ONNX Runtime"

    # -------------------------------------------------------------
    # INFERÊNCIA
    # -------------------------------------------------------------

    def _inferir(self, tensor):
        if hasattr(self, "_sessao"):
            return self._sessao.run(None, {self._nome_entrada: tensor})[0]
        self._requisicao.infer({0: tensor})
        return self._requisicao.get_output_tensor(0).data

    @staticmethod
    def letterbox(imagem, tamanho):
        """Redimensiona mantendo proporção e completa com cinza (114)."""
        alt_dest, larg_dest = tamanho
        alt, larg = imagem.shape[:2]
        r = min(alt_dest / alt, larg_dest / larg)
        nova_larg, nova_alt = int(round(larg * r)), int(round(alt * r))

        if (nova_larg, nova_alt) != (larg, alt):
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            imagem = cv2.resize(imagem, (nova_larg, nova_alt), interpolation=interp)

        pad_x = (larg_dest - nova_larg) / 2
        pad_y = (alt_dest - nova_alt) / 2
        topo, base = int(round(pad_y - 0.1)), int(round(pad_y + 0.1))
        esq, dir_ = int(round(pad_x - 0.1)), int(round(pad_x + 0.1))

        imagem = cv2.copyMakeBorder(
            imagem, topo, base, esq, dir_, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        return imagem, r, esq, topo

    def preparar(self, imagem_bgr):
        img, r, pad_x, pad_y = self.letterbox(imagem_bgr, self.tamanho_entrada)
        tensor = cv2.dnn.blobFromImage(img, 1 / 255.0, swapRB=True)  # 1x3xHxW float32
        return tensor, r, pad_x, pad_y

    def pos_processar(self, saida, r, pad_x, pad_y, largura, altura):
        """
        Saída do YOLOv8/YOLO11: (1, 4 + nc, N) com cx, cy, w, h em pixels
        da entrada e pontuações por classe (já com sigmoid).
        """
        saida = np.asarray(saida)
        if saida.ndim == 3:
            saida = saida[0]
        if saida.shape[0] < saida.shape[1]:
            saida = saida.T  # -> (N, 4 + nc)

        caixas = saida[:, :4]
        pont_modelo = saida[:, 4:]

        # Pontuações só das classes de veículo, agrupadas pelo nome interno
        pont_veic = pont_modelo[:, self._indices_modelo]
        pont = np.zeros((saida.shape[0], len(self.classes)), dtype=np.float32)
        for col, destino in enumerate(self._destino):
            pont[:, destino] = np.maximum(pont[:, destino], pont_veic[:, col])

        melhor = pont.max(axis=1)
        manter = melhor >= self.conf_minima
        if not np.any(manter):
            return []

        caixas, pont, melhor = caixas[manter], pont[manter], melhor[manter]

        cx, cy, w, h = caixas[:, 0], caixas[:, 1], caixas[:, 2], caixas[:, 3]
        x1 = (cx - w / 2 - pad_x) / r
        y1 = (cy - h / 2 - pad_y) / r
        x2 = (cx + w / 2 - pad_x) / r
        y2 = (cy + h / 2 - pad_y) / r

        x1 = np.clip(x1, 0, largura - 1)
        x2 = np.clip(x2, 0, largura - 1)
        y1 = np.clip(y1, 0, altura - 1)
        y2 = np.clip(y2, 0, altura - 1)

        # NMS sem considerar a classe: carro e caminhão no mesmo lugar
        # são o mesmo veículo com classificação em dúvida.
        lista_caixas = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
        indices = cv2.dnn.NMSBoxes(
            lista_caixas, melhor.tolist(), self.conf_minima, self.iou_nms
        )
        if len(indices) == 0:
            return []

        resultado = []
        for i in np.array(indices).reshape(-1):
            if x2[i] - x1[i] < 2 or y2[i] - y1[i] < 2:
                continue
            k = int(np.argmax(pont[i]))
            resultado.append(
                Deteccao(
                    float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i]),
                    float(melhor[i]), self.classes[k], pont[i].copy(),
                )
            )
        return resultado

    def detectar(self, imagem_bgr):
        altura, largura = imagem_bgr.shape[:2]
        tensor, r, pad_x, pad_y = self.preparar(imagem_bgr)
        saida = self._inferir(tensor)
        return self.pos_processar(saida, r, pad_x, pad_y, largura, altura)

    def descricao(self):
        nome = os.path.basename(self.caminho_modelo.rstrip("/\\"))
        alt, larg = self.tamanho_entrada
        return f"{nome} ({self.backend}, {larg}x{alt})"
