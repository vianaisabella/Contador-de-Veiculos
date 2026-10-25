"""
Configurações e utilidades compartilhadas.

Os valores padrão aqui foram escolhidos para começar. Depois de testar com
os vídeos reais, ajuste principalmente a seção "REGRAS DE DÚVIDA".
"""

import os
import sys
import unicodedata
from dataclasses import dataclass, field, asdict


# =================================================================
# CLASSES
# =================================================================

# Nome interno -> nome exibido na tela e no Excel
NOMES_EXIBICAO = {
    "carro": "Carro",
    "moto": "Moto",
    "caminhao": "Caminhão",
    "onibus": "Ônibus",
    "utilitario": "Utilitário",
    "van": "Van",
    "bicicleta": "Bicicleta",
    "indefinido": "Não classificado",
}

# Nomes de classe que o modelo pode ter -> nome interno.
# Inclui os nomes do COCO (modelo yolo11n.pt original) e nomes em
# português, para quando vocês treinarem um modelo próprio.
MAPA_NOMES_MODELO = {
    "car": "carro",
    "carro": "carro",
    "automovel": "carro",
    "motorcycle": "moto",
    "moto": "moto",
    "motocicleta": "moto",
    "truck": "caminhao",
    "caminhao": "caminhao",
    "bus": "onibus",
    "onibus": "onibus",
    "utilitario": "utilitario",
    "van": "van",
    "pickup": "utilitario",
    "caminhonete": "utilitario",
}

# Ordem preferida das colunas no Excel e dos botões na revisão
ORDEM_CLASSES = [
    "carro",
    "moto",
    "caminhao",
    "onibus",
    "utilitario",
    "van",
    "bicicleta",
    "indefinido",
]


def normalizar_texto(texto):
    texto = unicodedata.normalize("NFKD", str(texto))
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return texto.strip().lower().replace(" ", "_").replace("-", "_")


def mapear_nome_classe(nome_modelo):
    """Converte o nome de classe do modelo para o nome interno (ou None)."""
    return MAPA_NOMES_MODELO.get(normalizar_texto(nome_modelo))


def nome_exibicao(classe):
    return NOMES_EXIBICAO.get(classe, str(classe).capitalize())


def ordenar_classes(classes):
    def chave(c):
        return (
            ORDEM_CLASSES.index(c) if c in ORDEM_CLASSES else 99,
            c,
        )
    return sorted(set(classes), key=chave)


# =================================================================
# PARÂMETROS DO PROCESSAMENTO
# =================================================================

@dataclass
class ConfigProcessamento:

    # Quantos frames por segundo de vídeo são analisados.
    # 10 é um bom equilíbrio; suba para 15 se a via for rápida
    # (motos se perdendo), desça para 6-8 se a CPU for muito lenta.
    fps_analise: float = 10.0

    # Tamanho de entrada do modelo (precisa ser o mesmo usado no export).
    # Lido automaticamente do modelo; este valor é só o padrão.
    imgsz: int = 640

    # Detector
    conf_minima: float = 0.15       # detecções abaixo disso são ignoradas
    conf_nova_trilha: float = 0.35  # confiança mínima para criar um veículo novo
    iou_nms: float = 0.50

    # Rastreador
    max_frames_perdido: int = 8     # frames analisados sem ver o veículo antes de encerrá-lo
    iou_associacao: float = 0.20

    # Filtro de movimento (pula o detector quando a pista está vazia)
    usar_filtro_movimento: bool = True
    limiar_movimento: float = 0.004   # fração da área com movimento
    forcar_deteccao_a_cada: int = 15  # roda o detector a cada N frames mesmo sem movimento

    # Contagem por movimento (veículos que o modelo não reconhece viram
    # "não classificado" e vão para a revisão)
    contar_por_movimento: bool = True
    area_min_movimento: int = 2500        # pixels (no vídeo original)
    limiar_variacao_movimento: float = 25.0
    min_deteccoes_movimento: int = 3
    janela_fusao_seg: float = 2.0

    # Margem em volta da área ao recortar para o detector. O veículo é
    # acompanhado em toda essa região e contado quando cruza a linha.
    margem_recorte: int = 150            # mínimo, em pixels
    margem_proporcional: float = 0.6     # fração do tamanho da área
    # A linha de contagem é estendida um pouco além dos lados da área
    extensao_linha: float = 0.15

    # ---------------- REGRAS DE DÚVIDA ----------------
    # Classe vencedora com menos que esta fração dos votos -> dúvida
    limiar_votacao: float = 0.65
    # Confiança média da classe vencedora abaixo disto -> dúvida
    limiar_confianca: float = 0.40
    # Veículo visto em menos frames que isto -> dúvida
    min_deteccoes: int = 3
    # Outro veículo sobreposto no momento do cruzamento -> dúvida
    limiar_sobreposicao: float = 0.25
    # Dois cruzamentos muito próximos no tempo e no espaço -> possível contagem dupla
    janela_duplicada_seg: float = 1.0

    # Intervalo do resumo (segundos)
    intervalo_resumo: int = 300

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, dados):
        validos = {k: v for k, v in (dados or {}).items() if k in cls.__dataclass_fields__}
        return cls(**validos)


# =================================================================
# CAMINHOS
# =================================================================

def pasta_recursos():
    """Pasta onde ficam os arquivos empacotados (modelos)."""
    if getattr(sys, "frozen", False):
        return getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pasta_aplicativo():
    """Pasta ao lado do .exe (ou raiz do projeto), onde ficam os resultados."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pasta_resultados():
    caminho = os.path.join(pasta_aplicativo(), "resultados")
    os.makedirs(caminho, exist_ok=True)
    return caminho


def listar_modelos():
    """Todos os modelos em modelos/ (OpenVINO primeiro, depois ONNX), sem repetir nomes."""
    encontrados = []
    vistos = set()
    for pasta in (os.path.join(pasta_aplicativo(), "modelos"), os.path.join(pasta_recursos(), "modelos")):
        if not os.path.isdir(pasta):
            continue
        itens = sorted(os.listdir(pasta))
        for item in itens:
            caminho = os.path.join(pasta, item)
            if os.path.isdir(caminho) and item.endswith("_openvino_model") and item not in vistos:
                encontrados.append(caminho)
                vistos.add(item)
        for item in itens:
            if item.lower().endswith(".onnx") and item not in vistos:
                encontrados.append(os.path.join(pasta, item))
                vistos.add(item)
    return encontrados


def _arquivo_preferencias():
    return os.path.join(pasta_aplicativo(), "preferencias.json")


def ler_preferencia(chave, padrao=None):
    try:
        import json
        with open(_arquivo_preferencias(), encoding="utf-8") as f:
            return json.load(f).get(chave, padrao)
    except Exception:
        return padrao


def salvar_preferencia(chave, valor):
    import json
    try:
        try:
            with open(_arquivo_preferencias(), encoding="utf-8") as f:
                dados = json.load(f)
        except Exception:
            dados = {}
        dados[chave] = valor
        with open(_arquivo_preferencias(), "w", encoding="utf-8") as f:
            json.dump(dados, f, ensure_ascii=False, indent=1)
    except Exception:
        pass


def encontrar_modelo():
    """
    Procura um modelo em modelos/ (ao lado do exe primeiro, depois nos
    recursos empacotados). Preferência: OpenVINO > ONNX.
    """
    candidatos_pastas = [
        os.path.join(pasta_aplicativo(), "modelos"),
        os.path.join(pasta_recursos(), "modelos"),
    ]

    for pasta in candidatos_pastas:
        if not os.path.isdir(pasta):
            continue

        itens = sorted(os.listdir(pasta))

        for item in itens:
            caminho = os.path.join(pasta, item)
            if os.path.isdir(caminho) and item.endswith("_openvino_model"):
                return caminho

        for item in itens:
            if item.lower().endswith(".onnx"):
                return os.path.join(pasta, item)

    return None


def formatar_tempo(segundos, com_decimos=False):
    segundos = max(0.0, float(segundos))
    horas = int(segundos // 3600)
    minutos = int((segundos % 3600) // 60)
    seg = segundos % 60
    if com_decimos:
        return f"{horas:02d}:{minutos:02d}:{seg:04.1f}"
    return f"{horas:02d}:{minutos:02d}:{int(seg):02d}"


def interpretar_tempo(texto):
    """'1:02:03', '62:03' ou '3723' -> segundos (float)."""
    partes = str(texto).strip().replace(",", ".").split(":")
    if not partes or any(p == "" for p in partes):
        raise ValueError("Tempo inválido")
    valores = [float(p) for p in partes]
    total = 0.0
    for v in valores:
        total = total * 60 + v
    return total
