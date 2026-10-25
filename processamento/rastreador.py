"""
Rastreador simples de veículos (inspirado no ByteTrack), sem dependências
além de numpy.

- Associação por IoU com a posição prevista (velocidade constante).
- Segunda tentativa por distância do centro, para veículos rápidos
  (principalmente motos) quando analisamos poucos frames por segundo.
- Detecções de baixa confiança só "continuam" veículos já existentes,
  nunca criam veículos novos.
- A associação ignora a classe: se o modelo alterna entre carro e
  caminhão, continua sendo o mesmo veículo (e a classe é decidida por
  votação no final).
"""

import numpy as np


def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def matriz_iou(caixas_a, caixas_b):
    if len(caixas_a) == 0 or len(caixas_b) == 0:
        return np.zeros((len(caixas_a), len(caixas_b)), dtype=np.float32)
    a = np.asarray(caixas_a, dtype=np.float32)[:, None, :]
    b = np.asarray(caixas_b, dtype=np.float32)[None, :, :]
    x1 = np.maximum(a[..., 0], b[..., 0])
    y1 = np.maximum(a[..., 1], b[..., 1])
    x2 = np.minimum(a[..., 2], b[..., 2])
    y2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / (area_a + area_b - inter + 1e-9)


class Trilha:
    """Um veículo sendo acompanhado."""

    def __init__(self, id_trilha, det, frame, n_classes):
        self.id = id_trilha
        self.caixa = np.array(det.caixa, dtype=np.float32)
        self.vel = np.zeros(2, dtype=np.float32)   # pixels por frame do vídeo
        self.ultimo_frame = frame
        self.primeiro_frame = frame
        self.perdidos = 0
        self.historico = [(frame, *det.centro)]

        # Votação de classe
        self.votos = np.zeros(n_classes, dtype=np.float64)
        self.n_deteccoes = 0

        # Preenchidos pelo contador
        self.evento = None
        self.melhor_qualidade = -1.0
        self.recorte_jpg = None

    @property
    def centro(self):
        return ((self.caixa[0] + self.caixa[2]) / 2.0, (self.caixa[1] + self.caixa[3]) / 2.0)

    def caixa_prevista(self, frame):
        dt = frame - self.ultimo_frame
        dx, dy = self.vel * dt
        return self.caixa + np.array([dx, dy, dx, dy], dtype=np.float32)

    def atualizar(self, det, frame):
        cx_ant, cy_ant = self.centro
        nova = np.array(det.caixa, dtype=np.float32)
        cx, cy = (nova[0] + nova[2]) / 2.0, (nova[1] + nova[3]) / 2.0
        dt = max(1, frame - self.ultimo_frame)
        vel_nova = np.array([(cx - cx_ant) / dt, (cy - cy_ant) / dt], dtype=np.float32)
        if self.n_deteccoes <= 1:
            self.vel = vel_nova
        else:
            self.vel = 0.6 * vel_nova + 0.4 * self.vel

        self.caixa = nova
        self.ultimo_frame = frame
        self.perdidos = 0
        self.historico.append((frame, cx, cy))

    def registrar_voto(self, det):
        self.votos += det.pontuacoes
        self.n_deteccoes += 1


class Rastreador:

    def __init__(self, n_classes, conf_nova_trilha=0.35, iou_associacao=0.2, max_frames_perdido=8):
        self.n_classes = n_classes
        self.conf_nova_trilha = conf_nova_trilha
        self.iou_associacao = iou_associacao
        self.max_frames_perdido = max_frames_perdido
        self.trilhas = []
        self._proximo_id = 1

    def _associar_iou(self, trilhas, dets, frame, limiar):
        pares = []
        if not trilhas or not dets:
            return pares, list(range(len(trilhas))), list(range(len(dets)))

        previstas = [t.caixa_prevista(frame) for t in trilhas]
        m = matriz_iou(previstas, [d.caixa for d in dets])

        livres_t = set(range(len(trilhas)))
        livres_d = set(range(len(dets)))
        for idx in np.argsort(-m, axis=None):
            i, j = divmod(int(idx), m.shape[1])
            if m[i, j] < limiar:
                break
            if i in livres_t and j in livres_d:
                pares.append((i, j))
                livres_t.discard(i)
                livres_d.discard(j)

        return pares, sorted(livres_t), sorted(livres_d)

    def _associar_distancia(self, trilhas, dets, idx_t, idx_d, frame):
        """Para veículos rápidos cuja caixa prevista não se sobrepõe à nova."""
        candidatos = []
        for i in idx_t:
            t = trilhas[i]
            p = t.caixa_prevista(frame)
            pcx, pcy = (p[0] + p[2]) / 2, (p[1] + p[3]) / 2
            larg_t, alt_t = p[2] - p[0], p[3] - p[1]
            area_t = max(1.0, larg_t * alt_t)
            # Sem velocidade conhecida (veículo visto 1 vez), procura mais longe:
            # uma moto a 80 km/h anda mais que o próprio comprimento entre
            # dois frames analisados a 10 fps.
            fator = 2.5 if len(t.historico) <= 1 else 1.0
            limite = fator * max(larg_t, alt_t)
            for j in idx_d:
                d = dets[j]
                dcx, dcy = d.centro
                dist = ((pcx - dcx) ** 2 + (pcy - dcy) ** 2) ** 0.5
                area_d = max(1.0, (d.x2 - d.x1) * (d.y2 - d.y1))
                razao = area_d / area_t
                if dist < limite and 0.4 < razao < 2.5:
                    candidatos.append((dist / max(limite, 1.0), i, j))

        candidatos.sort()
        pares = []
        livres_t, livres_d = set(idx_t), set(idx_d)
        for _, i, j in candidatos:
            if i in livres_t and j in livres_d:
                pares.append((i, j))
                livres_t.discard(i)
                livres_d.discard(j)
        return pares, sorted(livres_t), sorted(livres_d)

    def atualizar(self, deteccoes, frame):
        """
        Retorna (pares, finalizadas):
          pares       -> lista de (trilha, deteccao) atualizadas neste frame
          finalizadas -> trilhas que sumiram por tempo demais
        """
        altas = [d for d in deteccoes if d.conf >= self.conf_nova_trilha]
        baixas = [d for d in deteccoes if d.conf < self.conf_nova_trilha]

        trilhas = self.trilhas
        pares_finais = []

        # 1) detecções confiáveis x todas as trilhas (IoU)
        pares, livres_t, livres_d = self._associar_iou(trilhas, altas, frame, self.iou_associacao)
        pares_finais += [(trilhas[i], altas[j]) for i, j in pares]

        # 2) detecções confiáveis restantes x trilhas restantes (distância)
        pares, livres_t, livres_d = self._associar_distancia(trilhas, altas, livres_t, livres_d, frame)
        pares_finais += [(trilhas[i], altas[j]) for i, j in pares]

        # 3) detecções fracas só continuam trilhas existentes
        restantes = [trilhas[i] for i in livres_t]
        pares, livres_r, _ = self._associar_iou(restantes, baixas, frame, self.iou_associacao + 0.1)
        pares_finais += [(restantes[i], baixas[j]) for i, j in pares]
        nao_vistas = [restantes[i] for i in livres_r]

        for trilha, det in pares_finais:
            trilha.atualizar(det, frame)

        # 4) trilhas não vistas
        finalizadas = []
        for t in nao_vistas:
            t.perdidos += 1
            if t.perdidos > self.max_frames_perdido:
                finalizadas.append(t)

        # 5) novas trilhas
        novas = []
        for j in livres_d:
            t = Trilha(self._proximo_id, altas[j], frame, self.n_classes)
            self._proximo_id += 1
            novas.append(t)
            pares_finais.append((t, altas[j]))

        ids_fim = {id(t) for t in finalizadas}
        self.trilhas = [t for t in self.trilhas if id(t) not in ids_fim] + novas

        return pares_finais, finalizadas

    def encerrar_todas(self):
        finalizadas = self.trilhas
        self.trilhas = []
        return finalizadas
