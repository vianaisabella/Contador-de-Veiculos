"""
Contagem por cruzamento de linha, classificação por votação e marcação
de dúvidas para conferência humana.
"""

import cv2
import numpy as np

from .config import nome_exibicao
from .rastreador import Rastreador, iou


def _lado(p, a, b):
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def cruzamento_segmentos(p1, p2, a, b):
    """
    Retorna (fração t ao longo de p1->p2 onde cruza a linha a-b, sentido)
    ou None se não cruza. sentido = +1 ou -1.
    """
    d1 = _lado(p1, a, b)
    d2 = _lado(p2, a, b)
    if d1 == 0 or d2 == 0 or (d1 > 0) == (d2 > 0):
        return None
    d3 = _lado(a, p1, p2)
    d4 = _lado(b, p1, p2)
    if (d3 > 0) == (d4 > 0) and d3 != 0 and d4 != 0:
        return None
    t = d1 / (d1 - d2)
    return t, (1 if d1 < 0 else -1)


class Contador:

    def __init__(self, classes, linha, fps, config, poligono=None):
        self.classes = list(classes)
        self.linha = (tuple(map(float, linha[0])), tuple(map(float, linha[1])))
        # Linha um pouco mais comprida para o cruzamento (o centro da caixa
        # fica acima do chão e pode passar perto das pontas)
        (ax, ay), (bx, by) = self.linha
        ext = getattr(config, "extensao_linha", 0.0)
        dx, dy = (bx - ax) * ext, (by - ay) * ext
        self.linha_cruzamento = ((ax - dx, ay - dy), (bx + dx, by + dy))
        self.fps = float(fps)
        self.config = config
        self.poligono = np.array(poligono, dtype=np.int32) if poligono is not None else None

        self.rastreador = Rastreador(
            n_classes=len(self.classes),
            conf_nova_trilha=config.conf_nova_trilha,
            iou_associacao=config.iou_associacao,
            max_frames_perdido=config.max_frames_perdido,
        )

        self.eventos = []              # eventos finalizados
        self._recentes = []            # (tempo, caixa) dos últimos cruzamentos
        self._proximo_evento = 1
        self.recortes = {}             # id_evento -> bytes JPG

    # -------------------------------------------------------------

    @property
    def ha_veiculos_ativos(self):
        return bool(self.rastreador.trilhas)

    def contagem_atual(self):
        """Contagem para mostrar durante o processamento."""
        cont = {c: 0 for c in self.classes}
        for e in self.eventos:
            cont[e["classe"]] += 1
        for t in self.rastreador.trilhas:
            if t.evento is not None:
                cont[self.classes[int(np.argmax(t.votos))]] += 1
        return cont

    def n_duvidas(self):
        return sum(1 for e in self.eventos if e["duvida"])

    # -------------------------------------------------------------

    def processar(self, deteccoes, frame_idx, frame):
        pares, finalizadas = self.rastreador.atualizar(deteccoes, frame_idx)


        for trilha, det in pares:
            trilha.registrar_voto(det)
            self._atualizar_recorte(trilha, det, frame)

            if trilha.evento is None and len(trilha.historico) >= 2:
                f_ant, x_ant, y_ant = trilha.historico[-2]
                f_at, x_at, y_at = trilha.historico[-1]
                cruz = cruzamento_segmentos((x_ant, y_ant), (x_at, y_at), *self.linha_cruzamento)
                if cruz is not None:
                    t, sentido = cruz
                    frame_cruz = f_ant + t * (f_at - f_ant)
                    self._registrar_cruzamento(trilha, det, frame_cruz, sentido, deteccoes)

        for trilha in finalizadas:
            self._finalizar(trilha)

    def encerrar(self):
        for trilha in self.rastreador.encerrar_todas():
            self._finalizar(trilha)
        self.eventos.sort(key=lambda e: e["tempo"])

    # -------------------------------------------------------------

    def _atualizar_recorte(self, trilha, det, frame):
        """Guarda a melhor imagem do veículo (maior confiança x tamanho, sem cortar na borda)."""
        alt, larg = frame.shape[:2]
        x1, y1, x2, y2 = det.caixa
        area = max(1.0, (x2 - x1) * (y2 - y1))
        na_borda = x1 <= 2 or y1 <= 2 or x2 >= larg - 3 or y2 >= alt - 3
        qualidade = det.conf * np.sqrt(area) * (0.5 if na_borda else 1.0)

        if qualidade <= trilha.melhor_qualidade:
            return

        mx, my = 0.25 * (x2 - x1), 0.25 * (y2 - y1)
        a = int(max(0, x1 - mx)), int(max(0, y1 - my))
        b = int(min(larg, x2 + mx)), int(min(alt, y2 + my))
        recorte = frame[a[1]:b[1], a[0]:b[0]]
        if recorte.size == 0:
            return

        lado = max(recorte.shape[:2])
        if lado > 320:
            esc = 320 / lado
            recorte = cv2.resize(recorte, None, fx=esc, fy=esc, interpolation=cv2.INTER_AREA)

        ok, jpg = cv2.imencode(".jpg", recorte, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            trilha.melhor_qualidade = qualidade
            trilha.recorte_jpg = jpg.tobytes()

    def _registrar_cruzamento(self, trilha, det, frame_cruz, sentido, deteccoes):
        frame_cruz = float(frame_cruz)
        tempo = frame_cruz / self.fps
        motivos = []

        # Outro veículo grudado neste no momento do cruzamento
        for outra in deteccoes:
            if outra is det:
                continue
            if iou(det.caixa, outra.caixa) > self.config.limiar_sobreposicao:
                motivos.append("veículos sobrepostos")
                break

        # Cruzamento quase no mesmo lugar e hora que outro: pode ser o
        # mesmo veículo contado duas vezes (troca de ID)
        janela = self.config.janela_duplicada_seg
        self._recentes = [(tp, cx) for tp, cx in self._recentes if tempo - tp <= janela]
        for tp, cx in self._recentes:
            if iou(det.caixa, cx) > 0.3:
                motivos.append("possível contagem dupla")
                break
        self._recentes.append((tempo, det.caixa))

        trilha.evento = {
            "frame": int(round(frame_cruz)),
            "tempo": round(tempo, 2),
            "sentido": int(sentido),
            "caixa": [round(float(v), 1) for v in det.caixa],
            "motivos": motivos,
        }

    def _finalizar(self, trilha):
        if trilha.evento is None:
            return

        ev = trilha.evento
        cfg = self.config

        total = float(trilha.votos.sum())
        if total > 0:
            proporcoes = trilha.votos / total
        else:
            proporcoes = np.ones(len(self.classes)) / len(self.classes)

        ordem = np.argsort(-proporcoes)
        k = int(ordem[0])
        classe = self.classes[k]
        proporcao = float(proporcoes[k])
        conf_media = float(trilha.votos[k] / max(1, trilha.n_deteccoes))

        motivos = list(ev["motivos"])
        if proporcao < cfg.limiar_votacao and len(self.classes) > 1:
            segunda = self.classes[int(ordem[1])]
            motivos.insert(
                0,
                f"{nome_exibicao(classe)} ou {nome_exibicao(segunda)}? "
                f"({proporcao:.0%} x {float(proporcoes[ordem[1]]):.0%})",
            )
        if conf_media < cfg.limiar_confianca:
            motivos.append(f"confiança baixa ({conf_media:.2f})")
        if trilha.n_deteccoes < cfg.min_deteccoes:
            motivos.append(f"visto em poucos frames ({trilha.n_deteccoes})")

        id_evento = self._proximo_evento
        self._proximo_evento += 1

        if trilha.recorte_jpg:
            self.recortes[id_evento] = trilha.recorte_jpg

        self.eventos.append({
            "id": id_evento,
            "trilha": trilha.id,
            "frame": ev["frame"],
            "tempo": ev["tempo"],
            "classe": classe,
            "classe_sugerida": classe,
            "votos": {self.classes[i]: round(float(proporcoes[i]), 3) for i in range(len(self.classes))},
            "confianca": round(conf_media, 3),
            "deteccoes": int(trilha.n_deteccoes),
            "sentido": ev["sentido"],
            "caixa": ev["caixa"],
            "duvida": bool(motivos),
            "motivos": motivos,
            "origem": "automatico",
            "revisado": False,
            "descartado": False,
            "recorte": None,
        })


# =================================================================
# FUSÃO: modelo + movimento
# =================================================================

def fundir_eventos(eventos_modelo, recortes_modelo, eventos_mov, recortes_mov, config):
    """
    Junta os cruzamentos do modelo com os detectados só por movimento.
    Um cruzamento por movimento sem nenhum do modelo por perto (tempo e
    sentido) vira um evento "não classificado" para revisão humana.
    Cruzamentos por movimento muito próximos entre si são agrupados
    (a mancha de um veículo às vezes se divide em várias).
    """
    janela = config.janela_fusao_seg
    min_det = config.min_deteccoes_movimento

    candidatos = sorted(
        (e for e in eventos_mov if e["deteccoes"] >= min_det), key=lambda e: e["tempo"])
    grupos = []
    for e in candidatos:
        if grupos and e["tempo"] - grupos[-1][-1]["tempo"] <= janela:
            grupos[-1].append(e)
        else:
            grupos.append([e])

    finais = [(e, recortes_modelo.get(e["id"])) for e in eventos_modelo]
    tempos_modelo = [e["tempo"] for e in eventos_modelo]

    for grupo in grupos:
        inicio, fim = grupo[0]["tempo"], grupo[-1]["tempo"]
        if any(inicio - janela <= t <= fim + janela for t in tempos_modelo):
            continue
        # a mancha que ficou mais tempo visível representa o grupo
        rep = max(grupo, key=lambda e: e["deteccoes"])
        novo = dict(rep)
        novo.update({
            "classe": "indefinido",
            "classe_sugerida": None,
            "votos": {},
            "confianca": None,
            "duvida": True,
            "motivos": ["só detectado por movimento: confirme se é veículo e a classe"],
            "origem": "movimento",
        })
        finais.append((novo, recortes_mov.get(rep["id"])))

    finais.sort(key=lambda par: par[0]["tempo"])
    eventos, recortes = [], {}
    for novo_id, (e, jpg) in enumerate(finais, start=1):
        e = dict(e)
        e["id"] = novo_id
        eventos.append(e)
        if jpg:
            recortes[novo_id] = jpg
    return eventos, recortes
