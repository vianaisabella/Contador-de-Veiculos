"""
Processamento do vídeo em segundo plano, o mais rápido que a CPU permitir.

Não segue o relógio: o tempo de cada veículo vem do número do frame,
então é exato mesmo pulando frames.

Comunicação com a interface pela fila `fila`:
  ("progresso", dict)    a cada ~0,5 s
  ("previa", imagem_bgr) a cada ~2 s (opcional)
  ("fim", Resultado)
  ("cancelado", None)
  ("erro", texto)
"""

import queue
import threading
import time
import traceback

import cv2
import numpy as np

from .contador import Contador, fundir_eventos
from .detector import Deteccao
from .resultado import Resultado


class Processador(threading.Thread):

    def __init__(self, caminho_video, poligono, linha, detector, config, fila, enviar_previa=True):
        super().__init__(daemon=True)
        self.caminho_video = caminho_video
        self.poligono = np.array(poligono, dtype=np.int32)
        self.linha = linha
        self.detector = detector
        self.config = config
        self.fila = fila
        self.enviar_previa = enviar_previa
        self._cancelar = threading.Event()

    def cancelar(self):
        self._cancelar.set()

    def _enviar(self, tipo, dados):
        try:
            if tipo == "previa":
                self.fila.put_nowait((tipo, dados))
            else:
                self.fila.put((tipo, dados))
        except queue.Full:
            pass

    # -------------------------------------------------------------

    def run(self):
        try:
            self._executar()
        except Exception:
            self._enviar("erro", traceback.format_exc())

    def _executar(self):
        cfg = self.config
        video = cv2.VideoCapture(self.caminho_video)
        if not video.isOpened():
            self._enviar("erro", "Não foi possível abrir o vídeo.")
            return

        fps = video.get(cv2.CAP_PROP_FPS) or 0
        if fps <= 1 or fps > 240:
            fps = 30.0
        total = int(video.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        larg = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        alt = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        passo = max(1, int(round(fps / max(0.5, cfg.fps_analise))))

        # Retângulo de recorte em volta da área (o detector só vê isso).
        # A margem é generosa para o veículo ser acompanhado ANTES e DEPOIS
        # da área marcada: se a área for uma faixa estreita, o veículo
        # passaria por ela em 1 ou 2 frames e não daria para rastrear.
        x, y, w, h = cv2.boundingRect(self.poligono)
        m = max(cfg.margem_recorte, int(cfg.margem_proporcional * max(w, h)))
        x0, y0 = max(0, x - m), max(0, y - m)
        x1, y1 = min(larg, x + w + m), min(alt, y + h + m)

        # Filtro de movimento numa versão reduzida do recorte inteiro
        escala_mov = min(1.0, 320.0 / max(1, x1 - x0))
        tam_mov = (max(1, int((x1 - x0) * escala_mov)), max(1, int((y1 - y0) * escala_mov)))
        area_mov = tam_mov[0] * tam_mov[1]
        sub_fundo = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=cfg.limiar_variacao_movimento, detectShadows=False)
        nucleo3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        nucleo7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        area_min_mancha = cfg.area_min_movimento * escala_mov ** 2

        contador = Contador(self.detector.classes, self.linha, fps, cfg, self.poligono)

        # Segundo contador, só por movimento: pega veículos que o modelo
        # não reconhece (ex.: câmera noturna). Vira "não classificado".
        contador_mov = Contador(["indefinido"], self.linha, fps, cfg, self.poligono) \
            if cfg.contar_por_movimento else None

        inicio = time.time()
        ultimo_progresso = 0.0
        ultima_previa = 0.0
        frame_idx = 0
        analisados = 0
        detectados = 0
        ultimo_lido = -1

        while not self._cancelar.is_set():

            if frame_idx % passo != 0:
                if not video.grab():
                    break
                frame_idx += 1
                continue

            ok, frame = video.read()
            if not ok:
                break
            ultimo_lido = frame_idx

            recorte = frame[y0:y1, x0:x1]
            analisados += 1

            pequeno = cv2.resize(recorte, tam_mov, interpolation=cv2.INTER_AREA)
            pequeno = cv2.GaussianBlur(cv2.cvtColor(pequeno, cv2.COLOR_BGR2GRAY), (5, 5), 0)
            frente = sub_fundo.apply(pequeno)
            movimento = np.count_nonzero(frente) / area_mov

            if contador_mov is not None and analisados > 30:
                manchas = self._manchas(frente, nucleo3, nucleo7, area_min_mancha, escala_mov, x0, y0)
                contador_mov.processar(manchas, frame_idx, frame)

            rodar = True
            if cfg.usar_filtro_movimento:
                if (
                    analisados > 30
                    and not contador.ha_veiculos_ativos
                    and movimento < cfg.limiar_movimento
                    and analisados % max(1, cfg.forcar_deteccao_a_cada) != 0
                ):
                    rodar = False

            dets_area = []
            if rodar:
                detectados += 1
                for d in self.detector.detectar(recorte):
                    d.x1 += x0
                    d.x2 += x0
                    d.y1 += y0
                    d.y2 += y0
                    dets_area.append(d)
                contador.processar(dets_area, frame_idx, frame)

            agora = time.time()

            if agora - ultimo_progresso >= 0.5:
                ultimo_progresso = agora
                decorrido = agora - inicio
                self._enviar("progresso", {
                    "frame": frame_idx,
                    "total_frames": total,
                    "tempo_video": frame_idx / fps,
                    "duracao_video": total / fps if total else 0,
                    "decorrido": decorrido,
                    "contagem": contador.contagem_atual(),
                    "duvidas": contador.n_duvidas(),
                    "fracao_detectada": detectados / max(1, analisados),
                })

            if self.enviar_previa and agora - ultima_previa >= 2.0:
                ultima_previa = agora
                self._enviar("previa", self._desenhar_previa(frame, contador, dets_area))

            frame_idx += 1

        video.release()

        if self._cancelar.is_set():
            self._enviar("cancelado", None)
            return

        contador.encerrar()
        eventos, recortes = contador.eventos, contador.recortes
        if contador_mov is not None:
            contador_mov.encerrar()
            eventos, recortes = fundir_eventos(
                contador.eventos, contador.recortes, contador_mov.eventos, contador_mov.recortes, cfg)

        total_real = max(total, ultimo_lido + 1)
        resultado = Resultado(
            video=self.caminho_video, fps=fps, total_frames=total_real,
            poligono=self.poligono.tolist(), linha=self.linha,
            classes=self.detector.classes, config=cfg,
            modelo=self.detector.descricao(), eventos=eventos,
            pasta=Resultado.criar_pasta(self.caminho_video),
            duracao_processamento=round(time.time() - inicio, 1),
        )
        resultado.salvar_recortes(recortes)
        resultado.salvar()
        self._enviar("fim", resultado)

    # -------------------------------------------------------------

    @staticmethod
    def _manchas(frente, nucleo3, nucleo7, area_min, escala, x0, y0):
        """Regiões em movimento -> 'detecções' sem classe."""
        mascara = cv2.morphologyEx(frente, cv2.MORPH_OPEN, nucleo3)
        mascara = cv2.dilate(mascara, nucleo7, iterations=2)
        n, _, est, _ = cv2.connectedComponentsWithStats(mascara)
        alt, larg = mascara.shape[:2]
        manchas = []
        for i in range(1, n):
            bx, by, bw, bh, area = est[i]
            if area < area_min or bw * bh > 0.5 * larg * alt:
                continue  # pequeno demais, ou mudança de luz na imagem toda
            manchas.append(Deteccao(
                bx / escala + x0, by / escala + y0,
                (bx + bw) / escala + x0, (by + bh) / escala + y0,
                0.9, "indefinido", np.array([0.9], dtype=np.float32),
            ))
        return manchas

    def _desenhar_previa(self, frame, contador, dets):
        img = frame.copy()
        cv2.polylines(img, [self.poligono], True, (0, 255, 255), 2)
        a, b = contador.linha
        cv2.line(img, tuple(map(int, a)), tuple(map(int, b)), (255, 0, 255), 3)
        for t in contador.rastreador.trilhas:
            if t.perdidos:
                continue
            x1, y1, x2, y2 = map(int, t.caixa)
            classe = contador.classes[int(np.argmax(t.votos))] if t.votos.sum() > 0 else "?"
            cor = (0, 200, 0) if t.evento else (0, 165, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), cor, 2)
            cv2.putText(img, f"{t.id} {classe}", (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)
        esc = min(1.0, 640 / img.shape[1])
        if esc < 1:
            img = cv2.resize(img, None, fx=esc, fy=esc, interpolation=cv2.INTER_AREA)
        return img
