import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np

from processamento.config import (
    ConfigProcessamento, encontrar_modelo, formatar_tempo, ler_preferencia, listar_modelos,
    nome_exibicao, pasta_resultados, salvar_preferencia,
)
from processamento.detector import Detector
from processamento.processador import Processador
from processamento.resultado import Resultado

from interface.janela_revisao import JanelaRevisao
from interface.util import abrir_no_sistema, bgr_para_tk


class JanelaPrincipal:

    def __init__(self, janela):

        self.janela = janela
        self.janela.title("Contador de Veículos")

        # Ajusta o tamanho do vídeo à tela (notebooks 1366x768 inclusive)
        tela_larg = self.janela.winfo_screenwidth()
        tela_alt = self.janela.winfo_screenheight()
        self.larg_canvas = max(640, min(1000, tela_larg - 120))
        self.alt_canvas = max(300, min(540, tela_alt - 340))
        self.janela.geometry(f"{self.larg_canvas + 100}x{self.alt_canvas + 275}+20+5")

        # Estado do vídeo e da área
        self.caminho_video = None
        self.frame_original = None
        self.info_video = {}
        self.imagem_tk = None
        self.escala = 1.0
        self.pontos = []             # coordenadas do canvas
        self.modo_linha = 0          # 0 ou 1: qual par de lados define a linha

        # Modelo
        self.detector = None
        self.erro_modelo = None

        # Processamento
        self.config = ConfigProcessamento()
        self.processador = None
        self.fila = queue.Queue(maxsize=50)
        self.resultado = None
        self.janela_revisao = None

        self._montar_tela()
        self._carregar_modelo_em_segundo_plano()

        self.janela.protocol("WM_DELETE_WINDOW", self.fechar_programa)

    # =============================================================
    # TELA
    # =============================================================

    def _montar_tela(self):

        tk.Label(
            self.janela, text="CONTADOR DE VEÍCULOS", font=("Arial", 18, "bold")
        ).pack(pady=(10, 4))

        topo = tk.Frame(self.janela)
        topo.pack(fill="x", padx=20)

        tk.Button(
            topo, text="SELECIONAR VÍDEO", font=("Arial", 11, "bold"),
            command=self.selecionar_video,
        ).pack(side="left", padx=(0, 10))

        tk.Button(
            topo, text="ABRIR RESULTADO SALVO", font=("Arial", 10),
            command=self.abrir_resultado_salvo,
        ).pack(side="left")

        self.rotulo_modelo = tk.Label(topo, text="Carregando modelo...", fg="gray", font=("Arial", 9))
        self.rotulo_modelo.pack(side="right")

        # Escolha do modelo (n = rápido, s/m = mais preciso e mais lento)
        self.modelos = listar_modelos()
        nomes = [os.path.basename(m) for m in self.modelos]
        preferido = ler_preferencia("modelo")
        self.var_modelo = tk.StringVar(
            value=preferido if preferido in nomes else (os.path.basename(encontrar_modelo() or "") or ""))
        self.combo_modelo = ttk.Combobox(
            topo, textvariable=self.var_modelo, values=nomes, state="readonly", width=24,
        )
        self.combo_modelo.pack(side="right", padx=(0, 8))
        self.combo_modelo.bind("<<ComboboxSelected>>", self._trocar_modelo)
        tk.Label(topo, text="Modelo:", font=("Arial", 9)).pack(side="right")

        self.instrucao = tk.Label(
            self.janela, text="Selecione um vídeo para começar.", font=("Arial", 11)
        )
        self.instrucao.pack(pady=4)

        self.canvas = tk.Canvas(
            self.janela, width=self.larg_canvas, height=self.alt_canvas, bg="black",
            highlightthickness=0,
        )
        self.canvas.pack(pady=4)
        self.canvas.bind("<Button-1>", self.clicar_video)

        # ---------------- Botões da área ----------------
        linha_botoes = tk.Frame(self.janela)
        linha_botoes.pack(pady=6)

        self.botao_limpar = tk.Button(
            linha_botoes, text="LIMPAR SELEÇÃO", font=("Arial", 10, "bold"),
            command=self.limpar_selecao, state="disabled",
        )
        self.botao_limpar.pack(side="left", padx=8)

        self.botao_girar = tk.Button(
            linha_botoes, text="GIRAR LINHA", font=("Arial", 10, "bold"),
            command=self.girar_linha, state="disabled",
        )
        self.botao_girar.pack(side="left", padx=8)

        tk.Label(linha_botoes, text="Frames analisados por segundo:", font=("Arial", 10)).pack(
            side="left", padx=(20, 4)
        )
        self.var_fps = tk.StringVar(value=str(int(self.config.fps_analise)))
        tk.Spinbox(
            linha_botoes, from_=3, to=30, width=4, textvariable=self.var_fps, font=("Arial", 10),
        ).pack(side="left")

        self.var_previa = tk.BooleanVar(value=True)
        tk.Checkbutton(
            linha_botoes, text="Mostrar prévia", variable=self.var_previa, font=("Arial", 10),
        ).pack(side="left", padx=12)

        self.botao_iniciar = tk.Button(
            linha_botoes, text="INICIAR CONTAGEM", font=("Arial", 11, "bold"),
            bg="#2e7d32", fg="white", activebackground="#1b5e20", activeforeground="white",
            command=self.iniciar_contagem, state="disabled",
        )
        self.botao_iniciar.pack(side="left", padx=8)

        # ---------------- Progresso ----------------
        self.quadro_progresso = tk.Frame(self.janela)
        self.quadro_progresso.pack(fill="x", padx=50, pady=(4, 0))

        self.barra = ttk.Progressbar(self.quadro_progresso, maximum=1000)
        self.barra.pack(side="left", fill="x", expand=True)

        self.botao_cancelar = tk.Button(
            self.quadro_progresso, text="CANCELAR", font=("Arial", 10, "bold"),
            command=self.cancelar, state="disabled",
        )
        self.botao_cancelar.pack(side="left", padx=(10, 0))

        self.rotulo_status = tk.Label(self.janela, text="", font=("Arial", 10))
        self.rotulo_status.pack()
        self.rotulo_contagem = tk.Label(self.janela, text="", font=("Arial", 11, "bold"))
        self.rotulo_contagem.pack()

        # ---------------- Resultado ----------------
        self.quadro_resultado = tk.Frame(self.janela)
        self.quadro_resultado.pack(pady=6)

        self.botao_revisar = tk.Button(
            self.quadro_resultado, text="REVISAR DÚVIDAS", font=("Arial", 11, "bold"),
            bg="#f9a825", command=self.abrir_revisao, state="disabled",
        )
        self.botao_revisar.pack(side="left", padx=8)

        self.botao_excel = tk.Button(
            self.quadro_resultado, text="EXPORTAR EXCEL", font=("Arial", 11, "bold"),
            command=self.exportar_excel, state="disabled",
        )
        self.botao_excel.pack(side="left", padx=8)

        self.botao_pasta = tk.Button(
            self.quadro_resultado, text="ABRIR PASTA DO RESULTADO", font=("Arial", 10),
            command=self.abrir_pasta_resultado, state="disabled",
        )
        self.botao_pasta.pack(side="left", padx=8)

    # =============================================================
    # MODELO
    # =============================================================

    def _carregar_modelo_em_segundo_plano(self):

        # O Tk só pode ser usado pela thread principal: a thread de carga
        # só preenche as variáveis e a tela confere periodicamente.
        self.detector = None
        self.erro_modelo = None
        self.rotulo_modelo.config(text="Carregando modelo...", fg="gray")
        nome = self.var_modelo.get()
        caminho = next((m for m in self.modelos if os.path.basename(m) == nome), None) or encontrar_modelo()

        def carregar():
            try:
                self.detector = Detector(
                    caminho, conf_minima=self.config.conf_minima, iou_nms=self.config.iou_nms,
                )
            except Exception as erro:
                self.erro_modelo = str(erro) or erro.__class__.__name__

        self._thread_modelo = threading.Thread(target=carregar, daemon=True)
        self._thread_modelo.start()
        self.janela.after(200, self._verificar_modelo)

    def _trocar_modelo(self, _evento=None):
        if self.processando:
            return
        salvar_preferencia("modelo", self.var_modelo.get())
        self._atualizar_botao_iniciar()
        self._carregar_modelo_em_segundo_plano()

    def _verificar_modelo(self):
        if self._thread_modelo.is_alive():
            self.janela.after(200, self._verificar_modelo)
            return
        self._modelo_carregado()

    def _modelo_carregado(self):
        if self.detector is not None:
            classes = ", ".join(nome_exibicao(c) for c in self.detector.classes)
            self.rotulo_modelo.config(text=f"{self.detector.backend} — {classes}", fg="#2e7d32")
        else:
            self.rotulo_modelo.config(text="Modelo não carregado", fg="red")
            messagebox.showerror("Modelo", f"Não foi possível carregar o modelo:\n\n{self.erro_modelo}")
        self._atualizar_botao_iniciar()

    # =============================================================
    # VÍDEO E ÁREA
    # =============================================================

    def selecionar_video(self):

        if self.processando:
            return

        caminho = filedialog.askopenfilename(
            title="Selecione o vídeo",
            filetypes=[("Arquivos de vídeo", "*.mp4 *.dav *.avi *.mov *.mkv *.m4v *.mts *.wmv *.h264 *.264"),
                       ("Todos os arquivos", "*.*")],
        )
        if not caminho:
            return

        video = cv2.VideoCapture(caminho)
        if not video.isOpened():
            messagebox.showerror("Erro", "Não foi possível abrir o vídeo.")
            return

        fps = video.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(video.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        # Usa um frame a ~2 s do início (o primeiro às vezes é preto)
        if total > fps * 3:
            video.set(cv2.CAP_PROP_POS_FRAMES, int(fps * 2))
        sucesso, frame = video.read()
        if not sucesso:
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            sucesso, frame = video.read()
        video.release()

        if not sucesso:
            messagebox.showerror("Erro", "Não foi possível ler o vídeo.")
            return

        self.caminho_video = caminho
        self.frame_original = frame
        self.info_video = {"fps": fps, "total": total, "duracao": total / fps if fps else 0}
        self.resultado = None
        self._habilitar_resultado(False)

        self.limpar_selecao()
        self.rotulo_status.config(
            text=f"{os.path.basename(caminho)} — {frame.shape[1]}x{frame.shape[0]}, "
                 f"{fps:.1f} fps, duração {formatar_tempo(self.info_video['duracao'])}"
        )
        self.rotulo_contagem.config(text="")
        self.barra["value"] = 0
        self.botao_limpar.config(state="normal")

    def mostrar_frame(self, frame):
        self.imagem_tk, self.escala = bgr_para_tk(frame, self.larg_canvas, self.alt_canvas, ampliar=True)
        self.canvas.config(width=self.imagem_tk.width(), height=self.imagem_tk.height())
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.imagem_tk)

    def clicar_video(self, evento):
        if self.frame_original is None or self.processando or len(self.pontos) >= 4:
            return

        self.pontos.append((evento.x, evento.y))
        self._desenhar_selecao()

        if len(self.pontos) == 4:
            self.botao_girar.config(state="normal")
            self.instrucao.config(
                text="Área marcada. A linha roxa é onde o veículo é contado — use GIRAR LINHA "
                     "se ela não atravessar a pista. Depois clique em INICIAR CONTAGEM."
            )
        else:
            self.instrucao.config(
                text=f"Ponto {len(self.pontos)} marcado. Marque mais {4 - len(self.pontos)}."
            )
        self._atualizar_botao_iniciar()

    def _linha_canvas(self, pontos):
        p = pontos
        if self.modo_linha == 0:
            a, b = (p[0], p[3]), (p[1], p[2])
        else:
            a, b = (p[0], p[1]), (p[3], p[2])
        meio = lambda u, v: ((u[0] + v[0]) / 2, (u[1] + v[1]) / 2)  # noqa: E731
        return meio(*a), meio(*b)

    def _desenhar_selecao(self):
        if self.frame_original is None:
            return
        self.mostrar_frame(self.frame_original)

        for i, (x, y) in enumerate(self.pontos, start=1):
            self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill="red", outline="white", width=2)
            self.canvas.create_text(x + 12, y - 12, text=str(i), fill="white", font=("Arial", 12, "bold"))

        if len(self.pontos) >= 2:
            self.canvas.create_line(*self.pontos, fill="yellow", width=2, dash=(4, 2))

        if len(self.pontos) == 4:
            self.canvas.create_polygon(self.pontos, outline="yellow", fill="", width=3)
            a, b = self._linha_canvas(self.pontos)
            self.canvas.create_line(*a, *b, fill="#ff00ff", width=4)

    def girar_linha(self):
        self.modo_linha = 1 - self.modo_linha
        self._desenhar_selecao()

    def limpar_selecao(self):
        if self.processando:
            return
        self.pontos = []
        self.modo_linha = 0
        self.botao_girar.config(state="disabled")
        if self.frame_original is not None:
            self._desenhar_selecao()
            self.instrucao.config(
                text="Clique em 4 pontos ao redor da pista (em ordem, contornando a área)."
            )
        self._atualizar_botao_iniciar()

    def _atualizar_botao_iniciar(self):
        pronto = (
            self.detector is not None
            and self.caminho_video is not None
            and len(self.pontos) == 4
            and not self.processando
        )
        self.botao_iniciar.config(state="normal" if pronto else "disabled")

    # =============================================================
    # PROCESSAMENTO
    # =============================================================

    @property
    def processando(self):
        return self.processador is not None and self.processador.is_alive()

    def iniciar_contagem(self):
        if len(self.pontos) != 4 or self.detector is None:
            return

        try:
            fps_analise = float(self.var_fps.get().replace(",", "."))
            if not 1 <= fps_analise <= 60:
                raise ValueError
        except ValueError:
            messagebox.showerror("Erro", "Frames analisados por segundo deve ser um número entre 1 e 60.")
            return

        self.config.fps_analise = fps_analise

        poligono = [(int(x / self.escala), int(y / self.escala)) for x, y in self.pontos]
        a, b = self._linha_canvas(self.pontos)
        linha = ((a[0] / self.escala, a[1] / self.escala), (b[0] / self.escala, b[1] / self.escala))

        self.resultado = None
        self._habilitar_resultado(False)
        self.fila = queue.Queue(maxsize=50)
        self.processador = Processador(
            self.caminho_video, poligono, linha, self.detector, self.config, self.fila,
            enviar_previa=self.var_previa.get(),
        )
        self.processador.start()

        self.botao_iniciar.config(state="disabled")
        self.botao_limpar.config(state="disabled")
        self.botao_girar.config(state="disabled")
        self.botao_cancelar.config(state="normal")
        self.instrucao.config(text="Processando... pode minimizar a janela.")
        self.rotulo_status.config(text="Iniciando...")
        self.barra["value"] = 0

        self.janela.after(200, self._verificar_fila)

    def cancelar(self):
        if self.processando and messagebox.askyesno("Cancelar", "Cancelar o processamento?"):
            self.processador.cancelar()
            self.rotulo_status.config(text="Cancelando...")

    def _verificar_fila(self):
        terminou = False
        try:
            while True:
                tipo, dados = self.fila.get_nowait()
                if tipo == "progresso":
                    self._mostrar_progresso(dados)
                elif tipo == "previa":
                    if self.var_previa.get():
                        self.mostrar_frame(dados)
                elif tipo == "fim":
                    self._processamento_concluido(dados)
                    terminou = True
                elif tipo == "cancelado":
                    self._processamento_encerrado("Processamento cancelado.")
                    terminou = True
                elif tipo == "erro":
                    self._processamento_encerrado("Erro no processamento.")
                    messagebox.showerror("Erro", dados[-3000:])
                    terminou = True
        except queue.Empty:
            pass

        if not terminou:
            self.janela.after(200, self._verificar_fila)

    def _mostrar_progresso(self, p):
        total = p["total_frames"] or 1
        fracao = min(1.0, p["frame"] / total)
        self.barra["value"] = fracao * 1000

        decorrido = p["decorrido"]
        velocidade = p["tempo_video"] / decorrido if decorrido > 0 else 0
        restante = (decorrido / fracao - decorrido) if fracao > 0.01 else None

        texto = (
            f"{fracao:.0%} — vídeo em {formatar_tempo(p['tempo_video'])} de "
            f"{formatar_tempo(p['duracao_video'])} — velocidade {velocidade:.1f}x"
        )
        if restante is not None:
            texto += f" — faltam ~{formatar_tempo(restante)}"
        texto += f" — detector rodou em {p['fracao_detectada']:.0%} dos frames analisados"
        self.rotulo_status.config(text=texto)

        self._mostrar_contagem(p["contagem"], p["duvidas"])

    def _mostrar_contagem(self, contagem, duvidas):
        partes = [f"{nome_exibicao(c)}: {n}" for c, n in contagem.items()]
        partes.append(f"Total: {sum(contagem.values())}")
        partes.append(f"Dúvidas: {duvidas}")
        self.rotulo_contagem.config(text="   |   ".join(partes))

    def _processamento_encerrado(self, texto):
        self.botao_cancelar.config(state="disabled")
        self.botao_limpar.config(state="normal")
        if len(self.pontos) == 4:
            self.botao_girar.config(state="normal")
        self.instrucao.config(text=texto)
        self._desenhar_selecao()
        self.janela.after(100, self._atualizar_botao_iniciar)

    def _processamento_concluido(self, resultado):
        self.resultado = resultado
        self.barra["value"] = 1000
        self._processamento_encerrado("Contagem concluída.")

        n_duvidas = len(resultado.duvidas())
        self._mostrar_contagem(resultado.totais(), n_duvidas)
        self.rotulo_status.config(
            text=f"Concluído em {formatar_tempo(resultado.duracao_processamento or 0)} "
                 f"(vídeo de {formatar_tempo(resultado.duracao_video)}). "
                 f"Resultado salvo em {resultado.pasta}"
        )
        self._habilitar_resultado(True)
        self._salvar_excel_automatico()

        if n_duvidas and messagebox.askyesno(
            "Contagem concluída",
            f"Foram contados {len(resultado.eventos)} veículos.\n\n"
            f"{n_duvidas} precisam de conferência. Abrir a revisão agora?",
        ):
            self.abrir_revisao()

    # =============================================================
    # RESULTADO
    # =============================================================

    def _habilitar_resultado(self, ativo):
        estado = "normal" if ativo else "disabled"
        for botao in (self.botao_revisar, self.botao_excel, self.botao_pasta):
            botao.config(state=estado)
        self._atualizar_botao_revisar()

    def _atualizar_botao_revisar(self):
        if self.resultado is None:
            self.botao_revisar.config(text="REVISAR DÚVIDAS")
            return
        pend = len(self.resultado.duvidas(so_pendentes=True))
        total = len(self.resultado.duvidas())
        self.botao_revisar.config(text=f"REVISAR DÚVIDAS ({pend} de {total} pendentes)")

    def _salvar_excel_automatico(self):
        """Mantém um contagem.xlsx sempre atualizado na pasta do resultado."""
        if self.resultado is None or not self.resultado.pasta:
            return
        try:
            self.resultado.exportar_excel(os.path.join(self.resultado.pasta, "contagem.xlsx"))
        except Exception:
            pass  # ex.: arquivo aberto no Excel; o botão EXPORTAR continua disponível

    def _ao_fechar_revisao(self):
        self._salvar_excel_automatico()

    def _ao_alterar_revisao(self):
        self._atualizar_botao_revisar()
        self._mostrar_contagem(self.resultado.totais(), len(self.resultado.duvidas(so_pendentes=True)))

    def abrir_revisao(self):
        if self.resultado is None:
            return
        if self.janela_revisao is not None and self.janela_revisao.existe():
            self.janela_revisao.trazer_para_frente()
            return
        self.janela_revisao = JanelaRevisao(
            self.janela, self.resultado, ao_alterar=self._ao_alterar_revisao, ao_fechar=self._ao_fechar_revisao)

    def exportar_excel(self):
        if self.resultado is None:
            return
        caminho = filedialog.asksaveasfilename(
            title="Salvar Excel", defaultextension=".xlsx",
            initialdir=self.resultado.pasta, initialfile="contagem.xlsx",
            filetypes=[("Excel", "*.xlsx")],
        )
        if not caminho:
            return

        pend = len(self.resultado.duvidas(so_pendentes=True))
        if pend and not messagebox.askyesno(
            "Dúvidas pendentes",
            f"Ainda há {pend} dúvidas não revisadas. Elas entram com a classe sugerida pelo modelo.\n\n"
            "Exportar mesmo assim?",
        ):
            return

        try:
            self.resultado.exportar_excel(caminho)
        except PermissionError:
            messagebox.showerror("Erro", "Não foi possível salvar. O arquivo está aberto no Excel?")
            return
        if messagebox.askyesno("Excel", "Planilha salva. Abrir agora?"):
            abrir_no_sistema(caminho)

    def abrir_pasta_resultado(self):
        if self.resultado is not None and self.resultado.pasta:
            abrir_no_sistema(self.resultado.pasta)

    def abrir_resultado_salvo(self):
        if self.processando:
            return
        caminho = filedialog.askopenfilename(
            title="Abrir resultado", initialdir=pasta_resultados(),
            filetypes=[("Resultado da contagem", "resultado.json"), ("JSON", "*.json")],
        )
        if not caminho:
            return
        try:
            resultado = Resultado.abrir(caminho)
        except Exception as erro:
            messagebox.showerror("Erro", f"Não foi possível abrir o resultado:\n{erro}")
            return

        self.resultado = resultado
        self.caminho_video = resultado.video if os.path.isfile(resultado.video) else None
        self.frame_original = None
        self.pontos = []
        self.botao_girar.config(state="disabled")
        self.botao_limpar.config(state="disabled")

        if self.caminho_video:
            video = cv2.VideoCapture(self.caminho_video)
            video.set(cv2.CAP_PROP_POS_FRAMES, int(resultado.fps * 2))
            ok, frame = video.read()
            video.release()
            if ok:
                cv2.polylines(frame, [np.array(resultado.poligono, dtype=np.int32)], True, (0, 255, 255), 3)
                a, b = resultado.linha
                cv2.line(frame, tuple(map(int, a)), tuple(map(int, b)), (255, 0, 255), 4)
                self.mostrar_frame(frame)
        else:
            self.canvas.delete("all")
            messagebox.showwarning(
                "Vídeo não encontrado",
                f"O vídeo original não foi encontrado em:\n{resultado.video}\n\n"
                "A revisão vai mostrar só os recortes, sem o trecho do vídeo.",
            )

        self.instrucao.config(text="Resultado aberto.")
        self.rotulo_status.config(text=f"{os.path.basename(resultado.video)} — {resultado.pasta}")
        self._mostrar_contagem(resultado.totais(), len(resultado.duvidas(so_pendentes=True)))
        self.barra["value"] = 0
        self._habilitar_resultado(True)
        self._atualizar_botao_iniciar()

    # =============================================================
    # FECHAR
    # =============================================================

    def fechar_programa(self):
        if self.processando:
            if not messagebox.askyesno("Sair", "O processamento está em andamento. Sair mesmo assim?"):
                return
            self.processador.cancelar()
        if self.resultado is not None and self.resultado.pasta:
            try:
                self.resultado.salvar()
            except Exception:
                pass
        self.janela.destroy()


if __name__ == "__main__":
    janela = tk.Tk()
    JanelaPrincipal(janela)
    janela.mainloop()
