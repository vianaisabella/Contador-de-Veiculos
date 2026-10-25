"""
Janela de conferência humana.

Atalhos:
  1, 2, 3...  escolhe a classe (na ordem dos botões)
  Enter       confirma a sugestão do modelo
  0 / Delete  descarta (não é veículo / contado duas vezes)
  Espaço      toca o trecho do vídeo (2 s antes e depois)
  ↑ / ↓       navega na lista
"""

import os
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
import numpy as np

from processamento.config import formatar_tempo, interpretar_tempo, nome_exibicao

from interface.util import bgr_para_tk


FILTROS = {
    "Dúvidas pendentes": lambda e: e.get("duvida") and not e.get("revisado"),
    "Todas as dúvidas": lambda e: e.get("duvida"),
    "Todos os veículos": lambda e: True,
}

LARG_VIDEO, ALT_VIDEO = 560, 315
LARG_RECORTE, ALT_RECORTE = 200, 200
SEGUNDOS_TRECHO = 2.0


class JanelaRevisao:

    def __init__(self, pai, resultado, ao_alterar=None, ao_fechar=None):
        self.resultado = resultado
        self.ao_alterar = ao_alterar
        self.ao_fechar = ao_fechar

        self.janela = tk.Toplevel(pai)
        self.janela.title(f"Revisão — {os.path.basename(resultado.video)}")
        self.janela.geometry("1260x710+10+5")
        self.janela.minsize(1150, 650)

        self.video = None
        if os.path.isfile(resultado.video):
            self.video = cv2.VideoCapture(resultado.video)
            if not self.video.isOpened():
                self.video = None

        self.evento_atual = None
        self._img_video = None
        self._img_recorte = None
        self._tocando = False
        self._id_after = None
        self._frames_trecho = []
        self._pos_trecho = 0

        self._montar_tela()
        self._atalhos()
        self.recarregar_lista()

        self.janela.protocol("WM_DELETE_WINDOW", self.fechar)
        self.arvore.focus_set()

    # =============================================================
    # TELA
    # =============================================================

    def _montar_tela(self):
        esquerda = tk.Frame(self.janela)
        esquerda.pack(side="left", fill="y", padx=(10, 5), pady=10)

        filtro = tk.Frame(esquerda)
        filtro.pack(fill="x")
        tk.Label(filtro, text="Mostrar:", font=("Arial", 10)).pack(side="left")
        self.var_filtro = tk.StringVar(value="Dúvidas pendentes")
        combo = ttk.Combobox(
            filtro, textvariable=self.var_filtro, values=list(FILTROS), state="readonly", width=20,
        )
        combo.pack(side="left", padx=5)
        combo.bind("<<ComboboxSelected>>", lambda e: self.recarregar_lista())

        self.rotulo_pendentes = tk.Label(esquerda, text="", font=("Arial", 10, "bold"))
        self.rotulo_pendentes.pack(anchor="w", pady=(6, 4))

        lista = tk.Frame(esquerda)
        lista.pack(fill="both", expand=True)

        colunas = ("n", "tempo", "classe", "motivo", "status")
        self.arvore = ttk.Treeview(lista, columns=colunas, show="headings", height=24, selectmode="browse")
        for col, titulo, larg in (
            ("n", "Nº", 45), ("tempo", "Tempo", 85), ("classe", "Classe", 85),
            ("motivo", "Motivo", 160), ("status", "Status", 75),
        ):
            self.arvore.heading(col, text=titulo)
            self.arvore.column(col, width=larg, anchor="w" if col == "motivo" else "center")
        self.arvore.tag_configure("pendente", background="#fff2cc")
        self.arvore.tag_configure("descartado", foreground="#999999")

        barra = ttk.Scrollbar(lista, orient="vertical", command=self.arvore.yview)
        self.arvore.configure(yscrollcommand=barra.set)
        self.arvore.pack(side="left", fill="both", expand=True)
        barra.pack(side="left", fill="y")
        self.arvore.bind("<<TreeviewSelect>>", self._ao_selecionar)

        # ---------------- Direita ----------------
        direita = tk.Frame(self.janela)
        direita.pack(side="left", fill="both", expand=True, padx=(5, 10), pady=10)

        imagens = tk.Frame(direita)
        imagens.pack(anchor="w")

        self._vazio_video = tk.PhotoImage(width=LARG_VIDEO, height=ALT_VIDEO)
        self._vazio_recorte = tk.PhotoImage(width=LARG_RECORTE, height=ALT_RECORTE)

        self.rotulo_video = tk.Label(imagens, bg="black", image=self._vazio_video, width=LARG_VIDEO, height=ALT_VIDEO)
        self.rotulo_video.pack(side="left")

        lado = tk.Frame(imagens)
        lado.pack(side="left", padx=10, anchor="n")
        tk.Label(lado, text="Melhor imagem", font=("Arial", 9)).pack()
        self.rotulo_recorte = tk.Label(lado, bg="#222222", image=self._vazio_recorte, width=LARG_RECORTE, height=ALT_RECORTE)
        self.rotulo_recorte.pack()

        controles = tk.Frame(direita)
        controles.pack(anchor="w", pady=6)
        self.botao_tocar = tk.Button(
            controles, text="▶  VER TRECHO (Espaço)", font=("Arial", 10, "bold"),
            command=self.alternar_trecho,
        )
        self.botao_tocar.pack(side="left")
        if self.video is None:
            self.botao_tocar.config(state="disabled", text="Vídeo original não encontrado")

        self.rotulo_info = tk.Label(direita, text="", font=("Arial", 11), justify="left", anchor="w")
        self.rotulo_info.pack(anchor="w", fill="x", pady=(4, 8))

        tk.Label(direita, text="Qual é a classe correta?", font=("Arial", 11, "bold")).pack(anchor="w")
        self.quadro_classes = tk.Frame(direita)
        self.quadro_classes.pack(anchor="w", pady=4)
        self._montar_botoes_classe()

        acoes = tk.Frame(direita)
        acoes.pack(anchor="w", pady=4)
        tk.Button(
            acoes, text="✔ CONFIRMAR SUGESTÃO (Enter)", font=("Arial", 10, "bold"),
            bg="#2e7d32", fg="white", command=self.confirmar_sugestao,
        ).pack(side="left", padx=(0, 8))
        tk.Button(
            acoes, text="✖ DESCARTAR — não é veículo / contado 2x (0)", font=("Arial", 10),
            command=self.descartar,
        ).pack(side="left")

        rodape = tk.Frame(direita)
        rodape.pack(side="bottom", fill="x", pady=(10, 0))
        tk.Button(
            rodape, text="+ ADICIONAR VEÍCULO NÃO DETECTADO", font=("Arial", 10),
            command=self.adicionar_manual,
        ).pack(side="left")
        tk.Button(rodape, text="FECHAR", font=("Arial", 10, "bold"), command=self.fechar).pack(side="right")
        tk.Label(
            rodape, text="As alterações são salvas automaticamente.", fg="gray", font=("Arial", 9),
        ).pack(side="right", padx=10)

    def _montar_botoes_classe(self):
        for w in self.quadro_classes.winfo_children():
            w.destroy()
        for i, classe in enumerate(self.resultado.classes, start=1):
            tk.Button(
                self.quadro_classes, text=f"{nome_exibicao(classe)} ({i})", font=("Arial", 11, "bold"),
                width=13, command=lambda c=classe: self.classificar(c),
            ).pack(side="left", padx=(0, 6))

    def _atalhos(self):
        j = self.janela
        for i, classe in enumerate(self.resultado.classes[:9], start=1):
            j.bind(str(i), lambda e, c=classe: self.classificar(c))
            j.bind(f"<KP_{i}>", lambda e, c=classe: self.classificar(c))
        j.bind("0", lambda e: self.descartar())
        j.bind("<KP_0>", lambda e: self.descartar())
        j.bind("<Delete>", lambda e: self.descartar())
        j.bind("<Return>", lambda e: self.confirmar_sugestao())
        j.bind("<KP_Enter>", lambda e: self.confirmar_sugestao())
        j.bind("<space>", lambda e: (self.alternar_trecho(), "break")[1])

    # =============================================================
    # LISTA
    # =============================================================

    def recarregar_lista(self, selecionar_id=None):
        filtro = FILTROS[self.var_filtro.get()]
        self.arvore.delete(*self.arvore.get_children())
        for e in self.resultado.eventos:
            if filtro(e):
                self.arvore.insert("", "end", iid=str(e["id"]), values=self._valores(e), tags=self._tags(e))

        self._atualizar_pendentes()
        itens = self.arvore.get_children()
        if selecionar_id is not None and self.arvore.exists(str(selecionar_id)):
            alvo = str(selecionar_id)
        elif itens:
            alvo = itens[0]
        else:
            alvo = None

        if alvo:
            self.arvore.selection_set(alvo)
            self.arvore.see(alvo)
        else:
            self._limpar_detalhe()

    def _valores(self, e):
        if e.get("descartado"):
            status = "descartado"
        elif e.get("revisado"):
            status = "conferido"
        elif e.get("duvida"):
            status = "pendente"
        else:
            status = ""
        motivo = "; ".join(e.get("motivos", []))
        return (e["id"], formatar_tempo(e["tempo"], com_decimos=True), nome_exibicao(e["classe"]), motivo, status)

    def _tags(self, e):
        if e.get("descartado"):
            return ("descartado",)
        if e.get("duvida") and not e.get("revisado"):
            return ("pendente",)
        return ()

    def _atualizar_pendentes(self):
        pend = len(self.resultado.duvidas(so_pendentes=True))
        total = len(self.resultado.duvidas())
        tot = self.resultado.totais()
        resumo = "  ".join(f"{nome_exibicao(c)}: {n}" for c, n in tot.items())
        self.rotulo_pendentes.config(
            text=f"Pendentes: {pend} de {total} dúvidas\n{resumo}  |  Total: {sum(tot.values())}"
        )

    def _evento_por_id(self, id_evento):
        for e in self.resultado.eventos:
            if e["id"] == id_evento:
                return e
        return None

    # =============================================================
    # DETALHE
    # =============================================================

    def _ao_selecionar(self, _evento=None):
        sel = self.arvore.selection()
        if not sel:
            return
        e = self._evento_por_id(int(sel[0]))
        if e is None or (e is self.evento_atual and _evento is not None):
            return
        self.parar_trecho()
        self.evento_atual = e
        self._mostrar_detalhe(e)

    def _limpar_detalhe(self):
        self.parar_trecho()
        self.evento_atual = None
        self.rotulo_video.config(image=self._vazio_video)
        self.rotulo_recorte.config(image=self._vazio_recorte)
        self.rotulo_info.config(text="Nada para revisar neste filtro.")

    def _mostrar_detalhe(self, e):
        # Recorte
        caminho = self.resultado.caminho_recorte(e)
        # cv2.imread não abre caminhos com acento no Windows ("Área de Trabalho")
        img = None
        if caminho:
            dados = np.fromfile(caminho, dtype=np.uint8)
            img = cv2.imdecode(dados, cv2.IMREAD_COLOR) if dados.size else None
        if img is not None:
            self._img_recorte, _ = bgr_para_tk(img, LARG_RECORTE, ALT_RECORTE, ampliar=True)
            self.rotulo_recorte.config(image=self._img_recorte)
        else:
            self.rotulo_recorte.config(image=self._vazio_recorte)

        # Frame do cruzamento
        frame = self._ler_frame(e["frame"])
        if frame is not None:
            self._mostrar_video(self._desenhar(frame, e, destacar=True))
        else:
            self.rotulo_video.config(image=self._vazio_video)

        # Texto
        linhas = [f"Nº {e['id']}   —   tempo {formatar_tempo(e['tempo'], com_decimos=True)}"]
        if e.get("classe_sugerida"):
            linhas.append(f"Sugestão do modelo: {nome_exibicao(e['classe_sugerida'])}")
        votos = sorted((e.get("votos") or {}).items(), key=lambda kv: -kv[1])
        votos = [f"{nome_exibicao(c)} {v:.0%}" for c, v in votos if v >= 0.05]
        if votos:
            linhas.append("Votos: " + ",  ".join(votos))
        if e.get("deteccoes"):
            linhas.append(f"Visto em {e['deteccoes']} frames analisados   |   confiança média {e.get('confianca') or 0:.2f}")
        if e.get("motivos"):
            linhas.append("Motivo: " + "; ".join(e["motivos"]))
        estado = "DESCARTADO" if e.get("descartado") else (
            f"CONFERIDO como {nome_exibicao(e['classe'])}" if e.get("revisado") else "")
        if estado:
            linhas.append(estado)
        self.rotulo_info.config(text="\n".join(linhas))

    def _ler_frame(self, numero):
        if self.video is None:
            return None
        self.video.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(numero)))
        ok, frame = self.video.read()
        return frame if ok else None

    def _desenhar(self, frame, e, destacar):
        img = frame.copy()
        cv2.polylines(img, [np.array(self.resultado.poligono, dtype=np.int32)], True, (0, 255, 255), 2)
        a, b = self.resultado.linha
        cv2.line(img, tuple(map(int, a)), tuple(map(int, b)), (255, 0, 255), 2)
        if destacar and e.get("caixa"):
            x1, y1, x2, y2 = map(int, e["caixa"])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        return img

    def _mostrar_video(self, img):
        self._img_video, _ = bgr_para_tk(img, LARG_VIDEO, ALT_VIDEO, ampliar=True)
        self.rotulo_video.config(image=self._img_video)

    # =============================================================
    # TRECHO DO VÍDEO
    # =============================================================

    def alternar_trecho(self):
        if self._tocando:
            self.parar_trecho()
        else:
            self.tocar_trecho()

    def tocar_trecho(self):
        e = self.evento_atual
        if e is None or self.video is None:
            return
        fps = self.resultado.fps
        inicio = max(0, int(e["frame"] - SEGUNDOS_TRECHO * fps))
        fim = int(e["frame"] + SEGUNDOS_TRECHO * fps)

        # Lê e reduz o trecho uma vez; depois toca em loop
        self.janela.config(cursor="watch")
        self.janela.update_idletasks()
        self.video.set(cv2.CAP_PROP_POS_FRAMES, inicio)
        frames = []
        for n in range(inicio, fim + 1):
            ok, frame = self.video.read()
            if not ok:
                break
            perto = abs(n - e["frame"]) <= max(1, int(fps * 0.15))
            img = self._desenhar(frame, e, destacar=perto)
            esc = min(LARG_VIDEO / img.shape[1], ALT_VIDEO / img.shape[0])
            img = cv2.resize(img, None, fx=esc, fy=esc, interpolation=cv2.INTER_AREA)
            frames.append(img)
        self.janela.config(cursor="")

        if not frames:
            return
        self._frames_trecho = frames
        self._pos_trecho = 0
        self._tocando = True
        self.botao_tocar.config(text="■  PARAR (Espaço)")
        self._proximo_frame_trecho()

    def _proximo_frame_trecho(self):
        if not self._tocando or not self._frames_trecho:
            return
        self._mostrar_video(self._frames_trecho[self._pos_trecho])
        self._pos_trecho = (self._pos_trecho + 1) % len(self._frames_trecho)
        atraso = int(1000 / max(1.0, self.resultado.fps))
        if self._pos_trecho == 0:
            atraso = 700  # pausa curta antes de repetir
        self._id_after = self.janela.after(atraso, self._proximo_frame_trecho)

    def parar_trecho(self):
        self._tocando = False
        if self._id_after is not None:
            try:
                self.janela.after_cancel(self._id_after)
            except Exception:
                pass
            self._id_after = None
        self._frames_trecho = []
        if self.video is not None:
            self.botao_tocar.config(text="▶  VER TRECHO (Espaço)")

    # =============================================================
    # AÇÕES
    # =============================================================

    def classificar(self, classe):
        e = self.evento_atual
        if e is None:
            return
        e["classe"] = classe
        e["revisado"] = True
        e["descartado"] = False
        self._apos_alteracao(e)

    def confirmar_sugestao(self):
        e = self.evento_atual
        if e is None:
            return
        sugestao = e.get("classe_sugerida") or e["classe"]
        if sugestao not in self.resultado.classes:
            messagebox.showinfo(
                "Revisão", "Este veículo não tem sugestão do modelo. Escolha a classe (1, 2, 3...) "
                "ou descarte (0) se não for veículo.", parent=self.janela)
            return
        self.classificar(sugestao)

    def descartar(self):
        e = self.evento_atual
        if e is None:
            return
        e["revisado"] = True
        e["descartado"] = True
        self._apos_alteracao(e)

    def _apos_alteracao(self, e):
        iid = str(e["id"])
        if self.arvore.exists(iid):
            self.arvore.item(iid, values=self._valores(e), tags=self._tags(e))
        self._salvar()
        self._atualizar_pendentes()
        if self.ao_alterar:
            self.ao_alterar()
        self._ir_para_proximo(iid)

    def _ir_para_proximo(self, iid_atual):
        itens = list(self.arvore.get_children())
        if iid_atual not in itens:
            return
        pos = itens.index(iid_atual)
        # próximo pendente depois do atual; senão, o próximo da lista
        for iid in itens[pos + 1:] + itens[:pos]:
            e = self._evento_por_id(int(iid))
            if e and e.get("duvida") and not e.get("revisado"):
                self.arvore.selection_set(iid)
                self.arvore.see(iid)
                self._ao_selecionar()
                return
        if pos + 1 < len(itens):
            self.arvore.selection_set(itens[pos + 1])
            self.arvore.see(itens[pos + 1])
            self._ao_selecionar()
        else:
            self._mostrar_detalhe(self._evento_por_id(int(iid_atual)))
            if not self.resultado.duvidas(so_pendentes=True):
                messagebox.showinfo("Revisão", "Todas as dúvidas foram conferidas.", parent=self.janela)

    def _salvar(self):
        try:
            self.resultado.salvar()
        except Exception as erro:
            messagebox.showerror("Erro ao salvar", str(erro), parent=self.janela)

    def adicionar_manual(self):
        tempo_inicial = formatar_tempo(self.evento_atual["tempo"]) if self.evento_atual else "00:00:00"
        dialogo = DialogoManual(self.janela, self.resultado.classes, tempo_inicial)
        self.janela.wait_window(dialogo.janela)
        if dialogo.resposta is None:
            return
        tempo, classe = dialogo.resposta
        if tempo > self.resultado.duracao_video + 1:
            messagebox.showerror("Erro", "Tempo maior que a duração do vídeo.", parent=self.janela)
            return
        evento = self.resultado.adicionar_manual(tempo, classe)
        self._salvar()
        if self.ao_alterar:
            self.ao_alterar()
        self.var_filtro.set("Todos os veículos")
        self.recarregar_lista(selecionar_id=evento["id"])

    # =============================================================

    def existe(self):
        try:
            return bool(self.janela.winfo_exists())
        except tk.TclError:
            return False

    def trazer_para_frente(self):
        self.janela.deiconify()
        self.janela.lift()
        self.janela.focus_force()

    def fechar(self):
        self.parar_trecho()
        self._salvar()
        if self.video is not None:
            self.video.release()
            self.video = None
        self.janela.destroy()
        if self.ao_fechar:
            self.ao_fechar()


class DialogoManual:
    """Pergunta tempo e classe de um veículo que o modelo não contou."""

    def __init__(self, pai, classes, tempo_inicial):
        self.resposta = None
        self.classes = classes

        self.janela = tk.Toplevel(pai)
        self.janela.title("Adicionar veículo")
        self.janela.transient(pai)
        self.janela.resizable(False, False)
        self.janela.grab_set()

        tk.Label(self.janela, text="Tempo no vídeo (hh:mm:ss):", font=("Arial", 10)).grid(
            row=0, column=0, padx=10, pady=(12, 4), sticky="w")
        self.var_tempo = tk.StringVar(value=tempo_inicial)
        entrada = tk.Entry(self.janela, textvariable=self.var_tempo, width=12, font=("Arial", 11))
        entrada.grid(row=0, column=1, padx=10, pady=(12, 4))
        entrada.focus_set()
        entrada.select_range(0, "end")

        tk.Label(self.janela, text="Classe:", font=("Arial", 10)).grid(row=1, column=0, padx=10, pady=4, sticky="w")
        self.var_classe = tk.StringVar(value=nome_exibicao(classes[0]))
        ttk.Combobox(
            self.janela, textvariable=self.var_classe, state="readonly", width=14,
            values=[nome_exibicao(c) for c in classes],
        ).grid(row=1, column=1, padx=10, pady=4)

        botoes = tk.Frame(self.janela)
        botoes.grid(row=2, column=0, columnspan=2, pady=12)
        tk.Button(botoes, text="ADICIONAR", font=("Arial", 10, "bold"), command=self._ok).pack(side="left", padx=6)
        tk.Button(botoes, text="Cancelar", command=self.janela.destroy).pack(side="left", padx=6)
        self.janela.bind("<Return>", lambda e: self._ok())
        self.janela.bind("<Escape>", lambda e: self.janela.destroy())

    def _ok(self):
        try:
            tempo = interpretar_tempo(self.var_tempo.get())
        except ValueError:
            messagebox.showerror("Erro", "Tempo inválido. Use hh:mm:ss.", parent=self.janela)
            return
        nomes = [nome_exibicao(c) for c in self.classes]
        classe = self.classes[nomes.index(self.var_classe.get())]
        self.resposta = (tempo, classe)
        self.janela.destroy()
