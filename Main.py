import tkinter as tk
from interface.janela_principal import JanelaPrincipal


if __name__ == "__main__":

    janela = tk.Tk()

    aplicativo = JanelaPrincipal(janela)

    janela.mainloop()