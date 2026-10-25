# Contador de Veículos

Conta carros, motos, caminhões e ônibus em gravações longas (1–2 h) de câmeras de monitoramento, em computador **sem placa de vídeo**.

O vídeo é processado em segundo plano, sem ser reproduzido, o mais rápido que a CPU permitir. Cada veículo que atravessa a linha de contagem é registrado com o tempo exato. Os casos duvidosos vão para uma tela de **conferência humana**, com a imagem do veículo e o trecho do vídeo. O resultado sai em Excel, em intervalos de 5 minutos.

> **Situação atual (out/2026):** em vídeo noturno com infravermelho, os modelos prontos (YOLO) reconhecem poucos veículos. Por isso o programa também conta **por movimento**: o que o modelo não reconhece entra como "Não classificado" e a pessoa escolhe a classe na revisão. Nesse modo, num trecho de teste com 28 veículos, a contagem final com conferência humana foi de 27. Detalhes em `docs/relatorio_contador_veiculos.docx`.

## Estrutura

```
Main.py                       inicia o programa
interface/
    janela_principal.py       vídeo, área de contagem, escolha do modelo, processamento
    janela_revisao.py         conferência humana (dúvidas, trecho do vídeo, comparação)
    util.py
processamento/
    config.py                 TODOS os parâmetros (limites de dúvida, movimento, etc.)
    detector.py               roda o modelo YOLO via OpenVINO/ONNX (sem PyTorch)
    rastreador.py             acompanha cada veículo entre os quadros
    contador.py               linha de contagem, votação de classe, dúvidas, fusão com movimento
    processador.py            leitura do vídeo em segundo plano
    resultado.py              salvar/abrir resultado e gerar o Excel
scripts/
    exportar_modelo.py        converte modelos YOLO (.pt) para OpenVINO/ONNX
    medir_velocidade.py       mede a velocidade na máquina de destino
    extrair_frames.py         extrai imagens dos vídeos para treinar um modelo próprio
docs/
    relatorio_contador_veiculos.docx   histórico de problemas, testes e alternativas
yolo11n.pt                    modelo base (os outros são baixados pelo script)
gerar_exe.bat                 gera o executável
```

Não vão para o Git (ver `.gitignore`): vídeos, `venv/`, `modelos/`, `resultados/` e `preferencias.json`. Cada pessoa gera os seus.

## Instalação (uma vez)

Requer Python 3.10 ou mais novo, no Windows.

```bat
git clone <endereço do repositório>
cd contador_veiculos
python -m venv venv
venv\Scripts\activate
pip install -r requirements-dev.txt

python scripts\exportar_modelo.py
python scripts\exportar_modelo.py --pesos yolo11s.pt --formato onnx
```

Os dois últimos comandos criam a pasta `modelos\`. O primeiro converte o `yolo11n.pt`; o segundo baixa e converte o `yolo11s`, que é o recomendado para vídeo noturno. A primeira execução pode instalar pacotes extras (onnx, onnxslim) sozinha; é normal.

## Como rodar

Sempre pelo terminal, com o ambiente ativo:

```bat
venv\Scripts\activate
python Main.py
```

> Não dê dois cliques no `Main.py` nem digite só `Main.py`: o Windows abre o arquivo no editor (VS Code) em vez de executar.

1. Escolha o modelo no campo **Modelo:** (canto superior direito). Para vídeo noturno use `yolo11s.onnx`. O programa lembra a escolha.
2. **SELECIONAR VÍDEO**. Aceita `.dav` (Intelbras/Dahua), `.mp4`, `.avi` etc. Confira a duração mostrada embaixo da imagem.
3. Clique 4 pontos contornando um trecho da pista, em ordem. Uma faixa atravessando a pista funciona bem. A **linha roxa** é onde o veículo é contado e precisa ir de um lado ao outro da pista; se não for, clique em **GIRAR LINHA**.
4. **INICIAR CONTAGEM**. A barra mostra progresso, velocidade e tempo restante.
5. Ao terminar, abra a **revisão**.
6. O Excel `contagem.xlsx` é salvo sozinho na pasta do resultado (botão **ABRIR PASTA DO RESULTADO**), ao terminar e ao fechar a revisão.

Resultados antigos podem ser reabertos em **ABRIR RESULTADO SALVO** (arquivo `resultado.json` dentro de `resultados\`), sem processar o vídeo de novo.

### Tela de revisão

| Tecla | Ação |
|---|---|
| `1` `2` `3` `4` | Escolhe a classe (carro, moto, caminhão, ônibus) |
| `Enter` | Confirma a sugestão do modelo |
| `0` | Descarta (não é veículo ou já foi contado) |
| `Espaço` | Toca o trecho do vídeo (2 s antes e depois) |

- O quadro **"Já contados perto deste (±10 s)"** lista os veículos contados perto do que está em revisão. **COMPARAR** toca um trecho com os dois: o veículo em revisão aparece com caixa **vermelha** e o já contado com caixa **verde**. Se for o mesmo carro, aperte `0`.
- **+ ADICIONAR VEÍCULO NÃO DETECTADO** registra um veículo que o programa perdeu.
- Tudo é salvo automaticamente.

## Como funciona

- **Processamento offline:** analisa 10 quadros por segundo de vídeo (ajustável na tela). O tempo vem do número do quadro, então é exato.
- **Rastreamento:** o veículo é acompanhado numa região maior em volta da área marcada e contado quando atravessa a linha.
- **Classe por votação:** soma as pontuações do modelo em todos os quadros em que o veículo apareceu.
- **Contagem por movimento:** regiões em movimento que atravessam a linha sem nenhuma detecção do modelo por perto viram eventos "Não classificado".
- **Dúvidas** vão para a revisão: classe dividida, confiança baixa, poucos quadros, veículos sobrepostos, possível contagem dupla e eventos só de movimento. Os limites ficam em `processamento/config.py`.

## Medir a velocidade na máquina de destino

```bat
python scripts\medir_velocidade.py "C:\caminho\do\video.dav"
```

## Gerar o .exe

Use um ambiente separado, sem torch/ultralytics, para o executável ficar pequeno:

```bat
python -m venv venv-exe
venv-exe\Scripts\activate
pip install -r requirements.txt pyinstaller
gerar_exe.bat
```

Copie a pasta `dist\ContadorVeiculos\` inteira para a outra máquina.

## Próximo passo: treinar com as imagens das câmeras

O que mais melhora a classificação é treinar um modelo com imagens dessas câmeras (dia e noite):

1. `python scripts\extrair_frames.py video1.dav video2.dav --a-cada 4`
2. Anotar (CVAT ou Label Studio, formato YOLO) com as classes `carro`, `moto`, `caminhao`, `onibus`.
3. Treinar no Google Colab: `yolo train model=yolo11s.pt data=dados.yaml epochs=100 imgsz=640`
4. `python scripts\exportar_modelo.py --pesos best.pt` e escolher o novo modelo na tela.

## Problemas comuns

- **Abre o VS Code em vez do programa:** use `python Main.py` no terminal com o `venv` ativo.
- **"Modelo não carregado":** falta a pasta `modelos\`; rode `python scripts\exportar_modelo.py`.
- **Excel não salva:** o arquivo `contagem.xlsx` está aberto no Excel; feche e use EXPORTAR EXCEL.
- **Projeto dentro do OneDrive:** funciona, mas a sincronização das imagens de cada resultado pode deixar o computador lento. Prefira uma pasta como `C:\projetos\`.

## Licença

O `ultralytics` e os pesos YOLO11 são AGPL-3.0. Para distribuição comercial, verifique a licença.
