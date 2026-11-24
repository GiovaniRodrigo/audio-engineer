# Reutilizar o Smart Audio Calibration em outro projeto

Este guia explica como incorporar o núcleo de calibração de áudio em outro sistema Python sem importar a aplicação Streamlit deste repositório.

## 1. O que pode ser reutilizado

O componente reutilizável é, principalmente, o módulo `smart_audio_calibration.py`. Ele concentra operações independentes da interface:

- `CalibrationProfile`: representa um conjunto de parâmetros DSP.
- `CANDIDATE_PROFILES`: grade pequena de perfis experimentais já definida.
- `apply_profile(samples, sample_rate, profile)`: processa uma cópia do sinal e retorna outro array.
- `prepare_for_asr(samples, sample_rate, target_rate=16000)`: prepara waveform mono para o formato de entrada do ASR.
- `signal_diagnostics(samples, sample_rate)`: calcula duração, pico, RMS, clipping suspeito e contraste dinâmico.
- `normalized_text(text)` e `transcript_error(reference, hypothesis)`: normalização e métricas WER/CER.
- `select_profile(results)`: seleciona pelo menor WER médio e mantém o baseline quando não há melhora.

A aplicação `smart_audio_calibration_app.py` é apenas uma interface de referência em Streamlit. O componente `browser_asr_component/` é opcional e não participa da calibração A/B do arquivo.

## 2. Dependências

Para usar o módulo DSP e as métricas:

```bash
python -m pip install numpy scipy jiwer noisereduce
```

O pacote `noisereduce` é necessário somente para os perfis que usam redução de ruído. Para avaliar transcrições com Faster-Whisper:

```bash
python -m pip install faster-whisper
```

A aplicação Streamlit completa tem dependências adicionais. Não é necessário instalar Streamlit, Plotly ou o componente Web Speech se você for integrar o núcleo a outra interface.

Recomenda-se criar um ambiente virtual próprio no projeto de destino e fixar versões testadas no arquivo de dependências desse projeto.

## 3. Formato esperado do áudio

Antes de chamar o módulo, converta a entrada para um array NumPy unidimensional (mono), com amostras numéricas finitas e taxa de amostragem conhecida.

- O array deve representar o áudio, não bytes comprimidos do arquivo.
- Preserve o áudio original; processe cópias.
- Se o arquivo tiver múltiplos canais, escolha uma estratégia explícita (por exemplo, selecionar o canal com maior RMS). Não faça média de canais sem considerar possível cancelamento de fase.
- Não informe uma taxa de amostragem diferente da taxa real.
- Faça validação de duração, tamanho, canais e valores não finitos na camada de entrada.

## 4. Integração mínima do núcleo DSP

Copie `smart_audio_calibration.py` para o projeto de destino (por exemplo, em `seu_projeto/audio_calibration/`) ou instale-o como módulo interno. Ajuste os imports conforme a estrutura do seu pacote.

Exemplo de aplicação de um perfil:

```python
import numpy as np

from smart_audio_calibration import (
    CANDIDATE_PROFILES,
    apply_profile,
    prepare_for_asr,
    signal_diagnostics,
)

# audio_mono deve ser um ndarray mono, com taxa real sample_rate.
audio_original = np.asarray(audio_mono, dtype=np.float32).copy()

diagnostico = signal_diagnostics(audio_original, sample_rate)

perfil = CANDIDATE_PROFILES[1]  # Exemplo: passa-alta 70 Hz
audio_processado = apply_profile(audio_original, sample_rate, perfil)

# Para um ASR que espera mono float32 a 16 kHz:
audio_para_asr = prepare_for_asr(audio_processado, sample_rate, 16000)

# audio_original permanece preservado.
```

O exemplo escolhe um perfil apenas para demonstrar a API. Em produção, não escolha um perfil fixo sem avaliação no seu conjunto de dados.

## 5. Conectar ao mecanismo de transcrição (ASR)

O calibrador precisa de um adaptador para o ASR escolhido. Esse adaptador deve receber um waveform e retornar texto. Exemplo conceitual com Faster-Whisper:

```python
from faster_whisper import WhisperModel
from smart_audio_calibration import (
    CANDIDATE_PROFILES,
    apply_profile,
    prepare_for_asr,
    transcript_error,
    select_profile,
)

model = WhisperModel("base", device="cpu", compute_type="int8")

def transcrever(samples, sample_rate):
    waveform = prepare_for_asr(samples, sample_rate, 16000)
    segments, _ = model.transcribe(
        waveform,
        language="pt",
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=False,
    )
    return " ".join(segment.text.strip() for segment in segments).strip()

# Cada item contém áudio e transcrição humana revisada.
amostras = [
    {"audio": audio_1, "sample_rate": 48000, "reference": "Texto falado na gravação"},
    # Inclua outras gravações representativas.
]

resultados = []
for perfil in CANDIDATE_PROFILES:
    erros = []
    for item in amostras:
        candidato = apply_profile(item["audio"], item["sample_rate"], perfil)
        hipotese = transcrever(candidato, item["sample_rate"])
        erros.append(transcript_error(item["reference"], hipotese))

    resultados.append({
        "profile": perfil.name,
        "mean_wer": sum(x["wer"] for x in erros) / len(erros),
        "mean_cer": sum(x["cer"] for x in erros) / len(erros),
        "per_sample": erros,
    })

recomendado = select_profile(resultados)
print("Perfil recomendado:", recomendado["profile"])
```

Esse exemplo é um esqueleto de integração, não um benchmark pronto. Em uma implementação robusta, trate referências vazias, exceções do ASR, transcrições vazias e métricas não finitas; registre o número de amostras válidas por perfil. Garanta também que todos os candidatos usem exatamente o mesmo modelo, idioma e parâmetros de decodificação.

Se usar outro ASR (serviço HTTP, modelo local ou API proprietária), mantenha a mesma interface conceitual `transcrever(waveform, sample_rate) -> str`. Documente se o áudio é enviado a um provedor externo, quais dados são transmitidos e obtenha consentimento antes de enviar voz.

## 6. Arquitetura recomendada no projeto de destino

Mantenha o DSP independente da interface e do fornecedor de ASR:

```text
seu_projeto/
├── audio_calibration/
│   ├── __init__.py
│   ├── dsp.py                 # núcleo adaptado de smart_audio_calibration.py
│   ├── asr_adapter.py         # integração com o ASR escolhido
│   ├── calibration_service.py # loop de avaliação e seleção
│   └── schemas.py             # formatos de entrada e saída
├── interfaces/
│   └── ...                    # API, CLI, web app ou Streamlit
└── tests/
    ├── test_dsp.py
    └── test_calibration.py
```

Essa é uma organização sugerida, não a estrutura obrigatória do repositório atual. Evite acoplar as funções DSP a Streamlit, banco de dados ou provedor de transcrição.

## 7. Fluxo de integração

1. **Entrada:** receber o arquivo ou stream e validar formato, duração, canais e taxa de amostragem.
2. **Decodificação:** converter para waveform NumPy e manter uma cópia imutável do original.
3. **Diagnóstico:** calcular métricas descritivas e exibir alertas sem tratá-las como prova de inteligibilidade.
4. **Conjunto de referência:** obter gravações com transcrições humanas revisadas.
5. **Candidatos:** executar baseline e perfis DSP em cópias equivalentes do mesmo áudio.
6. **ASR:** transcrever todos os candidatos com o mesmo modelo e parâmetros.
7. **Avaliação:** calcular WER/CER por gravação e agregado.
8. **Validação independente:** testar o perfil vencedor em gravações que não participaram da seleção.
9. **Aplicação:** aplicar o perfil apenas quando a validação indicar benefício; manter o original disponível e permitir reversão.
10. **Observabilidade:** registrar parâmetros, versões, métricas agregadas e condições do teste, evitando armazenar áudio ou texto sensível sem necessidade.

## 8. Calibração e validação não são o mesmo conjunto

Não selecione e valide o perfil usando exatamente as mesmas gravações. Separe os dados, por exemplo:

- **Calibração:** conjunto usado para comparar candidatos e escolher parâmetros.
- **Validação:** conjunto independente usado uma única vez para medir generalização.
- **Teste operacional:** gravações reais posteriores, monitoradas com consentimento e política de retenção definida.

Inclua diversidade de locutores, microfones, distâncias, ambientes, sotaques e condições acústicas compatíveis com o uso real. Um perfil que melhora uma frase ou um microfone pode piorar outros.

## 9. Como interpretar os resultados

- WER é a métrica primária usada pelo seletor atual; CER é complementar.
- Menor WER indica menos erros de palavra no conjunto avaliado, não uma garantia universal de transcrição.
- RMS e pico medem amplitude; não demonstram melhoria de inteligibilidade.
- O indicador de clipping é heurístico e não recupera informação já perdida.
- Contraste dinâmico não equivale a SNR.
- Redução de ruído pode remover componentes importantes da fala, especialmente consoantes.
- Empate ou ausência de melhora deve favorecer o baseline.

A grade de perfis atual é deliberadamente pequena e conservadora. Adapte os parâmetros somente com experimentos reproduzíveis e validação independente.

## 10. Testes mínimos antes de usar em produção

- Áudio silencioso, muito curto, longo e próximo do clipping.
- Mono e multicanal, com seleção de canal testada.
- Taxas de amostragem diferentes, incluindo conversão para 16 kHz quando exigida pelo ASR.
- Arrays com NaN/Inf, formatos inesperados e arquivos inválidos.
- Garantia de que `apply_profile` não modifica o array de entrada.
- Baseline e candidatos avaliados com o mesmo ASR/configuração.
- Referência vazia, hipótese vazia, erro do ASR e resultado não finito.
- Teste de regressão com gravações de validação independentes.
- Teste de latência, memória e custo do mecanismo ASR escolhido.

Não publique uma alegação de melhoria sem informar o conjunto de avaliação, número de gravações, modelo ASR, parâmetros, baseline e resultados de validação.

## 11. Privacidade, licenciamento e distribuição

Áudio de voz pode conter dados pessoais. Defina consentimento, finalidade, retenção, exclusão, controle de acesso e política de logs. Não envie áudio a serviços externos sem informar claramente o usuário.

**Atenção à licença:** no estado documentado deste repositório, ainda não há licença de software definida. Antes de copiar, redistribuir ou incorporar o código em um produto próprio ou comercial, resolva a titularidade e a licença com o responsável pelo repositório. A existência de código público no GitHub, por si só, não concede uma licença de reutilização.

## 12. Checklist de adoção

- [ ] Confirmar autorização/licença para reutilizar o código.
- [ ] Copiar o núcleo DSP e instalar apenas as dependências necessárias.
- [ ] Implementar decodificação e validação de áudio do projeto de destino.
- [ ] Implementar o adaptador do ASR e documentar o fluxo de dados.
- [ ] Criar conjunto de calibração com referências revisadas.
- [ ] Criar conjunto separado de validação.
- [ ] Implementar testes automatizados e comparação com baseline.
- [ ] Expor parâmetros e resultados na interface do projeto.
- [ ] Definir consentimento, retenção e exclusão de dados.
- [ ] Documentar versões, limitações e resultados reais.

## Referências internas

- [Implementação do núcleo DSP](../smart_audio_calibration.py)
- [Especificação técnica original](SMART_AUDIO_CALIBRATION.md)
- [Guia para agentes de desenvolvimento](../AGENTS.md)
