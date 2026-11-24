# Smart Audio Calibration — Especificação técnica

## Objetivo

Calibrar automaticamente áudio de fala para reduzir erros de transcrição. O critério de sucesso é a qualidade textual medida contra uma referência, não uma preferência subjetiva de timbre.

## Escopo da versão atual

- Captura pelo navegador via Streamlit `st.audio_input`.
- Seis frases de referência em português brasileiro.
- Diagnóstico de duração, pico, RMS, clipping suspeito e contraste dinâmico.
- Seis perfis DSP conservadores, incluindo baseline.
- Avaliação automática com Faster-Whisper e cálculo de WER/CER.
- Seleção do menor WER médio, mantendo o baseline quando não há melhora.
- Exportação de parâmetros do perfil selecionado em JSON.
- Reconhecimento independente ao vivo via Web Speech API.

## Frases de calibração

1. O rato roeu a roupa do rei.
2. Três pratos de trigo para três tigres.
3. A casa azul fica próxima à estação.
4. Papai comprou pipoca para Pedro.
5. Sessenta e seis sessões foram realizadas.
6. Amanhã enviarei os documentos necessários.

As frases são um conjunto inicial de estímulos fonéticos variados, não um benchmark validado. Não presumir que cubram todos os fonemas, sotaques, velocidades, gêneros de fala ou ambientes.

## Pipeline de avaliação

Para cada frase gravada:

1. Decodificar amostras e taxa de amostragem.
2. Se houver múltiplos canais, selecionar o canal com maior RMS.
3. Aplicar cada perfil em uma cópia do áudio.
4. Converter waveform para mono float32 a 16 kHz para o Whisper.
5. Transcrever com Faster-Whisper usando idioma `pt`, beam size 5, VAD habilitado e contexto anterior desabilitado.
6. Normalizar referência e hipótese.
7. Calcular WER e CER.
8. Agregar métricas por perfil e comparar com baseline.

O modelo e as configurações devem permanecer constantes durante uma mesma avaliação A/B.

## Perfis DSP da versão atual

| Perfil | Processamento |
|---|---|
| Original (baseline) | Sem filtro |
| Passa-alta 70 Hz | Butterworth passa-alta de 2ª ordem |
| Passa-alta 100 Hz | Butterworth passa-alta de 2ª ordem |
| Redução leve de ruído | noisereduce, prop_decrease 0,30 |
| 70 Hz + redução leve | Combinação dos dois processos |
| 100 Hz + redução moderada | Combinação, prop_decrease 0,45 |

O conjunto é intencionalmente pequeno. Não representa uma busca global nem uma otimização universal para qualquer microfone.

## Critério de escolha

O perfil selecionado é o de menor WER médio. Se nenhum candidato superar o WER do áudio original, o baseline é mantido. CER é exibido como métrica complementar e não substitui a regra de seleção atual.

## Mecanismos de transcrição

### Faster-Whisper

É a referência controlada para a comparação de perfis. A implementação atual executa em CPU/int8 e permite escolher tiny, base ou small. A primeira execução pode baixar pesos do modelo.

### Web Speech API

É oferecida para reconhecimento ao vivo e comparação qualitativa. Não transcreve o arquivo filtrado pelo pipeline Python e não participa da seleção do perfil. A API pode depender de serviços externos do navegador; não prometer operação offline.

## Limitações conhecidas

- O conjunto de frases é pequeno e pode levar a overfitting.
- O resultado depende do modelo ASR e do seu comportamento com a variedade de fala do usuário.
- A API do navegador e o Faster-Whisper não usam necessariamente a mesma captura nem o mesmo backend.
- Equalização não recupera informação perdida por clipping, não corrige microfone distante e não elimina todo tipo de ruído.
- Contraste dinâmico não é uma estimativa validada de SNR.
- O componente customizado do navegador precisa de validação manual em navegadores e versões suportados.
- A interface ainda precisa de testes sistemáticos de ponta a ponta.

## Protocolo mínimo de validação futura

1. Dividir frases em calibração e validação, sem reutilizar a validação na busca.
2. Gravar com microfone, distância, ganho e ambiente documentados.
3. Incluir diferentes condições controladas: ruído estacionário, ruído variável, reverberação e níveis de fala.
4. Revisar manualmente as transcrições de referência.
5. Comparar WER/CER por frase e por condição, além da média.
6. Reportar modelo, versão, idioma, parâmetros e taxa de amostragem.
7. Só recomendar o perfil se a melhoria aparecer no conjunto de validação; em caso contrário, manter o baseline.

## Interface e execução

```bash
./instalar.sh
./iniciar_calibracao.sh
```

A aplicação é servida localmente pelo Streamlit, normalmente em `http://localhost:8501`.

## Decisões arquiteturais

- O processamento DSP está separado da interface Streamlit para facilitar testes unitários.
- O áudio de entrada é preservado e os candidatos são processados em cópias.
- Faster-Whisper é o avaliador controlado da calibração.
- Web Speech API é um mecanismo auxiliar independente, não intercambiável com o Whisper.
- O formato JSON exportado descreve parâmetros DSP; não contém o áudio ou uma garantia de desempenho.
