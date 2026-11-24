# Agent Guide — Audio Engineering Lab

Este arquivo define como agentes de desenvolvimento (incluindo agentes de IA) devem compreender, modificar, testar e documentar este repositório.

## 1. Objetivo do produto

O projeto é um ambiente didático e experimental de engenharia de áudio, com três superfícies principais:

- Curso guiado: `audio_engineering_course.py`.
- Laboratório de análise e DSP: `audio_engineering_lab.py`.
- Smart Audio Calibration: calibração automática de áudio orientada à redução de erros de reconhecimento de fala.

O objetivo do Smart Audio Calibration não é deixar o áudio subjetivamente mais agradável. É encontrar um processamento que melhore a transcrição de fala, medido por WER/CER contra texto de referência revisado.

## 2. Regras obrigatórias para agentes

1. Inspecione os arquivos e o histórico relevantes antes de alterar código. Não presuma que a documentação esteja atualizada.
2. Preserve o áudio original. Todo processamento deve operar em uma cópia e permitir comparação com o baseline.
3. Não declare melhoria de inteligibilidade com base apenas em RMS, pico, espectro ou aparência da forma de onda.
4. Não declare SNR sem um método de estimação definido e documentado ou uma referência de ruído apropriada.
5. Compare candidatos com o mesmo mecanismo ASR, modelo, idioma e parâmetros de decodificação. Registre essas condições.
6. Não selecione um filtro só porque é comum em áudio. Exija evidência de melhora na métrica e valide contra o baseline.
7. Evite overfitting às frases de calibração. Proponha conjunto separado de validação, frases inéditas e, quando possível, vários locutores e ambientes.
8. Não trate equalização como solução universal. Clipping, reverberação, vento, microfone distante e ruído estacionário/dinâmico são problemas diferentes.
9. Não envie gravações, transcrições ou dados de voz para serviços externos sem explicar o fluxo e obter consentimento explícito do usuário.
10. Não afirme que interface, microfone, navegador, áudio ou pipeline foram testados se isso não foi realmente executado.
11. Mantenha as mudanças focadas. Não reescreva o curso ou laboratório existente como parte de uma tarefa isolada do calibrador.
12. Atualize esta documentação e o README quando arquitetura, comandos, dependências ou limitações mudarem.

## 3. Arquitetura atual

### Smart Audio Calibration

- `smart_audio_calibration_app.py`: interface Streamlit, coleta de frases, execução dos candidatos, resultados e exportação do perfil.
- `smart_audio_calibration.py`: funções de DSP, diagnóstico, preparação do waveform para ASR, normalização textual, métricas e seleção do perfil.
- `browser_asr_component/index.html`: componente customizado que usa Web Speech API do navegador para reconhecimento ao vivo.
- `iniciar_calibracao.sh`: inicializador dedicado.
- `requirements.txt`: dependências Python do projeto.

Fluxo atual:

1. Usuário grava frases de referência conhecidas.
2. Streamlit recebe os arquivos de áudio.
3. O decoder seleciona o canal com maior RMS em gravações multicanais, evitando média que possa causar cancelamento de fase.
4. Cada amostra é avaliada em uma grade de perfis DSP.
5. Cada versão é enviada ao mesmo Faster-Whisper (idioma português, configuração fixa).
6. WER/CER são calculados contra a frase de referência.
7. O perfil com menor WER médio é escolhido; se não superar o baseline, mantém-se o áudio original.
8. A interface apresenta diagnóstico, transcrição, áudio processado e exportação JSON.

### Mecanismos ASR

**Faster-Whisper**
- É o mecanismo de avaliação A/B usado pela calibração automática.
- Executa localmente via CPU/int8 na implementação atual.
- O tamanho do modelo pode ser selecionado na interface.
- A primeira execução pode baixar pesos e consumir tempo e memória.

**Web Speech API**
- É um reconhecimento ao vivo e independente, executado pelo navegador.
- Não recebe o áudio filtrado pelo pipeline Python.
- Pode depender de conectividade e de serviços do navegador; não presumir processamento local ou privacidade offline.
- Sua saída não deve ser comparada diretamente ao WER do Whisper sem controlar as diferenças de modelo, captura, texto e condições.

## 4. Princípios de processamento de áudio

- Trabalhe em mono apenas quando necessário e documente como os canais são combinados ou selecionados.
- Preserve sample rate e metadados originais; faça resampling explicitamente para o formato exigido pelo ASR.
- Whisper recebe waveform float32 mono a 16 kHz nesta implementação.
- Evite clipping após filtros, ganho ou normalização. Limitar pico não recupera clipping já gravado.
- Filtros passa-alta devem ter frequência de corte configurável e respeitar Nyquist.
- Redução de ruído deve ser conservadora: artefatos podem apagar consoantes e prejudicar WER.
- Não aplique compressor, gate, limiter ou EQ adicional sem experimento reproduzível que justifique a mudança.
- Diagnósticos devem distinguir métricas medidas de estimativas heurísticas.

## 5. Métricas e seleção

- WER (Word Error Rate) é a métrica primária de seleção atual.
- CER (Character Error Rate) é complementar.
- Normalize texto de referência e hipótese com a mesma função antes da comparação.
- Uma frase de referência vazia não deve produzir WER/CER artificialmente válidos.
- Faça média por amostra/frase e informe o tamanho do conjunto.
- Em caso de empate ou ausência de melhoria sobre o baseline, mantenha o baseline.
- Melhorias pequenas devem ser tratadas com cautela; para produção, adote validação separada e análise de variabilidade por frase/locutor.

## 6. Execução local

Requisitos documentados: Python 3.10+, ambiente virtual e dependências de `requirements.txt`. FFmpeg é necessário para formatos que não sejam suportados diretamente por libsndfile.

Instalação:

```bash
./instalar.sh
```

Curso:

```bash
./iniciar.sh
```

Laboratório:

```bash
.venv/bin/streamlit run audio_engineering_lab.py
```

Smart Audio Calibration:

```bash
chmod +x iniciar_calibracao.sh
./iniciar_calibracao.sh
```

## 7. Testes e validação

Antes de concluir uma mudança:

1. Execute verificação de sintaxe nos módulos Python alterados, por exemplo:
   ` .venv/bin/python -m compileall -q smart_audio_calibration.py smart_audio_calibration_app.py `
2. Faça testes unitários para funções puras: resampling, preservação do array de entrada, clipping/headroom, filtros, normalização textual, WER/CER e seleção de baseline.
3. Use fixtures sintéticas determinísticas para validar DSP básico; não dependa de gravações privadas do usuário.
4. Teste com pelo menos: áudio silencioso, áudio curto, mono, estéreo, taxa de amostragem diferente de 16 kHz, amostras inválidas e áudio próximo do clipping.
5. Para integração ASR, registre modelo, versão, idioma e parâmetros; não confunda teste de unidade com teste de qualidade de transcrição.
6. Execute o Streamlit e valide manualmente upload/gravação, permissões do microfone, reprodução, download JSON e comportamento de erro quando o navegador não suporta Web Speech API.
7. Registre claramente o que foi executado e o que ficou pendente. Não invente cobertura, resultados ou sucesso de testes.

Se não houver testes automatizados, crie testes pequenos e focados sem introduzir dependências pesadas desnecessárias.

## 8. Diretrizes de interface e experiência

- Use português brasileiro na interface e mensagens voltadas ao usuário.
- Explique cada ajuste com linguagem técnica acessível.
- Mostre baseline e candidato lado a lado sempre que possível.
- Exiba os parâmetros exatos do perfil recomendado.
- Não esconder incerteza: indique quantidade de frases, WER/CER e limitações da amostra.
- Diferencie “perfil recomendado” de “garantia de transcrição correta”.
- Permita ao usuário ouvir o áudio original e o processado.
- Não substituir silenciosamente gravações nem aplicar perfil automaticamente a arquivos futuros sem consentimento.

## 9. Segurança, privacidade e dados

- Áudio de voz pode conter dados pessoais. Não registrar áudio/transcrição em logs por padrão.
- Não adicionar upload remoto, analytics, telemetria ou armazenamento persistente sem consentimento e documentação.
- Se implementar persistência de perfis, documente localização, retenção, exclusão e associação ao dispositivo/microfone.
- Nunca incluir tokens, credenciais, caminhos pessoais ou gravações reais em commits e fixtures.
- Trate uploads como entrada não confiável: valide tamanho, duração, sample rate, canais e valores finitos.

## 10. Próximas evoluções recomendadas

Prioridade técnica sugerida:

1. Adicionar testes unitários e fixtures sintéticas.
2. Separar frases de calibração e validação para reduzir overfitting.
3. Avaliar por frase e apresentar intervalo/variabilidade, não apenas uma média.
4. Expandir diagnósticos: ruído de fundo, clipping, silêncio, reverberação e saturação, com métodos explicitados.
5. Construir busca controlada de parâmetros com limites e early stopping, mantendo baseline.
6. Comparar mecanismos ASR com protocolo equivalente e deixar claro quando a captura não é a mesma.
7. Persistir perfis por dispositivo/microfone somente com consentimento.
8. Integrar o calibrador ao laboratório principal depois de estabilizar a API e os testes.

## 11. Critérios de conclusão de tarefas

Uma tarefa só deve ser considerada concluída quando:
- a implementação corresponde ao pedido;
- os testes relevantes foram executados ou as limitações foram declaradas;
- a documentação foi atualizada;
- o áudio original continua preservado;
- nenhuma alegação de melhoria excede os dados medidos;
- o resumo final lista arquivos alterados, comandos/testes executados e pendências.
