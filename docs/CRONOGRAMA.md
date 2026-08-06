# Cronograma de criação

## Visão geral

- Início planejado: 3 de agosto de 2026
- Encerramento planejado: 11 de outubro de 2026
- Duração: 10 semanas
- Dedicação estimada: 8 a 12 horas por semana
- Objetivo: criar e validar um LLM especializado em documentar código legado e explicar
  regras de negócio observáveis em PHP, Python, JavaScript e SQL.

O cronograma separa pesquisa, construção e validação. Downloads grandes, treinamento em GPU
e publicação somente serão executados nas etapas correspondentes e mediante comando explícito.

## Cronograma semanal

| Semana | Período | Foco | Entregas | Critério de aceite |
|---:|---|---|---|---|
| 1 | 03/08 a 09/08 | Fundação do projeto | Repositório, estrutura, ambiente, escopo e formato documental | Instalação e testes básicos executam localmente |
| 2 | 10/08 a 16/08 | Seleção do modelo | Comparativo de modelos, baseline e decisão registrada | Modelo principal e alternativa aprovados |
| 3 | 17/08 a 23/08 | Aquisição dos dados | CodeXGLUE e fonte complementar para SQL | PHP, Python, JavaScript e SQL carregam de forma reproduzível |
| 4 | 24/08 a 30/08 | Análise exploratória | Notebook EDA, qualidade, duplicatas e distribuição de tokens | Relatório identifica filtros e riscos do dataset |
| 5 | 31/08 a 06/09 | Preparação SFT | Limpeza, divisão, prompts e conjunto curado em português | Dataset validado e sem vazamento conhecido |
| 6 | 07/09 a 13/09 | LoRA e QLoRA | Baseline e experimentos de treinamento | Adapter e métricas de treinamento preservados |
| 7 | 14/09 a 20/09 | Avaliação | Métricas automáticas e rubrica humana | Modelo comparado ao baseline por linguagem |
| 8 | 21/09 a 27/09 | Inferência | CLI, notebook e exemplos legados | Código novo gera documentação no padrão definido |
| 9 | 28/09 a 04/10 | Publicação | Model card, adapter, Hugging Face e teste GGUF/Ollama | Artefatos reproduzíveis e instruções verificadas |
| 10 | 05/10 a 11/10 | Consolidação | README, artigo, apresentação e demonstração | Projeto reproduzido do início ao fim |

## Marcos

### M1 — Fundação pronta (09/08)

- estrutura de diretórios;
- dependências e configuração;
- prompt e formato de saída;
- cronograma e critérios de sucesso;
- testes unitários básicos.

### M2 — Dados prontos (06/09)

- análise exploratória concluída;
- filtros justificados;
- divisões de treino, validação e teste;
- amostra curada em português;
- manifesto do dataset.

### M3 — Modelo validado (20/09)

- baseline registrado;
- experimentos LoRA/QLoRA rastreáveis;
- avaliação automática;
- avaliação humana por linguagem;
- limitações documentadas.

### M4 — Projeto publicado (11/10)

- adapter e model card;
- inferência local e no Colab;
- README final;
- artigo, apresentação e demonstração.

## Dependências entre atividades

```text
Fundação → seleção do modelo → dados → preparação SFT → treinamento → avaliação
                                                                    ↓
                                                    inferência → publicação → entrega
```

## Riscos e respostas

| Risco | Impacto | Resposta planejada |
|---|---|---|
| Docstrings do CodeXGLUE não possuem regras detalhadas | Alto | Criar subconjunto curado e não rotular inferências como fatos |
| Memória insuficiente no Colab | Alto | Usar QLoRA 4-bit, modelo de 1,7B e gradient accumulation |
| Modelo inventar regras | Alto | Prompt restritivo, seção de pontos não determinados e avaliação de fidelidade |
| Português inconsistente | Médio | Curadoria de exemplos em português e rubrica de clareza |
| Vazamento entre treino e teste | Alto | Deduplicação por hash do código antes do treinamento |
| Mudanças nas bibliotecas | Médio | Fixar versões após o primeiro treinamento reproduzível |

## Estado atual

| Item | Estado |
|---|---|
| Estrutura de diretórios | Concluído |
| Escopo, cronograma e decisões | Concluído |
| Configuração e prompt iniciais | Concluído |
| README, exemplo e rubrica | Concluído |
| Testes do contrato documental | Concluído |
| Candidatos e protocolo de baseline | Concluído |
| Execução e relatório do baseline | Concluído |
| Modelo-base Qwen3 1.7B | Selecionado |
| Pipeline de preparação inicial | Em andamento |
| Treinamento QLoRA inicial | Em revisão |
| Inferência inicial | Em revisão |
| EDA, métricas, notebook e publicação | Não iniciado |

O código criado antecipadamente faz parte do M1 e permanece sujeito à revisão durante as
semanas correspondentes. Nenhum treinamento ou download de dataset foi iniciado neste PC.
