# Seleção do modelo-base

## Objetivo

Escolher o modelo leve com melhor equilíbrio entre fidelidade ao código, português, aderência ao
formato, tempo de inferência, memória para QLoRA, licença e facilidade de publicação.

Nenhum candidato será escolhido apenas por benchmarks gerais. Todos devem receber o mesmo
prompt e os mesmos exemplos de PHP, Python, JavaScript e SQL.

## Candidatos

| Modelo | Porte | Licença | Acesso | Papel no teste |
|---|---:|---|---|---|
| Qwen/Qwen3-1.7B | 1,7B | Apache 2.0 | Aberto | Candidato principal inicial |
| mistralai/Ministral-3-3B-Instruct-2512-BF16 | 3B | Apache 2.0 | Aberto | Limite superior de qualidade/custo |

Os metadados executáveis ficam em `configs/model_candidates.yaml`. Os dois candidatos usam
licença Apache 2.0 e podem ser baixados sem token. A licença deve ser reconfirmada antes da
publicação do adapter.

## Protocolo

1. Executar o benchmark em ambiente Colab com a mesma GPU.
2. Usar quantização 4-bit em todos os modelos compatíveis.
3. Usar temperatura zero e limite de 700 novos tokens.
4. Processar os casos sempre na mesma ordem.
5. Preservar resposta, tempo e quantidade de tokens.
6. Avaliar as respostas sem identificar o modelo ao avaliador quando possível.
7. Aplicar a rubrica em `docs/RUBRICA.md`.

## Casos mínimos

O arquivo `dataset/benchmark/baseline.jsonl` começa com um caso de cada linguagem e registra:

- código analisado;
- fatos que a documentação deveria cobrir;
- armadilhas que o modelo não deve afirmar;
- identificador estável para comparação.

O benchmark expandido contém 100 casos por linguagem: PHP, Python e JavaScript amostrados do
CodeXGLUE e SQL amostrado do Spider. Os quatro casos manuais permanecem como conjunto de controle.
Como as referências públicas estão em inglês e não seguem o contrato documental, o conjunto
expandido mede estrutura, estabilidade e desempenho; a fidelidade exige revisão humana amostral.

## Comandos

No Google Colab, use `notebooks/01_baseline_colab.ipynb`. O notebook valida a GPU, instala as
dependências, recebe o projeto em ZIP, executa o Qwen e exporta a planilha de avaliação.

Executar somente o Qwen:

```bash
python scripts/run_baseline.py --model qwen3-1.7b
```

Executar os dois candidatos:

```bash
python scripts/run_baseline.py
```

Criar a planilha de avaliação:

```bash
python scripts/score_baseline.py
```

## Pontuação de decisão

| Critério | Peso |
|---|---:|
| Fidelidade ao código | 30% |
| Regras de negócio sem invenções | 20% |
| Cobertura | 15% |
| Português e clareza | 10% |
| Estrutura | 10% |
| Memória e tempo | 10% |
| Licença e distribuição | 5% |

Um modelo com erro crítico de fidelidade recorrente não poderá vencer apenas por desempenho ou
menor consumo de memória.

## Estado da decisão

- Modelo selecionado para o primeiro ciclo: `Qwen/Qwen3-1.7B`.
- Baseline alternativo: `mistralai/Ministral-3-3B-Instruct-2512-BF16`.
- Fundamentação: `docs/RELATORIO_BASELINE.md`.
- Dataset ou modelo baixado neste PC: não.
