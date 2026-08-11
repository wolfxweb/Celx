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

1. Preparar o projeto neste Mac e empacotar com `bash scripts/package_project.sh`.
2. Executar o benchmark no Kaggle (preferencial) ou Colab, sempre na mesma GPU da sessão.
3. Usar quantização 4-bit em todos os modelos compatíveis.
4. Usar temperatura zero; Qwen3 com `enable_thinking: true` (raciocínio sempre ativo) e até 1600 novos tokens.
5. Processar os casos sempre na mesma ordem.
6. Preservar resposta, tempo e quantidade de tokens em `outputs/`.
7. Avaliar as respostas sem identificar o modelo ao avaliador quando possível.
8. Aplicar a rubrica em `docs/RUBRICA.md`.

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

Notebook principal de seleção: `notebooks/01_benchmark.ipynb`.
Treino (depois): `notebooks/02_treino_qlora.ipynb`.

```bash
bash scripts/package_project.sh
# na GPU (Kaggle/Colab), a partir da raiz do projeto:
python scripts/run_baseline.py --model qwen3-1.7b
python scripts/run_baseline.py
python scripts/score_baseline.py
python scripts/compare_models.py
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

- Ciclo reiniciado: decisão do modelo deve ser **reconfirmada** neste benchmark.
- Candidato provisional (histórico): `Qwen/Qwen3-1.7B`.
- Alternativa: `mistralai/Ministral-3-3B-Instruct-2512-BF16`.
- Relatórios antigos: `arquivos/docs/RELATORIO_BASELINE.md`.
- Dataset ou modelo baixado neste PC: não.
