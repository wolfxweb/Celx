# Execução no Kaggle

Use um Kaggle Notebook comum criado em `https://www.kaggle.com/code/new`. Não use a área
Kaggle Benchmarks nem crie um Benchmark Task, pois esse ambiente pode não incluir PyTorch ou GPU.

## Configuração

1. Abra `03_benchmark_100_kaggle.ipynb` em um Notebook comum.
2. Em Settings, selecione GPU T4 ou P100.
3. Ative Internet.
4. Em Add Input, crie ou selecione um dataset privado contendo `Celx-colab-v8.zip`.
5. Execute as células em ordem.

O notebook usa `/kaggle/input` somente para leitura e `/kaggle/working` para arquivos gerados.
Ao final, salve uma versão para preservar os outputs.

## Continuação

Para continuar uma execução em outra sessão, adicione o ZIP ou JSONL da execução anterior como
Input e copie o resultado correspondente para `evaluation/results/benchmark_100` antes de rodar
o modelo. O executor ignora IDs já concluídos.

