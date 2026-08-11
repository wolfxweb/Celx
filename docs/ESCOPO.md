# Escopo do projeto

## Objetivo

Especializar um LLM para analisar código-fonte legado e produzir documentação técnica em
português que ajude pessoas desenvolvedoras a entender o funcionamento do sistema e as regras
de negócio sustentadas pelo código.

## Dentro do escopo

- funções e métodos em PHP, Python e JavaScript;
- consultas, views, procedures e regras observáveis em SQL;
- objetivo, parâmetros, retorno e fluxo de execução;
- validações, exceções, dependências e efeitos colaterais observáveis;
- regras de negócio diretamente identificáveis;
- pontos que não podem ser determinados pelo trecho analisado;
- treinamento LoRA/QLoRA, avaliação, inferência e publicação.

## Fora do escopo inicial

- gerar uma especificação normativa para desenvolvimento futuro;
- afirmar requisitos que não estejam sustentados pelo código;
- analisar uma aplicação inteira em uma única entrada;
- substituir revisão técnica ou conhecimento de especialistas do negócio;
- QLoRA 4-bit com bitsandbytes neste Mac (usar LoRA/MPS via `train_lora.py`);
- editar `arquivos/` como código ativo (é histórico congelado do reset).

## Formato mínimo da documentação

1. Método ou função / Consulta SQL
2. Objetivo
3. Parâmetros
4. Retorno
5. Funcionamento
6. Regras de negócio identificadas
7. Pontos não determinados

## Critério central de qualidade

A documentação deve ser fiel ao comportamento observável. Uma resposta incompleta que declare
uma incerteza é preferível a uma regra de negócio inventada.
