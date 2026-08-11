# Registro de decisões

## D001 — Foco documental

- Estado: aceita
- Data: 01/08/2026

O projeto documentará o funcionamento de sistemas legados e regras de negócio observáveis.
Não produzirá especificações normativas para desenvolvimento futuro.

## D002 — Linguagens iniciais

- Estado: aceita
- Data: 01/08/2026

PHP, Python, JavaScript e SQL serão atendidas na primeira versão. O CodeXGLUE será usado para
as três linguagens de programação. Como não oferece SQL, essa linguagem terá uma fonte de dados
complementar e uma etapa de preparação própria.

## D003 — Dataset inicial

- Estado: aceita com ressalvas
- Data: 01/08/2026

O CodeXGLUE Code-to-Text será usado para a etapa geral código-para-texto. Como as docstrings não
representam documentação completa nem garantem regras de negócio, um subconjunto curado em
português será necessário antes do treinamento final.

## D004 — Modelo inicial configurável

- Estado: provisória
- Data: 01/08/2026

`Qwen/Qwen3-1.7B` é o padrão inicial por ser leve o suficiente para experimentos com QLoRA.
A decisão final depende do comparativo da Semana 2 e poderá ser alterada somente na configuração.

## D005 — Execução pesada explícita

- Estado: aceita
- Data: 01/08/2026

Downloads, treinamento em GPU e publicação não serão disparados como efeito colateral de testes
ou instalação. Cada operação terá um comando explícito e artefatos fora do controle de versão.

## D006 — Modelo-base selecionado

- Estado: provisória (reabrir após reset)
- Data: 01/08/2026

No ciclo anterior, `Qwen/Qwen3-1.7B` venceu o baseline curto. Após o reset, a escolha deve ser
reconfirmada pelo benchmark da Fase 2 (`docs/PLANO.md`). Histórico em `arquivos/docs/`.

## D007 — Reset do repositório

- Estado: aceita
- Data: 06/08/2026

O estado anterior foi arquivado em `arquivos/`. A raiz contém apenas o ciclo ativo. Fonte da
verdade: raiz + `docs/PLANO.md`. Não editar `arquivos/` como código vivo.
