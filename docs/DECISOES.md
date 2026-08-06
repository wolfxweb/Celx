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

- Estado: aceita para o primeiro ciclo
- Data: 01/08/2026

O `Qwen/Qwen3-1.7B` será o modelo-base do primeiro Fine-Tuning QLoRA. No baseline com quatro
linguagens, foi aproximadamente 42,3% mais rápido que o Ministral, gerou 49,3% menos tokens e
apresentou maior aderência ao formato. O Ministral permanece apenas como referência comparativa.

A decisão será reavaliada após a expansão do benchmark e a avaliação humana independente.
