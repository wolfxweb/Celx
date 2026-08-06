# Rubrica de avaliação humana

Avalie cada critério de 1 a 5.

| Critério | 1 | 3 | 5 |
|---|---|---|---|
| Fidelidade ao código | Contradiz o código | Parcialmente correto | Totalmente sustentado pelo código |
| Cobertura das regras | Omite regras centrais | Cobre as principais | Cobre todas as regras observáveis |
| Ausência de alucinações | Inventa várias regras | Faz inferências discutíveis | Não inventa comportamento |
| Clareza | Confuso | Compreensível | Claro, direto e preciso |
| Estrutura | Ignora o formato | Segue parcialmente | Segue todas as seções |

## Procedimento

1. Selecione no mínimo 10 exemplos de cada linguagem.
2. Use os mesmos exemplos para os dois modelos.
3. O avaliador não deve saber qual modelo produziu cada resposta, se possível.
4. Registre justificativas para notas 1, 2 ou 5.
5. Considere regra de negócio somente o comportamento sustentado pelo código.
6. Marque como alucinação toda validação, efeito colateral ou restrição inventada.

Não use exemplos do conjunto de treinamento para a avaliação final.
