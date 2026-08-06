# Rubrica de avaliação

Cada resposta recebe nota de 1 a 5 em cinco dimensões. A avaliação deve ser feita sem conhecer
se a resposta veio do modelo-base ou do modelo ajustado.

| Dimensão | 1 | 3 | 5 |
|---|---|---|---|
| Fidelidade | Contradiz ou inventa comportamentos | Mistura fatos e inferências | Todas as afirmações são sustentadas pelo código |
| Cobertura | Omite o comportamento principal | Registra o fluxo principal | Cobre fluxo, entradas, saída, condições e efeitos observáveis |
| Clareza | Texto confuso ou ambíguo | Compreensível com pequenas ambiguidades | Direto, preciso e adequado ao público técnico |
| Regras de negócio | Inventa regras | Identifica parcialmente ou sem evidência clara | Identifica regras observáveis e declara limites |
| Estrutura | Ignora o formato | Usa a maioria das seções | Usa todas as seções de forma consistente |

## Erros críticos

Uma resposta é reprovada independentemente da média quando:

- inventa regra de negócio relevante;
- descreve retorno incompatível com o código;
- omite efeito colateral importante;
- afirma tratamento de erro que não existe;
- inclui dados sensíveis presentes apenas no contexto de avaliação.

## Aprovação inicial

- média geral mínima: 4,0;
- fidelidade mínima: 4;
- nenhuma ocorrência de erro crítico;
- formato completo em pelo menos 95% das respostas;
- resultado superior ao baseline nas três linguagens.

Os limites poderão ser recalibrados após a avaliação piloto, mas qualquer mudança deve ser
registrada antes da avaliação final.

