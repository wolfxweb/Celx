SELECT
    p.id,
    p.cliente_id,
    SUM(i.valor) AS valor_total
FROM pedidos AS p
JOIN itens_pedido AS i ON i.pedido_id = p.id
WHERE p.status = 'PENDENTE'
GROUP BY p.id, p.cliente_id
HAVING SUM(i.valor) > 0;

