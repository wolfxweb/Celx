export function calculaTotal(pedido) {
  return pedido.itens.reduce((total, item) => total + item.valor, 0);
}

