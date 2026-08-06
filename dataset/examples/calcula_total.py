from decimal import Decimal


def calcula_total(pedido):
    total = Decimal("0")
    for item in pedido.itens:
        total += item.valor
    return total

