<?php

function calculaTotal(Pedido $pedido): float
{
    $total = 0.0;
    foreach ($pedido->getItens() as $item) {
        $total += $item->getValor();
    }
    return $total;
}

