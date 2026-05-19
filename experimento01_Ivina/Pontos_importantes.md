# Pontos importantes para uma analise exploratória financeira

---

- Estacionariedade: testar com ADF (?) -- e provavelmente trabalhar com retornos logarítmicos: $$log(p_t / p_{t-1})$$ em vez do preço direto

- Análise de autocorrelação: ACF e PACF mostrariam quais lags realmente têm poder preditivo

- Decomposição da série:  separar tendência, sazonalidade e resíduo

- Distribuição dos retornos: A distribuição não é normal (fat tails), implica que afeta diretamente como interpretar o MAE e o MAPE

- Análise de votalidade

- Detecção de outliers e eventos
