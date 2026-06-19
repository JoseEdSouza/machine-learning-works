# %%
from statsmodels.tsa.stattools import adfuller
import numpy as np
from pathlib import Path
import os 
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats
import logging
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import pacf as compute_pacf
import joblib
from tensorflow import keras
from keras.callbacks import EarlyStopping
from statsmodels.stats.diagnostic import acorr_ljungbox
from datetime import timedelta
import matplotlib.gridspec as gridspec


# %%
logging.basicConfig(level=logging.INFO, 
format="%(asctime)s [%(levelname)s] %(message)s",
datefmt= "%d %H: %M: %S")

log = logging.getLogger(__name__)

#%% hiperparâmetros
SEED = 42
GOLD_DATA_PATH = Path(r"gold\data\final_gold_data.csv")
OUTPUT_DIR = Path("output_analisefinal")
OUTPUT_DIR.mkdir(exist_ok=True)

RAMDOM_SEED= 42
WINDOW_SIZE_MIN = 10  # Valor mínimo/fallback; será ajustado com base em lags significativos da PACF
WINDOW_SIZE_MAX = 50  # Limite máximo para evitar modelo muito pesado
WINDOW_SIZE = WINDOW_SIZE_MIN 

VAL_SPLIT = 0.1
LIMIAR_SIGMA = 2.0
JANELA_VOL = 21
LAGS = 60
HORIZONTE = 5
TRAIN_RATIO = 0.95
# %% configurando reprodutibilidade
log.info("\n config da reprodutibilidade")
np.random.seed(SEED)
try:
    tf.random.set_seed(SEED)
    log.info("Tensorflo seed configurado: %d", SEED)
except ImportError:
    log.warning("Tensorflow não disponível")
    
# %% carregar os dados
log.info("Carregaod os dados")

GOLD_DATA_PATH = Path(
    os.environ.get("GOLD_DATA_PATH", "../gold/data/final_gold_data.csv")
)

df = pd.read_csv(GOLD_DATA_PATH, sep=";", encoding="utf-8", parse_dates=["timestamp"])
log.info("Shape: %s", df.shape)
log.info("Tipos:\n%s", df.dtypes)
log.info("Estatísticas:\n%s", df.describe())

log.info("Nulos por coluna:\n%s", df.isnull().sum())

# %%
datas = pd.to_datetime(df["timestamp"])
gaps = datas.diff().dropna()
gaps_grandes = gaps[gaps > pd.Timedelta("3 days")]
log.info("Gaps > 3 dias: %d", len(gaps_grandes))

invalidos = df[df["close"] <= 0]
log.info("Linhas com close <= 0: %d", len(invalidos))

#%% visualizacao inicial
plt.figure(figsize=(14, 5))
plt.plot(datas, df["close"], linewidth=0.8)
plt.title("Preço de fechamento do ouro")
plt.xlabel("Data")
plt.ylabel("USD")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_preco_fechamento.png", dpi=150)
plt.close()
log.info("Gráfico salvo: 01_preco_fechamento.png")

# %% log
log.info("transformando em log")

df["log_return"] = np.log(df["close"] / df["close"].shift(1))
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

log.info("Série de retornos: primeiras linhas:\n%s",
         df[["timestamp", "close", "log_return"]].head())

# %% teste de estacionariedade (ADF)
log.info("\nTestando estacionariedade")

def testar_estacionariedade(serie, nome):
    resultado = adfuller(serie.dropna())
    p_value = resultado[1]
    estacionaria = p_value < 0.05
    log.info(
        "ADF [%s]: estatística: %.4f | p-value: %.4f | estacionária: %s",
        nome, resultado[0], p_value, estacionaria
    )
    return estacionaria

testar_estacionariedade(df["close"], "preco_fechamento")
testar_estacionariedade(df["log_return"], "log_return")

#%% distribuição dos log returns
log.info("\ndistribuição dos retornos")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(df["log_return"], bins=80, density=True, alpha=0.7, color="#378ADD")
mu, sigma = df["log_return"].mean(), df["log_return"].std()
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
axes[0].plot(x, stats.norm.pdf(x, mu, sigma), color="#E24B4A", linewidth=1.5,
             label="Normal teórica")
axes[0].set_title("Distribuição dos retornos")
axes[0].legend()
axes[0].grid(True)

stats.probplot(df["log_return"], plot=axes[1])
axes[1].set_title("Q-Q plot (vs normal)")
axes[1].grid(True)

axes[2].boxplot(df["log_return"], vert=True)
axes[2].set_title("Boxplot dos retornos")
axes[2].grid(True)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_distribuicao_retornos.png", dpi=150)
plt.close()

curtose = df["log_return"].kurt()
assimetria = df["log_return"].skew()
log.info("Curtose: %.4f | Assimetria: %.4f", curtose, assimetria)

# %% detecção de outliers (eventos extremos)
log.info("\nDetectando eventos extremos")

df["extremo"] = np.abs(df["log_return"]) > LIMIAR_SIGMA * df["log_return"].std()
eventos = df[df["extremo"]][["timestamp", "close", "log_return"]]
log.info("Eventos extremos (> %.1f desvios): %d ocorrências", LIMIAR_SIGMA, len(eventos))
log.info("Primeiros eventos:\n%s", eventos.head(10).to_string())

plt.figure(figsize=(14, 5))
plt.plot(df["timestamp"], df["log_return"], linewidth=0.6, alpha=0.8, label="Log return")
plt.scatter(eventos["timestamp"], eventos["log_return"],
            color="#E24B4A", s=20, zorder=5, label=f"Extremos (>{LIMIAR_SIGMA}σ)")
plt.axhline(0, color="gray", linewidth=0.5)
plt.title("Retornos logarítmicos e eventos extremos")
plt.xlabel("Data")
plt.ylabel("Log return")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_eventos_extremos.png", dpi=150)
plt.close()
log.info("Gráfico salvo: 03_eventos_extremos.png")

# %% volatilidade
log.info("\nAnalisando volatilidade")

df["volatilidade"] = df["log_return"].rolling(window=JANELA_VOL).std() * np.sqrt(252)

plt.figure(figsize=(14, 4))
plt.plot(df["timestamp"], df["volatilidade"], linewidth=0.8, color="#F39C12")
plt.title(f"Volatilidade anualizada (janela {JANELA_VOL} dias)")
plt.xlabel("Data")
plt.ylabel("Volatilidade")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_volatilidade.png", dpi=150)
plt.close()

vol_max_idx = df["volatilidade"].idxmax()
vol_max = df.loc[vol_max_idx, ["timestamp", "volatilidade"]]
log.info("Pico de volatilidade: %s", vol_max.to_dict())

# %% decomposição da série
log.info("\n[8] Decompondo série...")

decomp = seasonal_decompose(df["close"].values, model="multiplicative", period=252)

fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
componentes = ["Observado", "Tendência", "Sazonalidade", "Resíduo"]
dados = [df["close"].values, decomp.trend, decomp.seasonal, decomp.resid]

for ax, comp, dado in zip(axes, componentes, dados):
    ax.plot(dado, linewidth=0.7)
    ax.set_ylabel(comp)
    ax.grid(True)

axes[0].set_title("Decomposição da série (preço fechamento)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_decomposicao.png", dpi=150)
plt.close()
log.info("Gráfico salvo: 05_decomposicao.png")

#%%  acf e pacf
log.info("\nAnalisando ACF e PACF")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))
plot_acf(df["log_return"], lags=LAGS, ax=ax1, title="ACF — retorno logarítmico")
plot_pacf(df["log_return"], lags=LAGS, ax=ax2, title="PACF — retorno logarítmico")
ax1.grid(True)
ax2.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "06_acf_pacf.png", dpi=150)
plt.close()

pacf_vals = compute_pacf(df["log_return"].dropna(), nlags=LAGS)
conf_bound = 1.96 / np.sqrt(len(df["log_return"].dropna()))
lags_sig = [i for i, v in enumerate(pacf_vals) if abs(v) > conf_bound and i > 0]
log.info("Lags significativos na PACF: %s", lags_sig)


if lags_sig:
    window_sugerido = max(lags_sig)
else:
    window_sugerido = WINDOW_SIZE_MIN


WINDOW_SIZE = max(WINDOW_SIZE_MIN, min(window_sugerido, WINDOW_SIZE_MAX))
log.info("Window_size sugerido pela PACF: %d", window_sugerido)
log.info("Window_size final (após limites): %d", WINDOW_SIZE)

# %% correlação 
log.info("\ncorrelação entre variáveis")

df_num = df.select_dtypes(include=["int64", "float64"])

plt.figure(figsize=(9, 7))
sns.heatmap(df_num.corr(), annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.4, vmin=-1, vmax=1)
plt.title("Correlação entre variáveis numéricas")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "07_correlacao.png", dpi=150)
plt.close()
log.info("Gráfico salvo: 07_correlacao.png")

# %% preprocessamento para LSTM
log.info("\nPré-processando dados")
serie = df["log_return"].values
train_size = int(np.ceil(len(serie) * TRAIN_RATIO))

scaler = StandardScaler()
train_scaled = scaler.fit_transform(serie[:train_size].reshape(-1, 1))
test_scaled = scaler.transform(serie[train_size:].reshape(-1, 1))
full_scaled = np.concatenate([train_scaled, test_scaled], axis=0)

joblib.dump(scaler, OUTPUT_DIR / "scaler.pkl")
log.info("Scaler salvo. Média: %.6f | Desvio: %.6f", scaler.mean_[0], scaler.scale_[0])


def criar_janelas(dados, window):
    X, y = [], []
    for i in range(window, len(dados)):
        X.append(dados[i - window:i, 0])
        y.append(dados[i, 0])
    return np.array(X).reshape(-1, window, 1), np.array(y)

X_train, y_train = criar_janelas(full_scaled[:train_size], WINDOW_SIZE)
X_test, y_test = criar_janelas(full_scaled, WINDOW_SIZE)
X_test = X_test[train_size - WINDOW_SIZE:]
y_test = y_test[train_size - WINDOW_SIZE:]

log.info("X_train: %s | X_test: %s", X_train.shape, X_test.shape)

# %% modelo
log.info("\nConstruindo e treinando modelo LSTM")

try:
    model = keras.models.Sequential([
        keras.layers.LSTM(64, return_sequences=True, input_shape=(WINDOW_SIZE, 1)),
        keras.layers.LSTM(64, return_sequences=False),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1),
    ])
    model.summary()
    model.compile(optimizer="adam", loss="mae",
                  metrics=[keras.metrics.RootMeanSquaredError()])

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1,
    )

    log.info("Iniciando treinamento")
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=VAL_SPLIT,
        callbacks=[early_stop],
        verbose=1,
    )

    model.save(OUTPUT_DIR / "lstm_gold.keras")
    log.info("Modelo salvo: lstm_gold.keras")

except ImportError as e:
    log.error("TensorFlow não disponível: %s", e)
    log.warning("Pulando treinamento do modelo LSTM")
    model = None
    history = None

#%% 
#avaliação
if model is not None:
    log.info("\nAvaliando modelo")

    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import math

    previsoes_norm = model.predict(X_test, verbose=0)
    previsoes = scaler.inverse_transform(previsoes_norm)
    y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))

    mae = mean_absolute_error(y_test_real, previsoes)
    rmse = math.sqrt(mean_squared_error(y_test_real, previsoes))

    mask = np.abs(y_test_real) > 1e-6
    mape = np.mean(np.abs((y_test_real[mask] - previsoes[mask]) / y_test_real[mask])) * 100

    log.info("MAE : %.6f", mae)
    log.info("RMSE: %.6f", rmse)
    log.info("MAPE: %.2f%%", mape)

    # Curvas de aprendizado
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history["loss"], label="Treino")
    ax1.plot(history.history["val_loss"], label="Validação")
    ax1.set_title("Loss (MAE)")
    ax1.set_xlabel("Época")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history.history["root_mean_squared_error"], label="Treino", color="orange")
    ax2.plot(history.history["val_root_mean_squared_error"], label="Validação", color="red")
    ax2.set_title("RMSE")
    ax2.set_xlabel("Época")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "08_curvas_aprendizado.png", dpi=150)
    plt.close()
    log.info("Gráfico salvo: 08_curvas_aprendizado.png")


    plt.figure(figsize=(14, 5))
    plt.plot(y_test_real, color="#378ADD", linewidth=0.8, alpha=0.9, label="Real")
    plt.plot(previsoes, color="#E24B4A", linewidth=0.8, alpha=0.9,
             linestyle="--", label="Previsto")
    plt.title("Previsão vs real — conjunto de teste")
    plt.xlabel("Amostras")
    plt.ylabel("Log return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "09_previsao_vs_real.png", dpi=150)
    plt.close()
    log.info("Gráfico salvo: 09_previsao_vs_real.png")

    residuos = y_test_real.flatten() - previsoes.flatten()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(residuos, linewidth=0.6)
    axes[0].axhline(0, color="red", linewidth=0.8, linestyle="--")
    axes[0].set_title("Resíduos ao longo do tempo")
    axes[0].grid(True)

    axes[1].hist(residuos, bins=50, color="#378ADD", alpha=0.7)
    axes[1].set_title("Distribuição dos resíduos")
    axes[1].grid(True)

    plot_acf(residuos, lags=30, ax=axes[2])
    axes[2].set_title("ACF dos resíduos")
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "10_residuos.png", dpi=150)
    plt.close()
    log.info("Gráfico salvo: 10_residuos.png")


    lb_result = acorr_ljungbox(residuos, lags=[10, 20], return_df=True)
    log.info("Ljung-Box (ausência de autocorrelação):\n%s", lb_result)

#%% previsao futura
log.info("\nGerando previsões futuras")

def previsao_futuro(dias):
    janela = full_scaled[-WINDOW_SIZE:].copy()
    preds = []

    for _ in range(dias):
        entrada = janela.reshape(1, WINDOW_SIZE, 1)
        proximo = model.predict(entrada, verbose=0)
        preds.append(float(proximo[0, 0]))
        janela = np.append(janela[1:], proximo[0, 0]).reshape(-1, 1)

    return scaler.inverse_transform(np.array(preds).reshape(-1, 1))

log_returns_futuros = previsao_futuro(HORIZONTE)

# Converter retornos previstos para preço
ultimo_preco = df["close"].iloc[-1]
precos_futuros = [ultimo_preco]
for r in log_returns_futuros.flatten():
    precos_futuros.append(precos_futuros[-1] * np.exp(r))
precos_futuros = np.array(precos_futuros[1:])

log.info("Retornos previstos (log):\n%s", log_returns_futuros.flatten())
log.info("Preços estimados:\n%s", precos_futuros)



ultima_data = df["timestamp"].iloc[-1]
datas_futuras = [ultima_data + timedelta(days=i+1) for i in range(HORIZONTE)]


variacao_preco = precos_futuros - ultimo_preco
variacao_pct = (variacao_preco / ultimo_preco) * 100


previsoes_df = pd.DataFrame({
    "Data": datas_futuras,
    "Preço_USD": precos_futuros,
    "Log_Return_Previsto": log_returns_futuros.flatten(),
    "Variação_USD": variacao_preco,
    "Variação_%": variacao_pct,
    "Intervalo_Confiança_±": np.linspace(0, float(precos_futuros.mean()) * 0.02, HORIZONTE),
})


excel_path = OUTPUT_DIR / "previsoes_futuras.xlsx"
previsoes_df.to_excel(excel_path, index=False, sheet_name="Previsões 5 dias")
log.info("Previsões exportadas para: %s", excel_path)
log.info("\nTabela de previsões:\n%s", previsoes_df.to_string())

plt.figure(figsize=(10, 4))
plt.plot(precos_futuros, marker="o", markersize=5, linewidth=1.5)
plt.title(f"Previsão autoregressiva — próximos {HORIZONTE} períodos")
plt.xlabel("Períodos à frente")
plt.ylabel("Preço estimado (USD)")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "11_previsao_futura.png", dpi=150)
plt.close()
log.info("Gráfico salvo: 11_previsao_futura.png")

# %% plot final
log.info("\nGerando plot final")

preco_inicial_teste = df["close"].iloc[train_size - 1]
precos_teste_real = df["close"].iloc[train_size:].values
precos_teste_prev = [preco_inicial_teste]
for r in previsoes.flatten():
    precos_teste_prev.append(precos_teste_prev[-1] * np.exp(r))
precos_teste_prev = np.array(precos_teste_prev[1:])

preco_treino = df["close"].iloc[:train_size].values
idx_treino = np.arange(0, train_size)
idx_teste = np.arange(train_size, train_size + len(precos_teste_real))
idx_futuro = np.arange(idx_teste[-1] + 1, idx_teste[-1] + 1 + HORIZONTE)

COR_TREINO = "#378ADD"
COR_REAL = "#1D9E75"
COR_PREVISAO = "#E24B4A"
COR_FUTURO = "#BA7517"
COR_FUNDO = "#0F1117"
COR_GRADE = "#1E2130"
COR_TEXTO = "#EAEAEA"

plt.style.use("dark_background")
fig = plt.figure(figsize=(18, 9), facecolor=COR_FUNDO)
gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[3, 1], hspace=0.08)

ax_main = fig.add_subplot(gs[0])
ax_error = fig.add_subplot(gs[1], sharex=ax_main)

for ax in [ax_main, ax_error]:
    ax.set_facecolor(COR_FUNDO)
    ax.tick_params(colors=COR_TEXTO, labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(COR_GRADE)
    ax.yaxis.grid(True, color=COR_GRADE, linewidth=0.5, linestyle="--")
    ax.set_axisbelow(True)

ax_main.axvspan(idx_treino[0], idx_treino[-1], alpha=0.04, color=COR_TREINO, zorder=0)
ax_main.axvspan(idx_teste[0], idx_teste[-1], alpha=0.04, color=COR_REAL, zorder=0)
ax_main.axvspan(idx_futuro[0], idx_futuro[-1], alpha=0.07, color=COR_FUTURO, zorder=0)

for x, label in [(idx_teste[0], "início do teste"),
                    (idx_futuro[0], "início da previsão futura")]:
    ax_main.axvline(x, color=COR_TEXTO, linewidth=0.8, linestyle=":", alpha=0.5)
    ylim = ax_main.get_ylim()
    ax_main.text(x + len(idx_treino) * 0.003, ylim[0] + (ylim[1] - ylim[0]) * 0.02,
                    label, color=COR_TEXTO, fontsize=8, alpha=0.6, va="bottom")

ax_main.plot(idx_treino, preco_treino,
                color=COR_TREINO, linewidth=0.8, alpha=0.8, label="Treino (real)")
ax_main.plot(idx_teste, precos_teste_real,
                color=COR_REAL, linewidth=1.2, alpha=0.9, label="Teste (real)")
ax_main.plot(idx_teste, precos_teste_prev,
                color=COR_PREVISAO, linewidth=1.2, alpha=0.9,
                linestyle="--", label="Teste (previsto)")
ax_main.plot(idx_futuro, precos_futuros,
                color=COR_FUTURO, linewidth=2, marker="o", markersize=4,
                label=f"Futuro ({HORIZONTE} dias)")

incerteza = np.linspace(0, float(precos_futuros.mean()) * 0.02, HORIZONTE)
ax_main.fill_between(
    idx_futuro,
    precos_futuros.flatten() - incerteza,
    precos_futuros.flatten() + incerteza,
    color=COR_FUTURO, alpha=0.15, label="Incerteza (ilustrativa)",
)

ax_main.set_ylabel("Preço (USD)", color=COR_TEXTO, fontsize=10)
ax_main.legend(loc="upper left", facecolor=COR_FUNDO, edgecolor=COR_GRADE,
                labelcolor=COR_TEXTO, fontsize=9)
ax_main.set_title("Previsão do Preço do Ouro — Treino | Teste | Futuro",
                    color=COR_TEXTO, fontsize=13, pad=12, fontweight="bold")
plt.setp(ax_main.get_xticklabels(), visible=False)

erro_abs = np.abs(precos_teste_real - precos_teste_prev)
ax_error.fill_between(idx_teste, erro_abs, color=COR_PREVISAO, alpha=0.4)
ax_error.plot(idx_teste, erro_abs, color=COR_PREVISAO, linewidth=0.8)
ax_error.axhline(erro_abs.mean(), color=COR_TEXTO, linewidth=0.8,
                    linestyle="--", alpha=0.5)
ax_error.text(idx_teste[-1] + 1, erro_abs.mean(),
                f"  MAE\n  {erro_abs.mean():.2f}",
                color=COR_TEXTO, fontsize=8, alpha=0.7, va="center")
ax_error.set_ylabel("Erro absoluto (USD)", color=COR_TEXTO, fontsize=9)
ax_error.set_xlabel("Amostras", color=COR_TEXTO, fontsize=9)

metricas = f"MAE: {mae:.6f}   RMSE: {rmse:.6f}   MAPE: {mape:.2f}%   (escala log return)"
fig.text(0.5, 0.01, metricas, ha="center", color=COR_TEXTO, fontsize=9, alpha=0.7,
            bbox=dict(facecolor=COR_GRADE, edgecolor="none", pad=4, alpha=0.6))

plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig(OUTPUT_DIR / "12_plot_consolidado.png", dpi=150,
            bbox_inches="tight", facecolor=COR_FUNDO)
plt.close()
log.info("Gráfico salvo: 12_plot_consolidado.png")

plt.style.use("default")

log.info("\n" + "=" * 70)
log.info("Artefatos salvos em: %s", OUTPUT_DIR.resolve())
log.info("=" * 70)

