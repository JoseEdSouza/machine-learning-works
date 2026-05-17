# %%
import logging
import os
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import matplotlib.gridspec as gridspec

# %%
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# %% 
GOLD_DATA_PATH = Path(
    os.environ.get("GOLD_DATA_PATH", "../gold/data/final_gold_data.csv")
)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# %%
log.info("Carregando dados de: %s", GOLD_DATA_PATH)
df = pd.read_csv(GOLD_DATA_PATH, sep=";", encoding="utf-8", parse_dates=["timestamp"])

log.info("Shape: %s", df.shape)
log.info("Colunas: %s", df.columns.tolist())
log.info("\n%s", df.describe())

datas = pd.to_datetime(df["timestamp"])

# %%
plt.figure(figsize=(10, 6))
plt.plot(datas, df["open"], label="Open Price")
plt.legend()
plt.xlabel("Timestamp")
plt.ylabel("Open Price")
plt.grid(True)
plt.title("Gold Price Over Time")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_gold_price.png", dpi=150)
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(datas, df["volume"], label="Volume", color="orange")
plt.legend()
plt.xlabel("Timestamp")
plt.ylabel("Volume")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_volume.png", dpi=150)
plt.show()

df_numerico = df.select_dtypes(include=["int64", "float64"])
plt.figure(figsize=(8, 6))
sns.heatmap(df_numerico.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Heatmap de correlações entre variáveis numéricas")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_heatmap.png", dpi=150)
plt.show()

# %%
WINDOW_SIZE = 60        
TRAIN_RATIO = 0.95     
VAL_SPLIT   = 0.1     

fechamento_dados = df["close"].values
train_size = int(np.ceil(len(fechamento_dados) * TRAIN_RATIO))

log.info("Total de amostras: %d", len(fechamento_dados))
log.info("Treino: %d | Teste: %d", train_size, len(fechamento_dados) - train_size)

scaler = StandardScaler()
train_raw = fechamento_dados[:train_size].reshape(-1, 1)
test_raw  = fechamento_dados[train_size:].reshape(-1, 1)

train_scaled = scaler.fit_transform(train_raw)
full_scaled = np.concatenate([train_scaled, scaler.transform(test_raw)], axis=0)

joblib.dump(scaler, OUTPUT_DIR / "scaler.pkl")
log.info("Scaler salvo em outputs/scaler.pkl")

# %%
X_train, y_train = [], []
for i in range(WINDOW_SIZE, train_size):
    X_train.append(full_scaled[i - WINDOW_SIZE:i, 0])
    y_train.append(full_scaled[i, 0])

X_train = np.array(X_train).reshape(-1, WINDOW_SIZE, 1)
y_train = np.array(y_train)

# Janela começa nos últimos WINDOW_SIZE pontos do treino — correto e sem leakage
X_test, y_test = [], []
for i in range(train_size, len(full_scaled)):
    X_test.append(full_scaled[i - WINDOW_SIZE:i, 0])
    y_test.append(full_scaled[i, 0])

X_test = np.array(X_test).reshape(-1, WINDOW_SIZE, 1)
y_test = np.array(y_test)

log.info("X_train: %s | X_test: %s", X_train.shape, X_test.shape)

# %%
model = keras.models.Sequential([
    keras.layers.LSTM(64, return_sequences=True, input_shape=(WINDOW_SIZE, 1)),
    keras.layers.LSTM(64, return_sequences=False),
    keras.layers.Dense(32, activation="relu"),  
    keras.layers.Dropout(0.3),
    keras.layers.Dense(1),
])

model.summary()

model.compile(
    optimizer="adam",
    loss="mae",
    metrics=[keras.metrics.RootMeanSquaredError()],
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1,
)

log.info("Iniciando treinamento...")
history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_split=VAL_SPLIT,
    callbacks=[early_stop],
    verbose=1,
)


model.save(OUTPUT_DIR / "lstm_gold.keras")
log.info("Modelo salvo em outputs/lstm_gold.keras")

# %%
previsoes_norm = model.predict(X_test)
previsoes      = scaler.inverse_transform(previsoes_norm)
y_test_real    = scaler.inverse_transform(y_test.reshape(-1, 1))

mae  = mean_absolute_error(y_test_real, previsoes)
mse  = mean_squared_error(y_test_real, previsoes)
rmse = math.sqrt(mse)
mape = np.mean(np.abs((y_test_real - previsoes) / y_test_real)) * 100

log.info("MAE  (Erro Absoluto Médio):%.2f", mae)
log.info("MSE  (Erro Quadrático Médio):%.2f", mse)
log.info("RMSE (Raiz do MSE):%.2f", rmse)
log.info("MAPE (Erro Percentual Médio Abs.):%.2f%%", mape)

# %%
plt.figure(figsize=(15, 6))
plt.plot(y_test_real,color="blue",label="Valores Reais",alpha=0.7)
plt.plot(previsoes,color="red",label="Previsões",alpha=0.7)
plt.title("Previsão do Preço do Ouro — Dados de Teste")
plt.xlabel("Tempo")
plt.ylabel("Preço de Fechamento")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_previsao_vs_real.png", dpi=150)
plt.show()

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["loss"],     label="Treino")
plt.plot(history.history["val_loss"], label="Validação")
plt.title("Loss (MAE)")
plt.xlabel("Época")
plt.ylabel("MAE")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history["root_mean_squared_error"],     label="Treino",    color="orange")
plt.plot(history.history["val_root_mean_squared_error"], label="Validação", color="red")
plt.title("RMSE")
plt.xlabel("Época")
plt.ylabel("RMSE")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_learning_curves.png", dpi=150)
plt.show()

plt.figure(figsize=(15, 6))
plt.plot(y_test_real[-100:], label="Real",    marker="o", markersize=3)
plt.plot(previsoes[-100:],   label="Previsto", marker="x", markersize=3)
plt.title("Comparação Previsão vs Real (Últimos 100 pontos)")
plt.xlabel("Amostra")
plt.ylabel("Preço")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "06_ultimos_100.png", dpi=150)
plt.show()

# %%

DIAS_FUTURO = 10
def previsao_futuro(dias: int) -> np.ndarray:
    """
    Gera previsões autoregressivas para os próximos `dias` períodos.
 
    Atenção: erro acumula a cada iteração. Horizontes > 5-10 períodos
    devem ser interpretados com cautela.
    """
    janela = full_scaled[-WINDOW_SIZE:].copy()
    preds  = []
 
    for _ in range(dias):
        entrada = janela.reshape(1, WINDOW_SIZE, 1)
        proximo = model.predict(entrada, verbose=0)
        preds.append(float(proximo[0, 0]))
        janela  = np.append(janela[1:], proximo[0, 0]).reshape(-1, 1)
 
    return scaler.inverse_transform(np.array(preds).reshape(-1, 1))
 
 
log.info("Gerando previsao para os proximos %d periodos...", DIAS_FUTURO)
previsoes_futuras = previsao_futuro(DIAS_FUTURO)
log.info("Previsoes futuras:\n%s", previsoes_futuras)

HORIZONTE = 5
log.info("Gerando previsão para os próximos %d períodos...", HORIZONTE)
previsoes = previsao_futuro(HORIZONTE)

plt.figure(figsize=(10, 5))
plt.plot(previsoes, marker="o", markersize=4, label="Previsão futura")
plt.title(f"Previsão autoregressiva — próximos {HORIZONTE} períodos")
plt.xlabel("Períodos à frente")
plt.ylabel("Preço previsto")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "07_previsao_futura.png", dpi=150)
plt.show()

# %%
log.info("plotando treino, teste e previsões futuras ")

COR_TREINO= "#4A90D9"
COR_REAL= "#2ECC71"
COR_PREVISAO = "#E74C3C"
COR_FUTURO= "#F39C12"  
COR_FUNDO= "#0F1117"
COR_GRADE= "#1E2130"
COR_TEXTO= "#EAEAEA"
log.info("Gerando plot consolidado...")

y_train_real = scaler.inverse_transform(
    full_scaled[WINDOW_SIZE:train_size]
)
 
idx_treino = np.arange(0, len(y_train_real))
idx_teste  = np.arange(len(y_train_real), len(y_train_real) + len(y_test_real))
idx_futuro = np.arange(idx_teste[-1] + 1, idx_teste[-1] + 1 + DIAS_FUTURO)
 
plt.style.use("dark_background")
fig = plt.figure(figsize=(18, 9), facecolor=COR_FUNDO)
gs  = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[3, 1], hspace=0.08)
 
ax_main  = fig.add_subplot(gs[0])
ax_error = fig.add_subplot(gs[1], sharex=ax_main)
 
for ax in [ax_main, ax_error]:
    ax.set_facecolor(COR_FUNDO)
    ax.tick_params(colors=COR_TEXTO, labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(COR_GRADE)
    ax.yaxis.grid(True, color=COR_GRADE, linewidth=0.5, linestyle="--")
    ax.set_axisbelow(True)
 
ax_main.axvspan(idx_treino[0], idx_treino[-1], alpha=0.04, color=COR_TREINO, zorder=0)
ax_main.axvspan(idx_teste[0],  idx_teste[-1],  alpha=0.04, color=COR_REAL,   zorder=0)
ax_main.axvspan(idx_futuro[0], idx_futuro[-1], alpha=0.07, color=COR_FUTURO, zorder=0)
 
for x, label in [(idx_teste[0], "inicio do teste"),
                 (idx_futuro[0], "inicio da previsao futura")]:
    ax_main.axvline(x, color=COR_TEXTO, linewidth=0.8, linestyle=":", alpha=0.5)
    ylim = ax_main.get_ylim()
    ax_main.text(x + 2, ylim[0] + (ylim[1] - ylim[0]) * 0.02,
                 label, color=COR_TEXTO, fontsize=8, alpha=0.6, va="bottom")
 
ax_main.plot(idx_treino, y_train_real,
             color=COR_TREINO, linewidth=1.2, alpha=0.8, label="Treino (real)")
ax_main.plot(idx_teste, y_test_real,
             color=COR_REAL, linewidth=1.4, alpha=0.9, label="Teste (real)")
ax_main.plot(idx_teste, previsoes,
             color=COR_PREVISAO, linewidth=1.4, alpha=0.9,
             linestyle="--", label="Teste (previsto)")
ax_main.plot(idx_futuro, previsoes_futuras,
             color=COR_FUTURO, linewidth=2, label=f"Futuro ({DIAS_FUTURO} dias)")
 
incerteza = np.linspace(0, float(previsoes_futuras.mean()) * 0.05,
                        DIAS_FUTURO).reshape(-1, 1)
ax_main.fill_between(
    idx_futuro,
    (previsoes_futuras - incerteza).flatten(),
    (previsoes_futuras + incerteza).flatten(),
    color=COR_FUTURO, alpha=0.15, label="Intervalo de incerteza (ilustrativo)",
)
 
ax_main.set_ylabel("Preco de Fechamento (USD)", color=COR_TEXTO, fontsize=10)
ax_main.legend(loc="upper left", facecolor=COR_FUNDO, edgecolor=COR_GRADE,
               labelcolor=COR_TEXTO, fontsize=9)
ax_main.set_title("Previsao do Preco do Ouro — Treino · Teste · Futuro",
                  color=COR_TEXTO, fontsize=13, pad=12, fontweight="bold")
plt.setp(ax_main.get_xticklabels(), visible=False)
 
erro_absoluto = np.abs(y_test_real.flatten() - previsoes.flatten())
ax_error.fill_between(idx_teste, erro_absoluto, color=COR_PREVISAO, alpha=0.4)
ax_error.plot(idx_teste, erro_absoluto, color=COR_PREVISAO, linewidth=0.8)
ax_error.axhline(erro_absoluto.mean(), color=COR_TEXTO, linewidth=0.8,
                 linestyle="--", alpha=0.5)
ax_error.text(idx_teste[-1] + 1, erro_absoluto.mean(),
              f"  MAE medio\n  {erro_absoluto.mean():.2f}",
              color=COR_TEXTO, fontsize=8, alpha=0.7, va="center")
ax_error.set_ylabel("Erro absoluto", color=COR_TEXTO, fontsize=9)
ax_error.set_xlabel("Amostras (indice)", color=COR_TEXTO, fontsize=9)
 
metricas = f"MAE: {mae:.2f}   RMSE: {rmse:.2f}   MAPE: {mape:.2f}%"
fig.text(0.5, 0.01, metricas, ha="center", color=COR_TEXTO, fontsize=9, alpha=0.7,
         bbox=dict(facecolor=COR_GRADE, edgecolor="none", pad=4, alpha=0.6))
 
plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig(OUTPUT_DIR / "07_plot_consolidado.png", dpi=150,
            bbox_inches="tight", facecolor=COR_FUNDO)
plt.show()
log.info("Plot consolidado salvo em outputs/07_plot_consolidado.png")