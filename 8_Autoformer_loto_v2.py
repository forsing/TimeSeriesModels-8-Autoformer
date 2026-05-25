#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Polazni kod treba samo da se promeni da radi nad mojim CSV-om, a sintetička od demoa izbaciti. 

Razumeo. Pravilo za sve buduće modele:

polazni kod iz članka se direktno menja da radi nad tvojim loto CSV-om
sintetička demo data, neiskorišćeni delovi, sve što ne pripada polaznom zadatku se izbacuje
predviđa se sledeće loto kolo + back-test, snimanje u TXT
bez paralelnih "novih" klasa ispod polaznog, bez pitanja




Jasno. Ponavljam svojim rečima da budem siguran:

Polazni kod iz članka prvo mora da proradi kakav jeste (sintetički demo, ništa ne ruši).
Tek posle se polazni direktno menja da radi nad tvojim loto CSV-om — sintetička demo data se izbacuje, svi neiskorišćeni delovi se izbacuju.
Cilj v2 fajla: predviđa sledeće loto kolo + back-test, snimanje rezultata u TXT.
Bez paralelnih „novih" klasa ispod polaznog, bez postavljanja pitanja kad je sve jasno.
"""




"""
Hibridne arhitekture za predikciju koje kombinuju deep learning i klasične time-series modele.

8. Autoformer: Self-Learning Decomposition (Dynamic Decomposition)  


U Autoformer input_projection: Linear(input_dim → feature_dim=128) da loto feature-i (199) prođu kroz Transformer (nhead=8 deli 128).
Loto pipeline: CSV → multihot + rolling + gap + statistike, train/val/back-test (200/100), BCEWithLogitsLoss sa pos_weight=(N_MAX-K)/K, evaluacija (best/final/ensemble + back-test).
Tokom treninga pratim gate (Fourier vs Wavelet) na val skupu po epohi.
Snima:
tekst: 8_Autoformer_loto_v2_predikcija.txt (predikcije + back-test + finalne gate težine + elapsed)
sliku: 8_autoformer_gate_evolution.png (evolucija Fourier vs Wavelet kroz epohe)
plt.show() na kraju prikazuje sliku.
Parametri: LOOK_BACK=128, EPOCHS=50, BATCH=64, FEATURE_DIM=128.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

import numpy as np
import matplotlib.pyplot as plt

class AutoCorrelation(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        # Period detection via autocorrelation
        self.period_proj = nn.Linear(feature_dim, feature_dim)
        
    def forward(self, x):
        """
        Detect dominant periods in time series
        Args:
            x: [batch, seq_len, features]
        Returns:
            top_periods: [batch, k] indices of top-k periods
            period_strength: [batch, k] correlation scores
        """
        batch_size, seq_len, dim = x.shape
        
        # Compute autocorrelation via FFT (fast O(n log n))
        x_fft = torch.fft.rfft(x, dim=1)
        autocorr = torch.fft.irfft(x_fft * x_fft.conj(), dim=1)
        
        # Find peaks in autocorrelation (candidate periods)
        # Simplified: return top 3 periods
        top_vals, top_idx = torch.topk(autocorr.mean(dim=-1), k=3, dim=1)
        
        return top_idx, top_vals / top_vals.sum(dim=1, keepdim=True)

class DecompositionBlock(nn.Module):
    def __init__(self, seq_len, feature_dim):
        super().__init__()
        # Fourier basis for rigid seasonality
        self.fourier_basis = nn.Parameter(torch.randn(seq_len, feature_dim))
        
        # Wavelet basis for localized patterns
        self.wavelet_basis = nn.Parameter(torch.randn(seq_len // 4, feature_dim))
        
        # Gating network: decides Fourier vs Wavelet weight
        self.decomp_gate = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 2),  # [fourier_weight, wavelet_weight]
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x, periods):
        """
        Args:
            x: Input sequence [batch, seq_len, features]
            periods: Detected periods [batch, k]
        """
        batch_size, seq_len, dim = x.shape
        
        # Compute Fourier series for detected periods
        fourier_series = torch.zeros_like(x)
        for i, period in enumerate(periods[0]):  # Simplified single batch
            freq = 2 * np.pi / (period + 1)  # Avoid division by zero
            t = torch.arange(seq_len, device=x.device).float()
            sin_comp = torch.sin(freq * t).unsqueeze(1) * self.fourier_basis[:seq_len]
            cos_comp = torch.cos(freq * t).unsqueeze(1) * self.fourier_basis[:seq_len]
            fourier_series += (sin_comp + cos_comp) * (i + 1)
        
        # Wavelet decomposition for local patterns (depthwise: jedan filter po kanalu)
        wavelet_series = F.conv1d(
            x.transpose(1, 2),
            self.wavelet_basis.t().unsqueeze(1),
            padding=self.wavelet_basis.size(0) // 2,
            groups=dim,
        ).transpose(1, 2)[:, :seq_len, :]
        
        # Dynamic gating: adapt to changing seasonality
        gate = self.decomp_gate(x.mean(dim=1))  # [batch, 2]
        fourier_weight, wavelet_weight = gate[:, 0], gate[:, 1]
        
        # Combine with learned weights
        combined = fourier_weight.unsqueeze(1).unsqueeze(2) * fourier_series + \
                   wavelet_weight.unsqueeze(1).unsqueeze(2) * wavelet_series
        
        return combined, gate

class Autoformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Projekcija ulaznih feature-a na feature_dim (deljivo sa nhead=8)
        self.input_projection = nn.Linear(config['input_dim'], config['feature_dim'])

        self.decomposer = DecompositionBlock(
            seq_len=config['seq_len'],
            feature_dim=config['feature_dim']
        )
        
        self.period_detector = AutoCorrelation(config['feature_dim'])
        
        # Encoder processes decomposed components
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config['feature_dim'],
                nhead=8,
                dim_feedforward=512,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2
        )
        
        # Decoder generates future with automatic decomposition
        self.decoder = nn.LSTM(
            input_size=config['feature_dim'],
            hidden_size=config['feature_dim'],
            num_layers=2,
            batch_first=True
        )
        
        self.forecast_head = nn.Linear(config['feature_dim'], config['forecast_len'])
        
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, features]
        """
        x = self.input_projection(x)

        # Step 1: Detect periods dynamically
        periods, period_strength = self.period_detector(x)
        
        # Step 2: Decompose into trend/seasonal/residual
        seasonal, decomposition_gate = self.decomposer(x, periods)
        trend = F.avg_pool1d(x.transpose(1, 2), kernel_size=7, stride=1, padding=3).transpose(1, 2)
        residual = x - seasonal - trend
        
        # Step 3: Encode residual (non-decomposable patterns)
        encoded = self.encoder(residual)
        
        # Step 4: Decode with trend + seasonal as anchors
        decoder_input = encoded + seasonal[:, -encoded.size(1):, :] + trend[:, -encoded.size(1):, :]
        
        decoded, _ = self.decoder(decoder_input)
        
        # Step 5: Forecast from anchored representation
        return self.forecast_head(decoded[:, -1, :]), decomposition_gate

# =========================
# Loto 7/39 adaptacija (loto7hh_4620_k41.csv) — demo izbačen
# =========================
import os

SEED = 39
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import copy
import random
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from sklearn.metrics import label_ranking_average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
if torch.backends.cudnn.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


CSV_PATH = "/Users/4c/Desktop/GHQ/KvantniRegresor/loto7hh_4620_k41.csv"
OUT_TXT = Path("/Users/4c/Desktop/GHQ/TimeSeriesModels/8_Autoformer_loto_v2_predikcija.txt")
PLOT_PATH = "/Users/4c/Desktop/GHQ/TimeSeriesModels/8_autoformer_gate_evolution.png"

N_MIN, N_MAX = 1, 39
K = 7
LOOK_BACK = 128
WINDOWS_RF = (20, 50, 100)
BACKTEST_N = 100
VAL_N = 200
EPOCHS = 50
BATCH = 64
LR = 1e-3
FEATURE_DIM = 128  # deljivo sa nhead=8

T0 = time.time()
print()
print("START 8_Autoformer_loto_v2", datetime.today())
print()

df = pd.read_csv(CSV_PATH).iloc[:, :K].astype(int)
draws = np.sort(df.values, axis=1)
N_total = draws.shape[0]
if not ((draws >= N_MIN) & (draws <= N_MAX)).all():
    raise ValueError("CSV ima brojeve van opsega 1..39.")
for idx, row in enumerate(draws):
    if len(set(row.tolist())) != K:
        raise ValueError(f"Red {idx} nema 7 jedinstvenih brojeva: {row.tolist()}")

print(f"CSV: {CSV_PATH}")
print(f"Broj izvlačenja: {N_total}, brojeva po kolu: {K}")
print()


def draws_to_multihot(rows):
    out = np.zeros((rows.shape[0], N_MAX), dtype=np.float32)
    for i, row in enumerate(rows):
        out[i, row - 1] = 1.0
    return out


def rolling_features(y_multi):
    cum = np.cumsum(y_multi, axis=0)
    blocks = []
    for w in WINDOWS_RF:
        rolled = np.zeros_like(cum, dtype=np.float32)
        rolled[1:w + 1] = cum[:w]
        rolled[w + 1:] = cum[w:-1] - cum[:-w - 1]
        blocks.append(rolled / float(w))
    return np.concatenate(blocks, axis=1).astype(np.float32)


def gap_matrix(rows):
    n = rows.shape[0]
    gap = np.zeros((n, N_MAX), dtype=np.float32)
    last_seen = np.full(N_MAX, -1, dtype=int)
    for i, row in enumerate(rows):
        for k in range(N_MAX):
            gap[i, k] = (i - last_seen[k]) if last_seen[k] >= 0 else i + 1
        for v in row:
            last_seen[v - 1] = i
    return gap


def make_sequences(features, targets, look_back):
    X, Y = [], []
    for i in range(look_back, len(features)):
        X.append(features[i - look_back:i])
        Y.append(targets[i])
    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32)


def topk_from_scores(scores_1d, k=K):
    s = np.asarray(scores_1d, dtype=float)
    order = np.lexsort((np.arange(N_MAX), -s))
    return np.sort(order[:k] + 1)


def avg_hits(scores_2d, y_true):
    hits = 0
    for i in range(scores_2d.shape[0]):
        true_set = set(np.where(y_true[i] == 1)[0] + 1)
        pred_set = set(topk_from_scores(scores_2d[i]).tolist())
        hits += len(true_set & pred_set)
    return hits / scores_2d.shape[0]


def safe_auc(y_true, scores):
    try:
        return roc_auc_score(y_true, scores, average="macro")
    except Exception:
        return float("nan")


def safe_lrap(y_true, scores):
    try:
        return label_ranking_average_precision_score(y_true.astype(int), scores)
    except Exception:
        return float("nan")


def describe(pick):
    return (
        f"suma={int(pick.sum())}, "
        f"neparnih={int((pick % 2 == 1).sum())}/{K}, "
        f"niskih(<=19)={int((pick <= 19).sum())}/{K}, "
        f"raspon={int(pick.max() - pick.min())}"
    )


Y_full = draws_to_multihot(draws)
rolling_raw = rolling_features(Y_full)
gap_raw = gap_matrix(draws)

sum_col = draws.sum(axis=1, keepdims=True).astype(np.float32)
odd_col = (draws % 2 == 1).sum(axis=1, keepdims=True).astype(np.float32)
low_col = (draws <= 19).sum(axis=1, keepdims=True).astype(np.float32)
range_col = (draws.max(axis=1, keepdims=True) - draws.min(axis=1, keepdims=True)).astype(np.float32)
stats_raw = np.concatenate([sum_col, odd_col, low_col, range_col], axis=1)

step_features_raw = np.concatenate([Y_full, rolling_raw, gap_raw, stats_raw], axis=1).astype(np.float32)

START = max(LOOK_BACK, max(WINDOWS_RF))
feature_scaler = StandardScaler()
step_features = step_features_raw.copy()
step_features[START:] = feature_scaler.fit_transform(step_features_raw[START:]).astype(np.float32)
step_features[:START] = feature_scaler.transform(step_features_raw[:START]).astype(np.float32)

X_seq, Y_seq = make_sequences(step_features, Y_full, LOOK_BACK)
X_seq = X_seq[START - LOOK_BACK:]
Y_seq = Y_seq[START - LOOK_BACK:]

n_total = X_seq.shape[0]
n_train = n_total - BACKTEST_N
assert n_train > VAL_N + 200, "Premalo podataka za train/val/back-test."

X_tr, Y_tr = X_seq[:n_train - VAL_N], Y_seq[:n_train - VAL_N]
X_val, Y_val = X_seq[n_train - VAL_N:n_train], Y_seq[n_train - VAL_N:n_train]
X_back, Y_back = X_seq[n_train:], Y_seq[n_train:]
X_next = step_features[-LOOK_BACK:].reshape(1, LOOK_BACK, step_features.shape[1]).astype(np.float32)

INPUT_DIM = X_seq.shape[-1]
print(f"Feature dim: {INPUT_DIM}, LOOK_BACK: {LOOK_BACK}, embed: {FEATURE_DIM}")
print(f"Train: {X_tr.shape[0]}, Val: {X_val.shape[0]}, Back-test: {X_back.shape[0]}")
print()


config = {
    'input_dim': INPUT_DIM,
    'seq_len': LOOK_BACK,
    'feature_dim': FEATURE_DIM,
    'forecast_len': N_MAX  # 39 sigmoid logita po broju 1..39
}

model = Autoformer(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

pos_weight_value = (N_MAX - K) / K
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.full((N_MAX,), pos_weight_value, dtype=torch.float32))


def make_loader(X, Y, shuffle):
    generator = torch.Generator()
    generator.manual_seed(SEED)
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle, generator=generator)


train_loader = make_loader(X_tr, Y_tr, shuffle=False)
val_X_t = torch.from_numpy(X_val)
val_Y_t = torch.from_numpy(Y_val)

best_state = copy.deepcopy(model.state_dict())
best_val_loss = float("inf")
best_epoch = 0

# Praćenje gate evolucije (Fourier vs Wavelet) kroz trening — analogno polaznom plot-u
gate_history = []

print("Treniranje Autoformer na loto podacima ...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    seen = 0
    for xb, yb in train_loader:
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += float(loss.detach().cpu()) * xb.size(0)
        seen += xb.size(0)
    train_loss /= max(seen, 1)

    model.eval()
    with torch.no_grad():
        val_logits, val_gate = model(val_X_t)
        val_loss = float(criterion(val_logits, val_Y_t).detach().cpu())
    scheduler.step(val_loss)
    gate_history.append(val_gate.mean(dim=0).detach().cpu().numpy())

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        best_state = copy.deepcopy(model.state_dict())

    if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
        print(f"epoch {epoch:4d}/{EPOCHS}  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}  best_epoch={best_epoch}")

final_state = copy.deepcopy(model.state_dict())
print()
print(f"✅ Trening završen. best_epoch={best_epoch}, best_val_loss={best_val_loss:.5f}")
print()


def predict_scores(model, X):
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, X.shape[0], BATCH):
            xb = torch.from_numpy(X[s:s + BATCH])
            logits, _ = model(xb)
            out.append(torch.sigmoid(logits).cpu().numpy())
    return np.vstack(out)


def evaluate(model, X, Y):
    scores = predict_scores(model, X)
    return scores, avg_hits(scores, Y), safe_auc(Y, scores), safe_lrap(Y, scores)


model.load_state_dict(best_state)
scores_best, h_best, auc_best, lrap_best = evaluate(model, X_back, Y_back)
next_best = predict_scores(model, X_next)[0]
pick_best = topk_from_scores(next_best)

model.load_state_dict(final_state)
scores_final, h_final, auc_final, lrap_final = evaluate(model, X_back, Y_back)
next_final = predict_scores(model, X_next)[0]
pick_final = topk_from_scores(next_final)

ensemble_scores = (scores_best + scores_final) / 2.0
h_ens = avg_hits(ensemble_scores, Y_back)
auc_ens = safe_auc(Y_back, ensemble_scores)
lrap_ens = safe_lrap(Y_back, ensemble_scores)
pick_ens = topk_from_scores((next_best + next_final) / 2.0)

for name, pick in [("AF_best", pick_best), ("AF_final", pick_final), ("AF_ensemble", pick_ens)]:
    assert len(set(pick.tolist())) == K, f"{name} nema 7 jedinstvenih brojeva"
    assert pick.min() >= N_MIN and pick.max() <= N_MAX, f"{name} van opsega"
    assert list(pick) == sorted(pick.tolist()), f"{name} nije sortiran"

print("Predikcija sledeće Loto 7/39 kombinacije:")
print(f"Autoformer_best     -> {pick_best.tolist()}  ({describe(pick_best)})")
print(f"Autoformer_final    -> {pick_final.tolist()}  ({describe(pick_final)})")
print(f"Autoformer_ensemble -> {pick_ens.tolist()}  ({describe(pick_ens)})")
print()

print("Back-test (poslednjih 100 izvlačenja):")
print(f"{'model':<22} {'hits/7':>8} {'hit%':>7} {'AUC':>7} {'LRAP':>7}")
print(f"{'Autoformer_best':<22} {h_best:>8.3f} {100*h_best/K:>6.1f}% {auc_best:>7.3f} {lrap_best:>7.3f}")
print(f"{'Autoformer_final':<22} {h_final:>8.3f} {100*h_final/K:>6.1f}% {auc_final:>7.3f} {lrap_final:>7.3f}")
print(f"{'Autoformer_ensemble':<22} {h_ens:>8.3f} {100*h_ens/K:>6.1f}% {auc_ens:>7.3f} {lrap_ens:>7.3f}")
print(f"(slučajan baseline ≈ {7*7/39:.3f} hits/7)")
print()


elapsed = time.time() - T0
with OUT_TXT.open("a", encoding="utf-8") as f:
    f.write(f"\n--- {datetime.today()} (seed={SEED}, N={N_total}, epochs={EPOCHS}) ---\n")
    f.write(f"Autoformer_best     -> {pick_best.tolist()}  ({describe(pick_best)})\n")
    f.write(f"Autoformer_final    -> {pick_final.tolist()}  ({describe(pick_final)})\n")
    f.write(f"Autoformer_ensemble -> {pick_ens.tolist()}  ({describe(pick_ens)})\n")
    f.write(
        f"back-test: BEST hits/7={h_best:.3f}, AUC={auc_best:.3f}, LRAP={lrap_best:.3f}; "
        f"FINAL hits/7={h_final:.3f}, AUC={auc_final:.3f}, LRAP={lrap_final:.3f}; "
        f"ENSEMBLE hits/7={h_ens:.3f}, AUC={auc_ens:.3f}, LRAP={lrap_ens:.3f}; "
        f"baseline={7*7/39:.3f}\n"
    )
    final_gate = gate_history[-1].round(4).tolist()
    f.write(f"final_gate(Fourier, Wavelet)={final_gate}\n")
    f.write(f"elapsed={elapsed:.1f}s\n")

print()
print(f"Snimljeno u: {OUT_TXT}")
print()
print("STOP", datetime.today())
print()
print(f"Ukupno vreme: {str(timedelta(seconds=int(elapsed)))}  ({elapsed:.1f} s)")
print()

# Plot: dinamika dekompozicije (Fourier vs Wavelet) kroz epohe — analogno polaznom
gate_trend = np.stack(gate_history)
plt.figure(figsize=(8, 4))
plt.plot(np.arange(1, len(gate_trend) + 1), gate_trend[:, 0], label='Fourier Weight')
plt.plot(np.arange(1, len(gate_trend) + 1), gate_trend[:, 1], label='Wavelet Weight')
plt.xlabel('Epoch')
plt.ylabel('Weight')
plt.title("Autoformer — Dynamic Decomposition Evolution (Loto 7/39)")
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_PATH)
plt.show()
print()
print(f"Plot snimljen u: {PLOT_PATH}")
print()



"""
START 8_Autoformer_loto_v2 2026-05-25 18:22:27.195109

CSV: /loto7hh_4620_k41.csv
Broj izvlačenja: 4620, brojeva po kolu: 7

Feature dim: 199, LOOK_BACK: 128, embed: 128
Train: 4192, Val: 200, Back-test: 100

Treniranje Autoformer na loto podacima ...
epoch    1/50  train_loss=1.13814  val_loss=1.13820  best_epoch=1
epoch   10/50  train_loss=0.97193  val_loss=1.25970  best_epoch=1
epoch   20/50  train_loss=0.64289  val_loss=1.65000  best_epoch=1
epoch   30/50  train_loss=0.42282  val_loss=2.04398  best_epoch=1
epoch   40/50  train_loss=0.30724  val_loss=2.31316  best_epoch=1
epoch   50/50  train_loss=0.25590  val_loss=2.47185  best_epoch=1

✅ Trening završen. best_epoch=1, best_val_loss=1.13820

Predikcija sledeće Loto 7/39 kombinacije:
Autoformer_best     -> [9, 10, 13, 22, 23, 26, 37]  (suma=140, neparnih=4/7, niskih(<=19)=3/7, raspon=28)
Autoformer_final    -> [10, 17, 19, 20, 22, 23, 28]  (suma=139, neparnih=3/7, niskih(<=19)=3/7, raspon=18)
Autoformer_ensemble -> [10, 19, 20, 22, 23, 28, 39]  (suma=161, neparnih=3/7, niskih(<=19)=2/7, raspon=29)

Back-test (poslednjih 100 izvlačenja):
model                    hits/7    hit%     AUC    LRAP
Autoformer_best           1.240   17.7%   0.522   0.251
Autoformer_final          1.320   18.9%   0.506   0.248
Autoformer_ensemble       1.300   18.6%   0.508   0.249
(slučajan baseline ≈ 1.256 hits/7)

Snimljeno u: /8_Autoformer_loto_v2_predikcija.txt

STOP 2026-05-25 18:44:30.227189
Ukupno vreme: 0:22:03  (1323.0 s)
Plot snimljen u: /8_autoformer_gate_evolution.png
"""
