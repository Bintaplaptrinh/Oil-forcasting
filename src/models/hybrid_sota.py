"""
hybrid_sota.py
--------------
Mo hinh SOTA: iTransformer (ICLR 2024) + Custom Dual MAE Loss.

Triet ly (theo GUMNet-WF):
- iTransformer dao nguoc chieu Attention: thay vi attend theo TIME-STEP
  (nhu vanilla Transformer), no attend theo FEATURE dimension.
  => Voi 48+ features (USD, GPR, WTI, lag...), mo hinh hoc duoc
     "bien nao anh huong den bien nao" thay vi "ngay nao quan trong".
  => Phu hop hon nhieu cho multivariate exogenous forecasting.

- Dual MAE Loss (cam hung GUMNet-WF):
  Loss = Huber(pred, true) + lambda * |pred - true|
  He so lambda tang ap luc tai cac spike - nen mo hinh khong "tron"
  ve trung binh khi gap bien dong dot ngot.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# ------------------------------------------------------------------ #
#  Custom Loss: Dual MAE (cam hung GUMNet-WF Dual MAE Penalty)
# ------------------------------------------------------------------ #
def dual_mae_loss(spike_lambda: float = 3.0):
    """
    Dual MAE Loss:  Huber(pred, true) + lambda * MAE(pred, true)

    Huber: on dinh voi outlier
    lambda * MAE: tang ap luc tai cac diem spike,
                  ep mo hinh khong bi 'hut' ve trung binh (Safe Mean Trap)

    Theo GUMNet-WF: lambda = 3.0 hieu qua nhat cho H=5 (ngay).
    """
    huber_fn = tf.keras.losses.Huber(delta=1.0, reduction="none")

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        h  = huber_fn(y_true, y_pred)
        m  = tf.abs(y_true - y_pred)
        return tf.reduce_mean(h + spike_lambda * m)

    loss.__name__ = f"dual_mae_lambda{spike_lambda}"
    return loss


# ------------------------------------------------------------------ #
#  iTransformer Architecture
# ------------------------------------------------------------------ #
def build_itransformer(time_steps: int,
                        n_features: int,
                        d_model: int  = 64,
                        n_heads: int  = 4,
                        n_layers: int = 3,
                        dropout: float = 0.1,
                        ff_mult: int  = 2) -> Model:
    """
    iTransformer: Inverted Attention cho Multivariate Time Series.
    Liu et al., ICLR 2024.

    Ky thuat then chot:
      Input  [B, T, F]
      Permute [B, F, T]   <- Dao nguoc chieu
      Dense  [B, F, d_model]  <- Moi feature = 1 token co embedding T
      N x [MultiHeadAttention(F,F) + FFN]  <- Attend giua cac features
      GlobalAvgPool -> Dense -> output
    """
    inp = layers.Input(shape=(time_steps, n_features), name="input")

    # === INVERTED PROJECTION ===
    x = layers.Permute((2, 1), name="invert_dims")(inp)       # [B, F, T]
    x = layers.Dense(d_model, name="embed_proj")(x)            # [B, F, d_model]
    x = layers.LayerNormalization(epsilon=1e-6, name="embed_norm")(x)
    x = layers.Dropout(dropout)(x)

    # === STACKED INVERTED TRANSFORMER BLOCKS ===
    for i in range(n_layers):
        # Self-Attention tren FEATURE dimension
        attn = layers.MultiHeadAttention(
            num_heads=n_heads,
            key_dim=d_model // n_heads,
            dropout=dropout,
            name=f"mha_{i}",
        )(x, x)
        x = layers.Add(name=f"add1_{i}")([x, attn])
        x = layers.LayerNormalization(epsilon=1e-6, name=f"ln1_{i}")(x)

        # Position-wise Feed-Forward
        ff = layers.Dense(d_model * ff_mult, activation="gelu",
                          name=f"ff1_{i}")(x)
        ff = layers.Dropout(dropout, name=f"ffdr_{i}")(ff)
        ff = layers.Dense(d_model, name=f"ff2_{i}")(ff)
        x  = layers.Add(name=f"add2_{i}")([x, ff])
        x  = layers.LayerNormalization(epsilon=1e-6, name=f"ln2_{i}")(x)

    # === OUTPUT HEAD ===
    x   = layers.GlobalAveragePooling1D(name="feature_pool")(x)
    x   = layers.Dense(d_model // 2, activation="gelu", name="head_dense")(x)
    x   = layers.Dropout(dropout, name="head_drop")(x)
    out = layers.Dense(1, name="output")(x)

    return Model(inp, out, name="iTransformer")


# ------------------------------------------------------------------ #
#  Training
# ------------------------------------------------------------------ #
def train_itransformer(X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray,   y_val: np.ndarray,
                        time_steps: int,     n_features: int,
                        horizon: int = 1,
                        epochs: int = 150,
                        batch_size: int = 64,
                        lr: float = 5e-4,
                        spike_lambda: float = 3.0,
                        d_model: int = 64,
                        n_heads: int = 4,
                        n_layers: int = 3,
                        seed: int = 42):
    """
    Build + Compile + Train iTransformer.
    spike_lambda tang len khi horizon lon hon (phan ra tin hieu).
    AdamW (weight_decay=1e-4) theo GUMNet-WF hyperparameter config.
    """
    tf.random.set_seed(seed)
    tf.keras.backend.clear_session()

    model = build_itransformer(
        time_steps, n_features,
        d_model=d_model, n_heads=n_heads, n_layers=n_layers
    )
    model.compile(
        optimizer=AdamW(learning_rate=lr, weight_decay=1e-4, clipnorm=1.0),
        loss=dual_mae_loss(spike_lambda),
        metrics=["mae"],
    )

    callbacks = [
        EarlyStopping(patience=20, restore_best_weights=True,
                      monitor="val_loss", verbose=1),
        # ReduceLROnPlateau: Do day (theo GUMNet-WF) - giam LR khi val_loss
        # di ngang, giup hoi tu den cuc tieu toan cuc
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=5, min_lr=1e-6, verbose=1),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    return model, history

# ------------------------------------------------------------------ #
#  GUMNet-Lite: Gated CNN-BiGRU (Mixture of Experts)
# ------------------------------------------------------------------ #
def build_gumnet_lite(time_steps: int, n_features: int, 
                      cnn_filters: int = 64, gru_units: int = 64, 
                      dropout: float = 0.2) -> Model:
    """
    Mo phong kien truc GUMNet-WF (Gated CNN-GRU).
    Gom 2 experts:
      - Expert 1 (CNN): Trich xuat local features.
      - Expert 2 (BiGRU): Ghi nho long-term dependencies.
    Gating Network: Xac dinh trong so dong de mix 2 experts.
    """
    inp = layers.Input(shape=(time_steps, n_features), name="input")
    
    # --- Expert 1: CNN 1D ---
    x_cnn = layers.Conv1D(filters=cnn_filters, kernel_size=3, padding="same", activation="relu", name="expert1_cnn")(inp)
    x_cnn = layers.BatchNormalization()(x_cnn)
    x_cnn = layers.GlobalAveragePooling1D(name="expert1_pool")(x_cnn)
    expert_1 = layers.Dense(32, activation="relu", name="expert1_out")(x_cnn)
    
    # --- Expert 2: BiGRU ---
    x_gru = layers.Bidirectional(layers.GRU(gru_units, return_sequences=False), name="expert2_bigru")(inp)
    x_gru = layers.BatchNormalization()(x_gru)
    expert_2 = layers.Dense(32, activation="relu", name="expert2_out")(x_gru)
    
    # --- Gating Network ---
    gate_context = layers.Concatenate(name="gate_context")([x_cnn, x_gru])
    gate_weights = layers.Dense(2, activation="softmax", name="gating_weights")(gate_context)
    
    # --- Mix Experts ---
    w1 = layers.Lambda(lambda x: tf.expand_dims(x[:, 0], axis=-1))(gate_weights)
    w2 = layers.Lambda(lambda x: tf.expand_dims(x[:, 1], axis=-1))(gate_weights)
    
    mixed_out = layers.Add(name="mixture_of_experts")([
        layers.Multiply()([expert_1, w1]),
        layers.Multiply()([expert_2, w2])
    ])
    
    mixed_out = layers.Dropout(dropout)(mixed_out)
    out = layers.Dense(1, name="output")(mixed_out)
    
    return Model(inp, out, name="GUMNet_Lite")

def train_gumnet_lite(X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray,   y_val: np.ndarray,
                      time_steps: int,     n_features: int,
                      horizon: int = 1,
                      epochs: int = 150, batch_size: int = 64,
                      lr: float = 1e-3, spike_lambda: float = 3.0,
                      seed: int = 42):
    tf.random.set_seed(seed)
    tf.keras.backend.clear_session()
    
    model = build_gumnet_lite(time_steps, n_features)
    model.compile(
        optimizer=AdamW(learning_rate=lr, weight_decay=1e-4),
        loss=dual_mae_loss(spike_lambda),
        metrics=["mae"],
    )
    callbacks = [
        EarlyStopping(patience=20, restore_best_weights=True, monitor="val_loss", verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs, batch_size=batch_size,
        callbacks=callbacks, verbose=1,
    )
    return model, history

# ------------------------------------------------------------------ #
#  Optuna Tuning cho iTransformer
# ------------------------------------------------------------------ #
def tune_itransformer(X_train, y_train, X_val, y_val, time_steps, n_features, n_trials=30, seed=42):
    import optuna
    from sklearn.metrics import mean_squared_error
    import gc
    import os
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    def objective(trial):
        d_model = trial.suggest_categorical("d_model", [32, 64, 128])
        n_layers = trial.suggest_int("n_layers", 1, 4)
        n_heads = trial.suggest_categorical("n_heads", [2, 4, 8])
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        dropout = trial.suggest_float("dropout", 0.05, 0.3)
        
        if d_model % n_heads != 0:
            raise optuna.exceptions.TrialPruned()
            
        tf.random.set_seed(seed)
        tf.keras.backend.clear_session()
        
        model = build_itransformer(time_steps, n_features, d_model=d_model, n_heads=n_heads, n_layers=n_layers, dropout=dropout)
        model.compile(optimizer=AdamW(learning_rate=lr), loss="mse")
        
        callbacks = [EarlyStopping(patience=3, restore_best_weights=True, monitor="val_loss")]
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=15, batch_size=64, callbacks=callbacks, verbose=0)
        
        pred_val = model.predict(X_val, verbose=0)
        val_mse = mean_squared_error(y_val, pred_val)
        
        del model
        gc.collect()
        tf.keras.backend.clear_session()
        
        return val_mse

    print(f"  -> Dang chay {n_trials} trials Optuna cho iTransformer...")
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"  -> Best params iTransformer: {study.best_params}")
    return {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "study": study
    }

# ------------------------------------------------------------------ #
#  GUMNet-Ultra: Non-Linear Spline Attention (Mimic KAN)
# ------------------------------------------------------------------ #
def build_gumnet_ultra(time_steps: int, n_features: int, 
                       cnn_filters: int = 64, gru_units: int = 64, 
                       dropout: float = 0.2) -> Model:
    """
    GUMNet-Ultra: Nang cap tu Lite, su dung mang tinh toan phien ban phuc tap
    de mo phong KAN (Kolmogorov-Arnold Network) trong viec Gating.
    """
    inp = layers.Input(shape=(time_steps, n_features), name="input")
    
    # --- Expert 1: CNN 1D ---
    x_cnn = layers.Conv1D(filters=cnn_filters, kernel_size=3, padding="same", activation="swish", name="expert1_cnn")(inp)
    x_cnn = layers.BatchNormalization()(x_cnn)
    x_cnn = layers.GlobalAveragePooling1D(name="expert1_pool")(x_cnn)
    expert_1 = layers.Dense(32, activation="swish", name="expert1_out")(x_cnn)
    
    # --- Expert 2: BiGRU ---
    x_gru = layers.Bidirectional(layers.GRU(gru_units, return_sequences=False), name="expert2_bigru")(inp)
    x_gru = layers.BatchNormalization()(x_gru)
    expert_2 = layers.Dense(32, activation="swish", name="expert2_out")(x_gru)
    
    # --- Gating Network (Ultra/KAN-like Approximation) ---
    # Thay vi Dense tuyen tinh + Softmax, ta dung Swish + nhieu layers
    # de mo phong spline phi tuyen cua KAN.
    gate_context = layers.Concatenate(name="gate_context")([x_cnn, x_gru])
    g1 = layers.Dense(32, activation="swish")(gate_context)
    g2 = layers.Dense(16, activation="swish")(g1)
    gate_weights = layers.Dense(2, activation="softmax", name="gating_weights")(g2)
    
    # --- Mix Experts ---
    w1 = layers.Lambda(lambda x: tf.expand_dims(x[:, 0], axis=-1))(gate_weights)
    w2 = layers.Lambda(lambda x: tf.expand_dims(x[:, 1], axis=-1))(gate_weights)
    
    mixed_out = layers.Add(name="mixture_of_experts")([
        layers.Multiply()([expert_1, w1]),
        layers.Multiply()([expert_2, w2])
    ])
    
    mixed_out = layers.Dropout(dropout)(mixed_out)
    out = layers.Dense(1, name="output")(mixed_out)
    
    return Model(inp, out, name="GUMNet_Ultra")

def train_gumnet_ultra(X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray,   y_val: np.ndarray,
                       time_steps: int,     n_features: int,
                       horizon: int = 1,
                       epochs: int = 150, batch_size: int = 64,
                       cnn_filters: int = 64, gru_units: int = 64,
                       dropout: float = 0.2, lr: float = 1e-3, 
                       spike_lambda: float = 3.0, seed: int = 42):
    tf.random.set_seed(seed)
    tf.keras.backend.clear_session()
    
    model = build_gumnet_ultra(time_steps, n_features, cnn_filters, gru_units, dropout)
    model.compile(
        optimizer=AdamW(learning_rate=lr, weight_decay=1e-4),
        loss=dual_mae_loss(spike_lambda),
        metrics=["mae"],
    )
    callbacks = [
        EarlyStopping(patience=20, restore_best_weights=True, monitor="val_loss", verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    ]
    history = model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=1,
    )
    return model, history

def tune_gumnet_ultra(X_train, y_train, X_val, y_val, time_steps, n_features, n_trials=30, seed=42):
    import optuna
    from sklearn.metrics import mean_squared_error
    import gc
    import os
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    def objective(trial):
        cnn_filters = trial.suggest_categorical("cnn_filters", [32, 64, 128])
        gru_units   = trial.suggest_categorical("gru_units", [32, 64, 128])
        lr          = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        dropout     = trial.suggest_float("dropout", 0.05, 0.4)
        
        tf.random.set_seed(seed)
        tf.keras.backend.clear_session()
        
        model = build_gumnet_ultra(time_steps, n_features, cnn_filters, gru_units, dropout)
        model.compile(optimizer=AdamW(learning_rate=lr), loss="mse")
        
        callbacks = [EarlyStopping(patience=3, restore_best_weights=True, monitor="val_loss")]
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=15, batch_size=64, callbacks=callbacks, verbose=0)
        
        pred_val = model.predict(X_val, verbose=0)
        val_mse = mean_squared_error(y_val, pred_val)
        
        del model
        gc.collect()
        tf.keras.backend.clear_session()
        return val_mse

    print(f"  -> Dang chay {n_trials} trials Optuna cho GUMNet-Ultra...")
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"  -> Best params GUMNet-Ultra: {study.best_params}")
    return {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "study": study
    }


