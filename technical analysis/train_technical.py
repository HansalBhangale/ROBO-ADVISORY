"""
==============================================================================
S&P 500 Multi-Horizon Stock Predictor — Training Pipeline
==============================================================================
Retrained on the same US stock universe used by the fundamental analysis module.
Architecture: LSTM(256→128) + Multi-Head Attention (8 heads) + Ensemble of 5 models.

Usage:
    python train_technical.py                   # Train with default settings
    python train_technical.py --max_stocks 50   # Limit to fewer stocks (faster)
    python train_technical.py --epochs 30       # Custom epoch count

Output Files:
    scaler_v2.pkl                    — StandardScaler for feature normalization
    metrics_v2.pkl                   — Complete evaluation metrics
    ensemble_models/model_*.keras    — 5 trained ensemble models
    sp500_ensemble_info.pkl          — Ensemble weights and metadata
==============================================================================
"""

import os
import sys
import argparse
import warnings
import json
import time
import numpy as np
import pandas as pd
import joblib
import yfinance as yf
import tensorflow as tf
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout,
    BatchNormalization, MultiHeadAttention,
    LayerNormalization, Add
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from datetime import datetime

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    'SEQ_LENGTH': 60,
    'MAX_HORIZON': 360,
    'TRAIN_YEARS': 15,           # 15 years of US stock history
    'BATCH_SIZE': 1024,
    'EPOCHS': 50,
    'N_ENSEMBLE': 5,
    'ENABLE_ENSEMBLE': True,
    'ENABLE_MARKET_FEATURES': True,
    'ENABLE_CUSTOM_LOSS': True,

    'MODEL_PREFIX': 'sp500_integrated',
    'SCALER_NAME': os.path.join(BASE_DIR, 'scaler_v2.pkl'),
    'METRICS_NAME': os.path.join(BASE_DIR, 'metrics_v2.pkl'),
    'ENSEMBLE_DIR': os.path.join(BASE_DIR, 'ensemble_models'),
    'ENSEMBLE_INFO': os.path.join(BASE_DIR, 'sp500_ensemble_info.pkl'),

    # S&P 500 stock universe — same blue-chip focus as fundamental analysis
    'TICKERS': [
        'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA', 'BRK-B',
        'UNH', 'JNJ', 'JPM', 'V', 'PG', 'MA', 'HD', 'ABBV',
        'MRK', 'PEP', 'KO', 'COST', 'AVGO', 'LLY', 'WMT', 'BAC',
        'PFE', 'TMO', 'CSCO', 'ACN', 'MCD', 'ABT', 'DHR', 'TXN',
        'NEE', 'UPS', 'PM', 'HON', 'UNP', 'LOW', 'INTC', 'QCOM',
        'AMAT', 'IBM', 'GE', 'CAT', 'BA', 'GS', 'BLK', 'ISRG',
        'ADP', 'MDLZ', 'MMM', 'CVX', 'XOM', 'COP', 'DE', 'SBUX',
        'GILD', 'MMC', 'SYK', 'ADI', 'CI', 'CB', 'SO', 'DUK',
        'CME', 'CL', 'APD', 'REGN', 'UBER', 'CRWD', 'PANW', 'CRM',
        'NOW', 'ADBE', 'ORCL', 'NFLX', 'AMD', 'PYPL', 'SQ', 'SHOP',
    ],

    # Market index for context features
    'MARKET_INDEX': '^GSPC',    # S&P 500 index
    'MARKET_NAME': 'S&P 500',
}


# ==============================================================================
# CUSTOM LOSS FUNCTION
# ==============================================================================
class DirectionAwareLoss(tf.keras.losses.Loss):
    """Penalize wrong direction predictions more."""
    def __init__(self, direction_weight=0.3, name='direction_aware_loss', reduction='sum_over_batch_size'):
        super().__init__(name=name, reduction=reduction)
        self.direction_weight = direction_weight

    def call(self, y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        direction_penalty = tf.reduce_mean(
            tf.cast(tf.sign(y_true) != tf.sign(y_pred), tf.float32)
        )
        return (1 - self.direction_weight) * mse + self.direction_weight * direction_penalty * 100

    def get_config(self):
        config = super().get_config()
        config.update({'direction_weight': self.direction_weight})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# ==============================================================================
# ENHANCED TECHNICAL INDICATORS
# ==============================================================================
class EnhancedIndicators:
    @staticmethod
    def get_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def get_adx(high, low, close, period=14):
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm = plus_dm.where(plus_dm > 0, 0)
        minus_dm = minus_dm.where(minus_dm < 0, 0).abs()
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)) * 100
        return dx.ewm(alpha=1/period, adjust=False).mean()

    @staticmethod
    def get_natr(high, low, close, period=14):
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        return (atr / close) * 100

    @staticmethod
    def get_obv_slope(close, volume, period=14):
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        return obv.diff(period)

    @staticmethod
    def get_dist_sma(close, period=50):
        sma = close.rolling(period).mean()
        return (close - sma) / sma

    @staticmethod
    def get_macd(close, fast=12, slow=26, signal=9):
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        return macd - macd_signal

    @staticmethod
    def get_roc(close, period=10):
        return ((close - close.shift(period)) / close.shift(period)) * 100

    @staticmethod
    def get_volume_ratio(volume, period=20):
        vol_ma = volume.rolling(period).mean()
        return volume / vol_ma

    @staticmethod
    def get_bollinger_position(close, period=20, std=2):
        sma = close.rolling(period).mean()
        rolling_std = close.rolling(period).std()
        upper = sma + (rolling_std * std)
        lower = sma - (rolling_std * std)
        band_width = upper - lower
        band_width = band_width.replace(0, np.nan)
        return (close - lower) / band_width


# ==============================================================================
# MARKET CONTEXT FEATURES
# ==============================================================================
class MarketContextFeatures:
    @staticmethod
    def calculate_market_features(market_close):
        features = pd.DataFrame(index=market_close.index)
        for period in [20, 50, 200]:
            ma = market_close.rolling(period).mean()
            features[f'above_ma_{period}'] = (market_close > ma).astype(float)
        features['market_strength'] = features[[f'above_ma_{p}' for p in [20, 50, 200]]].mean(axis=1)
        features['market_momentum'] = market_close.pct_change(20) * 100
        returns = market_close.pct_change()
        features['market_volatility'] = returns.rolling(20).std() * np.sqrt(252) * 100
        rolling_std = returns.rolling(20).std()
        features['market_trend'] = returns.rolling(20).mean() / (rolling_std + 1e-8)
        return features[['market_strength', 'market_momentum', 'market_volatility', 'market_trend']]


# ==============================================================================
# DATA PREPARATION
# ==============================================================================
def prepare_stock_features(ticker, market_df):
    """Prepare all 15 features for a single stock."""
    try:
        df = yf.download(ticker, period=f"{CONFIG['TRAIN_YEARS']}y", interval="1d", progress=False)
        if len(df) < 500:
            print(f"    ⚠️  Insufficient data ({len(df)} days)")
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Stock-specific technical indicators
        df['RSI'] = EnhancedIndicators.get_rsi(df['Close'])
        df['ADX'] = EnhancedIndicators.get_adx(df['High'], df['Low'], df['Close'])
        df['NATR'] = EnhancedIndicators.get_natr(df['High'], df['Low'], df['Close'])
        df['OBV_Slope'] = EnhancedIndicators.get_obv_slope(df['Close'], df['Volume'])
        df['Dist_SMA'] = EnhancedIndicators.get_dist_sma(df['Close'])
        df['MACD'] = EnhancedIndicators.get_macd(df['Close'])
        df['ROC'] = EnhancedIndicators.get_roc(df['Close'])
        df['Vol_Ratio'] = EnhancedIndicators.get_volume_ratio(df['Volume'])
        df['BB_Position'] = EnhancedIndicators.get_bollinger_position(df['Close'])

        # Market context features
        if CONFIG['ENABLE_MARKET_FEATURES'] and market_df is not None:
            market_aligned = market_df.reindex(df.index, method='ffill')
            market_features = MarketContextFeatures.calculate_market_features(market_aligned['Close'])
            df = pd.concat([df, market_features], axis=1)

            stock_returns = df['Close'].pct_change(20)
            market_returns = market_aligned['Close'].pct_change(20)
            df['relative_strength'] = stock_returns - market_returns

            stock_ret = df['Close'].pct_change()
            market_ret = market_aligned['Close'].pct_change()
            df['beta'] = (
                stock_ret.rolling(60).cov(market_ret) /
                (market_ret.rolling(60).var() + 1e-8)
            )

        df.dropna(inplace=True)

        if len(df) < CONFIG['SEQ_LENGTH'] + 100:
            print(f"    ⚠️  Insufficient data after cleaning ({len(df)} rows)")
            return None

        return df

    except Exception as e:
        print(f"    ❌ Error: {e}")
        return None


def calculate_targets(df):
    """Calculate future returns for all horizons."""
    target_data = {}
    for i in range(1, CONFIG['MAX_HORIZON'] + 1):
        target_data[f'Target_Day_{i}'] = df['Close'].pct_change(periods=i).shift(-i) * 100
    return pd.DataFrame(target_data, index=df.index)


def create_sequences(features, targets, seq_length):
    """Create sequences without boundary crossing."""
    if len(features) < seq_length + 1:
        return np.array([]), np.array([])

    X, y = [], []
    for i in range(len(features) - seq_length):
        X.append(features[i:(i + seq_length)])
        y.append(targets[i + seq_length])

    return np.array(X), np.array(y)


def get_feature_columns():
    """Return list of feature column names."""
    features = [
        'RSI', 'ADX', 'NATR', 'OBV_Slope', 'Dist_SMA',
        'MACD', 'ROC', 'Vol_Ratio', 'BB_Position'
    ]
    if CONFIG['ENABLE_MARKET_FEATURES']:
        features.extend([
            'market_strength', 'market_momentum', 'market_volatility', 'market_trend',
            'relative_strength', 'beta'
        ])
    return features


# ==============================================================================
# MODEL ARCHITECTURE
# ==============================================================================
def build_enhanced_model(seq_length, n_features, n_outputs, seed=42):
    """LSTM + Multi-Head Attention model."""
    np.random.seed(seed)
    tf.random.set_seed(seed)

    input_layer = Input(shape=(seq_length, n_features))

    # First LSTM
    x = LSTM(256, return_sequences=True)(input_layer)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)

    # Multi-head attention
    attn_output = MultiHeadAttention(num_heads=8, key_dim=32, dropout=0.2)(x, x)
    x = Add()([x, attn_output])
    x = LayerNormalization()(x)

    # Second LSTM
    x = LSTM(128, return_sequences=False)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Dense layers
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu')(x)

    output_layer = Dense(n_outputs, activation='linear')(x)

    return Model(inputs=input_layer, outputs=output_layer)


# ==============================================================================
# ENSEMBLE TRAINING
# ==============================================================================
class ModelEnsemble:
    def __init__(self, n_models=5):
        self.n_models = n_models
        self.models = []
        self.weights = []
        self.histories = []

    def train(self, X_train, y_train, X_val, y_val, seq_length, n_features, n_outputs):
        print(f"\n{'='*80}")
        print(f"TRAINING ENSEMBLE OF {self.n_models} MODELS")
        print(f"{'='*80}\n")

        os.makedirs(CONFIG['ENSEMBLE_DIR'], exist_ok=True)

        for i in range(self.n_models):
            print(f"\n--- Model {i+1}/{self.n_models} (Seed: {42+i}) ---")

            model = build_enhanced_model(seq_length, n_features, n_outputs, seed=42+i)

            if CONFIG['ENABLE_CUSTOM_LOSS']:
                loss = DirectionAwareLoss(direction_weight=0.3)
            else:
                loss = 'mse'

            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss=loss,
                metrics=['mae']
            )

            model_path = os.path.join(CONFIG['ENSEMBLE_DIR'], f'model_{i+1}.keras')
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=0),
                ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss', verbose=0),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=0),
            ]

            indices = np.random.RandomState(42+i).permutation(len(X_train))
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            history = model.fit(
                X_shuffled, y_shuffled,
                validation_data=(X_val, y_val),
                epochs=CONFIG['EPOCHS'],
                batch_size=CONFIG['BATCH_SIZE'],
                callbacks=callbacks,
                verbose=1
            )

            val_loss = min(history.history['val_loss'])
            weight = 1.0 / (val_loss + 1e-6)

            self.models.append(model)
            self.weights.append(weight)
            self.histories.append(history.history)

            print(f"   Best Val Loss: {val_loss:.4f} | Weight: {weight:.4f}")

        self.weights = np.array(self.weights)
        self.weights = self.weights / self.weights.sum()

        print(f"\n✅ Ensemble complete!")
        print(f"   Normalized weights: {[f'{w:.3f}' for w in self.weights]}")

    def predict(self, X):
        predictions = []
        for model in self.models:
            pred = model.predict(X, verbose=0)
            predictions.append(pred)
        predictions = np.array(predictions)
        return np.average(predictions, axis=0, weights=self.weights)

    def predict_with_uncertainty(self, X):
        predictions = []
        for model in self.models:
            pred = model.predict(X, verbose=0)
            predictions.append(pred)
        predictions = np.array(predictions)
        mean_pred = np.average(predictions, axis=0, weights=self.weights)
        std_pred = np.std(predictions, axis=0)
        return mean_pred, std_pred


# ==============================================================================
# EVALUATION
# ==============================================================================
def evaluate_model(predictor, X_val, y_val, horizons=[1, 5, 10, 20, 60, 120, 180, 240, 300, 360]):
    """Comprehensive evaluation across horizons."""
    print(f"\n{'='*80}")
    print("VALIDATION METRICS BY HORIZON")
    print(f"{'='*80}")

    if hasattr(predictor, 'predict') and hasattr(predictor, 'weights'):
        predictions = predictor.predict(X_val)
    else:
        predictions = predictor.predict(X_val, verbose=0)

    metrics = {}
    for horizon in horizons:
        if horizon > y_val.shape[1]:
            continue

        actual = y_val[:, horizon-1]
        pred = predictions[:, horizon-1]

        # Direction accuracy
        direction_acc = float(np.mean(np.sign(actual) == np.sign(pred)))

        # Correlation
        mask = ~(np.isnan(actual) | np.isnan(pred))
        if mask.sum() > 10:
            corr = float(np.corrcoef(actual[mask], pred[mask])[0, 1])
        else:
            corr = 0.0

        # Spearman rank correlation
        if mask.sum() > 10:
            spearman_corr, _ = spearmanr(actual[mask], pred[mask])
            spearman_corr = float(spearman_corr) if not np.isnan(spearman_corr) else 0.0
        else:
            spearman_corr = 0.0

        # MAE and RMSE
        mae = float(np.mean(np.abs(actual - pred)))
        rmse = float(np.sqrt(np.mean((actual - pred)**2)))

        # R-squared
        ss_res = np.sum((actual - pred)**2)
        ss_tot = np.sum((actual - np.mean(actual))**2)
        r_squared = float(1 - (ss_res / (ss_tot + 1e-8)))

        # Sharpe-like metric
        pred_sharpe = float(np.mean(pred) / (np.std(pred) + 1e-8))

        metrics[f'Day_{horizon}'] = {
            'direction_accuracy': direction_acc,
            'pearson_correlation': corr,
            'spearman_correlation': spearman_corr,
            'mae': mae,
            'rmse': rmse,
            'r_squared': r_squared,
            'pred_sharpe': pred_sharpe,
        }

        print(f"Day {horizon:3d} | Dir: {direction_acc:.2%} | Pearson: {corr:+.3f} | "
              f"Spearman: {spearman_corr:+.3f} | MAE: {mae:.2f}% | RMSE: {rmse:.2f}% | R²: {r_squared:.4f}")

    print(f"{'='*80}\n")
    return metrics


# ==============================================================================
# MAIN TRAINING PIPELINE
# ==============================================================================
def train_system(max_stocks=None, epochs=None):
    """Main training pipeline."""

    if epochs:
        CONFIG['EPOCHS'] = epochs

    tickers = CONFIG['TICKERS'][:max_stocks] if max_stocks else CONFIG['TICKERS']

    print(f"\n{'='*80}")
    print(f" S&P 500 INTEGRATED TRAINING SYSTEM v2.0")
    print(f"{'='*80}")
    print(f"\nConfiguration:")
    print(f"  Market:          {CONFIG['MARKET_NAME']} ({CONFIG['MARKET_INDEX']})")
    print(f"  Stocks:          {len(tickers)}")
    print(f"  Ensemble:        {'✓ Enabled' if CONFIG['ENABLE_ENSEMBLE'] else '✗ Disabled'}")
    print(f"  Market Features: {'✓ Enabled' if CONFIG['ENABLE_MARKET_FEATURES'] else '✗ Disabled'}")
    print(f"  Custom Loss:     {'✓ Enabled' if CONFIG['ENABLE_CUSTOM_LOSS'] else '✗ Disabled'}")
    print(f"  N Models:        {CONFIG['N_ENSEMBLE']}")
    print(f"  Sequence Length:  {CONFIG['SEQ_LENGTH']}")
    print(f"  Max Horizon:     {CONFIG['MAX_HORIZON']} days")
    print(f"  Epochs:          {CONFIG['EPOCHS']}")
    print(f"{'='*80}")

    # Phase 1: Download market index
    print(f"\n=== PHASE 1: Downloading {CONFIG['MARKET_NAME']} Index Data ===")
    market_df = None
    if CONFIG['ENABLE_MARKET_FEATURES']:
        print(f"Downloading {CONFIG['MARKET_INDEX']}...")
        market_df = yf.download(CONFIG['MARKET_INDEX'], period=f"{CONFIG['TRAIN_YEARS']}y",
                                interval='1d', progress=False)
        if isinstance(market_df.columns, pd.MultiIndex):
            market_df.columns = market_df.columns.get_level_values(0)
        print(f"✓ {CONFIG['MARKET_NAME']} data: {len(market_df)} trading days")

    # Phase 2: Process all stocks
    print(f"\n=== PHASE 2: Processing {len(tickers)} Stocks ===")
    stock_data = []
    for i, ticker in enumerate(tickers):
        print(f"  → [{i+1}/{len(tickers)}] {ticker}... ", end='')
        df = prepare_stock_features(ticker, market_df)
        if df is not None:
            stock_data.append((ticker, df))
            print(f"✓ {len(df)} samples")
        else:
            print("✗ skipped")

    print(f"\n✅ Successfully processed {len(stock_data)}/{len(tickers)} stocks")

    if len(stock_data) == 0:
        print("❌ No valid data. Exiting.")
        return None

    # Phase 3: Create sequences
    print(f"\n=== PHASE 3: Creating Sequences ===")
    feature_cols = get_feature_columns()
    print(f"Using {len(feature_cols)} features: {feature_cols}")

    all_X = []
    all_y = []
    for ticker, df in stock_data:
        targets_df = calculate_targets(df)
        df = pd.concat([df, targets_df], axis=1)
        df.dropna(inplace=True)

        features = df[feature_cols].values
        targets = df[[f'Target_Day_{i}' for i in range(1, CONFIG['MAX_HORIZON']+1)]].values

        X_stock, y_stock = create_sequences(features, targets, CONFIG['SEQ_LENGTH'])
        if len(X_stock) > 0:
            all_X.append(X_stock)
            all_y.append(y_stock)
            print(f"  {ticker:15s}: {len(X_stock):6,} sequences")

    X_data = np.concatenate(all_X)
    y_data = np.concatenate(all_y)
    print(f"\n✅ Total sequences: {len(X_data):,}")
    print(f"   Shape: X={X_data.shape}, y={y_data.shape}")

    # Phase 4: Fit scaler
    print(f"\n=== PHASE 4: Fitting Scaler ===")
    n_samples, seq_len, n_features = X_data.shape
    X_reshaped = X_data.reshape(-1, n_features)

    split_idx = int(len(X_reshaped) * 0.8)
    scaler = StandardScaler()
    scaler.fit(X_reshaped[:split_idx])

    X_scaled = scaler.transform(X_reshaped)
    X_scaled = X_scaled.reshape(n_samples, seq_len, n_features)

    joblib.dump(scaler, CONFIG['SCALER_NAME'])
    print(f"✅ Scaler fitted and saved to {CONFIG['SCALER_NAME']}")

    # Phase 5: Train/Val Split
    print(f"\n=== PHASE 5: Creating Train/Val Split ===")
    split_idx = int(len(X_scaled) * 0.8)
    gap = CONFIG['MAX_HORIZON']

    X_train = X_scaled[:split_idx]
    y_train = y_data[:split_idx]
    X_val = X_scaled[split_idx + gap:]
    y_val = y_data[split_idx + gap:]

    print(f"Train: {len(X_train):,} samples")
    print(f"Val:   {len(X_val):,} samples")
    print(f"Gap:   {gap} days")

    # Phase 6: Train
    print(f"\n=== PHASE 6: Training Model(s) ===")

    ensemble = ModelEnsemble(n_models=CONFIG['N_ENSEMBLE'])
    ensemble.train(X_train, y_train, X_val, y_val,
                   CONFIG['SEQ_LENGTH'], n_features, CONFIG['MAX_HORIZON'])

    # Save ensemble info (compatible with generate_portfolio.py)
    ensemble_info = {
        'n_models': CONFIG['N_ENSEMBLE'],
        'weights': ensemble.weights.tolist(),
        'model_dir': CONFIG['ENSEMBLE_DIR'],
        'market_index': CONFIG['MARKET_INDEX'],
        'tickers': [t for t, _ in stock_data],
        'feature_cols': feature_cols,
        'trained_at': datetime.now().isoformat(),
    }

    # Save under both names for backward compatibility
    joblib.dump(ensemble_info, CONFIG['ENSEMBLE_INFO'])
    # Also save as the old name so generate_portfolio.py can find it
    old_info_path = os.path.join(BASE_DIR, 'nifty50_integrated_ensemble_info.pkl')
    joblib.dump(ensemble_info, old_info_path)

    print(f"✅ Ensemble info saved")

    # Phase 7: Evaluation
    print(f"\n=== PHASE 7: Model Evaluation ===")
    metrics = evaluate_model(ensemble, X_val, y_val)

    # Save metrics
    metrics_data = {
        'config': {k: v for k, v in CONFIG.items() if k != 'TICKERS'},
        'tickers_used': [t for t, _ in stock_data],
        'horizon_metrics': metrics,
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'n_features': n_features,
        'feature_columns': feature_cols,
        'ensemble_weights': ensemble.weights.tolist(),
        'trained_at': datetime.now().isoformat(),
    }
    joblib.dump(metrics_data, CONFIG['METRICS_NAME'])

    # Print summary
    print(f"\n{'='*80}")
    print(f" TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"\n📁 Generated Files:")
    print(f"   • Scaler: {CONFIG['SCALER_NAME']}")
    print(f"   • Metrics: {CONFIG['METRICS_NAME']}")
    print(f"   • Ensemble: {CONFIG['ENSEMBLE_DIR']}/ ({CONFIG['N_ENSEMBLE']} models)")
    print(f"   • Ensemble Info: {CONFIG['ENSEMBLE_INFO']}")
    print(f"\n📊 Key Metrics:")
    for horizon in [20, 60, 120, 180]:
        if f'Day_{horizon}' in metrics:
            m = metrics[f'Day_{horizon}']
            print(f"   Day {horizon:3d}: Dir={m['direction_accuracy']:.1%}, "
                  f"Corr={m['pearson_correlation']:+.3f}, MAE={m['mae']:.2f}%")
    print(f"\n{'='*80}")
    print(f"Ready for unified portfolio generation!")
    print(f"{'='*80}\n")

    return metrics_data


# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train technical analysis ensemble on S&P 500 stocks")
    parser.add_argument("--max_stocks", type=int, default=None,
                        help="Limit number of stocks to process (default: all)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of training epochs (default: 50)")
    args = parser.parse_args()

    train_system(max_stocks=args.max_stocks, epochs=args.epochs)
