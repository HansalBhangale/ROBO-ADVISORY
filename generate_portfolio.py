"""
============================================================================
UNIFIED ROBO-ADVISORY PORTFOLIO GENERATOR
============================================================================
End-to-end portfolio generation integrating ALL modules:
  1. Risk Score Prediction (trained RandomForest on SCF survey data)
  2. Technical Analysis (LSTM+Attention ensemble on S&P 500)
  3. Fundamental Analysis (Random Forest on SEC EDGAR data)
  4. Combined Scoring Engine (blends technical + fundamental)
  5. Allocation Engine (6-stage pipeline from methodology)

Usage:
  python generate_portfolio.py                     # Default moderate profile
  python generate_portfolio.py --profile aggressive # Pick a profile
  python generate_portfolio.py --compare            # Compare 3 profiles
  python generate_portfolio.py --capital 50000      # Custom capital (USD)
  python generate_portfolio.py --metrics            # Generate metrics report

Author: Robo-Advisory System
Date: 2026
============================================================================
"""

import os
import sys
import glob
import json
import pickle
import warnings
import argparse
import numpy as np
import pandas as pd
import joblib
import yfinance as yf
import tensorflow as tf
from scipy.stats import norm, spearmanr
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RISK_MODEL_PATH = os.path.join(BASE_DIR, 'risk prediction', 'risk_tolerance_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'technical analysis', 'scaler_v2.pkl')
ENSEMBLE_DIR = os.path.join(BASE_DIR, 'technical analysis', 'ensemble_models')
ENSEMBLE_INFO_PATH = os.path.join(BASE_DIR, 'technical analysis', 'sp500_ensemble_info.pkl')
ENSEMBLE_INFO_PATH_OLD = os.path.join(BASE_DIR, 'technical analysis', 'nifty50_integrated_ensemble_info.pkl')
FUND_MODEL_PATH = os.path.join(BASE_DIR, 'fundamental analysis', 'best_model.pkl')
FUND_META_PATH = os.path.join(BASE_DIR, 'fundamental analysis', 'best_model_meta.json')
OUTPUT_CSV = os.path.join(BASE_DIR, 'portfolio_output.csv')
METRICS_FILE = os.path.join(BASE_DIR, 'model_metrics.txt')

# ============================================================================
# CONFIGURATION
# ============================================================================
SEQ_LENGTH = 60
MAX_HORIZON = 360
INVESTMENT_HORIZON_DAYS = 180
INITIAL_CAPITAL = 100_000  # $100K USD

# Combined score weights
TECHNICAL_WEIGHT = 0.4   # 40% technical
FUNDAMENTAL_WEIGHT = 0.6 # 60% fundamental (longer-horizon, more robust)

# S&P 500 universe — top blue-chips with good fundamental + technical data
STOCK_UNIVERSE = [
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA', 'BRK-B',
    'UNH', 'JNJ', 'JPM', 'V', 'PG', 'MA', 'HD', 'ABBV',
    'MRK', 'PEP', 'KO', 'COST', 'LLY', 'WMT', 'BAC', 'CRM',
    'NFLX', 'AMD', 'ORCL', 'ADBE', 'NOW', 'UBER',
]

STOCK_NAMES = {
    'AAPL': 'Apple Inc.', 'MSFT': 'Microsoft', 'AMZN': 'Amazon',
    'GOOGL': 'Alphabet (Google)', 'META': 'Meta Platforms', 'NVDA': 'NVIDIA',
    'TSLA': 'Tesla', 'BRK-B': 'Berkshire Hathaway', 'UNH': 'UnitedHealth',
    'JNJ': 'Johnson & Johnson', 'JPM': 'JPMorgan Chase', 'V': 'Visa',
    'PG': 'Procter & Gamble', 'MA': 'Mastercard', 'HD': 'Home Depot',
    'ABBV': 'AbbVie', 'MRK': 'Merck & Co.', 'PEP': 'PepsiCo',
    'KO': 'Coca-Cola', 'COST': 'Costco', 'LLY': 'Eli Lilly',
    'WMT': 'Walmart', 'BAC': 'Bank of America', 'CRM': 'Salesforce',
    'NFLX': 'Netflix', 'AMD': 'AMD', 'ORCL': 'Oracle',
    'ADBE': 'Adobe', 'NOW': 'ServiceNow', 'UBER': 'Uber Technologies',
}

# ============================================================================
# INVESTOR PROFILES (SCF-compatible features)
# ============================================================================
INVESTOR_PROFILES = {
    'ultra_conservative': {
        'name': 'Ultra Conservative (Retired Senior)',
        'features': {
            'EDUC': 12, 'EMERGSAV': 1, 'HSAVFIN': 1, 'HNMMF': 0,
            'HRETQLIQ': 1, 'NWCAT': 2, 'INCCAT': 1, 'ASSETCAT': 2,
            'NINCCAT': 1, 'NINC2CAT': 1, 'NWPCTLECAT': 25, 'INCPCTLECAT': 20,
            'NINCPCTLECAT': 20, 'INCQRTCAT': 1, 'NINCQRTCAT': 1,
            'AGE': 72, 'AGECL': 6, 'OCCAT1': 4, 'OCCAT2': 4
        }
    },
    'conservative': {
        'name': 'Conservative (Risk-Averse Professional)',
        'features': {
            'EDUC': 14, 'EMERGSAV': 1, 'HSAVFIN': 1, 'HNMMF': 0,
            'HRETQLIQ': 1, 'NWCAT': 3, 'INCCAT': 2, 'ASSETCAT': 3,
            'NINCCAT': 2, 'NINC2CAT': 2, 'NWPCTLECAT': 40, 'INCPCTLECAT': 40,
            'NINCPCTLECAT': 40, 'INCQRTCAT': 2, 'NINCQRTCAT': 2,
            'AGE': 58, 'AGECL': 5, 'OCCAT1': 1, 'OCCAT2': 1
        }
    },
    'moderate': {
        'name': 'Moderate (Balanced Investor)',
        'features': {
            'EDUC': 16, 'EMERGSAV': 1, 'HSAVFIN': 1, 'HNMMF': 1,
            'HRETQLIQ': 1, 'NWCAT': 4, 'INCCAT': 4, 'ASSETCAT': 4,
            'NINCCAT': 3, 'NINC2CAT': 3, 'NWPCTLECAT': 60, 'INCPCTLECAT': 60,
            'NINCPCTLECAT': 60, 'INCQRTCAT': 3, 'NINCQRTCAT': 3,
            'AGE': 40, 'AGECL': 3, 'OCCAT1': 1, 'OCCAT2': 1
        }
    },
    'growth': {
        'name': 'Growth (Young High Earner)',
        'features': {
            'EDUC': 16, 'EMERGSAV': 1, 'HSAVFIN': 1, 'HNMMF': 1,
            'HRETQLIQ': 1, 'NWCAT': 5, 'INCCAT': 5, 'ASSETCAT': 5,
            'NINCCAT': 4, 'NINC2CAT': 4, 'NWPCTLECAT': 75, 'INCPCTLECAT': 75,
            'NINCPCTLECAT': 75, 'INCQRTCAT': 4, 'NINCQRTCAT': 4,
            'AGE': 32, 'AGECL': 2, 'OCCAT1': 1, 'OCCAT2': 1
        }
    },
    'aggressive': {
        'name': 'Aggressive (Young Finance Professional)',
        'features': {
            'EDUC': 17, 'EMERGSAV': 1, 'HSAVFIN': 1, 'HNMMF': 1,
            'HRETQLIQ': 1, 'NWCAT': 5, 'INCCAT': 6, 'ASSETCAT': 6,
            'NINCCAT': 5, 'NINC2CAT': 5, 'NWPCTLECAT': 85, 'INCPCTLECAT': 85,
            'NINCPCTLECAT': 85, 'INCQRTCAT': 4, 'NINCQRTCAT': 4,
            'AGE': 28, 'AGECL': 2, 'OCCAT1': 1, 'OCCAT2': 2
        }
    },
    'ultra_aggressive': {
        'name': 'Ultra Aggressive (Wealthy Risk-Seeker)',
        'features': {
            'EDUC': 17, 'EMERGSAV': 0, 'HSAVFIN': 1, 'HNMMF': 1,
            'HRETQLIQ': 1, 'NWCAT': 5, 'INCCAT': 6, 'ASSETCAT': 6,
            'NINCCAT': 6, 'NINC2CAT': 6, 'NWPCTLECAT': 95, 'INCPCTLECAT': 95,
            'NINCPCTLECAT': 95, 'INCQRTCAT': 4, 'NINCQRTCAT': 4,
            'AGE': 26, 'AGECL': 1, 'OCCAT1': 2, 'OCCAT2': 2
        }
    }
}

# ============================================================================
# STUB CLASSES (for pickle deserialization)
# ============================================================================
class PCABasedRiskScorer:
    def __init__(self, df=None):
        self.df = df; self.pca = None; self.scaler = StandardScaler(); self.feature_names = []
    def create_risk_score(self): pass
    def get_feature_loadings(self): pass

class EmpiricalCorrelationScorer:
    def __init__(self, df=None):
        self.df = df; self.weights = {}; self.correlations = {}
    def calculate_empirical_weights(self): pass
    def normalize_to_percentile(self, series): pass
    def create_risk_score(self): pass

class DirectionAwareLoss(tf.keras.losses.Loss):
    def __init__(self, direction_weight=0.3, name='direction_aware_loss', reduction='sum_over_batch_size'):
        super().__init__(name=name, reduction=reduction)
        self.direction_weight = direction_weight
    def call(self, y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        dp = tf.reduce_mean(tf.cast(tf.sign(y_true) != tf.sign(y_pred), tf.float32))
        return (1 - self.direction_weight) * mse + self.direction_weight * dp * 100
    def get_config(self):
        c = super().get_config(); c.update({'direction_weight': self.direction_weight}); return c
    @classmethod
    def from_config(cls, config): return cls(**config)

# ============================================================================
# TECHNICAL INDICATORS (same as training pipeline)
# ============================================================================
class EnhancedIndicators:
    @staticmethod
    def get_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
        return 100 - (100 / (1 + gain / loss))

    @staticmethod
    def get_adx(high, low, close, period=14):
        plus_dm = high.diff().where(high.diff() > 0, 0)
        minus_dm = low.diff().where(low.diff() < 0, 0).abs()
        tr = pd.concat([high-low, (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)) * 100
        return dx.ewm(alpha=1/period, adjust=False).mean()

    @staticmethod
    def get_natr(high, low, close, period=14):
        tr = pd.concat([high-low, (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
        return (tr.ewm(alpha=1/period, adjust=False).mean() / close) * 100

    @staticmethod
    def get_obv_slope(close, volume, period=14):
        return (np.sign(close.diff()) * volume).fillna(0).cumsum().diff(period)

    @staticmethod
    def get_dist_sma(close, period=50):
        sma = close.rolling(period).mean(); return (close - sma) / sma

    @staticmethod
    def get_macd(close, fast=12, slow=26, signal=9):
        macd = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
        return macd - macd.ewm(span=signal, adjust=False).mean()

    @staticmethod
    def get_roc(close, period=10):
        return ((close - close.shift(period)) / close.shift(period)) * 100

    @staticmethod
    def get_volume_ratio(volume, period=20):
        return volume / volume.rolling(period).mean()

    @staticmethod
    def get_bollinger_position(close, period=20, std=2):
        sma = close.rolling(period).mean()
        rs = close.rolling(period).std()
        bw = (sma + rs*std) - (sma - rs*std)
        bw = bw.replace(0, np.nan)
        return (close - (sma - rs*std)) / bw


class MarketContextFeatures:
    @staticmethod
    def calculate_market_features(market_close):
        f = pd.DataFrame(index=market_close.index)
        for p in [20, 50, 200]:
            f[f'above_ma_{p}'] = (market_close > market_close.rolling(p).mean()).astype(float)
        f['market_strength'] = f[[f'above_ma_{p}' for p in [20, 50, 200]]].mean(axis=1)
        f['market_momentum'] = market_close.pct_change(20) * 100
        ret = market_close.pct_change()
        f['market_volatility'] = ret.rolling(20).std() * np.sqrt(252) * 100
        f['market_trend'] = ret.rolling(20).mean() / (ret.rolling(20).std() + 1e-8)
        return f[['market_strength', 'market_momentum', 'market_volatility', 'market_trend']]


# ============================================================================
# MODULE 1: RISK SCORE PREDICTION
# ============================================================================
def predict_risk_score(profile_key='moderate'):
    print("\n" + "=" * 80)
    print(" MODULE 1: RISK SCORE PREDICTION")
    print("=" * 80)

    with open(RISK_MODEL_PATH, 'rb') as f:
        risk_data = pickle.load(f)

    model = risk_data['model']
    feature_names = risk_data['features']
    print(f"  ✓ Loaded risk model ({type(model).__name__})")

    profile = INVESTOR_PROFILES[profile_key]
    profile_name = profile['name']
    feature_vector = [profile['features'].get(feat, 0) for feat in feature_names]
    X = np.array([feature_vector])

    risk_score = float(np.clip(model.predict(X)[0], 0, 100))

    if risk_score <= 20: category = 'Conservative'
    elif risk_score <= 35: category = 'Conservative-Moderate'
    elif risk_score <= 50: category = 'Moderate'
    elif risk_score <= 70: category = 'Moderate-Aggressive'
    else: category = 'Aggressive'

    print(f"\n  📋 Profile: {profile_name}")
    print(f"  🎯 Risk Score: {risk_score:.1f} / 100")
    print(f"  📊 Category: {category}")
    return risk_score, profile_name, category


# ============================================================================
# MODULE 2: TECHNICAL ANALYSIS (ENSEMBLE INFERENCE)
# ============================================================================
def load_ensemble_models():
    scaler = joblib.load(SCALER_PATH)
    info_path = ENSEMBLE_INFO_PATH if os.path.exists(ENSEMBLE_INFO_PATH) else ENSEMBLE_INFO_PATH_OLD
    ensemble_info = joblib.load(info_path)
    weights = np.array(ensemble_info['weights'])
    custom_objects = {'DirectionAwareLoss': DirectionAwareLoss}
    model_files = sorted(glob.glob(os.path.join(ENSEMBLE_DIR, 'model_*.keras')))
    models = []
    for mf in model_files:
        try:
            models.append(load_model(mf, custom_objects=custom_objects, compile=False))
        except Exception:
            models.append(tf.keras.models.load_model(mf, compile=False))
    return models, weights, scaler


def prepare_stock_features(ticker, market_df):
    df = yf.download(ticker, period="2y", interval="1d", progress=False)
    if len(df) < 150: return None, None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    current_price = float(df['Close'].iloc[-1])

    df['RSI'] = EnhancedIndicators.get_rsi(df['Close'])
    df['ADX'] = EnhancedIndicators.get_adx(df['High'], df['Low'], df['Close'])
    df['NATR'] = EnhancedIndicators.get_natr(df['High'], df['Low'], df['Close'])
    df['OBV_Slope'] = EnhancedIndicators.get_obv_slope(df['Close'], df['Volume'])
    df['Dist_SMA'] = EnhancedIndicators.get_dist_sma(df['Close'])
    df['MACD'] = EnhancedIndicators.get_macd(df['Close'])
    df['ROC'] = EnhancedIndicators.get_roc(df['Close'])
    df['Vol_Ratio'] = EnhancedIndicators.get_volume_ratio(df['Volume'])
    df['BB_Position'] = EnhancedIndicators.get_bollinger_position(df['Close'])

    if market_df is not None:
        mkt = market_df.reindex(df.index, method='ffill')
        mf = MarketContextFeatures.calculate_market_features(mkt['Close'])
        df = pd.concat([df, mf], axis=1)
        df['relative_strength'] = df['Close'].pct_change(20) - mkt['Close'].pct_change(20)
        sr, mr = df['Close'].pct_change(), mkt['Close'].pct_change()
        df['beta'] = sr.rolling(60).cov(mr) / (mr.rolling(60).var() + 1e-8)

    df.dropna(inplace=True)
    return df, current_price


def get_market_baseline():
    try:
        sp = yf.download('^GSPC', period='10y', interval='1d', progress=False)
        if isinstance(sp.columns, pd.MultiIndex):
            sp.columns = sp.columns.get_level_values(0)
        if len(sp) < 252: return 10.0, 15.0
        ret = sp['Close'].pct_change().dropna()
        return float(sp['Close'].pct_change(252).mean() * 100), float(ret.std() * np.sqrt(252) * 100)
    except Exception:
        return 10.0, 15.0


def score_stock_technical(ticker, df, current_price, models, weights, scaler,
                          duration_days, market_annual_ret, market_annual_vol):
    feature_cols = [
        'RSI', 'ADX', 'NATR', 'OBV_Slope', 'Dist_SMA',
        'MACD', 'ROC', 'Vol_Ratio', 'BB_Position',
        'market_strength', 'market_momentum', 'market_volatility', 'market_trend',
        'relative_strength', 'beta'
    ]
    if len(df) < SEQ_LENGTH: return None

    last_seq = df[feature_cols].tail(SEQ_LENGTH)
    input_seq = np.array([scaler.transform(last_seq)])

    predictions = np.array([m.predict(input_seq, verbose=0) for m in models])
    mean_forecast = np.average(predictions, axis=0, weights=weights)[0]
    std_forecast = np.std(predictions, axis=0)[0]

    eff = min(duration_days, MAX_HORIZON)
    raw_return = float(mean_forecast[eff - 1])
    pred_std = float(std_forecast[eff - 1])

    tf_ = eff / 365.0
    baseline_ret = float(market_annual_ret * tf_)
    baseline_vol = float(market_annual_vol * np.sqrt(tf_))
    natr = float(df['NATR'].iloc[-1])
    vol_move = float(natr * np.sqrt(eff))

    total_unc = float(np.sqrt(vol_move**2 + baseline_vol**2 + pred_std**2))
    fz = float((raw_return - baseline_ret) / total_unc) if total_unc > 0 else 0.0

    adx = float(df['ADX'].iloc[-1])
    rsi = float(df['RSI'].iloc[-1])
    conf = 1.0
    if adx < 20: conf *= 0.7
    elif adx > 40: conf *= 1.1
    if rsi > 70 and raw_return > 0: conf *= 0.8
    elif rsi < 30 and raw_return < 0: conf *= 0.8
    if pred_std > 5: conf *= 0.85
    conf = min(conf * 1.05, 1.2)

    base_score = norm.cdf(fz) * 100
    final_score = float(np.clip(50 + (base_score - 50) * conf, 0, 100))

    if final_score > 75: signal = 'STRONG BUY'
    elif final_score > 65: signal = 'BUY'
    elif final_score > 55: signal = 'WEAK BUY'
    elif final_score > 45: signal = 'NEUTRAL'
    elif final_score > 35: signal = 'WEAK SELL'
    elif final_score > 25: signal = 'SELL'
    else: signal = 'STRONG SELL'

    return {
        'ticker': ticker, 'name': STOCK_NAMES.get(ticker, ticker),
        'current_price': round(current_price, 2),
        'technical_score': round(final_score, 2),
        'predicted_return': round(raw_return, 2),
        'uncertainty': round(pred_std, 2),
        'confidence': round(conf, 2), 'signal': signal,
        'rsi': round(rsi, 2), 'adx': round(adx, 2), 'natr': round(natr, 2),
    }


def generate_technical_scores():
    print("\n" + "=" * 80)
    print(" MODULE 2: TECHNICAL ANALYSIS (ENSEMBLE INFERENCE)")
    print("=" * 80)

    print("  Loading ensemble models...")
    models, weights, scaler = load_ensemble_models()
    print(f"  ✓ Loaded {len(models)} ensemble models")

    print("  Downloading S&P 500 index data...")
    sp_df = yf.download('^GSPC', period='2y', interval='1d', progress=False)
    if isinstance(sp_df.columns, pd.MultiIndex):
        sp_df.columns = sp_df.columns.get_level_values(0)
    print(f"  ✓ S&P 500 data: {len(sp_df)} trading days")

    market_ret, market_vol = get_market_baseline()
    print(f"  ✓ Market baseline: {market_ret:.1f}% return, {market_vol:.1f}% volatility")

    print(f"\n  Analyzing {len(STOCK_UNIVERSE)} stocks (horizon: {INVESTMENT_HORIZON_DAYS} days)...\n")
    results = []
    for ticker in STOCK_UNIVERSE:
        print(f"    → {ticker:10s} ", end='')
        try:
            df, price = prepare_stock_features(ticker, sp_df)
            if df is None:
                print("✗ Insufficient data"); continue
            r = score_stock_technical(ticker, df, price, models, weights, scaler,
                                      INVESTMENT_HORIZON_DAYS, market_ret, market_vol)
            if r:
                results.append(r)
                print(f"✓ Score: {r['technical_score']:5.1f}  Signal: {r['signal']}")
            else: print("✗ Scoring failed")
        except Exception as e:
            print(f"✗ Error: {e}")

    print(f"\n  ✅ Scored {len(results)}/{len(STOCK_UNIVERSE)} stocks")
    return results


# ============================================================================
# MODULE 3: FUNDAMENTAL ANALYSIS SCORING
# ============================================================================
def generate_fundamental_scores():
    print("\n" + "=" * 80)
    print(" MODULE 3: FUNDAMENTAL ANALYSIS (SEC EDGAR MODEL)")
    print("=" * 80)

    # Add fundamental analysis dir to path so we can import
    fund_dir = os.path.join(BASE_DIR, 'fundamental analysis')
    if fund_dir not in sys.path:
        sys.path.insert(0, fund_dir)

    try:
        from score import load_best_model, engineer_features, predictions_to_scores
        from prepare import prepare_dataset

        model, meta = load_best_model()
        feature_cols = meta['feature_cols']
        buy_thresh = meta['buy_threshold']
        sell_thresh = meta['sell_threshold']
        print(f"  ✓ Loaded fundamental model: '{meta['tag']}' (val_ic={meta['val_ic']:.4f})")

        df = prepare_dataset()
        df = engineer_features(df)

        # Get latest filing per company
        latest = df.sort_values("filed_date").groupby("ticker").last().reset_index()
        print(f"  ✓ {len(latest)} companies with latest 10-K filings")

        X = latest[feature_cols].values
        raw_preds = model.predict(X)
        latest = latest.copy()
        latest['fundamental_score'] = predictions_to_scores(raw_preds)
        latest['fundamental_signal'] = latest['fundamental_score'].apply(
            lambda s: "BUY" if s >= buy_thresh else ("SELL" if s < sell_thresh else "HOLD")
        )

        # Build lookup dict by ticker
        fund_scores = {}
        for _, row in latest.iterrows():
            fund_scores[row['ticker']] = {
                'fundamental_score': round(float(row['fundamental_score']), 2),
                'fundamental_signal': row['fundamental_signal'],
                'filed_date': str(row['filed_date'].date()) if hasattr(row['filed_date'], 'date') else str(row['filed_date']),
            }

        print(f"  ✅ Generated fundamental scores for {len(fund_scores)} companies")
        return fund_scores

    except Exception as e:
        print(f"  ⚠️  Fundamental analysis unavailable: {e}")
        print(f"  → Continuing with technical scores only")
        return {}


# ============================================================================
# MODULE 4: COMBINED SCORING ENGINE
# ============================================================================
def combine_scores(technical_results, fundamental_scores):
    print("\n" + "=" * 80)
    print(" MODULE 4: COMBINED SCORING ENGINE")
    print("=" * 80)
    print(f"  Weights: Technical={TECHNICAL_WEIGHT:.0%}, Fundamental={FUNDAMENTAL_WEIGHT:.0%}")

    combined_results = []
    for r in technical_results:
        ticker = r['ticker']
        tech_score = r['technical_score']
        fund_data = fundamental_scores.get(ticker, {})
        fund_score = fund_data.get('fundamental_score', None)

        if fund_score is not None:
            combined = TECHNICAL_WEIGHT * tech_score + FUNDAMENTAL_WEIGHT * fund_score
            source = 'BOTH'
        else:
            combined = tech_score * 0.85  # penalty for missing fundamental
            source = 'TECH_ONLY'

        combined = round(float(np.clip(combined, 0, 100)), 2)

        if combined > 75: combined_signal = 'STRONG BUY'
        elif combined > 65: combined_signal = 'BUY'
        elif combined > 55: combined_signal = 'WEAK BUY'
        elif combined > 45: combined_signal = 'NEUTRAL'
        elif combined > 35: combined_signal = 'WEAK SELL'
        elif combined > 25: combined_signal = 'SELL'
        else: combined_signal = 'STRONG SELL'

        entry = {**r}
        entry['fundamental_score'] = fund_score if fund_score else 'N/A'
        entry['fundamental_signal'] = fund_data.get('fundamental_signal', 'N/A')
        entry['combined_score'] = combined
        entry['combined_signal'] = combined_signal
        entry['score_source'] = source
        combined_results.append(entry)

        print(f"    {ticker:10s} Tech:{tech_score:5.1f}  Fund:{str(fund_score or 'N/A'):>5s}  "
              f"Combined:{combined:5.1f}  {combined_signal}")

    combined_results.sort(key=lambda x: x['combined_score'], reverse=True)
    n_both = sum(1 for r in combined_results if r['score_source'] == 'BOTH')
    print(f"\n  ✅ {len(combined_results)} stocks scored ({n_both} with both scores)")
    return combined_results


# ============================================================================
# MODULE 5: ALLOCATION ENGINE (6-Stage Pipeline)
# ============================================================================
def get_risk_parameters(risk_score):
    if risk_score <= 20:
        return {'category': 'Conservative', 'max_equity': 0.40, 'concentration_limit': 0.10, 'min_holdings': 6, 'min_score': 60}
    elif risk_score <= 35:
        return {'category': 'Conservative-Moderate', 'max_equity': 0.55, 'concentration_limit': 0.15, 'min_holdings': 8, 'min_score': 55}
    elif risk_score <= 50:
        return {'category': 'Moderate', 'max_equity': 0.70, 'concentration_limit': 0.20, 'min_holdings': 10, 'min_score': 50}
    elif risk_score <= 70:
        return {'category': 'Moderate-Aggressive', 'max_equity': 0.85, 'concentration_limit': 0.25, 'min_holdings': 12, 'min_score': 45}
    else:
        return {'category': 'Aggressive', 'max_equity': 0.95, 'concentration_limit': 0.30, 'min_holdings': 15, 'min_score': 45}


def allocate_portfolio(stock_results, risk_score, capital=INITIAL_CAPITAL):
    print("\n" + "=" * 80)
    print(" MODULE 5: ALLOCATION ENGINE (6-Stage Pipeline)")
    print("=" * 80)

    params = get_risk_parameters(risk_score)
    max_equity = min(0.95, (risk_score / 100) ** 0.8)
    gamma = 2.0
    min_position = 0.05

    print(f"\n  Risk Score: {risk_score:.1f} | Category: {params['category']}")
    print(f"  Max Equity: {max_equity*100:.1f}% | Concentration Limit: {params['concentration_limit']*100:.0f}%")

    # Stage 1: Quality Filtering — using combined_score
    print(f"\n  --- Stage 1: Quality Filtering ---")
    qualified = [s for s in stock_results
                 if s['combined_score'] >= 45
                 and s['combined_signal'] in ['NEUTRAL', 'WEAK BUY', 'BUY', 'STRONG BUY']]
    print(f"  Passed: {len(qualified)}/{len(stock_results)}")

    # Stage 2: Risk-Based Selection
    print(f"  --- Stage 2: Risk-Based Selection ---")
    risk_filtered = [s for s in qualified if s['combined_score'] >= params['min_score']]
    risk_filtered.sort(key=lambda x: x['combined_score'], reverse=True)
    print(f"  After risk filter (min {params['min_score']}): {len(risk_filtered)}")

    if len(risk_filtered) < 3:
        print("\n  ⚠️  Insufficient qualified stocks → 100% CASH")
        return {'allocations': [], 'cash_weight': 100.0, 'cash_amount': capital,
                'equity_weight': 0.0, 'equity_amount': 0.0, 'portfolio_return': 0.0,
                'portfolio_uncertainty': 0.0, 'sharpe_ratio': 0.0, 'params': params}

    # Stage 3: Weighting (exponential γ=2.0) — using combined_score
    print(f"  --- Stage 3: Technical Weighting (γ={gamma}) ---")
    scores = np.array([s['combined_score'] for s in risk_filtered])
    raw_w = (scores / 100.0) ** gamma
    weights = raw_w / raw_w.sum()

    # Stage 4: Risk Adjustment
    print(f"  --- Stage 4: Risk Adjustment ---")
    adj_w = np.zeros(len(risk_filtered))
    for i, s in enumerate(risk_filtered):
        w = weights[i]
        p_unc = max(0.5, 1 - (s['uncertainty'] / 100) * (1 - risk_score / 100))
        sig = s['combined_signal']
        p_sig = 1.10 if sig == 'STRONG BUY' and risk_score >= 50 else (0.95 if sig in ['WEAK BUY', 'NEUTRAL'] else 1.0)
        p_conf = min(1.2, 0.8 + 0.4 * s['confidence'])
        adj_w[i] = w * p_unc * p_sig * p_conf

    if adj_w.sum() > 0: adj_w /= adj_w.sum()

    # Stage 5: Constraints
    print(f"  --- Stage 5: Constraint Application ---")
    cl = params['concentration_limit']
    for _ in range(10):
        excess = np.maximum(adj_w - cl, 0)
        if excess.sum() == 0: break
        adj_w = np.minimum(adj_w, cl)
        if adj_w.sum() > 0:
            remaining = adj_w < cl
            if remaining.sum() > 0:
                adj_w[remaining] += excess.sum() * (adj_w[remaining] / adj_w[remaining].sum())

    if adj_w.sum() > 0: adj_w /= adj_w.sum()
    mask = adj_w >= min_position
    adj_w = adj_w * mask
    if adj_w.sum() > 0: adj_w /= adj_w.sum()
    risk_filtered = [s for s, m in zip(risk_filtered, mask) if m]
    adj_w = adj_w[mask]
    if adj_w.sum() > 0: adj_w /= adj_w.sum()
    print(f"  Final positions: {len(risk_filtered)} stocks")

    # Stage 6: Capital Allocation
    print(f"  --- Stage 6: Capital Allocation ---")
    eq_w = max_equity
    cash_w = 1.0 - eq_w
    eq_amt = capital * eq_w
    cash_amt = capital * cash_w

    allocations = []
    for i, s in enumerate(risk_filtered):
        sw = adj_w[i] * eq_w
        sa = capital * sw
        allocations.append({
            'ticker': s['ticker'], 'name': s['name'], 'price': s['current_price'],
            'technical_score': s['technical_score'],
            'fundamental_score': s['fundamental_score'],
            'combined_score': s['combined_score'],
            'predicted_return': s['predicted_return'],
            'uncertainty': s['uncertainty'],
            'signal': s['combined_signal'],
            'weight_pct': round(sw * 100, 2),
            'capital_allocated': round(sa, 2),
            'shares_approx': int(sa / s['current_price']) if s['current_price'] > 0 else 0
        })

    rets = np.array([s['predicted_return'] for s in risk_filtered])
    uncs = np.array([s['uncertainty'] for s in risk_filtered])
    port_ret = float(np.sum(adj_w * rets))
    port_unc = float(np.sqrt(np.sum((adj_w * uncs) ** 2)))
    sharpe = port_ret / (port_unc + 1e-8)

    print(f"\n  Equity: {eq_w*100:.1f}% (${eq_amt:,.0f}) | Cash: {cash_w*100:.1f}% (${cash_amt:,.0f})")

    return {
        'allocations': allocations,
        'cash_weight': round(cash_w * 100, 2), 'cash_amount': round(cash_amt, 2),
        'equity_weight': round(eq_w * 100, 2), 'equity_amount': round(eq_amt, 2),
        'portfolio_return': round(port_ret, 2), 'portfolio_uncertainty': round(port_unc, 2),
        'sharpe_ratio': round(sharpe, 2), 'params': params
    }


# ============================================================================
# PORTFOLIO REPORT & CSV
# ============================================================================
def print_portfolio_report(risk_score, profile_name, category, portfolio, capital):
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " UNIFIED ROBO-ADVISORY PORTFOLIO REPORT".center(78) + "║")
    print("║" + f" Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(78) + "║")
    print("╚" + "═" * 78 + "╝")

    print("\n┌─── INVESTOR PROFILE " + "─" * 57 + "┐")
    print(f"│  Profile:    {profile_name:<62}│")
    print(f"│  Risk Score: {risk_score:<5.1f} / 100    Category: {category:<30}│")
    print(f"│  Capital:    ${capital:>12,.0f}    Horizon: {INVESTMENT_HORIZON_DAYS} days{' '*24}│")
    print("└" + "─" * 78 + "┘")

    print("\n┌─── ALLOCATION SUMMARY " + "─" * 55 + "┐")
    print(f"│  Equity:  {portfolio['equity_weight']:>6.1f}%  (${portfolio['equity_amount']:>12,.0f}){' '*29}│")
    print(f"│  Cash:    {portfolio['cash_weight']:>6.1f}%  (${portfolio['cash_amount']:>12,.0f}){' '*29}│")
    print(f"│  Stocks:  {len(portfolio['allocations']):>5}    Max/Stock: {portfolio['params']['concentration_limit']*100:.0f}%{' '*31}│")
    print("└" + "─" * 78 + "┘")

    if portfolio['allocations']:
        print("\n┌─── PORTFOLIO HOLDINGS " + "─" * 55 + "┐")
        print(f"│ {'#':>2} {'Stock':<18} {'Tech':>5} {'Fund':>5} {'Comb':>5} {'Signal':<10} {'Wt%':>5} {'Capital':>10} │")
        print("│" + "─" * 78 + "│")
        for i, a in enumerate(portfolio['allocations'], 1):
            fs = f"{a['fundamental_score']:5.1f}" if isinstance(a['fundamental_score'], (int, float)) else '  N/A'
            print(f"│ {i:>2} {a['name'][:18]:<18} {a['technical_score']:5.1f} {fs} "
                  f"{a['combined_score']:5.1f} {a['signal']:<10} {a['weight_pct']:5.1f} ${a['capital_allocated']:>9,.0f} │")
        print("│" + "─" * 78 + "│")
        print(f"│    {'Cash Reserve':<18} {'':>5} {'':>5} {'':>5} {'':10} "
              f"{portfolio['cash_weight']:5.1f} ${portfolio['cash_amount']:>9,.0f} │")
        print("└" + "─" * 78 + "┘")

    print("\n┌─── PORTFOLIO METRICS " + "─" * 56 + "┐")
    print(f"│  Expected Return ({INVESTMENT_HORIZON_DAYS}d):  {portfolio['portfolio_return']:>+6.2f}%{' '*43}│")
    print(f"│  Portfolio Uncertainty:    {portfolio['portfolio_uncertainty']:>6.2f}%{' '*43}│")
    print(f"│  Sharpe Ratio:            {portfolio['sharpe_ratio']:>6.2f}{' '*44}│")
    print("└" + "─" * 78 + "┘")

    print("\n" + "═" * 80)
    print(" Disclaimer: AI-generated advisory. Past performance ≠ future results.")
    print("═" * 80 + "\n")


def save_portfolio_csv(portfolio):
    rows = []
    for a in portfolio['allocations']:
        rows.append({
            'Ticker': a['ticker'], 'Stock Name': a['name'], 'Price ($)': a['price'],
            'Technical Score': a['technical_score'], 'Fundamental Score': a['fundamental_score'],
            'Combined Score': a['combined_score'],
            'Predicted Return (%)': a['predicted_return'], 'Uncertainty (%)': a['uncertainty'],
            'Signal': a['signal'], 'Weight (%)': a['weight_pct'],
            'Capital ($)': a['capital_allocated'], 'Shares': a['shares_approx']
        })
    rows.append({
        'Ticker': 'CASH', 'Stock Name': 'Cash Reserve', 'Price ($)': '',
        'Technical Score': '', 'Fundamental Score': '', 'Combined Score': '',
        'Predicted Return (%)': 0, 'Uncertainty (%)': 0, 'Signal': 'N/A',
        'Weight (%)': portfolio['cash_weight'], 'Capital ($)': portfolio['cash_amount'], 'Shares': ''
    })
    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)
    print(f"  ✓ Portfolio saved to: {OUTPUT_CSV}")


# ============================================================================
# METRICS REPORT GENERATION
# ============================================================================
def generate_metrics_report():
    print("\n" + "=" * 80)
    print(" GENERATING COMPREHENSIVE MODEL METRICS REPORT")
    print("=" * 80)

    lines = []
    lines.append("=" * 80)
    lines.append("ROBO-ADVISORY SYSTEM — COMPREHENSIVE MODEL METRICS REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)

    # --- Risk Prediction Model ---
    lines.append("\n" + "=" * 80)
    lines.append("MODEL 1: RISK PREDICTION (Investor Risk Tolerance)")
    lines.append("=" * 80)
    try:
        with open(RISK_MODEL_PATH, 'rb') as f:
            rd = pickle.load(f)
        model = rd['model']
        feats = rd['features']
        lines.append(f"Model Type:      {type(model).__name__}")
        lines.append(f"Features Used:   {len(feats)}")
        lines.append(f"Feature Names:   {feats}")
        lines.append(f"Model File:      {RISK_MODEL_PATH}")
        lines.append(f"Model Size:      {os.path.getsize(RISK_MODEL_PATH) / 1024 / 1024:.1f} MB")
        if hasattr(model, 'feature_importances_'):
            imp = dict(zip(feats, model.feature_importances_))
            imp = dict(sorted(imp.items(), key=lambda x: x[1], reverse=True))
            lines.append("\nFeature Importances:")
            for f_, i_ in imp.items():
                lines.append(f"  {f_:<20s} {i_:.4f} {'█' * int(i_ * 50)}")
        if hasattr(model, 'oob_score_'):
            lines.append(f"\nOOB Score:       {model.oob_score_:.4f}")
        if hasattr(model, 'n_estimators'):
            lines.append(f"N Estimators:    {model.n_estimators}")
        # Profile validation
        lines.append("\nProfile Risk Score Predictions:")
        lines.append(f"  {'Profile':<25s} {'Risk Score':>12s} {'Category':<25s}")
        lines.append("  " + "-" * 62)
        for pk, pv in INVESTOR_PROFILES.items():
            fv = [pv['features'].get(f_, 0) for f_ in feats]
            sc = float(np.clip(model.predict(np.array([fv]))[0], 0, 100))
            cat = ('Conservative' if sc <= 20 else 'Cons-Mod' if sc <= 35
                   else 'Moderate' if sc <= 50 else 'Mod-Agg' if sc <= 70 else 'Aggressive')
            lines.append(f"  {pk:<25s} {sc:>12.2f} {cat:<25s}")
    except Exception as e:
        lines.append(f"  Error loading risk model: {e}")

    # --- Technical Analysis Model ---
    lines.append("\n" + "=" * 80)
    lines.append("MODEL 2: TECHNICAL ANALYSIS (LSTM+Attention Ensemble)")
    lines.append("=" * 80)
    try:
        metrics_path = os.path.join(BASE_DIR, 'technical analysis', 'metrics_v2.pkl')
        md = joblib.load(metrics_path)
        lines.append(f"Architecture:    LSTM(256→128) + Multi-Head Attention (8 heads)")
        lines.append(f"Loss Function:   DirectionAwareLoss (MSE + 0.3 × Direction Penalty)")
        lines.append(f"Train Samples:   {md.get('train_samples', 'N/A'):,}")
        lines.append(f"Val Samples:     {md.get('val_samples', 'N/A'):,}")
        lines.append(f"N Features:      {md.get('n_features', 15)}")
        if 'feature_columns' in md:
            lines.append(f"Features:        {md['feature_columns']}")
        if 'ensemble_weights' in md:
            lines.append(f"Ensemble Wts:    {[f'{w:.3f}' for w in md['ensemble_weights']]}")
        if 'trained_at' in md:
            lines.append(f"Trained At:      {md['trained_at']}")
        if 'tickers_used' in md:
            lines.append(f"Stocks Used:     {len(md['tickers_used'])} ({', '.join(md['tickers_used'][:10])}...)")

        hm = md.get('horizon_metrics', {})
        if hm:
            lines.append(f"\nValidation Metrics by Prediction Horizon:")
            lines.append(f"  {'Horizon':>8s} {'DirAcc':>8s} {'Pearson':>8s} {'Spearman':>9s} {'MAE':>8s} {'RMSE':>8s} {'R²':>8s}")
            lines.append("  " + "-" * 59)
            for k in sorted(hm.keys(), key=lambda x: int(x.split('_')[1])):
                m = hm[k]
                day = k.split('_')[1]
                lines.append(f"  Day {day:>3s}  {m['direction_accuracy']:>7.2%} "
                             f"{m.get('pearson_correlation', m.get('correlation', 0)):>+7.3f}  "
                             f"{m.get('spearman_correlation', 0):>+8.3f} "
                             f"{m['mae']:>7.2f}% {m['rmse']:>7.2f}%"
                             f" {m.get('r_squared', 0):>7.4f}")
    except Exception as e:
        lines.append(f"  Error loading technical metrics: {e}")

    # --- Fundamental Analysis Model ---
    lines.append("\n" + "=" * 80)
    lines.append("MODEL 3: FUNDAMENTAL ANALYSIS (SEC EDGAR Random Forest)")
    lines.append("=" * 80)
    try:
        meta = json.loads(Path(FUND_META_PATH).read_text())
        lines.append(f"Model Tag:       {meta['tag']}")
        lines.append(f"Validation IC:   {meta['val_ic']:.6f} (Spearman rank correlation)")
        lines.append(f"BUY Threshold:   score >= {meta['buy_threshold']}")
        lines.append(f"SELL Threshold:  score < {meta['sell_threshold']}")
        lines.append(f"Feature Count:   {len(meta['feature_cols'])}")
        lines.append(f"Features:        {meta['feature_cols']}")
        lines.append(f"Saved At:        {meta['saved_at']}")

        results_path = os.path.join(BASE_DIR, 'fundamental analysis', 'results.tsv')
        if os.path.exists(results_path):
            lines.append(f"\nExperiment History (from results.tsv):")
            lines.append(f"  {'Timestamp':<20s} {'Tag':<18s} {'Val IC':>8s} {'Train IC':>9s} {'Sharpe':>7s} {'Hit Rate':>9s}")
            lines.append("  " + "-" * 73)
            rdf = pd.read_csv(results_path, sep='\t')
            for _, r in rdf.iterrows():
                lines.append(f"  {str(r.get('timestamp','')):.<20s} {str(r.get('tag','')):.<18s} "
                             f"{r.get('val_ic', 0):>8.6f} {r.get('train_ic', 0):>9.6f} "
                             f"{r.get('sharpe', 0):>7.4f} {r.get('hit_rate', 0):>9.4f}")
            best = rdf.loc[rdf['val_ic'].idxmax()]
            lines.append(f"\n  BEST MODEL: {best['tag']} (val_ic={best['val_ic']:.6f})")
    except Exception as e:
        lines.append(f"  Error loading fundamental metrics: {e}")

    # --- Summary ---
    lines.append("\n" + "=" * 80)
    lines.append("SYSTEM SUMMARY")
    lines.append("=" * 80)
    lines.append(f"Combined Score: {TECHNICAL_WEIGHT:.0%} Technical + {FUNDAMENTAL_WEIGHT:.0%} Fundamental")
    lines.append(f"Stock Universe:  S&P 500 ({len(STOCK_UNIVERSE)} stocks)")
    lines.append(f"Market Index:    S&P 500 (^GSPC)")
    lines.append(f"Currency:        USD")
    lines.append("=" * 80)

    report = "\n".join(lines)
    with open(METRICS_FILE, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n  ✓ Metrics report saved to: {METRICS_FILE}")
    print(report)


# ============================================================================
# COMPARE MODE
# ============================================================================
def compare_profiles(capital):
    profiles_to_test = ['conservative', 'moderate', 'aggressive']
    print("\n" + "█" * 80)
    print("█" + " MULTI-PROFILE PORTFOLIO COMPARISON ".center(78, '█') + "█")
    print("█" * 80)

    tech_results = generate_technical_scores()
    fund_scores = generate_fundamental_scores()
    combined = combine_scores(tech_results, fund_scores)

    if not combined:
        print("\n  ❌ No stocks scored. Exiting."); sys.exit(1)

    for pk in profiles_to_test:
        rs, pn, cat = predict_risk_score(pk)
        port = allocate_portfolio(combined, rs, capital)
        print_portfolio_report(rs, pn, cat, port, capital)
        save_portfolio_csv(port)


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Unified Robo-Advisory Portfolio Generator")
    parser.add_argument("--profile", type=str, default="moderate", choices=list(INVESTOR_PROFILES.keys()))
    parser.add_argument("--compare", action="store_true", help="Compare conservative/moderate/aggressive")
    parser.add_argument("--capital", type=float, default=INITIAL_CAPITAL)
    parser.add_argument("--metrics", action="store_true", help="Generate model metrics report")
    args = parser.parse_args()

    if args.metrics:
        generate_metrics_report()
        return

    print("\n" + "█" * 80)
    print("█" + " UNIFIED ROBO-ADVISORY PORTFOLIO GENERATOR ".center(78, '█') + "█")
    print("█" * 80)
    print(f"\n  Capital: ${args.capital:,.0f}")
    print(f"  Horizon: {INVESTMENT_HORIZON_DAYS} days")
    print(f"  Universe: {len(STOCK_UNIVERSE)} S&P 500 stocks")
    print(f"  Scoring: {TECHNICAL_WEIGHT:.0%} Technical + {FUNDAMENTAL_WEIGHT:.0%} Fundamental")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if args.compare:
        compare_profiles(args.capital)
        return

    # Step 1: Risk Score
    risk_score, profile_name, category = predict_risk_score(args.profile)

    # Step 2: Technical Scores
    tech_results = generate_technical_scores()
    if not tech_results:
        print("\n  ❌ No stocks scored. Exiting."); sys.exit(1)

    # Step 3: Fundamental Scores
    fund_scores = generate_fundamental_scores()

    # Step 4: Combined Scores
    combined = combine_scores(tech_results, fund_scores)

    # Step 5: Allocation
    portfolio = allocate_portfolio(combined, risk_score, args.capital)

    # Step 6: Report
    print_portfolio_report(risk_score, profile_name, category, portfolio, args.capital)
    save_portfolio_csv(portfolio)

    # Step 7: Metrics report
    generate_metrics_report()

    print("\n  ✅ Portfolio generation complete!\n")
    return portfolio


if __name__ == '__main__':
    main()
