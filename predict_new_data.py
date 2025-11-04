"""
Script para usar o modelo treinado em dados novos fora do dataframe de teste.

Este script carrega o modelo salvo e faz previsões em novos dados,
garantindo que os mesmos pré-processamentos sejam aplicados.
"""

import numpy as np
import pandas as pd
import os
import json
import pickle
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras

# ============================================================================
# 1. CONFIGURAR CAMINHOS
# ============================================================================

DATA_DIR = "data"

# Caminhos dos arquivos salvos durante o treinamento
MODEL_PATH = os.path.join(DATA_DIR, "model_cnn_lstm_selected.keras")
SCALER_PATH = os.path.join(DATA_DIR, "scaler.pkl")
SPLIT_JSON_PATH = os.path.join(DATA_DIR, "split_boundaries.json")
SELECTED_FEATURES_PATH = os.path.join(DATA_DIR, "selected_features.txt")

# ============================================================================
# 2. CARREGAR MODELO E ARTEFATOS
# ============================================================================

def load_model_artifacts():
    """Carrega o modelo, scaler e configurações."""
    
    # Carregar modelo
    model = keras.models.load_model(MODEL_PATH)
    print(f"✓ Modelo carregado de: {MODEL_PATH}")
    
    # Carregar scaler
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    print(f"✓ Scaler carregado de: {SCALER_PATH}")
    
    # Carregar configurações
    with open(SPLIT_JSON_PATH, "r") as f:
        split_meta = json.load(f)
    lookback = split_meta["LOOKBACK"]
    feature_cols = split_meta["FEATURE_COLS"]
    print(f"✓ Configurações carregadas: LOOKBACK={lookback}")
    
    # Carregar features selecionadas
    with open(SELECTED_FEATURES_PATH, "r") as f:
        selected_features = [line.strip() for line in f.readlines()]
    print(f"✓ {len(selected_features)} features selecionadas")
    
    return model, scaler, lookback, feature_cols, selected_features


# ============================================================================
# 3. PRÉ-PROCESSAR NOVOS DADOS
# ============================================================================

def preprocess_new_data(df_new, scaler, feature_cols, selected_features, lookback):
    """
    Pré-processa dados novos para serem compatíveis com o modelo.
    
    Args:
        df_new: DataFrame com novos dados (deve ter as colunas 'date' e features)
        scaler: StandardScaler treinado
        feature_cols: Lista de todas as features usadas no treinamento
        selected_features: Lista de features selecionadas pelo Boruta
        lookback: Número de períodos para sequências
    
    Returns:
        X_processed: Dados processados prontos para previsão
        dates: Datas correspondentes às previsões
    """
    
    # Garantir que temos date como datetime
    df_new["date"] = pd.to_datetime(df_new["date"], errors="coerce", utc=True).dt.tz_localize(None)
    df_new = df_new.sort_values("date").reset_index(drop=True)
    
    # Verificar se todas as features necessárias estão presentes
    missing_features = set(feature_cols) - set(df_new.columns)
    if missing_features:
        raise ValueError(f"Features faltando no novo dataframe: {missing_features}")
    
    # Selecionar apenas as features necessárias
    df_process = df_new[["date"] + feature_cols].copy()
    
    # Converter para numérico (coagir erros a NaN)
    for col in feature_cols:
        df_process[col] = pd.to_numeric(df_process[col], errors="coerce")
    
    # Forward fill para preencher NaNs
    df_process[feature_cols] = df_process[feature_cols].ffill()
    
    # Remover linhas com NaN
    df_process = df_process.dropna(subset=feature_cols).reset_index(drop=True)
    
    if len(df_process) == 0:
        raise ValueError("Nenhuma linha válida após limpeza!")
    
    # Aplicar suavização EMA (mesmo que no treinamento)
    def ema_smooth(series, span=7):
        return series.ewm(span=span, adjust=False).mean()
    
    for col in feature_cols:
        df_process[col] = ema_smooth(pd.to_numeric(df_process[col], errors="coerce"))
    
    # Remover NaNs da suavização
    df_process = df_process.dropna(subset=feature_cols).reset_index(drop=True)
    
    # Normalizar usando o scaler treinado (IMPORTANTE: usar transform, não fit_transform)
    X_scaled = scaler.transform(df_process[feature_cols].values)
    
    # Selecionar apenas features selecionadas
    selected_idx = [feature_cols.index(f) for f in selected_features if f in feature_cols]
    X_scaled_selected = X_scaled[:, selected_idx]
    
    # Criar sequências
    X_sequences = []
    valid_dates = []
    
    for i in range(lookback - 1, len(X_scaled_selected)):
        X_sequences.append(X_scaled_selected[i - lookback + 1 : i + 1, :])
        valid_dates.append(df_process["date"].iloc[i])
    
    X_sequences = np.array(X_sequences)
    
    print(f"\n✓ Dados processados:")
    print(f"  - Shape final: {X_sequences.shape}")
    print(f"  - Datas: {valid_dates[0].date()} → {valid_dates[-1].date()}")
    
    return X_sequences, valid_dates


# ============================================================================
# 4. FAZER PREVISÕES
# ============================================================================

def predict_on_new_data(model, X_sequences, dates, threshold=0.5):
    """
    Faz previsões usando o modelo treinado.
    
    Args:
        model: Modelo Keras treinado
        X_sequences: Dados processados
        dates: Datas correspondentes
        threshold: Threshold para classificação (default 0.5)
    
    Returns:
        DataFrame com previsões
    """
    
    # Fazer previsões (retorna probabilidades)
    predictions = model.predict(X_sequences, verbose=0)
    
    # predictions é (N, 1) ou (N, 2) dependendo da saída do modelo
    if predictions.shape[1] == 2:
        prob_up = predictions[:, 1]  # Probabilidade da classe positiva
    else:
        prob_up = predictions[:, 0]  # Probabilidade direta
    
    # Classificação binária
    pred_class = (prob_up >= threshold).astype(int)
    
    # Criar DataFrame com resultados
    results_df = pd.DataFrame({
        "date": dates,
        "probability_up": prob_up,
        "prediction": pred_class,
        "signal": ["UP" if p == 1 else "DOWN" for p in pred_class]
    })
    
    return results_df


# ============================================================================
# 5. EXEMPLO DE USO
# ============================================================================

def main():
    """Exemplo completo de uso."""
    
    # Carregar artefatos
    model, scaler, lookback, feature_cols, selected_features = load_model_artifacts()
    
    # -------- OPÇÃO 1: Carregar novos dados de um CSV --------
    print("\n" + "="*70)
    print("OPÇÃO 1: Carregar novos dados de um arquivo CSV")
    print("="*70)
    
    # Exemplo: você tem um novo CSV com dados
    df_new = pd.read_csv(os.path.join(DATA_DIR, "merged_btc_features_CLEAN.csv"))
    
    # Pode pegar apenas os últimos N dias se quiser
    df_new = df_new.tail(100)  # Últimos 100 dias
    
    # Pré-processar
    X_sequences, dates = preprocess_new_data(
        df_new, scaler, feature_cols, selected_features, lookback
    )
    
    # Fazer previsões
    results = predict_on_new_data(model, X_sequences, dates)
    print("\nPrimeiras previsões:")
    print(results.head(10))
    
    # Salvar resultados
    results.to_csv(os.path.join(DATA_DIR, "predictions_new_data.csv"), index=False)
    print(f"\n✓ Previsões salvas em: {os.path.join(DATA_DIR, 'predictions_new_data.csv')}")
    
    # Estatísticas
    print(f"\nEstatísticas:")
    print(f"  - Total de previsões: {len(results)}")
    print(f"  - Sinais UP: {(results['signal'] == 'UP').sum()} ({(results['signal'] == 'UP').sum()/len(results)*100:.1f}%)")
    print(f"  - Sinais DOWN: {(results['signal'] == 'DOWN').sum()} ({(results['signal'] == 'DOWN').sum()/len(results)*100:.1f}%)")
    print(f"  - Probabilidade média: {results['probability_up'].mean():.3f}")
    
    
    # -------- OPÇÃO 2: Dados em tempo real / novo ponto único --------
    print("\n" + "="*70)
    print("OPÇÃO 2: Previsão para um novo ponto único de dados")
    print("="*70)
    
    # Criar um DataFrame mínimo com um novo ponto de dados
    df_single = pd.DataFrame({
        "date": pd.date_range(start="2025-11-01", periods=30, freq="D"),
        **{col: np.random.randn(30) for col in feature_cols}
    })
    
    X_seq_single, dates_single = preprocess_new_data(
        df_single, scaler, feature_cols, selected_features, lookback
    )
    
    pred_single = predict_on_new_data(model, X_seq_single, dates_single)
    print("\nÚltima previsão:")
    print(pred_single.iloc[-1])


if __name__ == "__main__":
    main()
