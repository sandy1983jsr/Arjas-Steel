import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def train_predict_model(blend_df):
    blend_cols = [c for c in blend_df.columns if c.startswith('Pct_')]
    coke_cols = ['CSR', 'CRI', 'Ash', 'Yield']  # add M10, M40 if present
    for col in ['M10', 'M40']:
        if col in blend_df.columns: coke_cols.append(col)
    X = blend_df[blend_cols]
    y = blend_df[coke_cols]
    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(X, y)
    return model, blend_cols, coke_cols

def predict_coke_properties(model, blend_vector, blend_cols, coke_cols):
    X_pred = pd.DataFrame([blend_vector], columns=blend_cols)
    pred = model.predict(X_pred)[0]
    return dict(zip(coke_cols, pred))
