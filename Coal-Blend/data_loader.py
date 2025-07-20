import pandas as pd

def load_coal_db(file):
    df = pd.read_csv(file)
    assert {'Coal', 'Volatile matter (%)', 'Ash content (%)', 'Fixed carbon (%)', 'Moisture (%)', 'Sulphur (%)', 'G-caking Index', 'CSR', 'Cost per ton (INR)'}.issubset(df.columns)
    return df

def load_blend_db(file, coal_df):
    df = pd.read_csv(file)
    # Expect blend_id, blend_%_CoalA,...,blend_%_CoalH, CSR, CRI, Ash, Yield, [M10, M40]
    blend_cols = []
    for c in coal_df['Coal']:
        col = f"Pct_{c}"
        if col not in df.columns:
            # Convert blend_%_CoalA to Pct_CoalA
            for cc in df.columns:
                if c in cc and '%' in cc: df.rename(columns={cc:col}, inplace=True)
        blend_cols.append(col)
    # Normalize percentages to 0-1
    for c in blend_cols:
        if df[c].max()>2: df[c] = df[c]/100
    return df
