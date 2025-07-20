import numpy as np
from scipy.optimize import minimize
from blend_model import predict_coke_properties

def optimize_blend(model, coal_df, blend_cols, coke_cols, constraints, avail_limits):
    n = len(coal_df)
    costs = coal_df['Cost per ton (INR)'].values
    bounds = []
    for c in coal_df["Coal"]:
        max_lim = avail_limits.get(c, 1.0)
        bounds.append((0, max_lim))
    def obj(x): return np.dot(costs, x)
    def eq(x): return np.sum(x)-1
    cons = [{'type':'eq','fun':eq}]
    # Quality constraints
    def quality_cons(x):
        blend_vector = dict(zip(blend_cols, x))
        pred = predict_coke_properties(model, blend_vector, blend_cols, coke_cols)
        return [
            pred['CSR'] - constraints['CSR'],
            constraints['CRI'] - pred['CRI'],
            constraints['Ash'] - pred['Ash'],
            pred['Yield'] - constraints['Yield']
        ]
    cons += [
        {'type':'ineq','fun':lambda x:quality_cons(x)[0]}, # CSR >= min
        {'type':'ineq','fun':lambda x:quality_cons(x)[1]}, # CRI <= max
        {'type':'ineq','fun':lambda x:quality_cons(x)[2]}, # Ash <= max
        {'type':'ineq','fun':lambda x:quality_cons(x)[3]}, # Yield >= min
    ]
    x0 = np.array([1/n]*n)
    res = minimize(obj, x0, bounds=bounds, constraints=cons)
    blend = dict(zip(coal_df['Coal'], res.x))
    blend_vector = dict(zip(blend_cols, res.x))
    quality = predict_coke_properties(model, blend_vector, blend_cols, coke_cols)
    cost = obj(res.x)
    return {'blend':blend, 'cost':cost, 'quality':quality}
