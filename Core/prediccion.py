import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import streamlit as st

# Intentar importar librerías ML opcionales de forma segura
try:
    from sklearn.ensemble import RandomForestRegressor as SklearnRF
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR as SklearnSVR
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
   
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────

ANIOS_FUTUROS_DEFAULT = (2026, 2027, 2028)
ANIOS_TEST_DEFAULT = 5  # Ventanas de validación temporal (2021..2025)


# ─────────────────────────────────────────────────────────────────────────────
# API PÚBLICA
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def predecir_serie(
    serie: List[Dict[str, Any]],
    campo_valor: str = "Valor",
    campo_anio: str = "Año",
    anios_futuros: Tuple[int, ...] = ANIOS_FUTUROS_DEFAULT,
    anios_test: int = ANIOS_TEST_DEFAULT,
) -> Dict[str, Any]:

    # 1. Limpieza y ordenamiento
    datos = _limpiar(serie, campo_valor, campo_anio)
    n = len(datos)

    if n < 5:
        return _resultado_invalido(datos, "Datos insuficientes (mínimo 5 observaciones anuales).")

    anios = np.array([d["Año"] for d in datos], dtype=float)
    vals = np.array([d["Valor"] for d in datos], dtype=float)

    # 2. Configurar validación rolling-origin (k ventanas)
    k = max(2, min(anios_test, n // 3))

    # 3. Evaluar los 8 modelos
    evaluaciones = []
    definiciones_modelos = _obtener_definiciones_modelos()

    for nombre, es_ml, fn_entrenar, fn_predecir_step in definiciones_modelos:
        try:
            res_val = _validar_rolling(
                anios=anios,
                vals=vals,
                k=k,
                es_ml=es_ml,
                fn_entrenar=fn_entrenar,
                fn_predecir_step=fn_predecir_step
            )
            evaluaciones.append({
                "Modelo": nombre,
                "MAE": round(res_val["MAE"], 5),
                "RMSE": round(res_val["RMSE"], 5),
                "R2": round(res_val["R2"], 4) if res_val["R2"] is not None else None,
                "Splits": res_val["Splits"],
                "Disponible": True
            })
        except Exception as err:
            evaluaciones.append({
                "Modelo": nombre,
                "MAE": None,
                "RMSE": None,
                "R2": None,
                "Splits": 0,
                "Disponible": False,
                "Detalle": str(err)
            })

    # Filtrar modelos que se pudieron evaluar con éxito
    disponibles = [m for m in evaluaciones if m.get("Disponible") and m.get("MAE") is not None]

    if not disponibles:
        return _resultado_invalido(datos, "Ningún modelo pudo entrenarse con los datos provistos.")

    # 4. Selección del mejor modelo: menor MAE (RMSE como desempate secundario)
    disponibles.sort(key=lambda m: (m["MAE"], m["RMSE"] if m["RMSE"] is not None else 1e9))
    mejor_modelo_info = disponibles[0]
    mejor_nombre = mejor_modelo_info["Modelo"]

    # 5. Reentrenamiento del modelo seleccionado con TODA la serie histórica (2000–2025)
    def_sel = next(d for d in definiciones_modelos if d[0] == mejor_nombre)
    _, es_ml_sel, fn_entrenar_sel, fn_predecir_sel = def_sel

    # Entrenamiento con todos los datos
    modelo_final = fn_entrenar_sel(anios, vals)

    # 6. Predicción recursiva 2026–2028
    predicciones = []
    anios_hist_list = anios.tolist()
    vals_hist_list = vals.tolist()

    for anio_fut in anios_futuros:
        y_hat = fn_predecir_sel(modelo_final, anios_hist_list, vals_hist_list, float(anio_fut))
        y_hat_val = float(y_hat)

        # Intervalos de incertidumbre
        min_val, max_val = _calcular_intervalos(mejor_nombre, modelo_final, anios_hist_list, vals_hist_list, float(anio_fut), y_hat_val)

        predicciones.append({
            "Año": int(anio_fut),
            "Valor": round(y_hat_val, 4),
            "Min": round(min_val, 4) if min_val is not None else None,
            "Max": round(max_val, 4) if max_val is not None else None,
        })

        # Para modelos autoregresivos (lags), incorporar la predicción al historial
        anios_hist_list.append(float(anio_fut))
        vals_hist_list.append(y_hat_val)

    # 7. Métricas y resumen final
    mae_sel = mejor_modelo_info["MAE"]
    rmse_sel = mejor_modelo_info["RMSE"]
    r2_sel = mejor_modelo_info["R2"]
    pendiente = _pendiente_lineal(anios, vals)

    # Ordenar la lista de evaluaciones por MAE ascendente
    evaluaciones.sort(key=lambda m: (0 if m.get("Disponible") and m.get("MAE") is not None else 1,
                                     m.get("MAE") if m.get("MAE") is not None else 1e9))

    return {
        "valido": True,
        "mensaje": f"Modelo seleccionado: {mejor_nombre} (MAE = {mae_sel:.5f})",
        "historico": datos,
        "predicciones": predicciones,
        "metricas": {
            "MAE": mae_sel,
            "RMSE": rmse_sel,
            "R2": r2_sel,
            "Modelo": mejor_nombre,
            "Pendiente_anual": round(pendiente, 6),
        },
        "modelos_evaluados": evaluaciones,
        "mensaje_seleccion": (
            f"Se evaluaron {len(evaluaciones)} modelos mediante validación rolling-origin ({k} splits temporales). "
            f"El modelo **{mejor_nombre}** obtuvo el menor MAE ({mae_sel:.5f}, RMSE = {rmse_sel:.5f}) "
            f"y fue seleccionado automáticamente para proyectar el periodo {anios_futuros[0]}–{anios_futuros[-1]}."
        ),
    }


def calcular_anomalias_proyectadas(
    historico: List[Dict[str, Any]],
    predicciones: List[Dict[str, Any]],
) -> Dict[str, Any]:

    valores_hist = [d["Valor"] for d in historico if d.get("Valor") is not None]
    if not valores_hist:
        return {"historico": [], "proyectado": [], "media": 0.0, "std": 0.0}

    n = len(valores_hist)
    media = sum(valores_hist) / n
    std = (sum((v - media) ** 2 for v in valores_hist) / n) ** 0.5
    if std == 0:
        std = 1e-6

    return {
        "historico": [
            {
                "Año": d["Año"],
                "Valor": d["Valor"],
                "Anomalia": round(d["Valor"] - media, 4),
                "Tipo": "Observada"
            }
            for d in historico
        ],
        "proyectado": [
            {
                "Año": d["Año"],
                "Valor": d["Valor"],
                "Anomalia": round(d["Valor"] - media, 4),
                "Tipo": "Proyectada"
            }
            for d in predicciones
        ],
        "media": round(media, 4),
        "std": round(std, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# GENERACIÓN DE CARACTERÍSTICAS (FEATURES) TEMPORALES PARA ML (RF/XGB)
# ─────────────────────────────────────────────────────────────────────────────

def _crear_dataset_lags(anios: np.ndarray, vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    n = len(vals)
    if n < 4:
        raise ValueError("Se requieren al menos 4 observaciones para construir lags.")

    X = []
    y = []
    for i in range(3, n):
        anio_i = anios[i]
        lag1 = vals[i - 1]
        lag2 = vals[i - 2]
        roll3 = (vals[i - 1] + vals[i - 2] + vals[i - 3]) / 3.0
        X.append([anio_i, lag1, lag2, roll3])
        y.append(vals[i])

    return np.array(X, dtype=float), np.array(y, dtype=float)


def _crear_vector_features_paso(anios_hist: List[float], vals_hist: List[float], anio_target: float) -> np.ndarray:

    lag1 = vals_hist[-1]
    lag2 = vals_hist[-2]
    roll3 = (vals_hist[-1] + vals_hist[-2] + vals_hist[-3]) / 3.0
    return np.array([[anio_target, lag1, lag2, roll3]], dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# IMPLEMENTACIONES DE LOS 8 MODELOS
# ─────────────────────────────────────────────────────────────────────────────

# 1. Naive (Persistencia)
def _naive_train(anios: np.ndarray, vals: np.ndarray) -> Dict[str, Any]:
    residuos = np.diff(vals) if len(vals) > 1 else np.array([0.0])
    s = float(np.std(residuos)) if len(residuos) else float(np.std(vals))
    return {"last_val": float(vals[-1]), "s": s, "n": len(vals)}

def _naive_predict(modelo: Dict[str, Any], anios_hist: List[float], vals_hist: List[float], anio_target: float) -> float:
    return float(vals_hist[-1])


# 2. Regresión Lineal (OLS)
def _lineal_train(anios: np.ndarray, vals: np.ndarray) -> Dict[str, Any]:
    m, b = _ajustar_ols(anios, vals)
    residuos = vals - (m * anios + b)
    s = float(np.std(residuos, ddof=2)) if len(vals) > 2 else float(np.std(residuos))
    mx = float(np.mean(anios))
    ssx = float(np.sum((anios - mx) ** 2))
    return {"m": m, "b": b, "s": s, "n": len(vals), "mean_x": mx, "ssx": ssx}

def _lineal_predict(modelo: Dict[str, Any], anios_hist: List[float], vals_hist: List[float], anio_target: float) -> float:
    return float(modelo["m"] * anio_target + modelo["b"])


# 3. Regresión Polinómica grado 2
def _poly2_train(anios: np.ndarray, vals: np.ndarray) -> Dict[str, Any]:
    x0 = float(anios[0])
    t = anios - x0
    coeffs = np.polyfit(t, vals, 2)
    residuos = vals - np.polyval(coeffs, t)
    s = float(np.std(residuos, ddof=3)) if len(vals) > 3 else float(np.std(residuos))
    return {"coeffs": coeffs.tolist(), "x0": x0, "s": s, "n": len(vals)}

def _poly2_predict(modelo: Dict[str, Any], anios_hist: List[float], vals_hist: List[float], anio_target: float) -> float:
    t = anio_target - modelo["x0"]
    return float(np.polyval(modelo["coeffs"], t))


# 4. Holt (Suavizado Exponencial Lineal)
def _holt_train(anios: np.ndarray, vals: np.ndarray) -> Dict[str, Any]:
    best = {"alpha": 0.3, "beta": 0.1, "mse": float("inf")}
    for alpha in np.linspace(0.05, 0.95, 10):
        for beta in np.linspace(0.01, 0.5, 6):
            L, B = vals[0], vals[1] - vals[0] if len(vals) > 1 else 0.0
            errs = []
            for v in vals[1:]:
                y_hat = L + B
                errs.append((v - y_hat) ** 2)
                L_new = alpha * v + (1 - alpha) * (L + B)
                B = beta * (L_new - L) + (1 - beta) * B
                L = L_new
            mse = float(np.mean(errs)) if errs else 0.0
            if mse < best["mse"]:
                best = {"alpha": alpha, "beta": beta, "mse": mse}

    alpha, beta = best["alpha"], best["beta"]
    L, B = vals[0], vals[1] - vals[0] if len(vals) > 1 else 0.0
    residuos = []
    for v in vals[1:]:
        y_hat = L + B
        residuos.append(v - y_hat)
        L_new = alpha * v + (1 - alpha) * (L + B)
        B = beta * (L_new - L) + (1 - beta) * B
        L = L_new

    s = float(np.std(residuos)) if residuos else float(np.std(vals) * 0.1)
    return {"L": L, "B": B, "alpha": alpha, "beta": beta, "s": s, "ultimo_anio": float(anios[-1]), "n": len(vals)}

def _holt_predict(modelo: Dict[str, Any], anios_hist: List[float], vals_hist: List[float], anio_target: float) -> float:
    h = anio_target - modelo["ultimo_anio"]
    return float(modelo["L"] + h * modelo["B"])


# 5. ARIMA(1,1,0)
def _arima_train(anios: np.ndarray, vals: np.ndarray) -> Dict[str, Any]:
    if len(vals) < 5:
        raise ValueError("ARIMA requiere al menos 5 observaciones.")
    diff = np.diff(vals)
    y = diff[1:]
    x_ = diff[:-1]
    phi = float(np.dot(x_, y) / np.dot(x_, x_)) if np.dot(x_, x_) > 0 else 0.0
    phi = max(-0.95, min(0.95, phi))
    residuos = y - phi * x_
    s = float(np.std(residuos)) if len(residuos) else float(np.std(diff))
    return {
        "phi": phi,
        "s": s,
        "last_val": float(vals[-1]),
        "last_diff": float(diff[-1]),
        "ultimo_anio": float(anios[-1]),
        "n": len(vals)
    }

def _arima_predict(modelo: Dict[str, Any], anios_hist: List[float], vals_hist: List[float], anio_target: float) -> float:
    h = int(anio_target - modelo["ultimo_anio"])
    phi = modelo["phi"]
    val = modelo["last_val"]
    d_prev = modelo["last_diff"]
    for _ in range(max(1, h)):
        d_next = phi * d_prev
        val += d_next
        d_prev = d_next
    return float(val)


# 6. SVR con kernel RBF (con StandardScaler dentro del pipeline)
def _svr_rbf_train(anios: np.ndarray, vals: np.ndarray) -> Dict[str, Any]:
    if HAS_SKLEARN:
        pipe = make_pipeline(StandardScaler(), SklearnSVR(kernel="rbf", C=1.0, epsilon=0.01))
        pipe.fit(anios.reshape(-1, 1), vals)
        residuos = vals - pipe.predict(anios.reshape(-1, 1))
        s = float(np.std(residuos))
        return {"tipo": "sklearn", "modelo": pipe, "s": s, "n": len(vals)}
    else:
        # Implementación de Ridge RBF con escalado estandarizado propio
        mu_x, std_x = float(np.mean(anios)), float(np.std(anios)) if np.std(anios) > 0 else 1.0
        mu_y, std_y = float(np.mean(vals)), float(np.std(vals)) if np.std(vals) > 0 else 1.0
        x_norm = (anios - mu_x) / std_x
        y_norm = (vals - mu_y) / std_y
        gamma = 0.5
        K = np.exp(-gamma * (x_norm[:, None] - x_norm[None, :]) ** 2)
        lam = 0.05
        alpha = np.linalg.solve(K + lam * np.eye(len(vals)), y_norm)
        pred_norm = K @ alpha
        pred_raw = pred_norm * std_y + mu_y
        s = float(np.std(vals - pred_raw))
        return {
            "tipo": "custom",
            "alpha": alpha.tolist(),
            "x_norm": x_norm.tolist(),
            "mu_x": mu_x, "std_x": std_x,
            "mu_y": mu_y, "std_y": std_y,
            "gamma": gamma,
            "s": s,
            "n": len(vals)
        }

def _svr_rbf_predict(modelo: Dict[str, Any], anios_hist: List[float], vals_hist: List[float], anio_target: float) -> float:
    if modelo.get("tipo") == "sklearn":
        return float(modelo["modelo"].predict(np.array([[anio_target]]))[0])
    else:
        x_target_norm = (anio_target - modelo["mu_x"]) / modelo["std_x"]
        x_norm_arr = np.array(modelo["x_norm"])
        k = np.exp(-modelo["gamma"] * (x_norm_arr - x_target_norm) ** 2)
        y_norm_hat = float(np.dot(modelo["alpha"], k))
        return float(y_norm_hat * modelo["std_y"] + modelo["mu_y"])


# 7. Random Forest Regressor
class _NumpyRandomForestRegressor:
    """Implementación pura en NumPy para Random Forest si sklearn no está disponible."""
    def __init__(self, n_estimators=50, max_depth=2, min_samples_split=4, min_samples_leaf=3, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.trees = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape
        self.trees = []
        for _ in range(self.n_estimators):
            idx = rng.choice(n_samples, size=n_samples, replace=True)
            tree = self._build_tree(X[idx], y[idx], depth=0, rng=rng)
            self.trees.append(tree)

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int, rng: np.random.RandomState):
        if depth >= self.max_depth or len(y) < self.min_samples_split:
            return {"leaf": True, "value": float(np.mean(y))}

        best_score = float("inf")
        best_split = None
        n_feat = X.shape[1]
        k_feat = max(1, int(math.sqrt(n_feat)))
        features = rng.choice(n_feat, size=k_feat, replace=False)

        for f in features:
            thresholds = np.unique(X[:, f])
            for th in thresholds:
                left_mask = X[:, f] <= th
                right_mask = ~left_mask
                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue
                y_l, y_r = y[left_mask], y[right_mask]
                var_score = np.sum((y_l - np.mean(y_l)) ** 2) + np.sum((y_r - np.mean(y_r)) ** 2)
                if var_score < best_score:
                    best_score = var_score
                    best_split = (f, th, left_mask, right_mask)

        if best_split is None:
            return {"leaf": True, "value": float(np.mean(y))}

        f, th, left_mask, right_mask = best_split
        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1, rng)
        right_tree = self._build_tree(X[right_mask], y[right_mask], depth + 1, rng)
        return {"leaf": False, "feature": f, "threshold": th, "left": left_tree, "right": right_tree}

    def _predict_row(self, node, row):
        if node["leaf"]:
            return node["value"]
        if row[node["feature"]] <= node["threshold"]:
            return self._predict_row(node["left"], row)
        return self._predict_row(node["right"], row)

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = []
        for row in X:
            tree_preds = [self._predict_row(t, row) for t in self.trees]
            preds.append(np.mean(tree_preds))
        return np.array(preds)


def _rf_train(anios: np.ndarray, vals: np.ndarray) -> Dict[str, Any]:
    X, y = _crear_dataset_lags(anios, vals)
    if len(y) < 5:
        raise ValueError("Observaciones insuficientes para Random Forest con lags.")

    if HAS_SKLEARN:
        modelo = SklearnRF(
            n_estimators=50,
            max_depth=2,
            min_samples_split=4,
            min_samples_leaf=3,
            max_features="sqrt",
            random_state=42
        )
    else:
        modelo = _NumpyRandomForestRegressor(
            n_estimators=50,
            max_depth=2,
            min_samples_split=4,
            min_samples_leaf=3,
            random_state=42
        )

    modelo.fit(X, y)
    # Compute residual standard deviation and sample size for confidence intervals
    preds = modelo.predict(X) if not HAS_SKLEARN else modelo.predict(X)
    residuals = y - preds
    s = float(np.std(residuals)) if residuals.size > 0 else 0.0
    n = len(y)
    return {"modelo": modelo, "s": s, "n": n}

def _rf_predict(modelo: Dict[str, Any], anios_hist: List[float], vals_hist: List[float], anio_target: float) -> float:
    x_feat = _crear_vector_features_paso(anios_hist, vals_hist, anio_target)
    y_hat = modelo["modelo"].predict(x_feat)[0]
    return float(y_hat)


# 8. XGBoost Regressor
class _NumpyXGBoostRegressor:
    """Implementación de Gradient Boosted Trees tipo XGBoost pura en NumPy."""
    def __init__(
        self,
        n_estimators=30,
        max_depth=2,
        learning_rate=0.05,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=1.0,
        random_state=42
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.trees = []
        self.base_pred = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape
        self.base_pred = float(np.mean(y))
        y_hat = np.full(n_samples, self.base_pred)
        self.trees = []

        for _ in range(self.n_estimators):
            residuals = y - y_hat
            sample_size = max(4, int(self.subsample * n_samples))
            sample_idx = rng.choice(n_samples, size=sample_size, replace=False)
            feat_size = max(1, int(self.colsample_bytree * n_features))
            feat_idx = rng.choice(n_features, size=feat_size, replace=False)

            tree = self._build_tree(
                X[sample_idx],
                residuals[sample_idx],
                depth=0,
                feat_idx=feat_idx,
                rng=rng
            )
            self.trees.append(tree)
            step_preds = np.array([self._predict_row(tree, row) for row in X])
            y_hat += self.learning_rate * step_preds

    def _build_tree(self, X: np.ndarray, residuals: np.ndarray, depth: int, feat_idx: np.ndarray, rng: np.random.RandomState):
        G = np.sum(residuals)
        H = len(residuals)

        if abs(G) < self.reg_alpha:
            leaf_weight = 0.0
        else:
            leaf_weight = float(np.sign(G) * (abs(G) - self.reg_alpha) / (H + self.reg_lambda))

        if depth >= self.max_depth or H < self.min_child_weight:
            return {"leaf": True, "value": leaf_weight}

        best_gain = 0.0
        best_split = None
        curr_score = (G ** 2) / (H + self.reg_lambda)

        for f in feat_idx:
            thresholds = np.unique(X[:, f])
            for th in thresholds:
                left_mask = X[:, f] <= th
                right_mask = ~left_mask
                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                if n_left < max(1, self.min_child_weight) or n_right < max(1, self.min_child_weight):
                    continue

                G_L, H_L = np.sum(residuals[left_mask]), n_left
                G_R, H_R = np.sum(residuals[right_mask]), n_right

                gain = 0.5 * (
                    (G_L ** 2) / (H_L + self.reg_lambda) +
                    (G_R ** 2) / (H_R + self.reg_lambda) -
                    curr_score
                )

                if gain > best_gain:
                    best_gain = gain
                    best_split = (f, th, left_mask, right_mask)

        if best_split is None:
            return {"leaf": True, "value": leaf_weight}

        f, th, left_mask, right_mask = best_split
        left_tree = self._build_tree(X[left_mask], residuals[left_mask], depth + 1, feat_idx, rng)
        right_tree = self._build_tree(X[right_mask], residuals[right_mask], depth + 1, feat_idx, rng)
        return {"leaf": False, "feature": f, "threshold": th, "left": left_tree, "right": right_tree}

    def _predict_row(self, node, row):
        if node["leaf"]:
            return node["value"]
        if row[node["feature"]] <= node["threshold"]:
            return self._predict_row(node["left"], row)
        return self._predict_row(node["right"], row)

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = np.full(len(X), self.base_pred)
        for tree in self.trees:
            step = np.array([self._predict_row(tree, row) for row in X])
            preds += self.learning_rate * step
        return preds


def _xgboost_train(anios: np.ndarray, vals: np.ndarray) -> Dict[str, Any]:
    X, y = _crear_dataset_lags(anios, vals)
    if len(y) < 5:
        raise ValueError("Observaciones insuficientes para XGBoost con lags.")

    if HAS_XGBOOST and HAS_SKLEARN:
        modelo = XGBRegressor(
            n_estimators=30,
            max_depth=2,
            learning_rate=0.05,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=1.0,
            random_state=42,
            objective="reg:squarederror"
        )
    else:
        modelo = _NumpyXGBoostRegressor(
            n_estimators=30,
            max_depth=2,
            learning_rate=0.05,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=1.0,
            random_state=42
        )

    modelo.fit(X, y)
    # Compute residual standard deviation and sample size for confidence intervals
    preds = modelo.predict(X)
    residuals = y - preds
    s = float(np.std(residuals)) if residuals.size > 0 else 0.0
    n = len(y)
    return {"modelo": modelo, "s": s, "n": n}

def _xgboost_predict(modelo: Dict[str, Any], anios_hist: List[float], vals_hist: List[float], anio_target: float) -> float:
    x_feat = _crear_vector_features_paso(anios_hist, vals_hist, anio_target)
    y_hat = modelo["modelo"].predict(x_feat)[0]
    return float(y_hat)


# ─────────────────────────────────────────────────────────────────────────────
# DEFINICIONES Y REGISTRO DE MODELOS
# ─────────────────────────────────────────────────────────────────────────────

def _obtener_definiciones_modelos():

    return [
        ("Naive", False, _naive_train, _naive_predict),
        ("Regresión Lineal", False, _lineal_train, _lineal_predict),
        ("Regresión Polinómica grado 2", False, _poly2_train, _poly2_predict),
        ("Holt", False, _holt_train, _holt_predict),
        ("ARIMA(1,1,0)", False, _arima_train, _arima_predict),
        ("SVR con kernel RBF", False, _svr_rbf_train, _svr_rbf_predict),
        ("Random Forest Regressor", True, _rf_train, _rf_predict),
        ("XGBoost Regressor", True, _xgboost_train, _xgboost_predict),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# VALIDACIÓN TEMPORAL (ROLLING-ORIGIN / EXPANDING WINDOW)
# ─────────────────────────────────────────────────────────────────────────────

def _validar_rolling(
    anios: np.ndarray,
    vals: np.ndarray,
    k: int,
    es_ml: bool,
    fn_entrenar: Any,
    fn_predecir_step: Any,
) -> Dict[str, Any]:

    n = len(anios)
    errores = []
    reales = []
    splits_validos = 0

    for i in range(k):
        idx_split = n - k + i
        min_requerido = 6 if es_ml else 4
        if idx_split < min_requerido:
            continue

        anios_tr = anios[:idx_split]
        vals_tr = vals[:idx_split]
        x_test = float(anios[idx_split])
        y_test = float(vals[idx_split])

        modelo = fn_entrenar(anios_tr, vals_tr)
        y_hat = fn_predecir_step(modelo, anios_tr.tolist(), vals_tr.tolist(), x_test)

        errores.append(float(y_hat) - y_test)
        reales.append(y_test)
        splits_validos += 1

    if splits_validos == 0:
        raise ValueError("No hubo suficientes observaciones para completar splits válidos.")

    errs_arr = np.array(errores)
    real_arr = np.array(reales)

    mae = float(np.mean(np.abs(errs_arr)))
    rmse = float(np.sqrt(np.mean(errs_arr ** 2)))

    media_real = float(np.mean(real_arr))
    ss_tot = float(np.sum((real_arr - media_real) ** 2))
    ss_res = float(np.sum(errs_arr ** 2))
    r2 = (1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else None

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Splits": splits_validos,
    }


# ─────────────────────────────────────────────────────────────────────────────
# INTERVALOS DE INCERTIDUMBRE
# ─────────────────────────────────────────────────────────────────────────────

def _calcular_intervalos(
    nombre_modelo: str,
    modelo: Dict[str, Any],
    anios_hist: List[float],
    vals_hist: List[float],
    anio_target: float,
    y_hat: float
) -> Tuple[Optional[float], Optional[float]]:

    if nombre_modelo in ("Random Forest Regressor", "XGBoost Regressor", "SVR con kernel RBF"):
        s = modelo.get("s", 0.0)
        n = modelo.get("n", len(vals_hist))
        se = s * math.sqrt(1.0 + 1.0 / n)
        return y_hat - 1.96 * se, y_hat + 1.96 * se

    if nombre_modelo == "Naive":
        s = modelo.get("s", 0.0)
        n = modelo.get("n", len(vals_hist))
        se = s * math.sqrt(1.0 + 1.0 / n)
        return y_hat - 1.96 * se, y_hat + 1.96 * se

    if nombre_modelo == "Regresión Lineal":
        s = modelo["s"]
        n = modelo["n"]
        mx = modelo["mean_x"]
        ssx = modelo["ssx"] if modelo["ssx"] > 0 else 1e-9
        se = s * math.sqrt(1.0 + 1.0 / n + (anio_target - mx) ** 2 / ssx)
        return y_hat - 1.96 * se, y_hat + 1.96 * se

    if nombre_modelo == "Regresión Polinómica grado 2":
        s = modelo["s"]
        n = modelo["n"]
        se = s * math.sqrt(1.0 + 1.0 / n)
        return y_hat - 1.96 * se, y_hat + 1.96 * se

    if nombre_modelo == "Holt":
        s = modelo["s"]
        h = max(1.0, anio_target - modelo["ultimo_anio"])
        se = s * math.sqrt(h)
        return y_hat - 1.96 * se, y_hat + 1.96 * se

    if nombre_modelo == "ARIMA(1,1,0)":
        s = modelo["s"]
        h = max(1.0, anio_target - modelo["ultimo_anio"])
        se = s * math.sqrt(h)
        return y_hat - 1.96 * se, y_hat + 1.96 * se

    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES AUXILIARES
# ─────────────────────────────────────────────────────────────────────────────

def _limpiar(serie: List[Dict[str, Any]], campo_valor: str, campo_anio: str) -> List[Dict[str, Any]]:
    result = []
    for d in serie:
        anio = d.get(campo_anio)
        val = d.get(campo_valor)
        if anio is None or val is None:
            continue
        try:
            v = float(val)
            if math.isnan(v) or math.isinf(v):
                continue
            result.append({"Año": int(anio), "Valor": v})
        except (TypeError, ValueError):
            continue
    result.sort(key=lambda x: x["Año"])
    return result


def _ajustar_ols(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mx = float(np.mean(x))
    my = float(np.mean(y))
    cov = float(np.dot(x - mx, y - my))
    var = float(np.dot(x - mx, x - mx))
    if var == 0:
        return 0.0, my
    m = cov / var
    return m, my - m * mx


def _pendiente_lineal(anios: np.ndarray, vals: np.ndarray) -> float:
    m, _ = _ajustar_ols(anios, vals)
    return float(m)


def _resultado_invalido(datos: List[Dict[str, Any]], mensaje: str) -> Dict[str, Any]:
    return {
        "valido": False,
        "mensaje": mensaje,
        "historico": datos,
        "predicciones": [],
        "metricas": {
            "MAE": None,
            "RMSE": None,
            "R2": None,
            "Modelo": "No disponible",
            "Pendiente_anual": 0.0
        },
        "modelos_evaluados": [],
        "mensaje_seleccion": mensaje,
    }
