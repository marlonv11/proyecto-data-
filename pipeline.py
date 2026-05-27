from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm

import io
import base64
import warnings

warnings.filterwarnings("ignore")

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    accuracy_score
)

app = FastAPI(
    title="Informalidad Laboral DANE",
    version="1.0"
)

templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# =============================================================================
# CARGA Y PREPARACIÓN DE DATOS
# =============================================================================
def preparar_datos():
    cols_caract = ["DIRECTORIO", "ORDEN", "SECUENCIA_P", "HOGAR", "P3271", "P6040", "P6080", "P3042", "P6070"]
    cols_ocup = ["DIRECTORIO", "ORDEN", "SECUENCIA_P", "HOGAR", "P6430", "P3069", "INGLABO", "RAMA2D_R4", "CLASE", "DPTO", "P6800"]

    caract1 = pd.read_csv("Base_Datos_Proyecto/Caracteristicas/Octubre_Características.CSV", encoding="latin-1", sep=";", usecols=cols_caract)
    caract2 = pd.read_csv("Base_Datos_Proyecto/Caracteristicas/Noviembre_Características.CSV", encoding="latin-1", sep=";", usecols=cols_caract)
    caract3 = pd.read_csv("Base_Datos_Proyecto/Caracteristicas/Diciembre_Características.CSV", encoding="latin-1", sep=";", usecols=cols_caract)

    ocup1 = pd.read_csv("Base_Datos_Proyecto/Ocupados/Octubre_Ocupados.CSV", encoding="latin-1", sep=";", low_memory=False, usecols=cols_ocup)
    ocup2 = pd.read_csv("Base_Datos_Proyecto/Ocupados/Noviembre_Ocupados.CSV", encoding="latin-1", sep=";", low_memory=False, usecols=cols_ocup)
    ocup3 = pd.read_csv("Base_Datos_Proyecto/Ocupados/Diciembre_Ocupados.CSV", encoding="latin-1", sep=";", low_memory=False, usecols=cols_ocup)

    claves = ["DIRECTORIO", "SECUENCIA_P", "HOGAR", "ORDEN"]
    oct_ = ocup1.merge(caract1, on=claves, how="left", validate="one_to_one")
    nov_ = ocup2.merge(caract2, on=claves, how="left", validate="one_to_one")
    dic_ = ocup3.merge(caract3, on=claves, how="left", validate="one_to_one")

    oct_["MES"] = "Octubre"
    nov_["MES"] = "Noviembre"
    dic_["MES"] = "Diciembre"

    data = pd.concat([oct_, nov_, dic_], ignore_index=True)

    rename = {
        "P6430": "POSICION_TRABAJO", "P3069": "CANTIDAD_EMPLEADOS", "RAMA2D_R4": "RAMA_ECONOMICA",
        "CLASE": "CLASE", "DPTO": "DEPARTAMENTO", "P6800": "HORAS_SEMANA", "P3271": "SEXO",
        "P6040": "EDAD", "P6080": "NIVEL_EDUCATIVO", "P3042": "GRADO_EDUC", "P6070": "SITUACION_SENTIMENTAL"
    }
    data.rename(columns=rename, inplace=True)

    data["EDAD2"] = data["EDAD"] ** 2

    cond_sector = [
        ((data["RAMA_ECONOMICA"] >= 1) & (data["RAMA_ECONOMICA"] <= 9)),
        (((data["RAMA_ECONOMICA"] >= 10) & (data["RAMA_ECONOMICA"] <= 33)) |
         ((data["RAMA_ECONOMICA"] >= 35) & (data["RAMA_ECONOMICA"] <= 39)) |
         ((data["RAMA_ECONOMICA"] >= 41) & (data["RAMA_ECONOMICA"] <= 43))),
        (data["RAMA_ECONOMICA"] >= 44)
    ]
    data["SECTOR_EMPRESA"] = np.select(cond_sector, [1, 2, 3], default=0)

    pos_informal = [1, 3, 4, 5, 6, 7, 8]
    data["INFORMAL"] = np.where((data["POSICION_TRABAJO"].isin(pos_informal)) & (data["CANTIDAD_EMPLEADOS"] <= 3), 1, 0)
    data.loc[data["POSICION_TRABAJO"] == 2, "INFORMAL"] = 0
    data.loc[(data["POSICION_TRABAJO"] == 4) & (data["NIVEL_EDUCATIVO"].between(10, 13)), "INFORMAL"] = 0

    data["GENERO"] = np.where(data["SEXO"] == 1, 0, 1)
    data["AREA_VIVIENDA"] = np.where(data["CLASE"] == 1, 0, 1)

    cond_educ = [
        (data["NIVEL_EDUCATIVO"].isin([1, 99])),
        (data["NIVEL_EDUCATIVO"].between(2, 4)),
        (data["NIVEL_EDUCATIVO"].between(5, 7)),
        (data["NIVEL_EDUCATIVO"].between(8, 10)),
        (data["NIVEL_EDUCATIVO"].between(11, 13))
    ]
    data["EDUCACION"] = np.select(cond_educ, [1, 2, 3, 4, 5])

    data["ESTADO_CIVIL"] = np.nan
    data.loc[data["SITUACION_SENTIMENTAL"].between(1, 3), "ESTADO_CIVIL"] = 1
    data.loc[data["SITUACION_SENTIMENTAL"].between(4, 5), "ESTADO_CIVIL"] = 2
    data.loc[data["SITUACION_SENTIMENTAL"] == 6, "ESTADO_CIVIL"] = 3

    data["INGLABO_IMP"] = data["INGLABO"]
    media_grupo = data.groupby("POSICION_TRABAJO")["INGLABO"].transform("mean")
    data.loc[data["POSICION_TRABAJO"].isin([1, 2, 3]) & data["INGLABO"].isna(), "INGLABO_IMP"] = media_grupo
    data.loc[data["POSICION_TRABAJO"].isin([4, 5, 6]), "INGLABO_IMP"] = 0

    data = data[data["SECTOR_EMPRESA"] != 0]

    data["MES"] = pd.Categorical(data["MES"], categories=["Octubre", "Noviembre", "Diciembre"], ordered=True)

    df = data[[
        "INFORMAL", "INGLABO_IMP", "HORAS_SEMANA", "EDAD", "EDAD2",
        "SECTOR_EMPRESA", "GENERO", "AREA_VIVIENDA", "EDUCACION", "ESTADO_CIVIL", "MES"
    ]].dropna()

    return df

# =============================================================================
# ENTRENAMIENTO
# =============================================================================
def entrenar_modelo(df):
    df_model = df.drop(columns=["MES"]).dropna()
    Y = df_model["INFORMAL"].astype(int)
    X = df_model.drop(columns=["INFORMAL"])

    CAT_VARS = ["SECTOR_EMPRESA", "GENERO", "AREA_VIVIENDA", "EDUCACION", "ESTADO_CIVIL"]
    X = pd.get_dummies(X, columns=CAT_VARS, drop_first=True)
    X = sm.add_constant(X).astype(float)

    result = sm.Logit(Y, X).fit(disp=0)
    return result, X, Y

print("Cargando datos y entrenando modelo...")
try:
    DF = preparar_datos()
    MODEL, X_MODEL, Y_MODEL = entrenar_modelo(DF)
    print(f"Listo. Observaciones: {len(DF)}")
except Exception as e:
    print(f"Error al cargar datos: {e}")
    DF, MODEL, X_MODEL, Y_MODEL = None, None, None, None

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="none")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded

# =============================================================================
# ENDPOINTS
# =============================================================================
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "dashboard.html")

@app.get("/api")
def api_status():
    return {"status": "ok", "mensaje": "API Informalidad Laboral DANE activa"}

# ── /metricas ─────────────────────────────────────────────────────────────────
@app.get("/metricas")
def metricas():
    if MODEL is None:
        return JSONResponse({"error": "Datos no disponibles"}, status_code=500)

    preds_prob = MODEL.predict(X_MODEL)
    preds      = (preds_prob >= 0.5).astype(int)
    auc        = roc_auc_score(Y_MODEL, preds_prob)
    acc        = accuracy_score(Y_MODEL, preds)
    cm         = confusion_matrix(Y_MODEL, preds).tolist()

    mfx = MODEL.get_margeff()
    ci  = mfx.conf_int()   # shape (n_vars, 2)

    efectos = []
    for i, (var, me, se, z, pval) in enumerate(zip(
        X_MODEL.columns,
        mfx.margeff,
        mfx.margeff_se,
        mfx.tvalues,
        mfx.pvalues,
    )):
        efectos.append({
            "variable":      str(var),
            "efecto":        float(me),
            "std_err":       float(se),
            "z_stat":        round(float(z),    3),
            "pvalor":        round(float(pval), 4),
            "ci_low":        float(ci[i, 0]),
            "ci_high":       float(ci[i, 1]),
            "significativa": bool(pval < 0.05),
        })

    return {
        "n_obs":              int(len(Y_MODEL)),
        "pseudo_r2":          round(float(MODEL.prsquared), 4),
        "accuracy":           round(float(acc), 4),
        "auc":                round(float(auc), 4),
        "efectos_marginales": efectos,
        "matriz_confusion": {
            "verdaderos_negativos": cm[0][0],
            "falsos_positivos":     cm[0][1],
            "falsos_negativos":     cm[1][0],
            "verdaderos_positivos": cm[1][1],
        },
        "tasa_informalidad": round(float(DF["INFORMAL"].mean()), 4),
        "por_mes": DF.groupby("MES", observed=False)["INFORMAL"].mean().round(4).to_dict(),
    }

# ── /graficas ─────────────────────────────────────────────────────────────────
@app.get("/graficas")
def obtener_graficas():
    if DF is None or MODEL is None:
        return JSONResponse({"error": "Datos no disponibles"}, status_code=500)

    plt.style.use('dark_background')
    matplotlib.rcParams['text.color']       = '#b5c7f3'
    matplotlib.rcParams['axes.labelcolor']  = '#b5c7f3'
    matplotlib.rcParams['xtick.color']      = '#8ea4d2'
    matplotlib.rcParams['ytick.color']      = '#8ea4d2'

    graficas = {}

    # 1. Ingreso por mes
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.boxplot(data=DF, x="MES", y="INGLABO_IMP", ax=ax, color="#1b2c50")
    ax.set_title("Distribución del Ingreso Laboral", pad=15)
    graficas["ingreso_mes"] = fig_to_b64(fig)

    # 2. Horas semana
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(DF["HORAS_SEMANA"], bins=30, kde=True, ax=ax, color="#4c8dff")
    ax.set_title("Distribución de Horas Semanales", pad=15)
    graficas["horas_semana"] = fig_to_b64(fig)

    # 3. Educación
    edu = DF.groupby("EDUCACION", observed=False)["INFORMAL"].mean()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    edu.plot(kind="bar", ax=ax, color="#ff6f61")
    ax.set_title("Proporción de Informalidad por Nivel Educativo", pad=15)
    graficas["educacion"] = fig_to_b64(fig)

    # 4. Edad
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.boxplot(data=DF, x="INFORMAL", y="EDAD", ax=ax, palette=["#17233d", "#1b2c50"])
    ax.set_title("Edad por Condición de Informalidad", pad=15)
    graficas["edad"] = fig_to_b64(fig)

    # 5. ROC
    probs = MODEL.predict(X_MODEL)
    fpr, tpr, _ = roc_curve(Y_MODEL, probs)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(fpr, tpr, color="#4c8dff", lw=2, label=f"AUC = {roc_auc_score(Y_MODEL, probs):.2f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#666666")
    ax.set_title("Curva ROC del Modelo Logit", pad=15)
    ax.legend(loc="lower right")
    graficas["roc"] = fig_to_b64(fig)

    # 6. Sector
    sector = DF.groupby("SECTOR_EMPRESA", observed=False)["INFORMAL"].mean()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sector.plot(kind="bar", ax=ax, color="#9cb4ea")
    ax.set_title("Tasa de Informalidad según Sector Económico", pad=15)
    graficas["sector"] = fig_to_b64(fig)

    return graficas