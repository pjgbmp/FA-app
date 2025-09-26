# app.py
import json, math, sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# ========================
# CONFIG
# ========================
st.set_page_config(
    page_title="Análisis Fundamental – Toolkit",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Compatibilidad cache (Streamlit antiguo)
if hasattr(st, "cache_data"):
    cache_deco = st.cache_data
else:
    cache_deco = st.cache

# ========================
# THEME / CSS
# ========================
def apply_theme(dark: bool):
    bg = "#0f1116" if dark else "#ffffff"
    text = "#e6e6e6" if dark else "#111111"
    card = "#171a21" if dark else "#f6f8fc"
    border = "#2a2f3a" if dark else "#e1e5f2"
    st.markdown(
        f"""
        <style>
        .block-container {{ padding-top: 1.0rem; padding-bottom: 2rem; }}
        .kpi-card {{
            background:{card}; border:1px solid {border};
            border-radius:14px; padding:0.9rem 1rem 0.6rem 1rem; margin-bottom:0.8rem; color:{text};
        }}
        .kpi-title {{ font-size:0.9rem; opacity:0.8; }}
        .kpi-value {{ font-size:1.4rem; font-weight:700; }}
        .soft-divider {{ height:1px; background:{border}; margin:12px 0 8px 0; }}
        .tag {{
            display:inline-block; padding:2px 8px; border-radius:999px;
            font-size:0.75rem; border:1px solid {border}; margin-right:6px; margin-bottom:6px; color:{text};
        }}
        .card {{
            background:{card}; padding:14px; border-radius:16px; border:1px solid {border}; margin-bottom: 12px; color:{text};
        }}
        .help {{
            display:inline-block; font-size:0.78rem; margin-left:6px; opacity:0.7;
            border:1px solid {border}; border-radius:999px; padding:0 6px; cursor:help;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    return "plotly_dark" if dark else "plotly"

def with_tooltip(label: str, desc: str) -> str:
    return f'{label}<span class="help" title="{desc}">ⓘ</span>'

# ========================
# GLOSARIO (siglas)
# ========================
ACRONYMS = {
    "LTM": "Last Twelve Months: últimos 12 meses.",
    "NTM": "Next Twelve Months: próximos 12 meses (estimados).",
    "FCF": "Free Cash Flow (Flujo de Caja Libre) = CFO – CapEx.",
    "CFO": "Cash Flow from Operations (Flujo de caja de operaciones).",
    "CapEx": "Capital Expenditures (Gastos de capital).",
    "EBIT": "Earnings Before Interest and Taxes (Utilidad operativa).",
    "EBITDA": "EBIT + Depreciación y Amortización.",
    "EV": "Enterprise Value = Market Cap + Deuda Neta + MI + Pref – Caja no operativa.",
    "ROIC": "Return on Invested Capital = NOPAT / Capital invertido.",
    "ROE": "Return on Equity = Utilidad Neta / Patrimonio.",
    "ROA": "Return on Assets = Utilidad Neta / Activos totales.",
    "WACC": "Weighted Average Cost of Capital (Costo promedio ponderado de capital).",
    "DSO": "Days Sales Outstanding (días de cobro).",
    "DIO": "Days Inventory Outstanding (días de inventario).",
    "DPO": "Days Payable Outstanding (días de pago a proveedores).",
    "SBC": "Stock-Based Compensation (compensación en acciones).",
    "MI": "Minority Interest (participación no controladora).",
    "DCF": "Discounted Cash Flow (Descuento de Flujos de Caja).",
    "DDM": "Dividend Discount Model (Modelo de Descuento de Dividendos).",
}

# ========================
# HELPERS
# ========================
def human_int(n):
    try:
        if n is None or (isinstance(n, float) and (math.isnan(n) or math.isinf(n))):
            return "—"
        abs_n = abs(n); sign = "-" if n < 0 else ""
        if abs_n >= 1_000_000_000_000: return f"{sign}{abs_n/1_000_000_000_000:.2f}T"
        if abs_n >= 1_000_000_000: return f"{sign}{abs_n/1_000_000_000:.2f}B"
        if abs_n >= 1_000_000: return f"{sign}{abs_n/1_000_000:.2f}M"
        if abs_n >= 1_000: return f"{sign}{abs_n/1_000:.2f}K"
        return f"{n:,.0f}"
    except Exception:
        return "—"

def pct(x, decimals=1):
    try:
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            return "—"
        return f"{100*x:.{decimals}f}%"
    except Exception:
        return "—"

def safe_get(series_or_val):
    try:
        if isinstance(series_or_val, pd.Series):
            s = series_or_val.dropna()
            if s.empty: return None
            return float(s.iloc[0])
        return None if pd.isna(series_or_val) else float(series_or_val)
    except Exception:
        return None

def bs_get(bs: pd.DataFrame, key: str):
    try:
        if isinstance(bs, pd.DataFrame) and not bs.empty:
            if key in bs.index:
                return safe_get(bs.loc[key])
        aliases = {
            "Total Liabilities Net Minority Interest": ["Total Liabilities Net Minority Interest","Total Liab","Total liabilities"],
            "Total Stockholder Equity": ["Total Stockholder Equity","Total Stockholders Equity","Stockholders' Equity"],
            "Cash And Cash Equivalents": ["Cash And Cash Equivalents","Cash","Cash & Cash Equivalents"],
            "Treasury Stock": ["Treasury Stock"],
            "Preferred Stock": ["Preferred Stock"],
            "Retained Earnings": ["Retained Earnings","Accumulated Deficit"],
            "Minority Interest": ["Minority Interest","Total Minority Interest"],
            "Total Debt": ["Total Debt","Short Long Term Debt","Long Term Debt"]
        }
        for a in aliases.get(key, []):
            if a in bs.index:
                return safe_get(bs.loc[a])
        return None
    except Exception:
        return None

def fin_get(fin: pd.DataFrame, key: str):
    try:
        if isinstance(fin, pd.DataFrame) and not fin.empty and key in fin.index:
            return safe_get(fin.loc[key])
        aliases = {
            "Net Income": ["Net Income","Net Income Applicable To Common Shares"],
            "EBIT": ["EBIT","Ebit","Operating Income"],
            "Interest Expense": ["Interest Expense","Interest Expense Non Operating"],
            "Gross Profit": ["Gross Profit"],
            "Total Revenue": ["Total Revenue","Revenue"],
            "Depreciation": ["Reconciled Depreciation","Depreciation","Depreciation & Amortization"]
        }
        for a in aliases.get(key, []):
            if isinstance(fin, pd.DataFrame) and a in fin.index:
                return safe_get(fin.loc[a])
        return None
    except Exception:
        return None

def cf_get(cf: pd.DataFrame, key: str):
    try:
        if isinstance(cf, pd.DataFrame) and not cf.empty and key in cf.index:
            return safe_get(cf.loc[key])
        aliases = {
            "Total Cash From Operating Activities": ["Total Cash From Operating Activities","Operating Cash Flow"],
            "Capital Expenditures": ["Capital Expenditures","Investments In Property Plant And Equipment"]
        }
        for a in aliases.get(key, []):
            if isinstance(cf, pd.DataFrame) and a in cf.index:
                return safe_get(cf.loc[a])
        return None
    except Exception:
        return None

# ========================
# FETCH DATOS
# ========================
@cache_deco(show_spinner=False)
def fetch_all(ticker: str) -> Dict[str, Any]:
    t = yf.Ticker(ticker)
    info = {}
    try:
        info = t.info or {}
        if not info and hasattr(t, "get_info"):
            info = t.get_info() or {}
    except Exception:
        info = {}
    try:
        hist = t.history(period="5y", interval="1d")
    except Exception:
        hist = pd.DataFrame()
    def safe_df(getter):
        try:
            df = getter
            return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
        except Exception:
            return pd.DataFrame()
    balance_a = safe_df(t.balance_sheet)
    balance_q = safe_df(t.quarterly_balance_sheet)
    fin_a = safe_df(t.financials)
    fin_q = safe_df(t.quarterly_financials)
    cf_a = safe_df(t.cashflow)
    cf_q = safe_df(t.quarterly_cashflow)
    try:
        rec = t.recommendations
    except Exception:
        rec = None
    return {"info": info, "hist": hist, "bs_a": balance_a, "bs_q": balance_q,
            "fin_a": fin_a, "fin_q": fin_q, "cf_a": cf_a, "cf_q": cf_q, "rec": rec}

def get_price_now(ticker: str):
    try:
        t = yf.Ticker(ticker)
        fi = getattr(t, "fast_info", None)
        if fi and "last_price" in fi:
            return float(fi["last_price"])
        h = t.history(period="1d", interval="1m")
        if not h.empty:
            return float(h["Close"].iloc[-1])
    except Exception:
        pass
    return None

# ========================
# CÁLCULOS
# ========================
def compute_margins(fin: pd.DataFrame):
    revenue = fin_get(fin, "Total Revenue")
    gross = fin_get(fin, "Gross Profit")
    ebit = fin_get(fin, "EBIT") or fin_get(fin, "Operating Income")
    net = fin_get(fin, "Net Income")
    return {
        "gross_margin": (gross / revenue) if (gross and revenue) else None,
        "ebit_margin": (ebit / revenue) if (ebit and revenue) else None,
        "net_margin": (net / revenue) if (net and revenue) else None,
        "revenue": revenue, "gross": gross, "ebit": ebit, "net": net
    }

def compute_fcf(cf: pd.DataFrame):
    cfo = cf_get(cf, "Total Cash From Operating Activities")
    capex = cf_get(cf, "Capital Expenditures")
    if cfo is None or capex is None:
        return None, cfo, capex
    return cfo + capex, cfo, capex  # CapEx suele negativo

def compute_net_debt(bs: pd.DataFrame):
    debt = bs_get(bs, "Total Debt") or 0.0
    cash = bs_get(bs, "Cash And Cash Equivalents") or 0.0
    return debt - cash, debt, cash

def compute_ev(info: dict, bs: pd.DataFrame):
    mc = info.get("marketCap") or info.get("market_cap")
    ev_info = info.get("enterpriseValue")
    if ev_info:
        try: return float(ev_info)
        except Exception: pass
    net_debt, _, _ = compute_net_debt(bs)
    minority = bs_get(bs, "Minority Interest") or 0.0
    pref = bs_get(bs, "Preferred Stock") or 0.0
    if mc is None:
        return None
    try:
        return float(mc + (net_debt or 0) + minority + pref)
    except Exception:
        return None

def compute_score_blocks(info, bs_a, fin_a, cf_a):
    net_debt, debt, cash = compute_net_debt(bs_a)
    cash_gt_debt = bool(cash is not None and debt is not None and cash > debt)
    liab = bs_get(bs_a, "Total Liabilities Net Minority Interest") or bs_get(bs_a, "Total Liab")
    equity = bs_get(bs_a, "Total Stockholder Equity")
    liab_equity_ok = bool(liab is not None and equity not in (None, 0) and (liab / equity < 0.80))
    pref = bs_get(bs_a, "Preferred Stock")
    no_pref = bool(pref is None or abs(pref) < 1e-6)
    treasury = bs_get(bs_a, "Treasury Stock")
    buybacks = bool(treasury is not None and treasury < 0)
    re = bs_get(bs_a, "Retained Earnings")
    retained_pos = bool(re is not None and re > 0)
    fcf, _, _ = compute_fcf(cf_a)
    fcf_pos = bool(fcf is not None and fcf > 0)
    margins = compute_margins(fin_a)
    net_margin_pos = bool(margins["net_margin"] is not None and margins["net_margin"] > 0)
    checks = [
        ("Caja > Deuda", cash_gt_debt),
        ("Pasivos/Patrimonio < 0.80", liab_equity_ok),
        ("Sin acciones preferentes", no_pref),
        ("Recompras (Treasury < 0)", buybacks),
        ("Retained Earnings > 0", retained_pos),
        ("FCF positivo", fcf_pos),
        ("Margen neto positivo", net_margin_pos),
    ]
    score = sum(1 for _, ok in checks if ok)
    return checks, score, len(checks)

def capm_cost_of_equity(info: dict, rf=0.03, mrp=0.05):
    beta = info.get("beta", None)
    try:
        return rf + float(beta) * mrp if beta is not None else 0.08
    except Exception:
        return 0.08

# DCF helpers
def dcf_project_flows(base_fcf: float, years: int, growth: float) -> list:
    flows = []
    f = base_fcf
    for t in range(1, years + 1):
        f = (base_fcf if t == 1 else f) * (1 + growth)
        flows.append(f)
    return flows

def dcf_value(base_fcf: float, years: int, growth: float, wacc: float, g_term: float) -> Tuple[float, float, float]:
    """Devuelve EV, PV_terminal, suma_flujos_PV; None si no se puede calcular."""
    if not base_fcf or base_fcf <= 0 or wacc is None or g_term is None or years <= 0:
        return None, None, None
    if wacc <= g_term:
        return None, None, None
    flows = dcf_project_flows(base_fcf, years, growth)
    disc = [(1 + wacc) ** t for t in range(1, years + 1)]
    pv_flows = [flows[i] / disc[i] for i in range(years)]
    terminal = (flows[-1] * (1 + g_term)) / (wacc - g_term)
    pv_term = terminal / ((1 + wacc) ** years)
    ev = sum(pv_flows) + pv_term
    return ev, pv_term, sum(pv_flows)

def dcf_equity_fairprice(base_fcf: float, years: int, growth: float, wacc: float, g_term: float,
                         net_debt: float, shares: float) -> float:
    ev, pv_term, pv_sum = dcf_value(base_fcf, years, growth, wacc, g_term)
    if ev is None or shares in (None, 0):
        return np.nan
    equity = ev - (net_debt or 0.0)
    return equity / shares

# ========================
# HISTÓRICOS
# ========================
def series_from_df(df: pd.DataFrame, row: str) -> pd.Series:
    if df is None or df.empty or row not in df.index:
        return pd.Series(dtype=float)
    s = df.loc[row].dropna()[::-1]
    try:
        s.index = pd.to_datetime(s.index)
    except Exception:
        s.index = pd.Index(s.index)
    return s

def build_historical(fin_a, fin_q, cf_a, cf_q) -> Dict[str, pd.Series]:
    hist = {}
    hist["Revenue_a"] = series_from_df(fin_a, "Total Revenue")
    hist["Gross_a"] = series_from_df(fin_a, "Gross Profit")
    ebit_a = series_from_df(fin_a, "EBIT")
    if ebit_a.empty:
        ebit_a = series_from_df(fin_a, "Operating Income")
    hist["EBIT_a"] = ebit_a
    hist["Net_a"] = series_from_df(fin_a, "Net Income")
    cfo_a = series_from_df(cf_a, "Total Cash From Operating Activities")
    capex_a = series_from_df(cf_a, "Capital Expenditures")
    if not cfo_a.empty and not capex_a.empty:
        hist["FCF_a"] = (cfo_a + capex_a).rename("Free Cash Flow")
    hist["Revenue_q"] = series_from_df(fin_q, "Total Revenue")
    hist["Net_q"] = series_from_df(fin_q, "Net Income")
    return hist

def margins_from_series(rev: pd.Series, gp: pd.Series, ebit: pd.Series, net: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame(index=rev.index)
    if not gp.empty: df["Gross %"] = (gp / rev).replace([np.inf,-np.inf], np.nan)
    if not ebit.empty: df["EBIT %"] = (ebit / rev).replace([np.inf,-np.inf], np.nan)
    if not net.empty: df["Net %"] = (net / rev).replace([np.inf,-np.inf], np.nan)
    return df

# ========================
# GUARDADO
# ========================
DB_PATH = Path("analyses_store.json")
def load_db() -> Dict[str, Any]:
    if not DB_PATH.exists(): return {}
    try: return json.loads(DB_PATH.read_text(encoding="utf-8"))
    except Exception: return {}
def save_db(db: Dict[str, Any]):
    try: DB_PATH.write_text(json.dumps(db, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e: st.warning(f"No se pudo guardar: {e}")
def snapshot_for_ticker(ticker: str, dcf_out: dict, ddm_out: dict, score_out: dict, notes: str):
    db = load_db()
    db[ticker.upper()] = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "dcf": dcf_out, "ddm": ddm_out, "score": score_out, "notes": notes,
    }
    save_db(db)

# ========================
# SIDEBAR
# ========================
with st.sidebar:
    st.markdown("## ⚙️ Configuración")
    ticker = st.text_input("Ticker principal", value="AAPL").strip().upper()
    peers_input = st.text_input("Peers (separados por comas)", value="MSFT, GOOGL, AMZN")
    peer_list = [p.strip().upper() for p in peers_input.split(",") if p.strip()]
    st.markdown("---")
    if hasattr(st, "toggle"):
        dark_mode = st.toggle("🌙 Modo oscuro (beta visual)", value=True)
    else:
        dark_mode = st.checkbox("🌙 Modo oscuro (beta visual)", value=True)
    plotly_theme = apply_theme(dark_mode)
    st.markdown("---")
    with st.expander("🧪 Diagnóstico"):
        import streamlit, plotly
        st.write({"python": sys.version.split()[0],
                  "streamlit": streamlit.__version__,
                  "yfinance": getattr(yf, "__version__", "unknown"),
                  "plotly": plotly.__version__})
    with st.expander("ℹ️ Glosario rápido (siglas)"):
        for k, v in ACRONYMS.items():
            st.markdown(f"- **{k}**: {v}")

if not ticker:
    st.info("Introduce un ticker para comenzar.")
    st.stop()

# ========================
# CARGA
# ========================
data = fetch_all(ticker)
info = data["info"]; bs_a, fin_a, cf_a = data["bs_a"], data["fin_a"], data["cf_a"]
price_now = get_price_now(ticker)
target_mean = info.get("targetMeanPrice")
recommendations = data["rec"]

# HEADER
st.markdown(
    f"""
    <h2 style="margin-bottom:0.1rem">📊 Análisis Fundamental – {ticker}</h2>
    <div class="tag">Sector: {info.get('sector','—')}</div>
    <div class="tag">Industria: {info.get('industry','—')}</div>
    <div class="tag">País: {info.get('country','—')}</div>
    """,
    unsafe_allow_html=True,
)

# ========================
# TABS (7)
# ========================
tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["0. Perfil", "1. Análisis absoluto", "2. Análisis relativo", "3. Score subjetivo",
     "4. Valoración FCF (DCF)", "5. Valoración Dividendos (DDM)", "6. Detalles & notas"]
)

# ---- TAB 0 (Perfil & Recomendaciones SOLO aquí)
with tab0:
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.markdown("### 🧾 Perfil y precio")
        c11, c12, c13, c14 = st.columns(4)
        c11.markdown(f'<div class="kpi-card"><div class="kpi-title">Precio actual</div><div class="kpi-value">{("$"+format(price_now, ".2f")) if price_now else "—"}</div></div>', unsafe_allow_html=True)
        c12.markdown(f'<div class="kpi-card"><div class="kpi-title">Target analistas</div><div class="kpi-value">{("$"+format(target_mean, ".2f")) if target_mean else "—"}</div></div>', unsafe_allow_html=True)
        mc = info.get("marketCap")
        c13.markdown(f'<div class="kpi-card"><div class="kpi-title">Market Cap</div><div class="kpi-value">{human_int(mc)}</div></div>', unsafe_allow_html=True)
        c14.markdown(f'<div class="kpi-card"><div class="kpi-title">Empleados</div><div class="kpi-value">{human_int(info.get("fullTimeEmployees"))}</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)
        st.markdown(f"**Nombre:** {info.get('longName', '—')}")
        st.markdown(f"**Moneda:** {info.get('currency', '—')} &nbsp;&nbsp; **Bolsa:** {info.get('exchange', '—')}")
        st.markdown(f"**Website:** {info.get('website','—')}")
        st.markdown("**Descripción del negocio**")
        st.write(info.get("longBusinessSummary", "—"))

        st.markdown("### 🧠 Opiniones recientes de analistas")
        try:
            recs = recommendations
            if isinstance(recs, pd.DataFrame) and not recs.empty:
                tail = recs.tail(10).copy()
                fecha_col_set = False
                if isinstance(tail.index, (pd.DatetimeIndex, pd.PeriodIndex)):
                    tail["Fecha"] = pd.to_datetime(tail.index, errors="coerce").strftime("%Y-%m-%d")
                    fecha_col_set = True
                else:
                    for cand in ["date", "Date", "Datetime", "datetime"]:
                        if cand in tail.columns:
                            tail["Fecha"] = pd.to_datetime(tail[cand], errors="coerce").dt.strftime("%Y-%m-%d")
                            fecha_col_set = True
                            break
                posibles = ["To Grade","ToGrade","toGrade","From Grade","FromGrade","fromGrade","Action","action","Firm","firm"]
                cols_exist = [c for c in posibles if c in tail.columns]
                if fecha_col_set:
                    view = tail[["Fecha"] + cols_exist] if cols_exist else tail[["Fecha"]]
                else:
                    view = tail[cols_exist] if cols_exist else tail.reset_index(drop=True)
                st.dataframe(view, use_container_width=True, hide_index=True)
            else:
                st.write("— No hay datos recientes de recomendaciones —")
        except Exception as e:
            st.caption(f"Recomendaciones no disponibles ({e})")

    with c2:
        st.markdown("### 🕯️ Gráfico de velas")
        period = st.selectbox("Periodo", ["1mo", "3mo", "6mo", "1y", "5y"], index=3, key="period0")
        interval = st.selectbox("Intervalo", ["1d", "1wk", "1mo"], index=0, key="interval0")
        try:
            t_hist = yf.Ticker(ticker).history(period=period, interval=interval)
        except Exception:
            t_hist = pd.DataFrame()
        if not t_hist.empty:
            fig = go.Figure(
                data=[go.Candlestick(
                    x=t_hist.index, open=t_hist["Open"], high=t_hist["High"],
                    low=t_hist["Low"], close=t_hist["Close"],
                    increasing_line_color="#19c37d", decreasing_line_color="#ef553b",
                )]
            )
            fig.update_layout(template=plotly_theme, height=380, xaxis_rangeslider_visible=False,
                              margin=dict(l=10, r=10, t=30, b=10), title=f"Velas {ticker} ({period}, {interval})")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hay historial para el rango seleccionado.")

# ---- TAB 1 (Análisis absoluto)
with tab1:
    st.markdown("### 🧩 Salud financiera y eficiencia (último año reportado)")
    margins = compute_margins(fin_a)
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="kpi-card"><div class="kpi-title">{with_tooltip("Margen bruto", "Gross Profit / Revenue")}</div><div class="kpi-value">{pct(margins["gross_margin"])}</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="kpi-card"><div class="kpi-title">{with_tooltip("Margen EBIT", "EBIT / Revenue")}</div><div class="kpi-value">{pct(margins["ebit_margin"])}</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="kpi-card"><div class="kpi-title">{with_tooltip("Margen neto", "Utilidad neta / Revenue")}</div><div class="kpi-value">{pct(margins["net_margin"])}</div></div>', unsafe_allow_html=True)
    fcf, cfo, capex = compute_fcf(cf_a)
    c4.markdown(f'<div class="kpi-card"><div class="kpi-title">{with_tooltip("FCF (CFO–CapEx)", ACRONYMS["FCF"])}</div><div class="kpi-value">{human_int(fcf)}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)
    st.markdown(f"#### 💵 Apalancamiento y liquidez")
    net_debt, debt, cash = compute_net_debt(bs_a)
    ebit = margins["ebit"]
    ebitda_info = info.get("ebitda")
    ebitda = ebitda_info if ebitda_info is not None else (ebit + (fin_get(fin_a, "Depreciation") or 0) if ebit is not None else None)
    nd_ebitda = (net_debt / ebitda) if (net_debt is not None and ebitda not in (None, 0)) else None
    interest = abs(fin_get(fin_a, "Interest Expense") or 0)
    int_cov = (ebit / interest) if (ebit not in (None, 0) and interest not in (None, 0)) else None

    colA, colB, colC, colD = st.columns(4)
    colA.markdown(f'<div class="kpi-card"><div class="kpi-title">Deuda total</div><div class="kpi-value">{human_int(debt)}</div></div>', unsafe_allow_html=True)
    colB.markdown(f'<div class="kpi-card"><div class="kpi-title">Caja</div><div class="kpi-value">{human_int(cash)}</div></div>', unsafe_allow_html=True)
    colC.markdown(f'<div class="kpi-card"><div class="kpi-title">{with_tooltip("Deuda neta / EBITDA", "Deuda neta dividida entre EBITDA")}</div><div class="kpi-value">{ "—" if nd_ebitda is None else f"{nd_ebitda:.2f}×"}</div></div>', unsafe_allow_html=True)
    colD.markdown(f'<div class="kpi-card"><div class="kpi-title">{with_tooltip("Cobertura de intereses", "EBIT / Gasto de intereses")}</div><div class="kpi-value">{ "—" if int_cov is None else f"{int_cov:.1f}×"}</div></div>', unsafe_allow_html=True)

    st.markdown(f"### 📈 Gráficos históricos")
    hist = build_historical(fin_a, data["fin_q"], cf_a, data["cf_q"])
    subc1, subc2 = st.columns(2)
    if not hist["Revenue_a"].empty:
        fig = go.Figure()
        fig.add_scatter(x=hist["Revenue_a"].index, y=hist["Revenue_a"].values, mode="lines+markers", name="Ingresos")
        if not hist["Gross_a"].empty:
            fig.add_scatter(x=hist["Gross_a"].index, y=hist["Gross_a"].values, mode="lines+markers", name="Gross Profit")
        if not hist["EBIT_a"].empty:
            fig.add_scatter(x=hist["EBIT_a"].index, y=hist["EBIT_a"].values, mode="lines+markers", name="EBIT")
        if not hist["Net_a"].empty:
            fig.add_scatter(x=hist["Net_a"].index, y=hist["Net_a"].values, mode="lines+markers", name="Net Income")
        fig.update_layout(template=plotly_theme, height=360, margin=dict(l=10, r=10, t=35, b=10), title="P&L anual (niveles)")
        subc1.plotly_chart(fig, use_container_width=True)
        mdf = margins_from_series(hist["Revenue_a"], hist["Gross_a"], hist["EBIT_a"], hist["Net_a"])
        if not mdf.empty:
            fig2 = go.Figure()
            for col in mdf.columns:
                fig2.add_scatter(x=mdf.index, y=mdf[col].values, mode="lines+markers", name=col)
            fig2.update_layout(template=plotly_theme, height=360, margin=dict(l=10, r=10, t=35, b=10), title="Márgenes anuales")
            subc2.plotly_chart(fig2, use_container_width=True)
    if not hist.get("FCF_a", pd.Series(dtype=float)).empty:
        fcf_a = hist["FCF_a"]; cfo_a = series_from_df(cf_a, "Total Cash From Operating Activities"); capex_a = series_from_df(cf_a, "Capital Expenditures")
        fig3 = go.Figure()
        fig3.add_bar(x=cfo_a.index, y=cfo_a.values, name="CFO")
        fig3.add_bar(x=capex_a.index, y=capex_a.values, name="CapEx")
        fig3.add_scatter(x=fcf_a.index, y=fcf_a.values, mode="lines+markers", name="FCF", yaxis="y2")
        fig3.update_layout(template=plotly_theme, height=380, margin=dict(l=10, r=10, t=35, b=10),
                           title="CFO, CapEx y FCF (anual)", barmode="relative",
                           yaxis=dict(title="CFO/Capex"), yaxis2=dict(overlaying="y", side="right", title="FCF"))
        st.plotly_chart(fig3, use_container_width=True)

# ---- TAB 2 (Relativo)
with tab2:
    st.markdown(f"### 🧭 Comparables (peers)")
    all_ticks = [ticker] + [p for p in peer_list if p and p != ticker]
    rows = []
    with st.spinner("Cargando datos de peers..."):
        for tk in all_ticks[:6]:
            d = fetch_all(tk); inf = d["info"]; bs = d["bs_a"]; fin = d["fin_a"]
            ev = compute_ev(inf, bs)
            rows.append({
                "Ticker": tk, "Nombre": inf.get("shortName", tk),
                "Precio": get_price_now(tk), "Market Cap": inf.get("marketCap"), "EV": ev,
                "EV/Ventas": inf.get("enterpriseToRevenue"),
                "EV/EBITDA": inf.get("enterpriseToEbitda"),
                "P/E (ttm)": inf.get("trailingPE"), "P/E (fwd)": inf.get("forwardPE"),
                "P/B": inf.get("priceToBook"),
                "ROE": inf.get("returnOnEquity"), "ROA": inf.get("returnOnAssets"), "Beta": inf.get("beta"),
                "Ingresos (LTM)": fin_get(fin, "Total Revenue"), "EBITDA (LTM)": inf.get("ebitda")
            })
    df_peers = pd.DataFrame(rows)
    if not df_peers.empty:
        show_cols = ["Ticker", "Nombre", "Precio", "Market Cap", "EV", "EV/Ventas", "EV/EBITDA", "P/E (ttm)", "P/E (fwd)", "P/B", "ROE", "ROA", "Beta"]
        st.dataframe(df_peers[show_cols], use_container_width=True, hide_index=True)
        st.download_button("⬇️ Descargar peers (CSV)", data=df_peers.to_csv(index=False).encode("utf-8"),
                           file_name=f"peers_{ticker}.csv", mime="text/csv")
        st.markdown("#### 📊 Comparación de múltiplos")
        mult = ["EV/EBITDA", "P/E (ttm)", "EV/Ventas", "P/B"]
        mcol1, mcol2 = st.columns(2)
        for i, m in enumerate(mult):
            fig = go.Figure()
            sub = df_peers[["Ticker", m]].copy()
            fig.add_bar(x=sub["Ticker"], y=sub[m])
            fig.update_layout(template=plotly_theme, height=300, margin=dict(l=10, r=10, t=30, b=10), title=m)
            (mcol1 if i % 2 == 0 else mcol2).plotly_chart(fig, use_container_width=True)
    else:
        st.info("No fue posible construir la tabla de comparables.")

# ---- TAB 3 (Score)
with tab3:
    st.markdown("### 🧮 Score subjetivo (checks rápidos)")
    checks, score, max_score = compute_score_blocks(info, bs_a, fin_a, cf_a)
    cols = st.columns(3)
    for idx, (label, ok) in enumerate(checks):
        icon = "✅" if ok else "❌"
        cols[idx % 3].markdown(f"**{icon} {label}**")
    st.markdown(f"#### Resultado: **{score}/{max_score}**")
    if score >= max_score - 1:
        st.success("Perfil financiero **sólido** a primera vista.")
    elif score >= max_score // 2:
        st.warning("Perfil **mixto**: amerita investigación adicional.")
    else:
        st.error("Perfil **débil**: múltiples señales de cautela.")

    with st.expander("¿Qué significa cada check? (explicaciones)"):
        st.markdown(f"- **Caja > Deuda**: Más liquidez que obligaciones brutas; margen de seguridad ante shocks.")
        st.markdown(f"- **Pasivos/Patrimonio < 0.80**: Endeudamiento conservador; menor riesgo de distress.")
        st.markdown(f"- **Sin acciones preferentes**: Estructura de capital simple; menos reclamantes anteriores al común.")
        st.markdown(f"- **Recompras (Treasury < 0)**: Señal de retorno de capital al accionista (si no compromete solidez).")
        st.markdown(f"- **Retained Earnings > 0**: Historial acumulado de utilidades; destrucción si negativo (déficit).")
        st.markdown(f"- **FCF positivo**: El negocio genera caja después de reinversión.")
        st.markdown(f"- **Margen neto positivo**: Rentabilidad contable después de intereses e impuestos.")

# ---- TAB 4 (DCF con Escenarios y Tornado)
with tab4:
    st.markdown(f"### 🧷 Valoración por FCF – supuestos editables y escenarios")
    default_coe = capm_cost_of_equity(info)
    templates = {
        "Conservadora": {"wacc": max(0.07, default_coe - 0.01), "g": 0.02, "years": 5, "growth": 0.03},
        "Base": {"wacc": max(0.08, default_coe), "g": 0.025, "years": 5, "growth": 0.05},
        "Agresiva": {"wacc": max(0.09, default_coe + 0.01), "g": 0.03, "years": 5, "growth": 0.08},
    }

    # Inputs principales (para bloque "Base" rápido)
    tcol1, tcol2 = st.columns([1.1, 1])
    with tcol1:
        tpl = st.selectbox("Plantilla de supuestos (rápido)", list(templates.keys()), index=1)
        tpl_vals = templates[tpl]
        fcf_info = info.get("freeCashflow")
        if fcf_info is None:
            fcf_calc, _, _ = compute_fcf(cf_a)
            fcf_info = fcf_calc
        base_fcf = st.number_input(with_tooltip("FCF base (LTM)", "Flujo de caja libre de los últimos 12 meses"), value=float(fcf_info or 0.0), step=1_000_000.0, format="%.2f")
        years = st.slider("Años de proyección", min_value=3, max_value=10, value=int(tpl_vals["years"]), step=1)
        growth = st.number_input("Crecimiento anual del FCF (periodo explícito)", value=float(tpl_vals["growth"]), step=0.005, format="%.3f")
        wacc = st.number_input(with_tooltip("WACC / tasa de descuento", "Weighted Average Cost of Capital"), value=float(tpl_vals["wacc"]), step=0.005, format="%.3f")
        g_term = st.number_input("Crecimiento a perpetuidad (g)", value=float(tpl_vals["g"]), step=0.005, format="%.3f")
        net_debt, _, _ = compute_net_debt(bs_a)
        shares = info.get("sharesOutstanding") or info.get("floatShares")
        st.caption(f"Deuda neta estimada: {human_int(net_debt)} | Acciones diluidas: {human_int(shares)}")

        # Proyección y valoración "Base"
        ev, pv_term, pv_sum = dcf_value(base_fcf, years, growth, wacc, g_term)
        equity = (ev - net_debt) if (ev is not None and net_debt is not None) else None
        fair = (equity / shares) if (equity is not None and shares not in (None, 0)) else None

    with tcol2:
        st.markdown("#### Resultado DCF (escenario rápido)")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"**Valor empresa (EV)**: {human_int(ev)}")
        st.markdown(f"**Equity value**: {human_int(equity)}")
        st.markdown(f"**Valor intrínseco / acción**: **{ '—' if fair is None else '$'+format(fair, '.2f') }**")
        pnow = get_price_now(ticker)
        if fair and pnow:
            upside = (fair / pnow) - 1
            color = "🟢" if upside > 0 else "🔴"
            st.markdown(f"**Diferencia vs precio actual**: {color} {pct(upside)}")
        st.markdown('</div>', unsafe_allow_html=True)
        with st.expander("🔎 Cómo funciona el DCF (explicación)"):
            st.latex(r"V_0=\sum_{t=1}^{N}\frac{FCF_t}{(1+r)^t} + \frac{FCF_N(1+g)}{(r-g)}\cdot\frac{1}{(1+r)^N}")
            st.markdown("""
- **Idea clave**: una empresa vale el **valor presente** de sus **FCF futuros**.
- **Dos partes**: (1) periodo explícito (años 1..N) y (2) **valor terminal** (crecimiento a perpetuidad *g*).
- **Descuento** a **r = WACC** (refleja riesgo y estructura de capital).
            """)
        with st.expander("🧭 Cómo elegir los supuestos (guía práctica)"):
            st.markdown(f"""
- **FCF base**: usa FCF **normalizado** (elimina efectos extraordinarios).
- **Crecimiento explícito**: basarse en histórico/consenso; maduros 3–6%, growth 8–12%.
- **WACC**: aproximar con beta (**CAPM**) o rango 8–10% (más riesgo ⇒ mayor WACC).
- **g a perpetuidad**: **< WACC**; típicamente 1.5–3.0% (PIB real + inflación).
- **Regla de sanity**: si el **valor terminal >70%** del EV, endurece *g* y/o *WACC*.
            """, unsafe_allow_html=True)

    # ========= ESCENARIOS =========
    st.markdown("### 🎬 Escenarios DCF (Bull / Base / Bear)")
    s1, s2, s3 = st.columns(3)
    # Defaults tomando los inputs del bloque "Base" para mantener consistencia
    default_params = {
        "fcf": base_fcf or 0.0,
        "years": years,
        "growth": growth,
        "wacc": wacc,
        "g": g_term
    }
    with s1:
        st.subheader("🐂 Bull")
        bull_fcf = st.number_input("FCF base (bull)", value=float(default_params["fcf"]*1.05), step=1_000_000.0, format="%.2f", key="bull_fcf")
        bull_growth = st.number_input("Growth explícito (bull)", value=float(min(default_params["growth"]+0.02, 0.25)), step=0.005, format="%.3f", key="bull_growth")
        bull_wacc = st.number_input("WACC (bull)", value=float(max(default_params["wacc"]-0.01, 0.04)), step=0.005, format="%.3f", key="bull_wacc")
        bull_g = st.number_input("g perpetuo (bull)", value=float(min(default_params["g"]+0.005, 0.05)), step=0.005, format="%.3f", key="bull_g")
        fp_bull = dcf_equity_fairprice(bull_fcf, years, bull_growth, bull_wacc, bull_g, compute_net_debt(bs_a)[0], info.get("sharesOutstanding") or info.get("floatShares"))
        st.markdown(f"**Precio justo (bull)**: { '—' if np.isnan(fp_bull) else '$'+format(fp_bull, '.2f') }")
    with s2:
        st.subheader("⚖️ Base")
        base2_fcf = st.number_input("FCF base (base)", value=float(default_params["fcf"]), step=1_000_000.0, format="%.2f", key="base_fcf2")
        base2_growth = st.number_input("Growth explícito (base)", value=float(default_params["growth"]), step=0.005, format="%.3f", key="base_growth2")
        base2_wacc = st.number_input("WACC (base)", value=float(default_params["wacc"]), step=0.005, format="%.3f", key="base_wacc2")
        base2_g = st.number_input("g perpetuo (base)", value=float(default_params["g"]), step=0.005, format="%.3f", key="base_g2")
        fp_base2 = dcf_equity_fairprice(base2_fcf, years, base2_growth, base2_wacc, base2_g, compute_net_debt(bs_a)[0], info.get("sharesOutstanding") or info.get("floatShares"))
        st.markdown(f"**Precio justo (base)**: { '—' if np.isnan(fp_base2) else '$'+format(fp_base2, '.2f') }")
    with s3:
        st.subheader("🐻 Bear")
        bear_fcf = st.number_input("FCF base (bear)", value=float(default_params["fcf"]*0.95), step=1_000_000.0, format="%.2f", key="bear_fcf")
        bear_growth = st.number_input("Growth explícito (bear)", value=float(max(default_params["growth"]-0.02, 0.00)), step=0.005, format="%.3f", key="bear_growth")
        bear_wacc = st.number_input("WACC (bear)", value=float(default_params["wacc"]+0.01), step=0.005, format="%.3f", key="bear_wacc")
        bear_g = st.number_input("g perpetuo (bear)", value=float(max(default_params["g"]-0.005, 0.00)), step=0.005, format="%.3f", key="bear_g")
        fp_bear = dcf_equity_fairprice(bear_fcf, years, bear_growth, bear_wacc, bear_g, compute_net_debt(bs_a)[0], info.get("sharesOutstanding") or info.get("floatShares"))
        st.markdown(f"**Precio justo (bear)**: { '—' if np.isnan(fp_bear) else '$'+format(fp_bear, '.2f') }")

    # Chart comparativo escenarios
    st.markdown("#### 📊 Comparativa de escenarios (precio justo)")
    scen_names = ["Bull","Base","Bear"]
    scen_vals = [fp_bull, fp_base2, fp_bear]
    fig_scen = go.Figure()
    fig_scen.add_bar(x=scen_names, y=scen_vals, name="Precio justo")
    if price_now:
        fig_scen.add_scatter(x=scen_names, y=[price_now]*3, mode="lines", name="Precio actual")
    fig_scen.update_layout(template=plotly_theme, height=350, margin=dict(l=10,r=10,t=30,b=10), title="DCF – Escenarios")
    st.plotly_chart(fig_scen, use_container_width=True)

    df_scen = pd.DataFrame({
        "Escenario": scen_names,
        "FCF_base": [bull_fcf, base2_fcf, bear_fcf],
        "Growth": [bull_growth, base2_growth, bear_growth],
        "WACC": [bull_wacc, base2_wacc, bear_wacc],
        "g_perpetuo": [bull_g, base2_g, bear_g],
        "Precio_justo": scen_vals
    })
    st.download_button("⬇️ Descargar escenarios (CSV)", data=df_scen.to_csv(index=False).encode("utf-8"),
                       file_name=f"escenarios_dcf_{ticker}.csv", mime="text/csv")

    # ========= TORNADO =========
    st.markdown("### 🌪️ Tornado de sensibilidad DCF")
    st.caption("Mide el impacto en el valor por acción al variar **cada** supuesto manteniendo los demás constantes.")
    tcolL, tcolR = st.columns(2)
    with tcolL:
        shock_fcf = st.slider("Shock en FCF base (±%)", 5, 50, 20, step=5) / 100.0
        shock_growth = st.slider("Shock en growth explícito (±puntos)", 1, 10, 3, step=1) / 100.0
    with tcolR:
        shock_wacc = st.slider("Shock en WACC (±puntos)", 0, 300, 100, step=25) / 10000.0  # 100 = 1.00% = 0.01
        shock_g = st.slider("Shock en g perpetuo (±puntos)", 0, 200, 50, step=25) / 10000.0

    # Base para tornado: la configuración "Base" rápida (fair)
    fair_base = fair if fair is not None else dcf_equity_fairprice(base_fcf, years, growth, wacc, g_term, compute_net_debt(bs_a)[0], info.get("sharesOutstanding") or info.get("floatShares"))
    nd = compute_net_debt(bs_a)[0]
    sh = info.get("sharesOutstanding") or info.get("floatShares")

    def tornado_range(label, low_params, high_params):
        low = dcf_equity_fairprice(low_params["fcf"], years, low_params["growth"], low_params["wacc"], low_params["g"], nd, sh)
        high = dcf_equity_fairprice(high_params["fcf"], years, high_params["growth"], high_params["wacc"], high_params["g"], nd, sh)
        return {"Variable": label, "Low": low, "High": high}

    # Construir rangos
    ranges = []
    # FCF
    ranges.append(tornado_range("FCF base",
                {"fcf": base_fcf*(1 - shock_fcf), "growth": growth, "wacc": wacc, "g": g_term},
                {"fcf": base_fcf*(1 + shock_fcf), "growth": growth, "wacc": wacc, "g": g_term}))
    # Growth explícito
    ranges.append(tornado_range("Growth explícito",
                {"fcf": base_fcf, "growth": max(growth - shock_growth, 0.0), "wacc": wacc, "g": g_term},
                {"fcf": base_fcf, "growth": growth + shock_growth, "wacc": wacc, "g": g_term}))
    # WACC
    ranges.append(tornado_range("WACC",
                {"fcf": base_fcf, "growth": growth, "wacc": wacc + shock_wacc, "g": g_term},
                {"fcf": base_fcf, "growth": growth, "wacc": max(wacc - shock_wacc, 0.01), "g": g_term}))
    # g perpetuo
    ranges.append(tornado_range("g a perpetuidad",
                {"fcf": base_fcf, "growth": growth, "wacc": wacc, "g": max(g_term - shock_g, 0.0)},
                {"fcf": base_fcf, "growth": growth, "wacc": wacc, "g": g_term + shock_g}))

    df_tor = pd.DataFrame(ranges)
    # Limpieza si NaN por casos wacc <= g
    df_tor = df_tor.replace([np.inf, -np.inf], np.nan).dropna()
    if not np.isnan(fair_base) and not df_tor.empty:
        df_tor["Down"] = np.maximum(0, fair_base - df_tor["Low"])
        df_tor["Up"] = np.maximum(0, df_tor["High"] - fair_base)
        df_tor = df_tor.sort_values(by=["Down","Up"], ascending=[False, False])
        fig_t = go.Figure()
        fig_t.add_bar(y=df_tor["Variable"], x=-df_tor["Down"], orientation="h", name="Baja", hovertemplate="Baja: -$%{x:.2f}<extra></extra>")
        fig_t.add_bar(y=df_tor["Variable"], x=df_tor["Up"], orientation="h", name="Sube", hovertemplate="Sube: $%{x:.2f}<extra></extra>")
        fig_t.add_vline(x=0, line_dash="dot")
        fig_t.update_layout(template=plotly_theme, height=420, barmode="overlay",
                            title="Tornado DCF – impacto en precio justo (desde escenario Base)",
                            margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig_t, use_container_width=True)
    else:
        st.info("No se pudo construir el tornado (revisa supuestos: WACC > g, FCF > 0, acciones válidas).")

    # Sensibilidades heatmap (ya existentes)
    st.markdown("### 🎯 Sensibilidad DCF (heatmaps)")
    scol1, scol2 = st.columns(2)
    def dcf_fairprice_given(wacc_x: float, g_x: float, growth_x: float) -> float:
        if not base_fcf or base_fcf<=0: return np.nan
        if not (info.get("sharesOutstanding") or info.get("floatShares")): return np.nan
        shares_local = info.get("sharesOutstanding") or info.get("floatShares")
        if wacc_x <= g_x: return np.nan
        f = base_fcf
        flows = []
        for _ in range(years):
            f = f * (1 + growth_x)
            flows.append(f)
        dfacts = [(1 + wacc_x) ** (i+1) for i in range(years)]
        pv = [flows[i] / dfacts[i] for i in range(years)]
        term = (flows[-1] * (1 + g_x)) / (wacc_x - g_x)
        ev_x = sum(pv) + term / ((1 + wacc_x) ** years)
        eq_x = ev_x - (compute_net_debt(bs_a)[0] or 0)
        return float(eq_x / shares_local) if shares_local else np.nan
    with scol1:
        st.subheader("WACC vs g (perpetuidad)")
        wacc_vals = np.round(np.arange(max(0.05, wacc-0.02), wacc+0.021, 0.005), 3)
        g_vals = np.round(np.arange(max(0.00, g_term-0.01), g_term+0.011, 0.005), 3)
        grid = np.zeros((len(g_vals), len(wacc_vals)))
        for i, g_ in enumerate(g_vals):
            for j, w_ in enumerate(wacc_vals):
                grid[i, j] = dcf_fairprice_given(w_, g_, growth)
        heat1 = go.Figure(data=go.Heatmap(
            z=grid, x=wacc_vals, y=g_vals, colorbar_title="Precio justo",
            hovertemplate="WACC=%{x:.3f} • g=%{y:.3f}<br>Fair=$%{z:.2f}<extra></extra>"))
        heat1.update_layout(template=plotly_theme, height=420, title="Sensibilidad DCF – WACC × g")
        st.plotly_chart(heat1, use_container_width=True)
    with scol2:
        st.subheader("WACC vs crecimiento explícito del FCF")
        growth_vals = np.round(np.arange(max(0.00, growth-0.03), growth+0.031, 0.005), 3)
        wacc_vals2 = np.round(np.arange(max(0.05, wacc-0.02), wacc+0.021, 0.005), 3)
        grid2 = np.zeros((len(growth_vals), len(wacc_vals2)))
        for i, gr in enumerate(growth_vals):
            for j, w_ in enumerate(wacc_vals2):
                grid2[i, j] = dcf_fairprice_given(w_, g_term, gr)
        heat2 = go.Figure(data=go.Heatmap(
            z=grid2, x=wacc_vals2, y=growth_vals, colorbar_title="Precio justo",
            hovertemplate="WACC=%{x:.3f} • growth=%{y:.3f}<br>Fair=$%{z:.2f}<extra></extra>"))
        heat2.update_layout(template=plotly_theme, height=420, title="Sensibilidad DCF – WACC × growth")
        st.plotly_chart(heat2, use_container_width=True)

    st.session_state["dcf_out"] = {
        "inputs": {"base_fcf": base_fcf, "years": years, "growth": growth, "wacc": wacc, "g": g_term},
        "outputs": {"EV": ev, "Equity": equity, "FairPrice": fair,
                    "UpsideVsPrice": (fair/get_price_now(ticker)-1) if (fair and get_price_now(ticker)) else None},
        "scenarios": df_scen.to_dict(orient="records")
    }

# ---- TAB 5 (DDM)
with tab5:
    st.markdown(f"### 💸 Valoración por dividendos – supuestos editables")
    templates_ddm = {
        "Conservadora": {"r": max(0.08, capm_cost_of_equity(info)), "g": 0.02},
        "Base": {"r": max(0.09, capm_cost_of_equity(info)+0.005), "g": 0.025},
        "Agresiva": {"r": max(0.10, capm_cost_of_equity(info)+0.01), "g": 0.03},
    }
    dcol1, dcol2 = st.columns([1.1, 1])
    with dcol1:
        tpl2 = st.selectbox("Plantilla DDM", list(templates_ddm.keys()), index=1)
        tv = templates_ddm[tpl2]
        div_rate = info.get("dividendRate") or 0.0
        div_yld = info.get("dividendYield")
        st.caption(f"Rendimiento actual reportado: {pct(div_yld) if div_yld else '—'}")
        D0 = st.number_input("Dividendo anual actual (D0)", value=float(div_rate or 0.0), step=0.01, format="%.4f")
        g = st.number_input("Crecimiento a perpetuidad (g)", value=float(tv["g"]), step=0.005, format="%.3f")
        r = st.number_input("Costo de equity (r)", value=float(tv["r"]), step=0.005, format="%.3f")
        D1 = D0 * (1 + g) if D0 else 0.0
        fair_ddm = (D1 / (r - g)) if (r > g and D1 > 0) else None
    with dcol2:
        st.markdown("#### Resultado DDM")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"**D1 (próximo dividendo)**: {D1:.4f}" if D1 else "**D1**: —")
        st.markdown(f"**Valor intrínseco (DDM)**: **{ '—' if fair_ddm is None else '$'+format(fair_ddm, '.2f') }**")
        pnow = get_price_now(ticker)
        if fair_ddm and pnow:
            up = (fair_ddm / pnow) - 1
            color = "🟢" if up > 0 else "🔴"
            st.markdown(f"**Diferencia vs precio actual**: {color} {pct(up)}")
        st.markdown('</div>', unsafe_allow_html=True)
        with st.expander("🔎 Cómo funciona el DDM (explicación)"):
            st.latex(r"P_0=\frac{D_1}{r-g}\quad (g<r)")
            st.markdown("""
- **Idea clave**: el precio justo es el **valor presente** de todos los **dividendos futuros**.
- **Gordon** (crecimiento constante): válido para empresas **maduras y estables** en dividendos.
- Si los dividendos no son estables, usar **multi-etapas** (alto crecimiento y luego estable).
            """)
        with st.expander("🧭 Cómo elegir supuestos (guía práctica)"):
            st.markdown("""
- **D0**: dividendo anual actual **normalizado** (excluye extraordinarios).
- **g**: crecimiento de dividendos sostenible (**largo plazo**). Suele estar **por debajo** del crecimiento de EPS; 1–4% típico en maduras.
- **r**: costo de equity (usa CAPM aproximado con beta o un rango 8–12% según riesgo).
- **Chequeo**: si **r ≈ g**, el valor se dispara y el modelo se vuelve inestable ⇒ usa **g más bajo** o **r mayor**.
            """)

    # Sensibilidad DDM
    st.markdown("### 🎯 Sensibilidad DDM (r × g)")
    r_vals = np.round(np.arange(max(0.06, r-0.03), r+0.031, 0.005), 3)
    g_vals = np.round(np.arange(max(0.00, g-0.02), g+0.021, 0.005), 3)
    grid_ddm = np.full((len(g_vals), len(r_vals)), np.nan)
    if D1 and D1 > 0:
        for i, g_ in enumerate(g_vals):
            for j, r_ in enumerate(r_vals):
                if r_ > g_:
                    grid_ddm[i, j] = D1 / (r_ - g_)
    heat_ddm = go.Figure(data=go.Heatmap(
        z=grid_ddm, x=r_vals, y=g_vals, colorbar_title="Precio justo",
        hovertemplate="r=%{x:.3f} • g=%{y:.3f}<br>Fair=$%{z:.2f}<extra></extra>"))
    heat_ddm.update_layout(template=plotly_theme, height=420, title="Sensibilidad DDM – r × g")
    st.plotly_chart(heat_ddm, use_container_width=True)

    st.session_state["ddm_out"] = {
        "inputs": {"D0": D0, "g": g, "r": r},
        "outputs": {"FairPrice": fair_ddm, "UpsideVsPrice": (fair_ddm/get_price_now(ticker)-1) if (fair_ddm and get_price_now(ticker)) else None},
    }

# ---- TAB 6 (Detalles, notas, descargas)
with tab6:
    st.markdown("### 📚 Detalles, notas y descargas")
    # Glosario
    with st.expander("📖 Glosario de siglas y métricas"):
        for k, v in ACRONYMS.items():
            st.markdown(f"- **{k}**: {v}")
        st.markdown("- **Cobertura de intereses**: EBIT / Intereses (solvencia operativa).")
        st.markdown("- **Pasivos/Patrimonio**: apalancamiento contable (más bajo = más conservador).")

    # Metodologías y armonizaciones
    with st.expander("🧪 Metodologías y armonizaciones (comparables)"):
        st.markdown("""
- **Moneda**: usa una **moneda común** para EV/Ventas, EV/EBITDA.
- **Horizonte**: **LTM** para todos; si comparas **NTM**, que sea consistente.
- **IFRS-16**: sé consistente con **arrendamientos** (incluir en deuda o no).
- **SBC**: decide si ajustas EBITDA y **aplícalo a todos**.
- **One-offs**: documenta **ajustes no recurrentes** si normalizas.
        """)

    # Riesgos y limitaciones
    with st.expander("⚠️ Limitaciones y buenas prácticas"):
        st.markdown("""
- **Datos de Yahoo Finance** pueden omitir partidas o cambiar etiquetas; valida con reportes oficiales.
- **Modelos (DCF/DDM)** son **sensibles** a supuestos; realiza **sensibilidades** y escenarios.
- Esta herramienta es **educativa**; no constituye **asesoría financiera**.
        """)

    st.markdown("#### 🗒️ Notas del análisis")
    user_notes = st.text_area("Notas para este ticker:", height=140)

    # Acciones
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("💾 Guardar análisis de este ticker"):
            snapshot_for_ticker(
                ticker,
                st.session_state.get("dcf_out", {}),
                st.session_state.get("ddm_out", {}),
                st.session_state.get("score_out", {}),
                user_notes,
            )
            st.success(f"Análisis de **{ticker}** guardado en {DB_PATH.resolve()}")
    with c2:
        out_json = json.dumps({
            "ticker": ticker,
            "timestamp": datetime.utcnow().isoformat()+"Z",
            "dcf": st.session_state.get("dcf_out", {}),
            "ddm": st.session_state.get("ddm_out", {}),
            "score": st.session_state.get("score_out", {}),
            "notes": user_notes
        }, indent=2, ensure_ascii=False)
        st.download_button("⬇️ Descargar snapshot (JSON)", data=out_json.encode("utf-8"),
                           file_name=f"snapshot_{ticker}.json", mime="application/json")
    with c3:
        price_now = get_price_now(ticker)
        summary = {
            "Ticker": [ticker],
            "Precio": [price_now],
            "DCF_Fair": [st.session_state.get("dcf_out", {}).get("outputs", {}).get("FairPrice")],
            "DDM_Fair": [st.session_state.get("ddm_out", {}).get("outputs", {}).get("FairPrice")],
            "Score": [st.session_state.get("score_out", {}).get("score")],
            "ScoreMax": [st.session_state.get("score_out", {}).get("max")]
        }
        df_summary = pd.DataFrame(summary)
        st.download_button("⬇️ Descargar resumen (CSV)", data=df_summary.to_csv(index=False).encode("utf-8"),
                           file_name=f"summary_{ticker}.csv", mime="text/csv")

    # Reporte Markdown
    st.markdown("#### 🧾 Exportar reporte (Markdown)")
    dcf = st.session_state.get("dcf_out", {})
    ddm = st.session_state.get("ddm_out", {})
    score = st.session_state.get("score_out", {})
    md = f"""# Reporte – {ticker}
Fecha: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC

## Resumen rápido
- Precio actual: {price_now if price_now else '—'}
- DCF (valor / acc): {dcf.get('outputs', {}).get('FairPrice', '—')}
- DDM (valor / acc): {ddm.get('outputs', {}).get('FairPrice', '—')}
- Score: {score.get('score', '—')}/{score.get('max', '—')}

## Notas
{user_notes if user_notes else '(sin notas)'}
"""
    st.download_button("⬇️ Descargar reporte.md", data=md.encode("utf-8"),
                       file_name=f"reporte_{ticker}.md", mime="text/markdown")

# ========================
# GUARDAR score en session
# ========================
checks_s, score_val, score_max = compute_score_blocks(info, bs_a, fin_a, cf_a)
st.session_state["score_out"] = {
    "score": score_val, "max": score_max,
    "checks": [{"label": c[0], "ok": bool(c[1])} for c in checks_s],
}
