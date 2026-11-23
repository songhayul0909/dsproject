import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

EXCEL_FILE = "제로음료_임상시험_원자료_체중혈당.xlsx"
Z95 = 1.96  # 95% 신뢰구간 계수
Z_WARN = 2.0  # 이상치 '주의' 기준

METRIC_KO = {
    "Body Weight": "체중",
    "BMI": "BMI",
    "Body Fat": "체지방률",
    "Waist Circumference": "허리둘레",
    "HbA1c": "당화혈색소",
    "Fasting Blood Glucose": "공복혈당",
    "2-Hour Post-Prandial Glucose": "식후 2시간 혈당",
}
COMP_KO = {
    "LNCSBs for SSBs": "설탕음료 → 제로음료로 바꿈",
    "Water for SSBs": "설탕음료 → 물로 바꿈",
    "LNCSBs for Water": "물 → 제로음료로 바꿈",
}
GROUP_KO = {"SSB": "설탕음료", "Water": "물", "LNCSB": "제로음료"}
COMP_COLORS = ["#2CA02C", "#4C78A8", "#F58518"]
BETTER, NEUTRAL, WORSE = "#8FD18F", "#E3E3E3", "#F4A6A6"
# 제거할 당화혈색소 이상치 (지표, 연구) — 다른 이상치는 그대로 둠
HBA1C_METRIC, HBA1C_OUTLIER_STUDY = "HbA1c", "Madjd 2017"
FLAG_COLOR = {"정상": "#5B8DB8", "주의": "#F58518", "이상치": "#D62728"}


# ------------------------------------------------------------------ 데이터
@st.cache_data
def load_data():
    study = pd.read_excel(EXCEL_FILE, sheet_name="개별연구_원자료").dropna(subset=["지표"])
    summ = pd.read_excel(EXCEL_FILE, sheet_name="종합분석_결과").dropna(subset=["지표"])

    summ["지표_한글"] = summ["지표"].map(METRIC_KO)
    summ["재계산"] = False  # 이상치 제거 후 직접 재계산한 값이면 True
    summ["비교_한글"] = summ["비교"].map(COMP_KO)
    # 모든 지표는 '낮아질수록 개선'으로 가정 (체중, BMI, 체지방, 허리둘레, HbA1c, 혈당)
    summ["판정"] = "유의한 차이 없음"
    summ.loc[summ["95% CI 상한"] < 0, "판정"] = "개선(유의)"
    summ.loc[summ["95% CI 하한"] > 0, "판정"] = "악화(유의)"
    # 종합 CI에서 표준오차를 역산해 z값(MD÷SE)을 구함 → 단위가 달라도 비교 가능
    summ["SE"] = (summ["95% CI 상한"] - summ["95% CI 하한"]) / (2 * Z95)
    summ["z"] = summ["종합 평균차이(MD)"] / summ["SE"]

    study["지표_한글"] = study["지표"].map(METRIC_KO)
    study["비교"] = study["Treatment 1"].map(GROUP_KO) + " vs " + study["Treatment 2"].map(GROUP_KO)
    study["y라벨"] = study["연구(저자·연도)"] + "  [" + study["Treatment 1"] + " vs " + study["Treatment 2"] + "]"
    study["CI 하한"] = study["Mean Difference"] - Z95 * study["Standard Error"]
    study["CI 상한"] = study["Mean Difference"] + Z95 * study["Standard Error"]
    return study, summ


# ------------------------------------------------------------------ 이상치 통계
def re_pool(md, se):
    """DerSimonian-Laird 랜덤효과 합산 (정규근사 95% CI)"""
    md, se = np.asarray(md, float), np.asarray(se, float)
    k = len(md)
    if k == 0:
        return dict(mu=np.nan, se=np.nan, lo=np.nan, hi=np.nan, tau2=np.nan, i2=np.nan, k=0)
    w = 1 / se**2
    mu_f = (w * md).sum() / w.sum()
    q = (w * (md - mu_f) ** 2).sum()
    c = w.sum() - (w**2).sum() / w.sum()
    tau2 = max(0.0, (q - (k - 1)) / c) if k > 1 else 0.0
    ws = 1 / (se**2 + tau2)
    mu = (ws * md).sum() / ws.sum()
    s = np.sqrt(1 / ws.sum())
    i2 = max(0.0, (q - (k - 1)) / q) * 100 if k > 1 and q > 0 else 0.0
    return dict(mu=mu, se=s, lo=mu - Z95 * s, hi=mu + Z95 * s, tau2=tau2, i2=i2, k=k)


def loo_z(md, se):
    """한 연구를 뺀 나머지로 합산한 값과 얼마나 다른지(표준화). 연구 3개 미만이면 계산 불가(NaN)"""
    md, se = np.asarray(md, float), np.asarray(se, float)
    out = np.full(len(md), np.nan)
    if len(md) < 3:
        return out
    for i in range(len(md)):
        m = np.arange(len(md)) != i
        p = re_pool(md[m], se[m])
        out[i] = (md[i] - p["mu"]) / np.sqrt(se[i] ** 2 + p["tau2"] + p["se"] ** 2)
    return out


def drop_hba1c_outlier(study, summ):
    """당화혈색소 이상치(Madjd 2017, 물 vs 제로음료) 한 건만 제거.
    - 개별 연구 자료에서 그 행을 삭제
    - 종합 결과 중 영향받는 '물 → 제로음료로 바꿈' HbA1c 한 칸만 남은 연구로 다시 합산(랜덤효과 근사)
      (논문의 네트워크 합산값은 재현할 수 없어, 영향받는 칸만 직접 계산값으로 교체하고 '재계산'으로 표시)
    """
    mask = (study["지표"] == HBA1C_METRIC) & (study["연구(저자·연도)"] == HBA1C_OUTLIER_STUDY)
    removed = study[mask].copy()
    study2 = study[~mask].reset_index(drop=True)

    s2 = summ.copy()
    g = study2[(study2["지표"] == HBA1C_METRIC) & (study2["Treatment 1"] == "Water") & (study2["Treatment 2"] == "LNCSB")]
    p = re_pool(g["Mean Difference"], g["Standard Error"])
    idx = s2.index[(s2["지표"] == HBA1C_METRIC) & (s2["비교"] == "LNCSBs for Water")]
    z = p["mu"] / p["se"]
    s2.loc[idx, "종합 평균차이(MD)"] = p["mu"]
    s2.loc[idx, "95% CI 하한"] = p["lo"]
    s2.loc[idx, "95% CI 상한"] = p["hi"]
    s2.loc[idx, "p값"] = math.erfc(abs(z) / math.sqrt(2))
    s2.loc[idx, "비교 수"] = len(g)
    s2.loc[idx, "재계산"] = True
    # 판정·SE·z 다시 계산
    s2["판정"] = "유의한 차이 없음"
    s2.loc[s2["95% CI 상한"] < 0, "판정"] = "개선(유의)"
    s2.loc[s2["95% CI 하한"] > 0, "판정"] = "악화(유의)"
    s2["SE"] = (s2["95% CI 상한"] - s2["95% CI 하한"]) / (2 * Z95)
    s2["z"] = s2["종합 평균차이(MD)"] / s2["SE"]
    return study2, s2, removed


@st.cache_data
def flag_outliers(study, z_out):
    """지표 × 비교 그룹 안에서 연구별 이상치 점수(z)와 판정(정상/주의/이상치)을 붙임"""
    d = study.rename(columns={"연구(저자·연도)": "study", "Mean Difference": "MD", "Standard Error": "SE"}).copy()
    d["metric"], d["comp"] = d["지표_한글"], d["비교"]
    d["lo"], d["hi"] = d["CI 하한"], d["CI 상한"]
    d["z"] = np.nan
    for _, idx in d.groupby(["metric", "comp"]).groups.items():
        d.loc[idx, "z"] = loo_z(d.loc[idx, "MD"], d.loc[idx, "SE"])
    d["판정"] = np.where(d.z.abs() >= z_out, "이상치", np.where(d.z.abs() >= Z_WARN, "주의", "정상"))
    return d


def fmt_ci(p):
    return f"{p['mu']:+.2f} [{p['lo']:.2f}, {p['hi']:.2f}]"


def ci_verdict(p):
    if np.isnan(p["mu"]):
        return "-"
    if p["hi"] < 0:
        return "유의하게 낮아짐"
    if p["lo"] > 0:
        return "유의하게 높아짐"
    return "차이 확신 못함(CI가 0 포함)"


# ------------------------------------------------------------------ 그래프
def scorecard(summ):
    """지표 × 비교 성적표 (색 = 판정, 글자 = MD [95% CI] · GRADE)"""
    metrics = list(dict.fromkeys(summ["지표"]))
    comps = list(COMP_KO)
    code = {"개선(유의)": -1, "유의한 차이 없음": 0, "악화(유의)": 1}
    z, text, hover, ylabels = [], [], [], []
    for m in metrics:
        zr, tr, hr = [], [], []
        for c in comps:
            r = summ[(summ["지표"] == m) & (summ["비교"] == c)].iloc[0]
            zr.append(code[r["판정"]])
            tr.append(
                f"<b>{r['종합 평균차이(MD)']:+.2f}</b><br>"
                f"[{r['95% CI 하한']:.2f}, {r['95% CI 상한']:.2f}]<br>"
                f"근거 {r['근거 확실성(논문 GRADE 표기)']}"
                + ("<br>※이상치 제거 후 재계산" if r["재계산"] else "")
            )
            hr.append(
                f"<b>{METRIC_KO[m]} · {COMP_KO[c]}</b><br>판정: {r['판정']}<br>"
                f"MD {r['종합 평균차이(MD)']:+.3f} {r['단위']}<br>"
                f"95% CI {r['95% CI 하한']:.3f} ~ {r['95% CI 상한']:.3f}<br>"
                f"p={r['p값']:.3f} · 비교 {r['비교 수']}건 · 근거 {r['근거 확실성(논문 GRADE 표기)']}"
            )
        z.append(zr)
        text.append(tr)
        hover.append(hr)
        ylabels.append(f"{METRIC_KO[m]} ({summ.loc[summ['지표'] == m, '단위'].iloc[0]})")

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=[COMP_KO[c] for c in comps],
            y=ylabels,
            text=text,
            texttemplate="%{text}",
            hovertext=hover,
            hoverinfo="text",
            zmin=-1,
            zmax=1,
            colorscale=[[0, BETTER], [0.5, NEUTRAL], [1, WORSE]],
            showscale=False,
            xgap=4,
            ygap=4,
        )
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(side="top")
    fig.update_layout(height=640, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def zscore_bars(summ):
    """단위가 다른 지표를 같은 눈금(z값)으로 비교"""
    metrics = [METRIC_KO[m] for m in dict.fromkeys(summ["지표"])]
    fig = go.Figure()
    for i, (c, ko) in enumerate(COMP_KO.items()):
        g = summ[summ["비교"] == c].set_index("지표_한글").loc[metrics].reset_index()
        fig.add_trace(
            go.Bar(
                x=g["지표_한글"],
                y=g["z"],
                name=ko,
                marker_color=COMP_COLORS[i],
                customdata=g[["종합 평균차이(MD)", "단위", "p값"]].values,
                hovertemplate=(
                    "<b>%{x}</b><br>z=%{y:.2f}<br>MD %{customdata[0]:+.3f} %{customdata[1]}"
                    "<br>p=%{customdata[2]:.3f}<extra></extra>"
                ),
            )
        )
    fig.add_hrect(y0=-Z95, y1=Z95, fillcolor="gray", opacity=0.15, line_width=0)
    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.add_annotation(
        xref="paper", x=1.0, y=Z95, yanchor="bottom", xanchor="right",
        text="회색 띠 안쪽 = 우연으로 설명 가능(유의하지 않음)", showarrow=False, font=dict(size=11),
    )
    fig.update_layout(
        barmode="group",
        yaxis_title="z값 = 평균차이 ÷ 표준오차  (음수 = 수치가 낮아짐)",
        legend=dict(orientation="h", y=-0.2),
        height=460,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig


def metric_forest(df, unit):
    """선택한 지표의 종합 결과: 비교 3종의 MD와 95% CI"""
    df = df.reset_index(drop=True)
    color_of = {"개선(유의)": "#2CA02C", "유의한 차이 없음": "#7F7F7F", "악화(유의)": "#D62728"}
    fig = go.Figure()
    shown = set()
    for _, r in df.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[r["종합 평균차이(MD)"]],
                y=[r["비교_한글"]],
                mode="markers+text",
                name=r["판정"],
                legendgroup=r["판정"],
                showlegend=r["판정"] not in shown,
                marker=dict(color=color_of[r["판정"]], size=15, symbol="diamond"),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=[r["95% CI 상한"] - r["종합 평균차이(MD)"]],
                    arrayminus=[r["종합 평균차이(MD)"] - r["95% CI 하한"]],
                    color=color_of[r["판정"]],
                    thickness=2.4,
                ),
                text=[f"{r['종합 평균차이(MD)']:+.2f} (p={r['p값']:.3f}, 근거 {r['근거 확실성(논문 GRADE 표기)']})"],
                textposition="top center",
                hovertemplate=(
                    f"<b>{r['비교_한글']}</b><br>MD {r['종합 평균차이(MD)']:+.3f} {unit}"
                    f"<br>95% CI {r['95% CI 하한']:.3f} ~ {r['95% CI 상한']:.3f}"
                    f"<br>비교 {r['비교 수']}건<extra></extra>"
                ),
            )
        )
        shown.add(r["판정"])
    fig.add_vline(x=0, line_dash="dash", line_color="black")
    fig.update_layout(
        xaxis_title=f"평균차이 MD ({unit})  ← 낮아짐 | 높아짐 →",
        yaxis=dict(autorange="reversed"),
        height=360,
        legend=dict(orientation="h", y=-0.3),
        margin=dict(l=10, r=10, t=20, b=10),
    )
    return fig


def study_forest(df, unit):
    """개별 연구: MD ± 1.96×SE, 점 크기 = 정밀도(1/SE)"""
    df = df.sort_values(["비교", "연구(저자·연도)"]).reset_index(drop=True)
    palette = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756"]
    prec = 1 / df["Standard Error"]
    sizes = 8 + 16 * prec / prec.max()
    fig = go.Figure()
    for i, (comp, g) in enumerate(df.groupby("비교", sort=False)):
        fig.add_trace(
            go.Scatter(
                x=g["Mean Difference"],
                y=g["y라벨"],
                mode="markers",
                name=comp,
                marker=dict(color=palette[i % len(palette)], size=sizes[g.index].tolist(), symbol="square"),
                error_x=dict(
                    type="data",
                    array=(Z95 * g["Standard Error"]).tolist(),
                    color=palette[i % len(palette)],
                    thickness=1.6,
                ),
                customdata=g[["Standard Error", "CI 하한", "CI 상한", "원문 PDF 페이지"]].values,
                hovertemplate=(
                    "<b>%{y}</b><br>MD %{x:.2f} " + unit + "<br>SE %{customdata[0]:.2f}"
                    "<br>95% CI %{customdata[1]:.2f} ~ %{customdata[2]:.2f}"
                    "<br>원문 PDF %{customdata[3]}쪽<extra></extra>"
                ),
            )
        )
    fig.add_vline(x=0, line_dash="dash", line_color="black")
    fig.update_layout(
        xaxis_title=f"평균차이 MD ({unit}) — 점 크기가 클수록 정밀한 연구",
        yaxis=dict(autorange="reversed", categoryorder="array", categoryarray=df["y라벨"].tolist()),
        height=max(380, 30 * len(df) + 170),
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=10, r=10, t=20, b=10),
    )
    return fig


GRADE_KO = {"High": "높음", "Moderate": "보통", "Low": "낮음", "Very low": "매우 낮음"}
VERDICT_EMOJI = {"개선(유의)": "✅", "유의한 차이 없음": "➖", "악화(유의)": "⚠️"}
VERDICT_TEXT = {"개선(유의)": "유의하게 낮아짐", "유의한 차이 없음": "뚜렷한 차이 없음", "악화(유의)": "유의하게 높아짐"}
VERDICT_COLOR = {"개선(유의)": "#2CA02C", "유의한 차이 없음": "#9AA5B1", "악화(유의)": "#D62728"}


def simple_bars(df, unit, metric_ko):
    """쉬운 종합 결과: 막대 = 평균 변화, 검은 선 = 오차 범위, 색 = 판정"""
    df = df.reset_index(drop=True)
    labels = [
        f"{r['비교_한글']}<br>{VERDICT_EMOJI[r['판정']]} {VERDICT_TEXT[r['판정']]} ({r['종합 평균차이(MD)']:+.2f}{unit})"
        for _, r in df.iterrows()
    ]
    md, lo, hi = df["종합 평균차이(MD)"], df["95% CI 하한"], df["95% CI 상한"]
    fig = go.Figure(
        go.Bar(
            x=md,
            y=labels,
            orientation="h",
            marker_color=[VERDICT_COLOR[j] for j in df["판정"]],
            error_x=dict(
                type="data",
                symmetric=False,
                array=(hi - md).tolist(),
                arrayminus=(md - lo).tolist(),
                color="#333333",
                thickness=2.2,
                width=8,
            ),
            customdata=df[["95% CI 하한", "95% CI 상한", "p값", "비교 수", "근거 확실성(논문 GRADE 표기)"]].values,
            hovertemplate=(
                "평균 변화 %{x:+.2f} " + unit + "<br>오차 범위 %{customdata[0]:.2f} ~ %{customdata[1]:.2f}"
                "<br>p=%{customdata[2]:.3f} · 비교 %{customdata[3]}건 · 근거 %{customdata[4]}<extra></extra>"
            ),
        )
    )
    fig.add_vline(x=0, line_color="black", line_width=2)
    fig.update_layout(
        xaxis_title=f"{metric_ko} 변화 ({unit})   ◀ 왼쪽(−): 낮아짐   |   오른쪽(+): 높아짐 ▶",
        yaxis=dict(autorange="reversed"),
        height=330,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
    )
    return fig


def study_bars(df, unit, show_ci, with_comp):
    """쉬운 개별 연구: 연구별 막대(왼쪽=음수, 오른쪽=양수), 색 = 혼자서도 유의한지"""
    df = df.copy().sort_values("Mean Difference").reset_index(drop=True)
    df["라벨"] = df["y라벨"] if with_comp else df["연구(저자·연도)"]

    def kind(r):
        if r["CI 상한"] < 0:
            return "혼자서도 유의하게 낮음"
        if r["CI 하한"] > 0:
            return "혼자서도 유의하게 높음"
        return "혼자서는 불확실 (오차 범위가 0 포함)"

    df["구분"] = df.apply(kind, axis=1)
    colors = {
        "혼자서도 유의하게 낮음": "#2CA02C",
        "혼자서도 유의하게 높음": "#D62728",
        "혼자서는 불확실 (오차 범위가 0 포함)": "#B8C4D6",
    }
    fig = go.Figure()
    for k, c in colors.items():
        g = df[df["구분"] == k]
        if g.empty:
            continue
        fig.add_trace(
            go.Bar(
                x=g["Mean Difference"],
                y=g["라벨"],
                orientation="h",
                name=k,
                marker_color=c,
                error_x=dict(type="data", array=(Z95 * g["Standard Error"]).tolist(), color="#555555", thickness=1.2)
                if show_ci
                else None,
                customdata=g[["Standard Error", "CI 하한", "CI 상한", "원문 PDF 페이지"]].values,
                hovertemplate=(
                    "<b>%{y}</b><br>MD %{x:+.2f} " + unit + "<br>오차 범위 %{customdata[1]:.2f} ~ %{customdata[2]:.2f}"
                    "<br>원문 PDF %{customdata[3]}쪽<extra></extra>"
                ),
            )
        )
    fig.add_vline(x=0, line_color="black", line_width=2)
    fig.update_layout(
        xaxis_title=f"비교별 평균차이 MD ({unit})   ◀ 왼쪽(−)  |  오른쪽(+) ▶",
        yaxis=dict(autorange="reversed", categoryorder="array", categoryarray=df["라벨"].tolist()),
        height=max(360, 26 * len(df) + 150),
        legend=dict(orientation="h", y=-0.15),
        margin=dict(l=10, r=10, t=10, b=10),
        bargap=0.25,
    )
    return fig, df


def verdict_markdown(summ):
    """데이터에서 자동으로 만든 한 줄 결론"""
    lines = []
    for c, ko in COMP_KO.items():
        sub = summ[summ["비교"] == c]
        good = sub.loc[sub["판정"] == "개선(유의)", "지표_한글"].tolist()
        bad = sub.loc[sub["판정"] == "악화(유의)", "지표_한글"].tolist()
        rest = len(sub) - len(good) - len(bad)
        parts = []
        if good:
            parts.append(f"✅ 유의하게 낮아진 지표: **{', '.join(good)}**")
        if bad:
            parts.append(f"⚠️ 유의하게 높아진 지표: **{', '.join(bad)}**")
        parts.append(f"나머지 {rest}개 지표는 유의한 차이 없음")
        lines.append(f"- **{ko}**: " + " / ".join(parts))
    return "\n".join(lines)


# ---- 이상치 분석용 그래프
def outlier_heatmap(d, comps):
    """연구 × 지표 이상치 점수(z) 히트맵"""
    h = d[d["comp"].isin(comps)].copy()
    h["row"] = h["study"] + " (" + h["comp"] + ")"
    piv = h.pivot_table(index="row", columns="metric", values="z", aggfunc="first")
    piv = piv.reindex(columns=[m for m in METRIC_KO.values() if m in piv.columns])
    fig = px.imshow(piv, color_continuous_scale="RdBu_r", zmin=-5, zmax=5, aspect="auto",
                    labels=dict(color="z"), text_auto=".1f")
    fig.update_layout(height=max(480, 22 * len(piv)), xaxis_side="top", margin=dict(l=10, r=10, t=40, b=10))
    return fig


def outlier_forest(g, unit, p_all, p_ex):
    """연구별 MD ± 95% CI (이상치 색 구분) + 합산(전체 / 이상치 제외)"""
    prec = 1 / g["SE"]
    g = g.assign(size=8 + 16 * prec / prec.max())
    fig = go.Figure()
    for lab, col in FLAG_COLOR.items():
        s = g[g["판정"] == lab]
        if s.empty:
            continue
        fig.add_trace(go.Scatter(
            x=s["MD"], y=s["study"], mode="markers", name=lab,
            marker=dict(color=col, symbol="square", size=s["size"].tolist()),
            error_x=dict(type="data", symmetric=False, array=(s["hi"] - s["MD"]).tolist(),
                         arrayminus=(s["MD"] - s["lo"]).tolist(), color=col, thickness=2),
            customdata=np.c_[s["SE"], s["z"], s["원문 PDF 페이지"]],
            hovertemplate="<b>%{y}</b><br>MD %{x:+.2f}  (SE %{customdata[0]:.2f})"
                          "<br>이상치 점수 z=%{customdata[1]:.1f}<br>원문 PDF %{customdata[2]}쪽<extra></extra>"))
    for p, name, col, sym in [(p_all, "합산(전체)", "#333333", "diamond"),
                              (p_ex, "합산(이상치 제외)", "#2CA02C", "diamond-open")]:
        fig.add_trace(go.Scatter(
            x=[p["mu"]], y=[name], mode="markers", name=name,
            marker=dict(color=col, symbol=sym, size=16, line=dict(width=2)),
            error_x=dict(type="data", symmetric=False, array=[p["hi"] - p["mu"]],
                         arrayminus=[p["mu"] - p["lo"]], color=col, thickness=3)))
    fig.add_vline(x=0, line_dash="dash", line_color="black")
    order = ["합산(이상치 제외)", "합산(전체)"] + g["study"].tolist()[::-1]
    fig.update_layout(
        height=max(400, 32 * (len(g) + 3) + 100),
        xaxis_title=f"평균차이 MD ({unit})   ◀ 왼쪽(−): 제로음료가 더 낮음  |  오른쪽(+): 더 높음 ▶",
        yaxis=dict(categoryorder="array", categoryarray=order),
        legend=dict(orientation="h", y=1.08), margin=dict(l=10, r=10, t=40, b=10))
    return fig


def exclusion_chart(res):
    """지표·비교별 합산 MD: 전체 vs 이상치 제외"""
    n = len(res)
    y = np.arange(n)
    fig = go.Figure()
    for key, name, col, off in [("pa", "전체", "#333333", 0.15), ("pe", "이상치 제외", "#2CA02C", -0.15)]:
        ps = res[key].tolist()
        fig.add_trace(go.Scatter(
            x=[p["mu"] for p in ps], y=y + off, mode="markers", name=name, marker=dict(color=col, size=10),
            error_x=dict(type="data", symmetric=False, array=[p["hi"] - p["mu"] for p in ps],
                         arrayminus=[p["mu"] - p["lo"] for p in ps], color=col)))
    fig.add_vline(x=0, line_dash="dash", line_color="black")
    fig.update_layout(
        height=max(480, 56 * n), legend=dict(orientation="h", y=1.05),
        xaxis_title="합산 MD (지표마다 단위가 다름)   ◀ 제로음료가 더 낮음  |  더 높음 ▶",
        yaxis=dict(tickmode="array", tickvals=y.tolist(),
                   ticktext=(res["지표"] + " · " + res["비교"]).tolist(), autorange="reversed"),
        margin=dict(l=10, r=10, t=40, b=10))
    return fig


# ------------------------------------------------------------------ 화면
st.set_page_config(page_title="제로칼로리 음료의 효과가 있을까?", layout="wide")
st.title("제로칼로리 음료, 효과가 있을까?")
st.caption(
    "성인 대상 임상시험 메타분석(McGlynn et al., JAMA Netw Open 2022) 보충자료의 체중·혈당 지표 · "
    "청소년에게 그대로 일반화할 수 없습니다."
)

study_full, summ_full = load_data()

remove_hba1c = st.toggle(
    f"당화혈색소 이상치 제거 ({HBA1C_OUTLIER_STUDY}, 물 vs 제로음료)", value=True,
    help="다른 이상치(체중·허리둘레의 Peters 2016 등)는 그대로 둡니다. 끄면 원래 자료로 돌아갑니다.",
)
if remove_hba1c:
    study_df, summ_df, removed_df = drop_hba1c_outlier(study_full, summ_full)
    st.success(
        f"당화혈색소 이상치 1건({HBA1C_OUTLIER_STUDY}, MD=+0.75%)을 제거했습니다. "
        "영향받는 '물 → 제로음료로 바꿈' 당화혈색소 칸은 남은 연구로 다시 합산했고(※재계산 표시), "
        "나머지 칸은 논문 값 그대로입니다.",
        icon="✅",
    )
else:
    study_df, summ_df = study_full, summ_full
    removed_df = study_full.iloc[0:0]
metric_options = list(dict.fromkeys(summ_df["지표"]))

tab1, tab2, tab3, tab4 = st.tabs(["🎯 한눈에 보는 결론", "🔍 지표별 자세히", "🔎 이상치 분석", "📄 원자료·한계"])

# ---- 탭 1: 결론
with tab1:
    st.subheader("무엇과 무엇을 비교했나요?")
    st.markdown(
        "논문은 음료를 **서로 바꿨을 때** 몸의 수치가 어떻게 달라지는지를 비교했습니다. "
        "모든 지표는 **낮아질수록 개선**이라고 보고 색을 칠했습니다."
    )
    st.markdown(verdict_markdown(summ_df))

    st.subheader("① 성적표: 지표 × 비교")
    st.markdown("🟩 유의하게 낮아짐(개선)  ⬜ 유의한 차이 없음  🟥 유의하게 높아짐(악화) · 칸 안 숫자는 평균차이 [95% 신뢰구간], 근거 확실성")
    st.plotly_chart(scorecard(summ_df), width="stretch")

    st.subheader("② 단위가 달라도 비교하기: 효과 신호의 세기(z값)")
    st.markdown(
        "체중(kg)과 혈당(mmol/L)은 단위가 달라 평균차이를 바로 비교할 수 없습니다. "
        "평균차이를 표준오차로 나눈 z값으로 바꾸면 한 그래프에서 견줄 수 있습니다. "
        "막대가 회색 띠(±1.96)를 벗어나야 유의한 차이입니다."
    )
    st.plotly_chart(zscore_bars(summ_df), width="stretch")

# ---- 탭 2: 지표별
with tab2:
    metric = st.selectbox("지표 선택", metric_options, format_func=lambda m: f"{METRIC_KO[m]} ({m})")
    unit = summ_df.loc[summ_df["지표"] == metric, "단위"].iloc[0]
    metric_ko = METRIC_KO[metric]
    t_df = summ_df[summ_df["지표"] == metric]
    s_df = study_df[study_df["지표"] == metric]

    st.subheader(f"{metric_ko} ({unit})")
    st.markdown("### 한 줄 결론")
    cols = st.columns(3)
    for col, (_, r) in zip(cols, t_df.iterrows()):
        with col:
            with st.container(border=True):
                st.markdown(f"**{r['비교_한글']}**")
                st.markdown(f"## {r['종합 평균차이(MD)']:+.2f} {unit}")
                st.markdown(f"{VERDICT_EMOJI[r['판정']]} **{VERDICT_TEXT[r['판정']]}**")
                grade = GRADE_KO.get(r["근거 확실성(논문 GRADE 표기)"], r["근거 확실성(논문 GRADE 표기)"])
                st.caption(
                    f"오차 범위 {r['95% CI 하한']:.2f} ~ {r['95% CI 상한']:.2f} · "
                    f"근거 확실성 {grade} · 비교 {r['비교 수']}건"
                )
                if r["재계산"]:
                    st.caption("※ 이상치 제거 후 직접 재계산한 값(랜덤효과 근사)")

    st.markdown("### ① 음료를 바꾸면 평균 얼마나 달라졌나")
    st.plotly_chart(simple_bars(t_df, unit, metric_ko), width="stretch")
    st.caption(
        "막대 = 평균 변화, 검은 선 = 오차 범위(95% 신뢰구간). "
        "🟩 유의하게 낮아짐 · ⬜ 뚜렷한 차이 없음 · 🟥 유의하게 높아짐. "
        "검은 선이 0(굵은 세로선)을 지나면 '차이가 있다고 확신할 수 없다'는 뜻입니다."
    )

    st.markdown("### ② 비교 하나하나는 어땠나")
    comps = sorted(s_df["비교"].unique())
    choice = st.radio("비교 선택", ["전체"] + comps, horizontal=True)
    show_ci = st.checkbox("오차 범위 선도 함께 보기", value=False)
    view = s_df if choice == "전체" else s_df[s_df["비교"] == choice]
    fig_s, view = study_bars(view, unit, show_ci, with_comp=(choice == "전체"))
    n = len(view)
    n_neg = int((view["Mean Difference"] < 0).sum())
    n_sig = int((view["구분"] != "혼자서는 불확실 (오차 범위가 0 포함)").sum())
    n_papers = view["연구(저자·연도)"].str.replace(r"\s*\(.*\)", "", regex=True).nunique()
    st.markdown(
        f"**{n}개 비교**(서로 다른 논문 {n_papers}편) 중 **{n_neg}개**가 음수(−, 왼쪽), 혼자서도 유의한 비교는 **{n_sig}개**입니다. "
        "한 논문이 여러 비교(남녀, 감미료 종류, 용량)를 담고 있어서 논문 수보다 비교 수가 많습니다. "
        "대부분 혼자서는 불확실하지만, 여러 비교가 같은 방향이면 합쳐서 볼 때 유의해질 수 있습니다."
    )
    st.plotly_chart(fig_s, width="stretch")
    st.caption(
        "개별 연구의 MD 부호는 Treatment 1/2 순서와 논문 정의를 확인해야 해석할 수 있습니다. "
        "같은 연구가 여러 행(예: 남성·여성, 여러 감미료)으로 들어 있어 행끼리 독립적이지 않습니다."
    )

    with st.expander("전문가용: 포리스트 플롯으로 보기"):
        st.plotly_chart(metric_forest(t_df, unit), width="stretch")
        st.plotly_chart(study_forest(s_df, unit), width="stretch")
        st.caption("점 = 평균차이, 선 = 95% 신뢰구간(MD ± 1.96×SE 근사), 점 크기 = 연구의 정밀도.")

# ---- 탭 3: 이상치 분석
with tab3:
    st.subheader("연구별 이상치 찾기")
    st.markdown(
        "같은 지표·같은 비교 안에서 **다른 연구들과 크게 벗어난 값**을 찾고, "
        "그 값을 빼면 제로음료 효과의 결론이 바뀌는지 확인합니다."
    )
    st.info(
        "부호 해석 가정: 제로음료가 들어간 비교는 **MD < 0 이면 제로음료 쪽이 더 낮음**으로 봤습니다"
        "(논문 합산 결과와 방향 일치 확인). 인용 전에 원문 PDF로 확인하세요.",
        icon="ℹ️",
    )

    if remove_hba1c:
        st.markdown("### 당화혈색소 이상치 제거 전 · 후")
        full = flag_outliers(study_full, 3.0)
        hg = full[(full["metric"] == "당화혈색소") & (full["comp"] == "물 vs 제로음료")].sort_values("MD").reset_index(drop=True)
        hg["판정"] = np.where(hg["study"] == HBA1C_OUTLIER_STUDY, "이상치", "정상")
        hg_keep = hg[hg["판정"] != "이상치"]
        hp_all, hp_ex = re_pool(hg["MD"], hg["SE"]), re_pool(hg_keep["MD"], hg_keep["SE"])
        st.plotly_chart(outlier_forest(hg, "%", hp_all, hp_ex), width="stretch")
        b1, b2 = st.columns(2)
        b1.metric("제거 전 (물 → 제로음료, 직접 합산)", fmt_ci(hp_all), ci_verdict(hp_all), delta_color="off")
        b2.metric("제거 후", fmt_ci(hp_ex), ci_verdict(hp_ex), delta_color="off")
        st.caption(
            "빨간 사각형이 제거한 연구입니다. 이 한 건이 '제로음료를 마시면 당화혈색소가 높아진다'는 신호를 거의 혼자 만들었고, "
            "빼면 0 근처로 돌아옵니다. 아래 분석(한눈에 보기 등)은 이 연구를 뺀 자료 기준입니다."
        )
        st.divider()

    oc1, oc2 = st.columns([2, 1])
    z_out = oc1.slider(
        "이상치 기준 |z|", 2.0, 5.0, 3.0, 0.5,
        help="나머지 연구들로 합산한 값에서 벗어난 정도(표준화 점수). 2 이상은 '주의'로 표시",
    )
    with_sw = oc2.checkbox("설탕음료 vs 물 비교도 포함", value=False)

    od = flag_outliers(study_df, z_out)
    zero_comps = [c for c in sorted(od["comp"].unique()) if "제로음료" in c]
    o_comps = zero_comps + ([c for c in sorted(od["comp"].unique()) if "제로음료" not in c] if with_sw else [])
    o_metrics = [m for m in METRIC_KO.values() if m in set(od["metric"])]

    o1, o2, o3, o4 = st.tabs(["한눈에 보기", "지표별 포레스트", "이상치 빼면 결론이 바뀌나?", "데이터 점검"])

    with o1:
        st.markdown("진한 빨강·파랑일수록 같은 지표의 다른 연구들과 많이 다른 값입니다. (연구 3개 미만인 칸은 빈칸)")
        st.plotly_chart(outlier_heatmap(od, o_comps), width="stretch")
        found = od[(od["판정"] == "이상치") & od["comp"].isin(o_comps)]
        if len(found):
            st.error("이상치: " + " · ".join(
                f"**{r.study}** ({r.metric}, {r.comp}, MD={r.MD:+.2f}, z={r.z:.1f})" for r in found.itertuples()))
        else:
            st.success("현재 기준에서는 이상치가 없습니다.")
        warn = od[(od["판정"] == "주의") & od["comp"].isin(o_comps)]
        if len(warn):
            st.warning("주의(경계): " + " · ".join(
                f"{r.study} ({r.metric}, z={r.z:.1f})" for r in warn.itertuples()))

    with o2:
        f1, f2 = st.columns(2)
        o_metric = f1.selectbox("지표", o_metrics, key="o_metric")
        o_comp = f2.radio("비교", o_comps, horizontal=True, key="o_comp")
        og = od[(od["metric"] == o_metric) & (od["comp"] == o_comp)].sort_values("MD").reset_index(drop=True)
        if og.empty:
            st.info("해당 조합의 연구가 없습니다.")
        else:
            o_unit = og["단위"].iloc[0]
            keep = og[og["판정"] != "이상치"]
            p_all, p_ex = re_pool(og["MD"], og["SE"]), re_pool(keep["MD"], keep["SE"])
            st.plotly_chart(outlier_forest(og, o_unit, p_all, p_ex), width="stretch")
            m1, m2, m3 = st.columns(3)
            m1.metric("합산 MD (전체)", fmt_ci(p_all), ci_verdict(p_all), delta_color="off")
            m2.metric("합산 MD (이상치 제외)", fmt_ci(p_ex), ci_verdict(p_ex), delta_color="off")
            m3.metric("연구 간 이질성 I²", f"{p_all['i2']:.0f}%")
            st.caption(
                "사각형 크기 = 연구 정밀도, 선 = 95% 신뢰구간. 이상치 빨강, 주의 주황. "
                "합산은 랜덤효과(DerSimonian-Laird) 근사라 논문의 네트워크 메타분석 값과 조금 다를 수 있습니다."
            )

    with o3:
        rows = []
        for m in o_metrics:
            for c in zero_comps:
                g = od[(od["metric"] == m) & (od["comp"] == c)]
                if g.empty:
                    continue
                k = g[g["판정"] != "이상치"]
                pa, pe = re_pool(g["MD"], g["SE"]), re_pool(k["MD"], k["SE"])
                rows.append(dict(
                    지표=m, 비교=c, 연구수=len(g),
                    제외한연구=", ".join(g[g["판정"] == "이상치"]["study"]) or "-",
                    전체=fmt_ci(pa), 이상치제외=fmt_ci(pe),
                    판정_전체=ci_verdict(pa), 판정_제외=ci_verdict(pe),
                    변화="⚠️ 바뀜" if ci_verdict(pa) != ci_verdict(pe) else "", pa=pa, pe=pe,
                ))
        res = pd.DataFrame(rows)
        st.plotly_chart(exclusion_chart(res), width="stretch")
        st.dataframe(res.drop(columns=["pa", "pe"]), width="stretch", hide_index=True)
        changed = res[res["변화"] != ""]
        if len(changed):
            st.warning("이상치를 빼면 판정이 바뀌는 항목: " + ", ".join(
                f"{r.지표}({r.비교})" for r in changed.itertuples()))
        else:
            st.success("이상치를 빼도 유의성 판정은 바뀌지 않습니다.")
        st.caption("연구가 2~4개뿐인 칸은 이상치 판정·합산 모두 불안정하니 참고용으로만 보세요.")

    with o4:
        st.markdown("**1) 비교 수: 개별 자료 행 수 vs 종합분석표의 '비교 수'**")
        cnt = od.groupby("metric").size().rename("개별 자료 행수")
        sc = summ_df.groupby("지표_한글")["비교 수"].first().rename("종합분석표 비교 수")
        chk = pd.concat([cnt, sc], axis=1).reindex(o_metrics)
        chk["일치"] = np.where(chk.iloc[:, 0] == chk.iloc[:, 1], "✅", "❌ 불일치")
        st.dataframe(chk, width="stretch")

        st.markdown("**2) 삼각 일관성**: 같은 연구에서 (설탕-물) ≈ (설탕-제로) − (물-제로) 이어야 합니다.")
        tri = []
        for (m, stu), g in od.groupby(["metric", "study"]):
            v = g.set_index("comp")["MD"]
            need = {"설탕음료 vs 제로음료", "물 vs 제로음료", "설탕음료 vs 물"}
            if need <= set(v.index):
                exp = v["설탕음료 vs 제로음료"] - v["물 vs 제로음료"]
                tri.append(dict(지표=m, 연구=stu, 기대값=round(exp, 2), 보고값=v["설탕음료 vs 물"],
                                차이=round(abs(exp - v["설탕음료 vs 물"]), 2)))
        tri = pd.DataFrame(tri)
        if len(tri):
            tri["판정"] = np.where(tri["차이"] > 0.15, "❌ 불일치 - 원문 확인", "✅")
            st.dataframe(tri, width="stretch", hide_index=True)

        st.markdown("**3) 판정 포함 전체 데이터**")
        show_cols = ["metric", "단위", "study", "comp", "MD", "SE", "z", "판정", "원문 PDF 페이지"]
        st.dataframe(od[show_cols].round(2), width="stretch", hide_index=True)
        st.download_button("판정 포함 CSV 다운로드", od[show_cols].round(3).to_csv(index=False).encode("utf-8-sig"),
                           "outlier_flags.csv", "text/csv")

# ---- 탭 4: 원자료·한계
with tab4:
    st.subheader("결과를 읽을 때 주의할 점")
    st.markdown(
        "- 대상은 **성인 임상시험**입니다. 청소년에게 그대로 적용할 수 없습니다.\n"
        "- 근거 확실성(GRADE)이 대부분 **Low**입니다. 연구가 더 나오면 결과가 바뀔 수 있습니다.\n"
        "- 신뢰구간이 0을 지나면 '효과가 없다'가 아니라 **'이 자료로는 차이를 확신할 수 없다'**는 뜻입니다.\n"
        "- 이 자료는 음료를 **바꿨을 때**의 변화입니다. 제로음료를 추가로 마시는 경우와는 다릅니다.\n"
        "- 모든 지표를 '낮을수록 개선'으로 가정한 색 구분입니다. 건강한 성인에게는 작은 차이가 임상적으로 의미 없을 수 있습니다.\n"
        "- 이상치 분석의 합산값은 직접 계산한 근사치라, 위의 논문 종합 결과와 소수점 단위로 다를 수 있습니다."
    )
    st.subheader("종합분석 표")
    st.dataframe(
        summ_df[["지표_한글", "단위", "비교_한글", "종합 평균차이(MD)", "95% CI 하한", "95% CI 상한",
                 "p값", "비교 수", "근거 확실성(논문 GRADE 표기)", "판정", "재계산"]],
        width="stretch",
        hide_index=True,
    )
    if remove_hba1c and len(removed_df):
        st.subheader("제거한 데이터")
        st.dataframe(removed_df.drop(columns=["y라벨", "해석 주의"], errors="ignore"), width="stretch", hide_index=True)
    st.subheader("개별 연구 원자료")
    st.dataframe(study_df.drop(columns=["y라벨", "해석 주의"]), width="stretch", hide_index=True)
    st.subheader("출처")
    st.dataframe(pd.read_excel(EXCEL_FILE, sheet_name="출처와_주의사항"), width="stretch", hide_index=True)
    # ───────────────────────── 로지스틱 회귀 임계값 튜닝 (추가) ─────────────────────────
# 로지스틱 회귀 하나만 leave-one-paper-out으로 예측 확률을 만든 뒤, 임계값을 바꿔 가며 성능이 어떻게 변하는지 보고
# 기준(Youden J / F1 / MCC / 비용 / 재현율 하한)에 따라 최적 임계값을 찾는다.
THR_GRID = np.round(np.arange(0.01, 1.00, 0.01), 2)
TARGETS = {"방향: 제로음료 쪽이 낮음(−)": "dir", "유의성: '개선'(CI 전체가 0 아래) vs 나머지": "sig"}
CRITERIA = ["Youden J (민감도+특이도−1)", "F1", "MCC", "비용 최소화 (FN·FP 비용)", "재현율 하한 아래 정밀도 최대"]


@st.cache_data
def logit_oof(x, target, C, balanced):
    """로지스틱 회귀의 leave-one-paper-out 양성 확률"""
    y = ((x.true_dir == DIRS[0]) if target == "dir" else (x.true_sig == "개선")).astype(int).values
    F, oof = x[CAT + NUM], np.zeros(len(x))
    for tr, te in LeaveOneGroupOut().split(F, y, groups=x.paper):
        if len(np.unique(y[tr])) < 2:  # 학습에 클래스가 하나뿐이면 평활된 사전확률로 대체
            oof[te] = (y[tr].sum() + 1) / (len(tr) + 2)
            continue
        pipe = clf_pipe(CLF_NAMES[0]).set_params(clf__C=C, clf__class_weight="balanced" if balanced else None)
        oof[te] = pipe.fit(F.iloc[tr], y[tr]).predict_proba(F.iloc[te])[:, 1]
    return y, oof


def sweep(y, p, grid=THR_GRID):
    """임계값 격자 전체에 대한 혼동행렬·지표를 한 번에 계산"""
    pred, pos = p[None, :] >= grid[:, None], (y == 1)[None, :]
    tp, fp = (pred & pos).sum(1).astype(float), (pred & ~pos).sum(1).astype(float)
    fn, tn = (~pred & pos).sum(1).astype(float), (~pred & ~pos).sum(1).astype(float)
    div = lambda a, b: np.divide(a, b, out=np.zeros_like(a), where=b > 0)
    rec, spec, prec = div(tp, tp + fn), div(tn, tn + fp), div(tp, tp + fp)
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return pd.DataFrame(dict(thr=grid, tp=tp, fp=fp, fn=fn, tn=tn, 재현율=rec, 특이도=spec, 정밀도=prec,
                             F1=div(2 * prec * rec, prec + rec), 균형정확도=(rec + spec) / 2, 정확도=(tp + tn) / len(y),
                             MCC=div(tp * tn - fp * fn, den), Youden=rec + spec - 1))


def pick_thr(sw, crit, c_fn=1.0, c_fp=1.0, min_rec=0.8):
    """기준별 최적 임계값 (동점이면 가운데 값)"""
    if crit == CRITERIA[0]:
        s = sw.Youden.values
    elif crit == CRITERIA[1]:
        s = sw.F1.values
    elif crit == CRITERIA[2]:
        s = sw.MCC.values
    elif crit == CRITERIA[3]:
        s = -(c_fn * sw.fn.values + c_fp * sw.fp.values)
    else:
        s = np.where(sw.재현율.values >= min_rec, sw.정밀도.values, -np.inf)
    if not np.isfinite(s).any():  # 재현율 하한을 만족하는 임계값이 없음
        return float(sw.thr.min())
    idx = np.flatnonzero(s >= s.max() - 1e-12)
    return float(sw.thr.values[idx[len(idx) // 2]])


def boot_thr(papers, y, p, crit, c_fn, c_fp, min_rec, B=300, seed=0):
    """논문 단위 부트스트랩: 논문을 복원추출해 최적 임계값이 얼마나 흔들리는지 확인"""
    rng, uniq = np.random.default_rng(seed), pd.unique(papers)
    grp, out = {k: np.flatnonzero(papers == k) for k in uniq}, []
    for _ in range(B):
        ids = np.concatenate([grp[k] for k in rng.choice(uniq, len(uniq))])
        if y[ids].min() == y[ids].max():
            continue
        out.append(pick_thr(sweep(y[ids], p[ids]), crit, c_fn, c_fp, min_rec))
    return np.array(out)


def render_threshold_tab():
    st.subheader("로지스틱 회귀 · 분류 임계값 튜닝")
    st.caption("확률이 임계값 이상이면 '양성'으로 분류합니다. 임계값을 낮추면 놓치는 양성(FN)이 줄고 오경보(FP)가 늘며, 높이면 반대가 됩니다. "
               "아래 예측 확률은 leave-one-paper-out으로 만든 것이라 같은 논문의 정답이 학습에 들어가지 않습니다.")
    if len(X) < 10 or X.paper.nunique() < 4:
        st.info("데이터가 부족합니다(논문 4편 이상, 10행 이상 필요).")
        return

    c1, c2, c3 = st.columns([2, 1, 1])
    tgt_name = c1.radio("예측 대상 (양성 클래스)", list(TARGETS), key="th_tgt")
    C = c2.select_slider("규제 강도 C", options=[0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0], value=1.0, key="th_C",
                         help="작을수록 규제가 강해 확률이 0.5 근처로 모입니다.")
    bal = c3.checkbox("클래스 가중치 balanced", value=True, key="th_bal",
                      help="켜면 소수 클래스를 더 중요하게 학습해 확률이 한쪽으로 쏠리는 것을 줄입니다.")
    target = TARGETS[tgt_name]
    pos_nm, neg_nm = ("낮음(−)", "높음(+)") if target == "dir" else ("개선", "그 외")

    y, p = logit_oof(X, target, C, bal)
    if y.sum() in (0, len(y)):
        st.warning("이 대상에서는 한 클래스만 존재해 임계값을 튜닝할 수 없습니다. 다른 대상을 선택하세요.")
        return

    k1, k2 = st.columns([2, 3])
    crit = k1.selectbox("최적 임계값 기준", CRITERIA, key="th_crit")
    c_fn, c_fp, min_rec = 1.0, 1.0, 0.8
    with k2:
        if crit == CRITERIA[3]:
            a, b = st.columns(2)
            c_fn = a.number_input(f"FN 비용 ('{pos_nm}'을 놓침)", min_value=0.0, value=2.0, step=0.5, key="th_cfn")
            c_fp = b.number_input(f"FP 비용 ('{pos_nm}'이 아닌데 양성으로 분류)", min_value=0.0, value=1.0, step=0.5, key="th_cfp")
        elif crit == CRITERIA[4]:
            min_rec = st.slider("재현율 하한", 0.1, 1.0, 0.8, 0.05, key="th_minrec")
        else:
            st.caption("균형정확도는 Youden J와 같은 지점에서 최대가 됩니다. 불균형이 심하면 정확도 대신 J·F1·MCC를 쓰세요.")

    sw = sweep(y, p)
    best = pick_thr(sw, crit, c_fn, c_fp, min_rec)
    thr = st.slider("임계값 (직접 조절)", 0.01, 0.99, float(best), 0.01, key=f"th_thr_{crit}_{best}",
                    help="기본값은 위에서 고른 기준의 최적값입니다. 움직여서 지표가 어떻게 변하는지 확인하세요.")
    row = sw.loc[(sw.thr - thr).abs().idxmin()]
    r05 = sw.loc[(sw.thr - 0.5).abs().idxmin()]
    st.caption(f"기준 '{crit}'의 최적 임계값 = **{best:.2f}** · 현재 {thr:.2f} · 양성 비율 {y.mean():.0%} (n={len(y)}, 양성 {int(y.sum())}) · "
               f"AUC {roc_auc_score(y, p):.2f}  (AUC는 임계값과 무관한 순위 성능)")

    for col, k in zip(st.columns(6), ["재현율", "정밀도", "특이도", "F1", "균형정확도", "MCC"]):
        col.metric(k, f"{row[k]:.2f}", f"{row[k] - r05[k]:+.2f} vs 0.5", delta_color="off")

    # 1) 임계값별 지표 곡선
    mcol = {"재현율": "#E45756", "정밀도": "#4C78A8", "특이도": "#54A24B", "F1": "#F58518", "균형정확도": "#B279A2", "MCC": "#7F7F7F"}
    f = go.Figure()
    for k, col in mcol.items():
        f.add_trace(go.Scatter(x=sw.thr, y=sw[k], mode="lines", name=k, line=dict(color=col, width=2.5)))
    f.add_vline(x=thr, line_color="black")
    f.add_vline(x=best, line_color="red", line_dash="dash", annotation_text=f"최적 {best:.2f}", annotation_position="top")
    f.update_layout(title="임계값에 따른 지표 변화 (검은 선=현재, 빨간 점선=최적)", height=380, xaxis_title="임계값",
                    margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(f, width="stretch")

    # 2) ROC · PR · 비용 곡선
    g1, g2, g3 = st.columns(3)
    roc = pd.concat([pd.DataFrame(dict(x=[1.0], y=[1.0])), pd.DataFrame(dict(x=1 - sw.특이도, y=sw.재현율)),
                     pd.DataFrame(dict(x=[0.0], y=[0.0]))])
    f = go.Figure([go.Scatter(x=roc.x, y=roc.y, mode="lines", line=dict(color="#4C78A8", width=3), name="ROC"),
                   go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color="gray"), showlegend=False),
                   go.Scatter(x=[1 - row.특이도], y=[row.재현율], mode="markers", marker=dict(size=14, color="black"), name=f"현재 {thr:.2f}")])
    f.update_layout(title="ROC", height=330, xaxis_title="거짓양성률", yaxis_title="재현율", margin=dict(l=10, r=10, t=50, b=10),
                    legend=dict(orientation="h", y=-0.3))
    g1.plotly_chart(f, width="stretch")
    pr = sw[(sw.tp + sw.fp) > 0]
    f = go.Figure([go.Scatter(x=pr.재현율, y=pr.정밀도, mode="lines", line=dict(color="#54A24B", width=3), name="PR"),
                   go.Scatter(x=[0, 1], y=[y.mean()] * 2, mode="lines", line=dict(dash="dot", color="gray"), name="무작위(양성 비율)"),
                   go.Scatter(x=[row.재현율], y=[row.정밀도], mode="markers", marker=dict(size=14, color="black"), name=f"현재 {thr:.2f}")])
    f.update_layout(title="정밀도–재현율", height=330, xaxis_title="재현율", yaxis_title="정밀도", yaxis=dict(range=[0, 1.02]),
                    margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.3))
    g2.plotly_chart(f, width="stretch")
    cost = c_fn * sw.fn + c_fp * sw.fp
    f = go.Figure([go.Scatter(x=sw.thr, y=cost, mode="lines", line=dict(color="#E45756", width=3), name="총비용"),
                   go.Scatter(x=[thr], y=[c_fn * row.fn + c_fp * row.fp], mode="markers", marker=dict(size=14, color="black"), name="현재")])
    f.update_layout(title=f"총비용 (FN×{c_fn:g} + FP×{c_fp:g})", height=330, xaxis_title="임계값", margin=dict(l=10, r=10, t=50, b=10),
                    legend=dict(orientation="h", y=-0.3))
    g3.plotly_chart(f, width="stretch")
    if crit != CRITERIA[3]:
        g3.caption("비용 곡선은 FN·FP 비용을 직접 정하려면 '비용 최소화' 기준을 선택하세요(지금은 1:1 가정).")

    # 3) 확률 분포 · 혼동행렬
    h1, h2 = st.columns([3, 2])
    f = go.Figure()
    for v, nm, col in [(1, pos_nm, "#4C78A8"), (0, neg_nm, "#F58518")]:
        f.add_trace(go.Histogram(x=p[y == v], name=f"실제 {nm}", marker_color=col, opacity=0.65, xbins=dict(start=0, end=1, size=0.05)))
    f.add_vline(x=thr, line_color="black")
    f.update_layout(barmode="overlay", title="예측 확률 분포 (두 분포가 겹칠수록 어떤 임계값도 완벽할 수 없음)", height=330,
                    xaxis_title=f"P({pos_nm})", margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.3))
    h1.plotly_chart(f, width="stretch")
    z = [[row.tn, row.fp], [row.fn, row.tp]]
    f = go.Figure(go.Heatmap(z=z, x=[f"예측 {neg_nm}", f"예측 {pos_nm}"], y=[f"실제 {neg_nm}", f"실제 {pos_nm}"], text=z,
                             texttemplate="%{text:.0f}", showscale=False, colorscale="Blues", xgap=2, ygap=2))
    f.update_yaxes(autorange="reversed")
    f.update_layout(title=f"혼동행렬 (임계값 {thr:.2f})", height=330, margin=dict(l=10, r=10, t=50, b=10))
    h2.plotly_chart(f, width="stretch")

    # 4) 최적 임계값의 안정성 (논문 단위 부트스트랩)
    if st.checkbox("최적 임계값이 얼마나 안정적인지 확인 (논문 단위 부트스트랩 300회)", key="th_boot"):
        bt = boot_thr(X.paper.values, y, p, crit, c_fn, c_fp, min_rec)
        if len(bt) < 10:
            st.info("유효한 부트스트랩 표본이 부족합니다.")
        else:
            lo, md, hi = np.percentile(bt, [5, 50, 95])
            f = go.Figure(go.Histogram(x=bt, xbins=dict(start=0, end=1, size=0.02), marker_color="#B279A2"))
            f.add_vline(x=best, line_color="red", line_dash="dash")
            f.update_layout(title=f"부트스트랩 최적 임계값 분포: 중앙값 {md:.2f}, 90% 구간 [{lo:.2f}, {hi:.2f}]", height=300,
                            xaxis_title="최적 임계값", margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(f, width="stretch")
            st.caption("구간이 넓으면 최적 임계값이 표본 몇 건에 좌우된다는 뜻입니다. 이때는 최적값 하나보다 '넓게 평평한 구간'을 고르는 편이 안전합니다.")

    with st.expander("연구별 예측 결과 (현재 임계값 기준)"):
        pred = p >= thr
        out = X[["metric", "study", "edge"]].assign(실제=np.where(y == 1, pos_nm, neg_nm), 확률=p, 예측=np.where(pred, pos_nm, neg_nm),
                                                   판정=np.select([pred & (y == 1), pred & (y == 0), ~pred & (y == 1)], ["TP", "FP", "FN"], "TN"))
        out["edge"] = out.edge.map(lambda e: EDGES[e][2])
        st.dataframe(out.rename(columns={"metric": "지표", "study": "연구", "edge": "비교"}).sort_values("확률", ascending=False)
                     .style.format({"확률": "{:.2f}"}), hide_index=True, width="stretch")
    with st.expander("임계값별 전체 지표표"):
        st.dataframe(sw.drop(columns=["Youden"]).style.format({c: "{:.2f}" for c in sw.columns if c not in ("tp", "fp", "fn", "tn")}
                                                               | {c: "{:.0f}" for c in ("tp", "fp", "fn", "tn")}),
                     hide_index=True, width="stretch")


with tt:
    render_threshold_tab()
