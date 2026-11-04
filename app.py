import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

EXCEL_FILE = "제로음료_임상시험_원자료_체중혈당.xlsx"
Z95 = 1.96  # 95% 신뢰구간 근사 (MD ± 1.96 × SE)

METRIC_KO = {
    "Body Weight": "체중",
    "BMI": "BMI",
    "Body Fat": "체지방률",
    "Waist Circumference": "허리둘레",
    "HbA1c": "당화혈색소(HbA1c)",
    "Fasting Blood Glucose": "공복혈당",
    "2-Hour Post-Prandial Glucose": "식후 2시간 혈당",
}
GROUP_KO = {"SSB": "설탕음료(SSB)", "Water": "물", "LNCSB": "제로음료(LNCSB)"}
COMP_KO = {
    "LNCSBs for SSBs": "SSB → 제로음료 대체",
    "Water for SSBs": "SSB → 물 대체",
    "LNCSBs for Water": "물 → 제로음료 대체",
}
PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756"]
SIG_COLOR, NONSIG_COLOR = "#D62728", "#7F7F7F"


@st.cache_data
def load_data():
    study = pd.read_excel(EXCEL_FILE, sheet_name="개별연구_원자료").dropna(subset=["지표"])
    summary = pd.read_excel(EXCEL_FILE, sheet_name="종합분석_결과").dropna(subset=["지표"])

    study["지표_한글"] = study["지표"].map(METRIC_KO)
    study["비교"] = study["Treatment 1"].map(GROUP_KO) + " vs " + study["Treatment 2"].map(GROUP_KO)
    study["y라벨"] = study["연구(저자·연도)"] + "  [" + study["Treatment 1"] + " vs " + study["Treatment 2"] + "]"
    study["CI 하한"] = study["Mean Difference"] - Z95 * study["Standard Error"]
    study["CI 상한"] = study["Mean Difference"] + Z95 * study["Standard Error"]

    summary["지표_한글"] = summary["지표"].map(METRIC_KO)
    summary["비교_한글"] = summary["비교"].map(COMP_KO)
    summary["유의"] = (summary["95% CI 하한"] > 0) | (summary["95% CI 상한"] < 0)
    return study, summary


def study_forest(df, unit, title):
    """개별 연구 포리스트 플롯: 연구별 평균차이(MD)와 95% CI."""
    df = df.sort_values(["비교", "연구(저자·연도)"]).reset_index(drop=True)
    fig = go.Figure()
    for i, (comp, g) in enumerate(df.groupby("비교", sort=False)):
        fig.add_trace(
            go.Scatter(
                x=g["Mean Difference"],
                y=g["y라벨"],
                mode="markers",
                name=comp,
                marker=dict(color=PALETTE[i % len(PALETTE)], size=10, symbol="square"),
                error_x=dict(
                    type="data",
                    array=(Z95 * g["Standard Error"]).tolist(),
                    color=PALETTE[i % len(PALETTE)],
                    thickness=1.8,
                ),
                customdata=g[["Standard Error", "CI 하한", "CI 상한", "원문 PDF 페이지"]].values,
                hovertemplate=(
                    "<b>%{y}</b><br>MD: %{x:.2f} " + unit + "<br>SE: %{customdata[0]:.2f}"
                    "<br>95% CI: %{customdata[1]:.2f} ~ %{customdata[2]:.2f}"
                    "<br>원문 PDF %{customdata[3]}쪽<extra></extra>"
                ),
            )
        )
    fig.add_vline(x=0, line_dash="dash", line_color="black")
    fig.update_layout(
        title=title,
        xaxis_title=f"평균차이 MD ({unit})  — 점: MD, 선: 95% CI",
        yaxis=dict(autorange="reversed", categoryorder="array", categoryarray=df["y라벨"].tolist()),
        height=max(380, 30 * len(df) + 170),
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def summary_forest(df, unit, title):
    """종합분석 포리스트 플롯: 비교 3종의 종합 MD와 95% CI."""
    df = df.reset_index(drop=True)
    fig = go.Figure()
    shown = set()
    for _, r in df.iterrows():
        label = "95% CI가 0을 포함하지 않음(유의)" if r["유의"] else "95% CI가 0을 포함(비유의)"
        color = SIG_COLOR if r["유의"] else NONSIG_COLOR
        fig.add_trace(
            go.Scatter(
                x=[r["종합 평균차이(MD)"]],
                y=[r["비교_한글"]],
                mode="markers+text",
                name=label,
                legendgroup=label,
                showlegend=label not in shown,
                marker=dict(color=color, size=14, symbol="diamond"),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=[r["95% CI 상한"] - r["종합 평균차이(MD)"]],
                    arrayminus=[r["종합 평균차이(MD)"] - r["95% CI 하한"]],
                    color=color,
                    thickness=2.2,
                ),
                text=[f"GRADE {r['근거 확실성(논문 GRADE 표기)']} · p={r['p값']:.3f}"],
                textposition="top center",
                hovertemplate=(
                    f"<b>{r['비교']}</b><br>MD: {r['종합 평균차이(MD)']:.3f} {unit}"
                    f"<br>95% CI: {r['95% CI 하한']:.3f} ~ {r['95% CI 상한']:.3f}"
                    f"<br>p값: {r['p값']:.3f}<br>비교 수: {r['비교 수']}"
                    f"<br>원문 PDF {r['원문 PDF 페이지']}쪽<extra></extra>"
                ),
            )
        )
        shown.add(label)
    fig.add_vline(x=0, line_dash="dash", line_color="black")
    fig.update_layout(
        title=title,
        xaxis_title=f"종합 평균차이 MD ({unit})",
        yaxis=dict(autorange="reversed"),
        height=360,
        legend=dict(orientation="h", y=-0.3),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def overview_grid(summary):
    """7개 지표의 종합분석 결과를 한 화면에 모은 작은 포리스트 플롯 모음."""
    metrics = list(dict.fromkeys(summary["지표"]))
    titles = [
        f"{METRIC_KO[m]} ({summary.loc[summary['지표'] == m, '단위'].iloc[0]})" for m in metrics
    ]
    fig = make_subplots(rows=2, cols=4, subplot_titles=titles, horizontal_spacing=0.08, vertical_spacing=0.18)
    shown = set()
    comps = list(COMP_KO.values())
    for idx, m in enumerate(metrics):
        row, col = idx // 4 + 1, idx % 4 + 1
        for _, r in summary[summary["지표"] == m].iterrows():
            label = "유의(CI가 0 제외)" if r["유의"] else "비유의(CI가 0 포함)"
            color = SIG_COLOR if r["유의"] else NONSIG_COLOR
            fig.add_trace(
                go.Scatter(
                    x=[r["종합 평균차이(MD)"]],
                    y=[r["비교_한글"]],
                    mode="markers",
                    name=label,
                    legendgroup=label,
                    showlegend=label not in shown,
                    marker=dict(color=color, size=10, symbol="diamond"),
                    error_x=dict(
                        type="data",
                        symmetric=False,
                        array=[r["95% CI 상한"] - r["종합 평균차이(MD)"]],
                        arrayminus=[r["종합 평균차이(MD)"] - r["95% CI 하한"]],
                        color=color,
                    ),
                    hovertemplate=(
                        f"<b>{METRIC_KO[m]} · {r['비교_한글']}</b><br>MD: {r['종합 평균차이(MD)']:.3f}"
                        f"<br>95% CI: {r['95% CI 하한']:.3f} ~ {r['95% CI 상한']:.3f}"
                        f"<br>p={r['p값']:.3f} · GRADE {r['근거 확실성(논문 GRADE 표기)']}<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )
            shown.add(label)
        fig.add_vline(x=0, line_dash="dash", line_color="black", row=row, col=col)
    fig.update_yaxes(autorange="reversed", categoryorder="array", categoryarray=comps)
    for c in (2, 3, 4):
        fig.update_yaxes(showticklabels=False, row=1, col=c)
        fig.update_yaxes(showticklabels=False, row=2, col=c)
    fig.update_layout(height=620, legend=dict(orientation="h", y=-0.08), margin=dict(l=10, r=10, t=60, b=10))
    return fig


# ---------------------------------------------------------------- 화면 구성
st.set_page_config(page_title="제로음료 임상시험 메타분석 대시보드", layout="wide")
st.title("제로음료(LNCSB) 임상시험 메타분석 대시보드")
st.caption(
    "McGlynn et al., JAMA Netw Open 2022;5(3):e222092 보충자료 기반 · "
    "성인 대상 임상시험 결과이므로 청소년에게 그대로 일반화할 수 없습니다."
)

study_df, summary_df = load_data()
metric_options = list(dict.fromkeys(summary_df["지표"]))

tab_detail, tab_overview, tab_raw = st.tabs(["📊 지표별 상세", "🧭 전체 지표 한눈에", "📄 원자료"])

with tab_detail:
    metric = st.sidebar.selectbox(
        "지표 선택", metric_options, format_func=lambda m: f"{METRIC_KO[m]} ({m})"
    )
    unit = summary_df.loc[summary_df["지표"] == metric, "단위"].iloc[0]
    s_df = study_df[study_df["지표"] == metric]
    t_df = summary_df[summary_df["지표"] == metric]

    st.subheader(f"{METRIC_KO[metric]} ({unit})")
    st.plotly_chart(
        summary_forest(t_df, unit, "① 종합분석 결과 (논문의 풀링 결과)"), use_container_width=True
    )

    comps = sorted(s_df["비교"].unique())
    chosen = st.multiselect("② 개별 연구에서 볼 비교 그룹", comps, default=comps)
    st.plotly_chart(
        study_forest(s_df[s_df["비교"].isin(chosen)], unit, f"② 개별 연구 결과 (연구 {len(s_df)}건)"),
        use_container_width=True,
    )
    st.info(
        "개별 연구의 95% CI는 MD ± 1.96×SE로 계산한 근사값입니다. "
        "Mean Difference의 부호는 Treatment 1/2 순서와 논문의 분석 정의를 확인해서 해석하세요. "
        "개별 연구 행을 단순 평균해도 종합분석 결과는 재현되지 않습니다."
    )

with tab_overview:
    st.plotly_chart(overview_grid(summary_df), use_container_width=True)
    st.dataframe(
        summary_df[
            ["지표_한글", "단위", "비교_한글", "종합 평균차이(MD)", "95% CI 하한", "95% CI 상한",
             "p값", "비교 수", "근거 확실성(논문 GRADE 표기)"]
        ],
        use_container_width=True,
        hide_index=True,
    )

with tab_raw:
    st.markdown("**개별 연구 원자료**")
    st.dataframe(
        study_df.drop(columns=["y라벨", "해석 주의"]), use_container_width=True, hide_index=True
    )
    st.markdown("**출처와 주의사항**")
    st.dataframe(
        pd.read_excel(EXCEL_FILE, sheet_name="출처와_주의사항"), use_container_width=True, hide_index=True
    )
