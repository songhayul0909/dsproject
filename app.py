import pandas as pd
import plotly.graph_objects as go
import streamlit as st

EXCEL_FILE = "제로음료_임상시험_원자료_체중혈당.xlsx"
Z95 = 1.96  # 95% 신뢰구간 계수

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


# ------------------------------------------------------------------ 데이터
@st.cache_data
def load_data():
    study = pd.read_excel(EXCEL_FILE, sheet_name="개별연구_원자료").dropna(subset=["지표"])
    summ = pd.read_excel(EXCEL_FILE, sheet_name="종합분석_결과").dropna(subset=["지표"])

    summ["지표_한글"] = summ["지표"].map(METRIC_KO)
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


# ------------------------------------------------------------------ 화면
st.set_page_config(page_title="제로칼로리 음료의 효과가 있을까?", layout="wide")
st.title("제로칼로리 음료, 효과가 있을까?")
st.caption(
    "성인 대상 임상시험 메타분석(McGlynn et al., JAMA Netw Open 2022) 보충자료의 체중·혈당 지표 · "
    "청소년에게 그대로 일반화할 수 없습니다."
)

study_df, summ_df = load_data()
metric_options = list(dict.fromkeys(summ_df["지표"]))

tab1, tab2, tab3 = st.tabs(["🎯 한눈에 보는 결론", "🔍 지표별 자세히", "📄 원자료·한계"])

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
    st.subheader(f"{METRIC_KO[metric]} ({unit})")

    st.markdown("**① 종합 결과** (논문이 여러 연구를 합친 값)")
    st.plotly_chart(metric_forest(summ_df[summ_df["지표"] == metric], unit), width="stretch")

    s_df = study_df[study_df["지표"] == metric]
    comps = sorted(s_df["비교"].unique())
    st.markdown(f"**② 개별 연구** ({len(s_df)}건)")
    chosen = st.multiselect("볼 비교 그룹", comps, default=comps)
    st.plotly_chart(study_forest(s_df[s_df["비교"].isin(chosen)], unit), width="stretch")
    st.info(
        "개별 연구의 95% CI는 MD ± 1.96×SE로 계산한 근사값입니다. "
        "개별 연구 표의 MD 부호는 Treatment 1/2 순서와 논문 정의를 확인해야 해석할 수 있고, "
        "행들을 단순 평균해도 종합 결과는 재현되지 않습니다."
    )

# ---- 탭 3: 원자료·한계
with tab3:
    st.subheader("결과를 읽을 때 주의할 점")
    st.markdown(
        "- 대상은 **성인 임상시험**입니다. 청소년에게 그대로 적용할 수 없습니다.\n"
        "- 근거 확실성(GRADE)이 대부분 **Low**입니다. 연구가 더 나오면 결과가 바뀔 수 있습니다.\n"
        "- 신뢰구간이 0을 지나면 '효과가 없다'가 아니라 **'이 자료로는 차이를 확신할 수 없다'**는 뜻입니다.\n"
        "- 이 자료는 음료를 **바꿨을 때**의 변화입니다. 제로음료를 추가로 마시는 경우와는 다릅니다.\n"
        "- 모든 지표를 '낮을수록 개선'으로 가정한 색 구분입니다. 건강한 성인에게는 작은 차이가 임상적으로 의미 없을 수 있습니다."
    )
    st.subheader("종합분석 표")
    st.dataframe(
        summ_df[["지표_한글", "단위", "비교_한글", "종합 평균차이(MD)", "95% CI 하한", "95% CI 상한",
                 "p값", "비교 수", "근거 확실성(논문 GRADE 표기)", "판정"]],
        width="stretch",
        hide_index=True,
    )
    st.subheader("개별 연구 원자료")
    st.dataframe(study_df.drop(columns=["y라벨", "해석 주의"]), width="stretch", hide_index=True)
    st.subheader("출처: https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2790045?utm_source=chatgpt.com")
    st.dataframe(pd.read_excel(EXCEL_FILE, sheet_name="출처와_주의사항"), width="stretch", hide_index=True)
