"""제로음료(LNCSB) 효과 확인용 - 이상치 시각화 Streamlit 앱
실행:  pip install -r requirements.txt  →  streamlit run app.py
"""
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DEFAULT_FILE = "제로음료_임상시험_원자료_체중혈당__1__1.xlsx"
IND_KO = {
    "Body Weight": "체중", "BMI": "BMI", "Body Fat": "체지방률",
    "Waist Circumference": "허리둘레", "HbA1c": "당화혈색소",
    "Fasting Blood Glucose": "공복혈당", "2-Hour Post-Prandial Glucose": "식후2시간혈당",
}
COMP_MAP = {("SSB", "LNCSB"): "제로음료 vs 설탕음료", ("Water", "LNCSB"): "제로음료 vs 물",
            ("SSB", "Water"): "설탕음료 vs 물"}
ZERO_COMPS = ["제로음료 vs 설탕음료", "제로음료 vs 물"]
COLORS = {"정상": "#5B8DB8", "주의": "#F29E38", "이상치": "#D62728"}
Z_WARN = 2.0


# ───────────────────────── 통계 함수 ─────────────────────────
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
    return dict(mu=mu, se=s, lo=mu - 1.96 * s, hi=mu + 1.96 * s, tau2=tau2, i2=i2, k=k)


def loo_z(md, se):
    """각 연구를 뺀 나머지로 합산한 값과의 차이를 표준화 (leave-one-out z)"""
    md, se = np.asarray(md, float), np.asarray(se, float)
    out = np.full(len(md), np.nan)
    if len(md) < 3:
        return out
    for i in range(len(md)):
        m = np.arange(len(md)) != i
        p = re_pool(md[m], se[m])
        out[i] = (md[i] - p["mu"]) / np.sqrt(se[i] ** 2 + p["tau2"] + p["se"] ** 2)
    return out


@st.cache_data
def load(src):
    raw = pd.read_excel(src, sheet_name="개별연구_원자료")
    summ = pd.read_excel(src, sheet_name="종합분석_결과")
    d = raw.rename(columns={"연구(저자·연도)": "study", "Mean Difference": "MD",
                            "Standard Error": "SE", "Treatment 1": "t1", "Treatment 2": "t2"})
    d["지표"] = d["지표"].map(IND_KO)
    d["comp"] = [COMP_MAP[(a, b)] for a, b in zip(d.t1, d.t2)]
    d["lo"], d["hi"] = d.MD - 1.96 * d.SE, d.MD + 1.96 * d.SE
    summ["지표"] = summ["지표"].map(IND_KO)
    return d[["지표", "단위", "study", "comp", "MD", "SE", "lo", "hi"]], summ


def flag(d, z_out):
    d = d.copy()
    d["z"] = np.nan
    for _, idx in d.groupby(["지표", "comp"]).groups.items():
        d.loc[idx, "z"] = loo_z(d.loc[idx, "MD"], d.loc[idx, "SE"])
    d["판정"] = np.where(d.z.abs() >= z_out, "이상치", np.where(d.z.abs() >= Z_WARN, "주의", "정상"))
    return d


def fmt(p):
    return f"{p['mu']:.2f} [{p['lo']:.2f}, {p['hi']:.2f}]"


def verdict(p, unit_note=""):
    if np.isnan(p["mu"]):
        return "-"
    if p["hi"] < 0:
        return "제로음료 ↓ (유의)"
    if p["lo"] > 0:
        return "제로음료 ↑ (유의)"
    return "차이 없음(CI가 0 포함)"


# ───────────────────────── 앱 ─────────────────────────
st.set_page_config(page_title="제로음료 효과 · 이상치 시각화", layout="wide")
st.title("🥤 제로음료(LNCSB) 효과 확인 — 이상치 시각화")
st.caption("McGlynn et al., JAMA Netw Open 2022 보충자료 기반 · 연구 단위 요약값(MD, SE) 분석 · 성인 대상")

with st.sidebar:
    st.header("설정")
    up = st.file_uploader("엑셀 업로드 (선택)", type="xlsx")
    z_out = st.slider("이상치 기준 |z| (leave-one-out)", 2.0, 5.0, 3.0, 0.5,
                      help="나머지 연구들로 합산한 값에서 얼마나 벗어났는지의 표준화 점수. 2 이상은 '주의'로 표시")
    show_ssb_water = st.checkbox("설탕음료 vs 물 비교도 포함", value=False)

src = up if up is not None else (DEFAULT_FILE if os.path.exists(DEFAULT_FILE) else None)
if src is None:
    st.info("분석할 엑셀 파일을 왼쪽에서 업로드하세요.")
    st.stop()

df, summ = load(src)
df = flag(df, z_out)
comps = ZERO_COMPS + (["설탕음료 vs 물"] if show_ssb_water else [])
inds = list(IND_KO.values())

st.warning("**부호 해석 가정**: 제로음료가 들어간 비교는 MD < 0 이면 '제로음료 쪽이 더 낮음'으로 해석했습니다 "
           "(논문 합산 결과와 방향이 맞는 것을 확인). 인용 전에 원문 PDF로 꼭 확인하세요.", icon="⚠️")

t1, t2, t3, t4, t5 = st.tabs(["① 한눈에 보기(히트맵)", "② 지표별 포레스트 플롯", "③ 이상치 빼면 결론이 바뀌나?",
                              "④ 논문 보고 결과", "⑤ 데이터 점검"])

# ── ① 히트맵 ──
with t1:
    st.subheader("연구별 이상치 점수 (leave-one-out z)")
    st.write("빨강/파랑이 진할수록 같은 지표의 다른 연구들과 많이 다른 값입니다. 테두리 표시된 칸이 이상치입니다.")
    h = df[df.comp.isin(comps)].copy()
    h["row"] = h.study + " (" + h.comp.str.replace("제로음료", "제로").str.replace("설탕음료", "설탕") + ")"
    piv = h.pivot_table(index="row", columns="지표", values="z", aggfunc="first").reindex(columns=inds)
    fig = px.imshow(piv, color_continuous_scale="RdBu_r", zmin=-5, zmax=5, aspect="auto",
                    labels=dict(color="z"), text_auto=".1f")
    fig.update_layout(height=max(500, 22 * len(piv)), xaxis_side="top")
    st.plotly_chart(fig)
    out = df[(df.판정 == "이상치") & df.comp.isin(comps)]
    if len(out):
        st.error("이상치: " + " · ".join(f"**{r.study}** ({r.지표}, {r.comp}, MD={r.MD:.2f}, z={r.z:.1f})"
                                       for r in out.itertuples()))
    else:
        st.success("현재 기준에서는 이상치가 없습니다.")

# ── ② 포레스트 ──
with t2:
    c1, c2 = st.columns(2)
    ind = c1.selectbox("지표", inds)
    comp = c2.radio("비교", comps, horizontal=True)
    g = df[(df.지표 == ind) & (df.comp == comp)].sort_values("MD").reset_index(drop=True)
    if g.empty:
        st.info("해당 조합의 연구가 없습니다.")
    else:
        unit = g.단위.iloc[0]
        keep = g[g.판정 != "이상치"]
        p_all, p_ex = re_pool(g.MD, g.SE), re_pool(keep.MD, keep.SE)
        fig = go.Figure()
        for lab, col in COLORS.items():
            s = g[g.판정 == lab]
            if s.empty:
                continue
            fig.add_trace(go.Scatter(
                x=s.MD, y=s.study, mode="markers", name=lab,
                marker=dict(color=col, symbol="square", size=8 + 22 * (1 / s.SE**2) ** 0.5 / (1 / g.SE.min()**2) ** 0.5),
                error_x=dict(type="data", symmetric=False, array=s.hi - s.MD, arrayminus=s.MD - s.lo,
                             color=col, thickness=2),
                customdata=np.c_[s.SE, s.z], hovertemplate="%{y}<br>MD=%{x:.2f} (SE %{customdata[0]:.2f})"
                                                           "<br>z=%{customdata[1]:.1f}<extra></extra>"))
        for p, name, col, sym in [(p_all, "합산(전체)", "#333", "diamond"),
                                  (p_ex, "합산(이상치 제외)", "#2CA02C", "diamond-open")]:
            fig.add_trace(go.Scatter(x=[p["mu"]], y=[name], mode="markers", name=name,
                                     marker=dict(color=col, symbol=sym, size=16, line=dict(width=2)),
                                     error_x=dict(type="data", symmetric=False, array=[p["hi"] - p["mu"]],
                                                  arrayminus=[p["mu"] - p["lo"]], color=col, thickness=3)))
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        fig.update_layout(height=max(420, 32 * (len(g) + 3)), xaxis_title=f"평균차이 MD ({unit}) · 0보다 왼쪽 = 제로음료가 더 낮음",
                          yaxis=dict(categoryorder="array", categoryarray=["합산(이상치 제외)", "합산(전체)"] + list(g.study[::-1])),
                          legend=dict(orientation="h", y=1.08), margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig)
        m1, m2, m3 = st.columns(3)
        m1.metric("합산 MD (전체)", fmt(p_all), verdict(p_all), delta_color="off")
        m2.metric("합산 MD (이상치 제외)", fmt(p_ex), verdict(p_ex), delta_color="off")
        m3.metric("이질성 I² (전체)", f"{p_all['i2']:.0f}%")
        st.caption("사각형 크기 = 연구 비중(1/SE²), 막대 = 95% CI. 이상치는 빨강, 주의는 주황. "
                   "합산은 DerSimonian-Laird 랜덤효과 근사값이며 논문의 네트워크 메타분석 결과와 다를 수 있습니다.")

# ── ③ 이상치 제외 비교 ──
with t3:
    st.subheader("이상치를 빼면 '제로음료 효과' 결론이 달라지나?")
    rows = []
    for i in inds:
        for c in ZERO_COMPS:
            g = df[(df.지표 == i) & (df.comp == c)]
            if g.empty:
                continue
            k = g[g.판정 != "이상치"]
            pa, pe = re_pool(g.MD, g.SE), re_pool(k.MD, k.SE)
            names = ", ".join(g[g.판정 == "이상치"].study)
            rows.append(dict(지표=i, 비교=c, 단위=g.단위.iloc[0], 연구수=len(g), 제외연구=names or "-",
                             전체=fmt(pa), 이상치제외=fmt(pe), 판정_전체=verdict(pa), 판정_제외=verdict(pe),
                             변화="⚠️ 바뀜" if verdict(pa) != verdict(pe) else "", _pa=pa, _pe=pe))
    res = pd.DataFrame(rows)
    if res.empty:
        st.info("데이터가 없습니다.")
    else:
        fig = go.Figure()
        ylab = res.지표 + " · " + res.비교.str.replace("제로음료 ", "")
        for key, name, col, off in [("_pa", "전체", "#333", 0.15), ("_pe", "이상치 제외", "#2CA02C", -0.15)]:
            mu = [p["mu"] for p in res[key]]
            fig.add_trace(go.Scatter(
                x=mu, y=list(range(len(res))), mode="markers", name=name, marker=dict(color=col, size=10),
                error_x=dict(type="data", symmetric=False, array=[p["hi"] - p["mu"] for p in res[key]],
                             arrayminus=[p["mu"] - p["lo"] for p in res[key]], color=col)))
        # 같은 y에서 겹치지 않도록 미세 이동
        for tr, off in zip(fig.data, (0.15, -0.15)):
            tr.y = [y + off for y in tr.y]
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        fig.update_layout(height=max(500, 55 * len(res)), yaxis=dict(tickmode="array", tickvals=list(range(len(res))),
                          ticktext=list(ylab), autorange="reversed"),
                          xaxis_title="합산 MD (지표마다 단위가 다름 · 0보다 왼쪽 = 제로음료가 더 낮음)",
                          legend=dict(orientation="h", y=1.06))
        st.plotly_chart(fig)
        st.dataframe(res[["지표", "비교", "단위", "연구수", "제외연구", "전체", "이상치제외", "판정_전체", "판정_제외", "변화"]],
                     hide_index=True)
        changed = res[res.변화 != ""]
        if len(changed):
            st.warning("이상치 제외 시 결론이 바뀌는 항목: " + ", ".join(f"{r.지표}({r.비교})" for r in changed.itertuples()))
        else:
            st.success("이상치를 제외해도 유의성 판정은 바뀌지 않습니다.")
        st.caption("MD 단위: 체중 kg · BMI kg/m² · 체지방 % · 허리둘레 cm · HbA1c % · 혈당 mmol/L. 연구가 2~3개뿐인 칸은 해석에 주의.")

# ── ④ 논문 보고 결과 ──
with t4:
    st.subheader("논문이 보고한 종합 결과 (95% CI)")
    s = summ.copy()
    s["비교"] = s["비교"].replace({"LNCSBs for SSBs": "제로음료로 설탕음료 대체", "LNCSBs for Water": "제로음료로 물 대체",
                                 "Water for SSBs": "물로 설탕음료 대체"})
    s["유의"] = np.where((s["95% CI 상한"] < 0) | (s["95% CI 하한"] > 0), "유의(p<0.05)", "비유의")
    fig = px.scatter(s, x="종합 평균차이(MD)", y="비교", color="유의", facet_col="지표", facet_col_wrap=3,
                     error_x=s["95% CI 상한"] - s["종합 평균차이(MD)"], error_x_minus=s["종합 평균차이(MD)"] - s["95% CI 하한"],
                     color_discrete_map={"유의(p<0.05)": "#D62728", "비유의": "#7F7F7F"}, hover_data=["p값", "근거 확실성(논문 GRADE 표기)"])
    fig.update_xaxes(matches=None, showticklabels=True)
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_layout(height=700)
    st.plotly_chart(fig)
    st.dataframe(s.drop(columns=["원문 PDF 페이지"]), hide_index=True)

# ── ⑤ 데이터 점검 ──
with t5:
    st.subheader("정합성 점검")
    st.markdown("**1) 비교 수: 개별 자료 행 수 vs 종합분석표의 '비교 수'**")
    cnt = df.groupby("지표").size().rename("개별 자료 행수")
    sc = summ.groupby("지표")["비교 수"].first().rename("종합분석표 비교 수")
    chk = pd.concat([cnt, sc], axis=1).reindex(inds)
    chk["일치"] = np.where(chk.iloc[:, 0] == chk.iloc[:, 1], "✅", "❌ 불일치")
    st.dataframe(chk)
    st.markdown("**2) 삼각 일관성**: 같은 연구에서 (설탕-물) ≈ (설탕-제로) − (물-제로) 이어야 합니다.")
    tri = []
    for (i, stu), g in df.groupby(["지표", "study"]):
        m = g.set_index("comp").MD
        if {"제로음료 vs 설탕음료", "제로음료 vs 물", "설탕음료 vs 물"} <= set(m.index):
            exp = m["제로음료 vs 설탕음료"] - m["제로음료 vs 물"]
            tri.append(dict(지표=i, 연구=stu, 기대값=round(exp, 2), 보고값=m["설탕음료 vs 물"],
                            차이=round(abs(exp - m["설탕음료 vs 물"]), 2)))
    tri = pd.DataFrame(tri)
    if len(tri):
        tri["판정"] = np.where(tri.차이 > 0.15, "❌ 불일치 - 원문 확인", "✅")
        st.dataframe(tri, hide_index=True)
    st.markdown("**3) 전체 데이터 (판정 포함)**")
    st.dataframe(df.round(2), hide_index=True)
    st.download_button("판정 포함 CSV 다운로드", df.round(3).to_csv(index=False).encode("utf-8-sig"),
                       "outlier_flags.csv", "text/csv")
