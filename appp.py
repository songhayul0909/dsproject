import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="제로음료 영양성분 탐색", page_icon="🥤", layout="wide")

# 데이터 위치: 저장소 안의 파일 경로입니다.
# GitHub raw 링크를 쓰려면 아래 줄을 링크로 바꾸세요.
# 예) "https://raw.githubusercontent.com/내계정/저장소이름/main/data/zero_drinks.csv"
DATA_SOURCE = "data/zero_drinks.csv"

NUTRIENTS = ["에너지_kcal", "단백질_g", "지방_g", "탄수화물_g", "당류_g", "나트륨_mg", "포화지방_g"]
NUMERIC = NUTRIENTS + ["식품중량"]


@st.cache_data
def load_data(source):
    df = pd.read_csv(source, encoding="utf-8-sig")
    df["데이터생성일"] = pd.to_datetime(df["데이터생성일"])
    df["생성월"] = df["데이터생성일"].dt.strftime("%Y-%m")
    return df


# ---------------------------------------------------------------- 데이터 불러오기
st.title("🥤 제로음료 영양성분 탐색")
st.caption("출처: 식품의약품안전처 식품영양성분 DB (화면을 보고 직접 옮겨 적은 사본, 자세한 내용은 data/출처.md)")

uploaded = st.sidebar.file_uploader("다른 CSV로 바꿔 보기 (선택)", type="csv")
try:
    df_all = load_data(uploaded if uploaded is not None else DATA_SOURCE)
except FileNotFoundError:
    st.error(f"'{DATA_SOURCE}' 파일을 찾을 수 없습니다. app.py와 같은 위치에 data 폴더가 있는지 확인하세요.")
    st.stop()

type_order = list(dict.fromkeys(df_all["음료종류"]))  # 파일에 나온 순서대로

# ---------------------------------------------------------------- 사이드바 필터
st.sidebar.header("필터")
unit_choice = st.sidebar.radio(
    "기준단위",
    ["100ml 기준만", "100g 기준만", "전체"],
    help="영양성분 값이 100ml 기준인 제품과 100g 기준인 제품이 섞여 있어서, 기본은 100ml 기준만 봅니다.",
)
types = st.sidebar.multiselect("음료종류", type_order, default=type_order)
bins = st.sidebar.slider("히스토그램 구간 수", 5, 20, 10)

df = df_all[df_all["음료종류"].isin(types)]
if unit_choice == "100ml 기준만":
    df = df[df["기준단위"] == "100ml"]
elif unit_choice == "100g 기준만":
    df = df[df["기준단위"] == "100g"]

st.sidebar.caption(f"전체 {len(df_all)}개 중 {len(df)}개를 보여주고 있습니다.")
if df.empty:
    st.warning("조건에 맞는 제품이 없습니다. 왼쪽 필터를 바꿔 보세요.")
    st.stop()


# ---------------------------------------------------------------- 그래프 도우미
def show_hist(col, why):
    """숫자 열: 값이 어디에 몰려 있는지 보는 히스토그램"""
    d = df.dropna(subset=[col])
    st.subheader(col)
    st.caption(f"히스토그램 · {why}")
    if d.empty:
        st.info("표시할 값이 없습니다.")
        return
    fig = px.histogram(d, x=col, nbins=bins)
    fig.update_layout(yaxis_title="제품 수", bargap=0.05, margin=dict(t=10, b=10))
    st.plotly_chart(fig)
    st.caption(f"값이 있는 제품 {len(d)}개 · 결측 {len(df) - len(d)}개")


# ---------------------------------------------------------------- 탭
tab1, tab2, tab3, tab4 = st.tabs(["열별 분포", "음료종류별 비교", "관계 보기(산점도)", "데이터 표"])

# ===== 탭 1: 열마다 알맞은 그래프 =====
with tab1:
    st.markdown("열의 성격(범주, 숫자, 날짜)에 맞는 그래프를 하나씩 보여줍니다. 번호, 식품코드, 식품명은 제품마다 달라서 그래프 없이 '데이터 표' 탭에서 확인합니다.")

    st.subheader("결측치 현황")
    st.caption("막대그래프 · 열마다 비어 있는 칸이 몇 개인지. 비어 있는 것은 0이 아니라 원본에 값이 없다는 뜻입니다.")
    miss = df[NUMERIC].isna().sum().reset_index()
    miss.columns = ["열", "결측 개수"]
    fig = px.bar(miss, x="열", y="결측 개수", text="결측 개수")
    fig.update_layout(margin=dict(t=10, b=10))
    st.plotly_chart(fig)

    left, right = st.columns(2)

    with left:
        st.subheader("음료종류")
        st.caption("가로 막대그래프 · 종류가 많은 범주형이라 원그래프보다 막대가 비교하기 쉽습니다.")
        cnt = df["음료종류"].value_counts().rename_axis("음료종류").reset_index(name="제품 수")
        fig = px.bar(cnt, x="제품 수", y="음료종류", orientation="h", text="제품 수")
        fig.update_layout(yaxis=dict(categoryorder="total ascending"), margin=dict(t=10, b=10))
        st.plotly_chart(fig)

    with right:
        st.subheader("기준단위")
        st.caption("도넛(원) 그래프 · 범주가 2개뿐이라 전체에서 차지하는 비율을 보기 좋습니다. (필터와 상관없이 전체 데이터)")
        cnt = df_all["기준단위"].value_counts().rename_axis("기준단위").reset_index(name="제품 수")
        fig = px.pie(cnt, names="기준단위", values="제품 수", hole=0.5)
        fig.update_layout(margin=dict(t=10, b=10))
        st.plotly_chart(fig)
        st.caption("중량단위(ml, g)는 기준단위와 똑같이 나뉘어서 따로 그리지 않았습니다.")

    left, right = st.columns(2)
    with left:
        show_hist("식품중량", "제품 1개 용량이 어느 크기에 몰려 있는지 (기준단위가 섞이면 ml와 g이 함께 나옵니다)")
    with right:
        st.subheader("데이터생성일")
        st.caption("막대그래프 · 원본 데이터가 언제 만들어졌는지 월별 제품 수")
        cnt = df["생성월"].value_counts().rename_axis("생성월").reset_index(name="제품 수").sort_values("생성월")
        fig = px.bar(cnt, x="생성월", y="제품 수", text="제품 수")
        fig.update_xaxes(type="category")
        fig.update_layout(margin=dict(t=10, b=10))
        st.plotly_chart(fig)

    st.divider()
    st.markdown("#### 영양성분 (숫자 열)")
    pairs = [
        ("에너지_kcal", "대부분 낮은 값에 몰리는지, 튀는 제품이 있는지"),
        ("단백질_g", "제품별 단백질 분포"),
        ("지방_g", "값이 있는 제품이 적을 수 있어 개수를 함께 확인하세요"),
        ("탄수화물_g", "값이 있는 제품이 적을 수 있어 개수를 함께 확인하세요"),
        ("당류_g", "제로 음료라도 당류가 0이 아닌 제품이 있는지"),
        ("나트륨_mg", "제품별 나트륨 분포"),
        ("포화지방_g", "제품별 포화지방 분포"),
    ]
    for i in range(0, len(pairs), 2):
        cols = st.columns(2)
        for c, (col, why) in zip(cols, pairs[i:i + 2]):
            with c:
                show_hist(col, why)

# ===== 탭 2: 음료종류별 비교 =====
with tab2:
    st.markdown("음료종류에 따라 영양성분이 어떻게 다른지 비교합니다.")
    if unit_choice == "전체":
        st.warning("100ml 기준과 100g 기준이 섞여 있어 커피 값이 왜곡될 수 있습니다. 왼쪽에서 기준단위를 하나로 고르는 것을 권합니다.")

    metric = st.selectbox("비교할 영양성분", NUTRIENTS)
    d = df.dropna(subset=[metric]).copy()
    if d.empty:
        st.info("이 영양성분은 지금 조건에서 값이 있는 제품이 없습니다.")
    else:
        counts = d["음료종류"].value_counts()
        d["종류(n)"] = d["음료종류"].map(lambda t: f"{t} (n={counts[t]})")
        order = [f"{t} (n={counts[t]})" for t in type_order if t in counts.index]

        c1, c2 = st.columns(2)
        with c1:
            st.subheader(f"음료종류별 {metric} 평균")
            st.caption("막대그래프 · 종류별 평균 비교 (n은 값이 있는 제품 수)")
            avg = d.groupby("종류(n)")[metric].mean().reindex(order).reset_index()
            fig = px.bar(avg, x="종류(n)", y=metric, text_auto=".2f")
            fig.update_layout(xaxis_title="", yaxis_title=f"평균 {metric}", margin=dict(t=10, b=10))
            st.plotly_chart(fig)
        with c2:
            st.subheader(f"음료종류별 {metric} 분포")
            st.caption("상자그림 · 제품 하나하나의 점과 퍼짐 (점에 마우스를 올리면 제품명)")
            fig = px.box(d, x="종류(n)", y=metric, points="all", hover_name="식품명",
                         category_orders={"종류(n)": order})
            fig.update_layout(xaxis_title="", margin=dict(t=10, b=10))
            st.plotly_chart(fig)

        no_value = [t for t in types if t not in counts.index]
        if no_value:
            st.caption("값이 하나도 없어 그래프에서 빠진 종류: " + ", ".join(no_value))
        st.caption("제품이 1~2개뿐인 종류는 평균이 그 제품의 값과 거의 같습니다. 종류 사이의 차이를 말할 때는 n을 함께 보세요.")

# ===== 탭 3: 산점도 =====
with tab3:
    st.markdown("두 숫자 열의 관계를 점으로 봅니다. 점 하나가 제품 하나입니다.")
    c1, c2 = st.columns(2)
    x = c1.selectbox("가로축", NUMERIC, index=NUMERIC.index("당류_g"))
    y = c2.selectbox("세로축", NUMERIC, index=0)
    if x == y:
        st.info("가로축과 세로축을 서로 다른 열로 골라 주세요.")
    else:
        d = df.dropna(subset=[x, y])
        if d.empty:
            st.info("두 열 모두 값이 있는 제품이 없습니다.")
        else:
            fig = px.scatter(d, x=x, y=y, color="음료종류", hover_name="식품명",
                             category_orders={"음료종류": type_order})
            fig.update_traces(marker=dict(size=10))
            st.plotly_chart(fig)
            st.caption(f"두 열 모두 값이 있는 제품 {len(d)}개 (전체 {len(df)}개 중). 점이 몰려 보이는 것은 같은 값의 제품이 겹쳐 있기 때문일 수 있습니다.")

# ===== 탭 4: 데이터 표 =====
with tab4:
    shown = df.drop(columns=["생성월"])
    st.dataframe(shown)
    st.download_button(
        "지금 보이는 표를 CSV로 받기",
        shown.to_csv(index=False).encode("utf-8-sig"),
        file_name="zero_drinks_filtered.csv",
        mime="text/csv",
    )
