import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from matplotlib.ticker import FuncFormatter
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib.cm import rainbow
import matplotlib.dates as mdates
import geopandas as gpd
import unidecode
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import ttest_rel
from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import f_oneway
from scipy.stats import kruskal


# Tải dữ liệu từ file CSV hoặc Excel
st.title("Phân Tích Tình Hình Dịch COVID-19")


@st.cache_data
def load_data():
    World_data = pd.read_csv(
        r"D:\Python\Project Data Analysis\Data-Covid-19\World data.csv"
    )  # dữ liệu thế giới
    country_data = pd.read_csv(
        r"D:\Python\Project Data Analysis\Data-Covid-19\Countries in the world.csv"
    )  # dữ liệu  các quốc gia
    df_territory = pd.read_csv(
        r"D:\Python\Project Data Analysis\Data-Covid-19\Map World.csv"
    )  # dữ liệu bản đồ thế giới
    continent_data = pd.read_csv(
        r"D:\Python\Project Data Analysis\Data-Covid-19\Regions of the world.csv"
    )  # dữ liệu châu lục
    df_support = pd.read_csv(
        r"D:\Python\Project Data Analysis\Data-Covid-19\Employee Support.csv"
    )  # dữ liệu hỗ trợ quốc gia
    gdp1 = pd.read_csv(
        r"D:\Python\Project Data Analysis\Data-Covid-19\GDP Q2 2020.csv"
    )  # dữ liệu gdp các nước quý 2 2020
    covid_data = pd.read_csv(
        r"D:\Python\Project Data Analysis\Data-Covid-19\Countries in the world.csv"
    )  # dữ liệu các quốc gia
    data_co2 = pd.read_csv(
        r"D:\Python\Project Data Analysis\Data-Covid-19\Amount of CO2 gas.csv"
    )  # dữ liệu co2
    travel = pd.read_csv(
        r"D:\Python\Project Data Analysis\Data-Covid-19\Travel countries.csv"
    )  # dữ liệu chuyến đi
    gdp_data = pd.read_csv(
        r"D:\Python\Project Data Analysis\Data-Covid-19\World bank data.csv"
    )  # dữ liệu gdp và tỉ lệ thất nghiệp 2020-2021

    return (
        World_data,
        country_data,
        df_territory,
        continent_data,
        df_support,
        gdp1,
        data_co2,
        travel,
        gdp_data,
        covid_data,
    )


# Tải dữ liệu khi người dùng mở ứng dụng
(
    World_data,
    country_data,
    df_territory,
    continent_data,
    df_support,
    gdp1,
    data_co2,
    travel,
    gdp_data,
    covid_data,
) = load_data()
FILE_PATH = r"D:\Python\Project Data Analysis\Data-Covid-19\World data.csv"  # đổi đường dẫn file fath để thao tác trực tiếp với dữ liệu
FILE_PATH2 = r"D:\Python\Project Data Analysis\Data-Covid-19\World data.csv"


# Tạo các tab trên màn hình chính
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "Home",
        "Tổng quan về tình hình dịch COVID-19",
        "Phân tích tương quan",
        "Phân tích",
        "Ảnh hưởng",
        "Kết luận",
    ]
)
World_data["date"] = pd.to_datetime(World_data["date"])

# Nội dung cho từng tab
with tab1:

    st.title("Home")

    # Hàng đầu tiên gồm 3 cột
    row1_col1, row1_col2, row1_col3 = st.columns(3)

    with row1_col1:
        # Tính tổng số ca nhiễm theo các châu lục
        df_grouped = (
            country_data.groupby("continent")
            .agg(
                {
                    "total_cases": "sum",
                }
            )
            .reset_index()
        )

        # -- Biểu đồ donut cho số ca nhiễm --
        fig_cases = px.pie(
            df_grouped,
            names="continent",
            values="total_cases",
            color_discrete_sequence=px.colors.sequential.Plasma,
            title="Tỉ lệ ca nhiễm giữa các châu lục",
            hole=0.4,  # Tạo hiệu ứng donut
        )
        fig_cases.update_layout(showlegend=False)

        st.plotly_chart(fig_cases, use_container_width=True)
    with row1_col2:
        # Tính tổng số tử vong theo các châu lục
        df_grouped = (
            country_data.groupby("continent").agg({"total_deaths": "sum"}).reset_index()
        )
        # --Biểu đồ donut cho số ca tử vong ----
        fig_deaths = px.pie(
            df_grouped,
            names="continent",
            values="total_deaths",
            color_discrete_sequence=px.colors.sequential.Viridis,
            title="Tỉ lệ ca tử vong giữa cac châu lục",
            hole=0.4,  # Tạo hiệu ứng donut
        )
        fig_deaths.update_layout(showlegend=False)  # Ẩn chú thích

        st.plotly_chart(fig_deaths, use_container_width=True)
    with row1_col3:
        # country_data = pd.read_csv(r'A:\UET-VNU\Covid19_KNK\Datacovid19\countries_data.csv')

        # Tổng số ca nhiễm toàn cầu
        world_total_cases = 775_000_000
        # ... (mã khác)

        # Xử lý NaN trước khi nhóm (chọn một trong những cách sau):
        country_data = country_data.dropna(
            subset=["total_cases"]
        )  # Loại bỏ các hàng có NaN
        # HOẶC
        country_data["total_cases"] = country_data["total_cases"].fillna(
            0
        )  # Điền NaN bằng 0

        country_data = country_data.loc[
            country_data.groupby("location")["total_cases"].idxmax()
        ]

        # ... (phần còn lại của mã)

        # Tính tỷ lệ total_case của từng quốc gia so với thế giới
        country_data["global_ratio"] = (
            country_data["total_cases"] / world_total_cases
        ) * 100

        # Lấy top 20 quốc gia có total_cases cao nhất
        top_20_countries = country_data.nlargest(20, "total_cases")
        st.dataframe(
            top_20_countries,
            column_order=["location", "total_cases", "global_ratio"],  # Hiển thị cột
            hide_index=True,
            column_config={
                "location": st.column_config.TextColumn("Country"),
                "total_cases": st.column_config.NumberColumn(
                    "Total Cases",
                    format="%d",
                ),
                "global_ratio": st.column_config.ProgressColumn(
                    "Global Ratio (%)",
                    format="%.2f %%",
                    min_value=0.0,
                    max_value=100.0,
                ),
            },
            use_container_width=True,
        )
    row2_col1, row2_col2, row2_col3 = st.columns(3)
    with row2_col1:
        # -- Biểu đồ số ca nhiễm theo thời gian
        World_data["date"] = pd.to_datetime(World_data["date"])
        cases_over_time = World_data.groupby("date")["new_cases"].sum().reset_index()

        fig = px.line(
            cases_over_time,
            x="date",
            y="new_cases",
            title="Số ca nhiễm theo thời gian",
            labels={"new_cases": "Số Ca Nhiễm Mới", "date": "Ngày"},
        )

        st.plotly_chart(fig)
    with row2_col2:
        # --Biểu đồ só ca tử vong theo thời gian
        World_data["date"] = pd.to_datetime(World_data["date"])
        deaths_over_time = World_data.groupby("date")["new_deaths"].sum().reset_index()

        fig = px.line(
            deaths_over_time,
            x="date",
            y="new_deaths",
            title="Số Ca Tử Vong Mới Theo Thời Gian",
            labels={"new_deaths": "Số Ca Tử Vong Mới", "date": "Ngày"},
            line_shape="linear",
        )

        fig.update_traces(line=dict(color="red"))

        st.plotly_chart(fig)

    with row2_col3:
        # -- Biều đồ Tương Quan ca nhiễm và hồi phục
        World_data["month"] = World_data["date"].dt.to_period("M")

        # Tính toán tổng quan theo tháng
        monthly_summary = (
            World_data.groupby("month")
            .agg(
                total_cases_start=("total_cases", "first"),
                total_cases_end=("total_cases", "last"),
                total_deaths_start=("total_deaths", "first"),
                total_deaths_end=("total_deaths", "last"),
                new_cases=("new_cases", "sum"),
            )
            .reset_index()
        )

        # Tính số ca hồi phục trong tháng
        monthly_summary["recovered_in_month"] = (
            monthly_summary["total_cases_end"] - monthly_summary["total_deaths_end"]
        ) - (
            monthly_summary["total_cases_start"] - monthly_summary["total_deaths_start"]
        )

        # Đảm bảo không có giá trị âm
        monthly_summary["recovered_in_month"] = monthly_summary[
            "recovered_in_month"
        ].clip(lower=0)
        monthly_summary["new_cases"] = monthly_summary["new_cases"].clip(lower=0)

        # Lọc các tháng đại diện cho quý
        quarters = ["01", "04", "07", "10"]
        monthly_summary["month_str"] = monthly_summary["month"].astype(str)
        filtered_months = monthly_summary[
            monthly_summary["month_str"].str[-2:].isin(quarters)
        ]
        fig = go.Figure()
        # Thêm dữ liệu "Ca Nhiễm Mới"
        fig.add_trace(
            go.Bar(
                x=filtered_months["month_str"],
                y=filtered_months["new_cases"],
                name="Ca Nhiễm Mới",
                marker_color="orange",
            )
        )
        # Thêm dữ liệu "Ca Hồi Phục"
        fig.add_trace(
            go.Bar(
                x=filtered_months["month_str"],
                y=filtered_months["recovered_in_month"],
                name="Ca Hồi Phục",
                marker_color="green",
                base=filtered_months["new_cases"],
            )
        )
        fig.update_layout(
            title="Tương quan ca nhiễm và hồi phục",
            xaxis_title="Tháng",
            yaxis_title="Số Ca",
            barmode="stack",
            template="plotly_white",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


with tab2:
    st.header("Tổng quan về tình hình dịch COVID-19")
    # Các tùy chọn cho phần tổng quan
    sub_option = st.selectbox(
        "Chọn phân tích chi tiết:",
        [
            "Số ca nhiễm và tử vong theo thời gian",
            "Các châu lục và quốc gia",
            "Bản đồ thế giới (số ca nhiễm, tử vong,hỗ trợ chính phủ)",
        ],
    )
    if sub_option == "Số ca nhiễm và tử vong theo thời gian":
        # Đọc dữ liệu trực tiếp từ file CSV
        data = pd.read_csv(FILE_PATH)
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        data = data.dropna(subset=["date"])
        st.title("Phân Tích Dữ Liệu COVID-19")
        # Giao diện chọn khoảng thời gian (chuyển vào chính trang)
        st.subheader("Chọn Khoảng Thời Gian")
        min_date = data["date"].min()
        max_date = data["date"].max()
        start_date = st.date_input(
            "Ngày bắt đầu", value=min_date, min_value=min_date, max_value=max_date
        )
        end_date = st.date_input(
            "Ngày kết thúc", value=max_date, min_value=min_date, max_value=max_date
        )
        # Lọc dữ liệu theo khoảng thời gian
        if start_date > end_date:
            st.error("Ngày bắt đầu không thể lớn hơn ngày kết thúc.")
        else:
            filtered_data = data[
                (data["date"] >= pd.Timestamp(start_date))
                & (data["date"] <= pd.Timestamp(end_date))
            ]

            # Tính toán dữ liệu theo nhóm thời gian
            cases_over_time = (
                filtered_data.groupby("date")["new_cases"].sum().reset_index()
            )
            deaths_over_time = (
                filtered_data.groupby("date")["new_deaths"].sum().reset_index()
            )
            vaccinations_over_time = (
                filtered_data.groupby("date")["new_vaccinations"].sum().reset_index()
            )
            # -- Biểu đồ số ca nhiễm --
            st.subheader("Số Ca Nhiễm Mới Theo Thời Gian")
            fig_cases = px.line(
                cases_over_time,
                x="date",
                y="new_cases",
                title="Số Ca Nhiễm Mới Theo Thời Gian",
                labels={"new_cases": "Số Ca Nhiễm Mới", "date": "Ngày"},
            )
            st.plotly_chart(fig_cases, key="fig_cases")

            # --Biểu đồ số ca tử vong --
            st.subheader("Số Ca Tử Vong Mới Theo Thời Gian")
            fig_deaths = px.line(
                deaths_over_time,
                x="date",
                y="new_deaths",
                title="Số Ca Tử Vong Mới Theo Thời Gian",
                labels={"new_deaths": "Số Ca Tử Vong Mới", "date": "Ngày"},
                line_shape="linear",
            )
            fig_deaths.update_traces(line=dict(color="red"))
            st.plotly_chart(fig_deaths, key="fig_deaths")

            # -- Biểu đồ số liều vaccine --
            st.subheader("Số Liều Vaccine Được Tiêm Theo Thời Gian")
            fig_vaccinations = px.line(
                vaccinations_over_time,
                x="date",
                y="new_vaccinations",
                title="Số Liều Vaccine Được Tiêm Theo Thời Gian",
                labels={"new_vaccinations": "Số Liều Vaccine", "date": "Ngày"},
                line_shape="linear",
            )
            fig_vaccinations.update_traces(line=dict(color="green"))
            st.plotly_chart(fig_vaccinations, key="fig_vaccinations")
    elif sub_option == "Bản đồ thế giới (số ca nhiễm, tử vong,hỗ trợ chính phủ)":
        st.subheader("Bản đồ thế giới về COVID-19")
        # -- Tổng số ca nhiễm covid trên toàn thế giới --
        # Đầu tiên nhóm dữ liệu và tính cố ca max cho mỗi quốc gia
        # tiến hành merge các chỉ số iso code của df và map cho khớp
        df_cases = country_data.groupby("iso_code")["total_cases"].max().reset_index()
        df_merged = df_cases.merge(
            df_territory, left_on="iso_code", right_on="ISO3166-1-Alpha-3", how="left"
        )

        fig_cases = px.choropleth(
            df_merged,
            locations="iso_code",
            color="total_cases",
            hover_name="official_name_en",
            color_continuous_scale="Reds",
            title="Tổng số ca nhiễm covid trên toàn thế giới",
            projection="natural earth",
        )
        fig_cases.update_layout(
            height=700,
            width=1200,
        )
        st.plotly_chart(fig_cases)
        # -- Tổng số ca tử vong covid trên toàn thế giới --
        # nhóm và láy max dữ liệu quốc gia sau đó merge các chỉ số iso code cho khớp
        df_deaths = country_data.groupby("iso_code")["total_deaths"].max().reset_index()
        df_merged = df_deaths.merge(
            df_territory, left_on="iso_code", right_on="ISO3166-1-Alpha-3", how="left"
        )

        fig_deaths = px.choropleth(
            df_merged,
            locations="iso_code",
            color="total_deaths",
            hover_name="official_name_en",
            color_continuous_scale="Reds",
            title="Tổng số ca tử vong covid trên toàn thế giới",
            projection="natural earth",
        )
        fig_deaths.update_layout(
            height=700,
            width=1200,
        )
        st.plotly_chart(fig_deaths)

        # Bản đồ hỗ trợ của chính phủ trên toàn cầu

        gov_support = (
            df_support.groupby("Code")["e1_income_support"].max().reset_index()
        )
        df_merged = gov_support.merge(
            df_territory, left_on="Code", right_on="ISO3166-1-Alpha-3", how="left"
        )
        fig_support = px.choropleth(
            df_merged,
            locations="Code",
            color="e1_income_support",
            hover_name="official_name_en",
            color_continuous_scale="Blues",
            title="Hỗ trợ của chính phủ",
            projection="natural earth",
        )
        fig_support.update_layout(
            height=700,
            width=1200,
        )
        # chú thích
        fig_support.update_layout(
            coloraxis_colorbar=dict(
                title="Hỗ trợ chính phủ",
                tickvals=[0, 1, 2],
                ticktext=[
                    "0: không hỗ trợ",
                    "1: Hỗ trợ dưới 50% thu nhập",
                    "2:Hỗ trợ hơn 50% thu nhập",
                ],
            )
        )

        st.plotly_chart(fig_support)
    elif sub_option == "Các châu lục và quốc gia":

        st.subheader("Tỉ lệ nhiễm và tử vong của các châu lục")
        # Tính tổng số ca nhiễm và tử vong theo các châu lục
        df_grouped = (
            country_data.groupby("continent")
            .agg({"total_cases": "sum", "total_deaths": "sum"})
            .reset_index()
        )

        # --- Biểu đồ tròn tương tác cho số ca nhiễm ---

        st.subheader("Tỉ Lệ Số Ca Nhiễm Theo Châu Lục")
        fig_cases = px.pie(
            df_grouped,
            names="continent",
            values="total_cases",
            title="Total Cases by Continent",
            color_discrete_sequence=px.colors.sequential.Plasma,
        )
        st.plotly_chart(fig_cases, use_container_width=True)

        # --- Biểu đồ tròn tương tác cho số ca tử vong ---
        st.subheader("Tỉ Lệ Số Ca Tử Vong Theo Châu Lục")
        fig_deaths = px.pie(
            df_grouped,
            names="continent",
            values="total_deaths",
            title="Total Deaths by Continent",
            color_discrete_sequence=px.colors.sequential.Viridis,
        )
        st.plotly_chart(fig_deaths, use_container_width=True)

        st.subheader("Các nước dẫn đầu về số ca nhiễm và tử vong")

        # Dữ liệu Top 10 quốc gia
        top_countries_cases = (
            country_data.groupby("location")["total_cases"].max().nlargest(10)
        )
        top_countries_deaths = (
            country_data.groupby("location")["total_deaths"].max().nlargest(10)
        )

        # --- Biểu đồ Top 10 quốc gia có số ca nhiễm cao nhất ---

        st.subheader("Top 10 Quốc Gia Có Số Ca Nhiễm Cao Nhất")
        fig_cases = px.bar(
            top_countries_cases.reset_index(),
            x="total_cases",
            y="location",
            orientation="h",
            title="Top 10 Quốc Gia Có Số Ca Nhiễm Cao Nhất",
            color="total_cases",
            color_continuous_scale="Plasma",
            labels={"location": "Quốc gia", "total_cases": "Số ca nhiễm"},
        )
        fig_cases.update_layout(
            xaxis_title="Số ca nhiễm",
            yaxis_title="Quốc gia",
            yaxis=dict(categoryorder="total ascending"),
            template="plotly_white",
        )
        st.plotly_chart(fig_cases, use_container_width=True)

        # --- Biểu đồ Top 10 quốc gia có số ca tử vong cao nhất ---
        st.subheader("Top 10 Quốc Gia Có Số Ca Tử Vong Cao Nhất")
        fig_deaths = px.bar(
            top_countries_deaths.reset_index(),
            x="total_deaths",
            y="location",
            orientation="h",
            title="Top 10 Quốc Gia Có Số Ca Tử Vong Cao Nhất",
            color="total_deaths",
            color_continuous_scale="Viridis",
            labels={"location": "Quốc gia", "total_deaths": "Số ca tử vong"},
        )
        fig_deaths.update_layout(
            xaxis_title="Số ca tử vong",
            yaxis_title="Quốc gia",
            yaxis=dict(categoryorder="total ascending"),
            template="plotly_white",
        )
        st.plotly_chart(fig_deaths, use_container_width=True)

        st.subheader("Xu hướng của các châu lục")

        # Chuyển đổi cột 'date' thành kiểu datetime và tạo cột 'year_quarter'
        # Chuyển đổi cột 'date' thành kiểu datetime và tạo cột 'year_quarter'
        continent_data["date"] = pd.to_datetime(continent_data["date"], errors="coerce")
        continent_data.dropna(subset=["date"], inplace=True)
        continent_data["year_quarter"] = (
            continent_data["date"].dt.to_period("Q").astype(str)
        )

        # Thay thế giá trị NaN bằng 0 cho các cột cần thiết
        for col in ["new_cases", "new_deaths", "total_vaccinations"]:
            if col in continent_data.columns:
                continent_data[col] = continent_data[col].fillna(0)

        # Tạo dữ liệu theo quý cho các chỉ số
        metrics = ["new_cases", "new_deaths", "total_vaccinations"]
        quarterly_data = {}
        for metric in metrics:
            if metric in continent_data.columns:
                quarterly_data[metric] = (
                    continent_data.groupby(["location", "year_quarter"])[metric]
                    .sum()
                    .reset_index()
                )
            else:
                st.warning(f"Dữ liệu cho chỉ số '{metric}' không tồn tại.")

        # Hàm vẽ biểu đồ cho từng chỉ số với Plotly
        def plot_metric(data, metric, title, ylabel, color):
            if data.empty:
                st.warning(f"Không có dữ liệu để hiển thị {ylabel}.")
                return None
            fig = px.line(
                data,
                x="year_quarter",
                y=metric,
                color="location",
                markers=True,
                labels={
                    "year_quarter": "Năm - Quý",
                    metric: ylabel,
                    "location": "Châu lục",
                },
                title=title,
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig.update_traces(marker=dict(size=8, line=dict(width=1, color="black")))
            fig.update_layout(
                title=dict(
                    text=title,
                    font=dict(size=18, color=color),  # Định dạng font cho tiêu đề
                ),
                xaxis=dict(
                    title="Năm - Quý",
                    title_font=dict(size=12),  # Font-size cho tiêu đề trục X
                    tickangle=-45,
                ),
                yaxis=dict(
                    title=ylabel,
                    title_font=dict(size=12),  # Font-size cho tiêu đề trục Y
                    tickformat=".1fM",  # Định dạng số liệu trục Y
                ),
                template="plotly_white",
                legend=dict(
                    title=dict(
                        text="Châu lục", font=dict(size=12)
                    ),  # Font-size cho tiêu đề legend
                    font=dict(size=10),  # Font-size cho các mục trong legend
                ),
            )
            return fig

        # Biểu đồ các chỉ số

        if "new_cases" in quarterly_data:
            fig_cases = plot_metric(
                quarterly_data["new_cases"],
                "new_cases",
                "Tổng số ca nhiễm COVID-19 theo quý tại các châu lục (2020-2024)",
                "Tổng số ca nhiễm",
                "darkblue",
            )
            if fig_cases:
                st.plotly_chart(fig_cases, use_container_width=True)

        if "new_deaths" in quarterly_data:
            fig_deaths = plot_metric(
                quarterly_data["new_deaths"],
                "new_deaths",
                "Tổng số ca tử vong COVID-19 theo quý tại các châu lục (2020-2024)",
                "Tổng số ca tử vong",
                "darkred",
            )
            if fig_deaths:
                st.plotly_chart(fig_deaths, use_container_width=True)

        if "total_vaccinations" in quarterly_data:
            fig_vaccinations = plot_metric(
                quarterly_data["total_vaccinations"],
                "total_vaccinations",
                "Tổng số liều tiêm chủng COVID-19 theo quý tại các châu lục (2020-2024)",
                "Tổng số liều tiêm chủng",
                "darkgreen",
            )
            if fig_vaccinations:
                st.plotly_chart(fig_vaccinations, use_container_width=True)

with tab3:

    st.subheader("Biểu Đồ Tương Quan Giữa các Chỉ số")

    fig = px.scatter(
        World_data,
        x="new_cases",
        y="new_deaths",
        title=f"Tương Quan Số Ca Nhiễm Mới và Số Ca Tử Vong Mới  ",
        labels={"new_cases": "Số Ca Nhiễm Mới", "new_deaths": "Số Ca Tử Vong Mới"},
        opacity=0.7,
    )

    fig.update_traces(marker=dict(size=8, line=dict(width=1, color="black")))
    fig.update_layout(
        xaxis_title="Số Ca Nhiễm Mới",
        yaxis_title="Số Ca Tử Vong Mới",
        template="plotly_white",
        title_font=dict(size=18),
        xaxis=dict(title_font=dict(size=12), tickformat=".1f"),
        yaxis=dict(title_font=dict(size=12), tickformat=".1f"),
    )

    st.plotly_chart(fig, use_container_width=True)

    # -- biều đồ tương quan giữa ca hồi phục và ca nhiễm mới
    World_data["month"] = World_data["date"].dt.to_period("M")

    # Tính toán tổng quan theo tháng
    monthly_summary = (
        World_data.groupby("month")
        .agg(
            total_cases_start=("total_cases", "first"),
            total_cases_end=("total_cases", "last"),
            total_deaths_start=("total_deaths", "first"),
            total_deaths_end=("total_deaths", "last"),
            new_cases=("new_cases", "sum"),
        )
        .reset_index()
    )

    # Tính số ca hồi phục trong tháng
    monthly_summary["recovered_in_month"] = (
        monthly_summary["total_cases_end"] - monthly_summary["total_deaths_end"]
    ) - (monthly_summary["total_cases_start"] - monthly_summary["total_deaths_start"])

    # Đảm bảo không có giá trị âm
    monthly_summary["recovered_in_month"] = monthly_summary["recovered_in_month"].clip(
        lower=0
    )
    monthly_summary["new_cases"] = monthly_summary["new_cases"].clip(lower=0)

    # Lọc các tháng đại diện cho quý
    quarters = ["01", "04", "07", "10"]
    monthly_summary["month_str"] = monthly_summary["month"].astype(str)
    filtered_months = monthly_summary[
        monthly_summary["month_str"].str[-2:].isin(quarters)
    ]

    # Vẽ biểu đồ tương tác bằng Plotly
    st.subheader("Biểu Đồ Tương Tác Ca Nhiễm và Hồi Phục Theo Quý")

    fig = go.Figure()

    # Thêm dữ liệu "Ca Nhiễm Mới"
    fig.add_trace(
        go.Bar(
            x=filtered_months["month_str"],
            y=filtered_months["new_cases"],
            name="Ca Nhiễm Mới",
            marker_color="orange",
        )
    )

    # Thêm dữ liệu "Ca Hồi Phục"
    fig.add_trace(
        go.Bar(
            x=filtered_months["month_str"],
            y=filtered_months["recovered_in_month"],
            name="Ca Hồi Phục",
            marker_color="green",
            base=filtered_months["new_cases"],  # Đặt giá trị cơ sở là số ca nhiễm mới
        )
    )

    # Tùy chỉnh biểu đồ
    fig.update_layout(
        title="Tương quan ca nhiễm mới và hồi phục",
        xaxis_title="Tháng",
        yaxis_title="Số Ca",
        barmode="stack",  # Hiển thị kiểu stacked bar
        template="plotly_white",
        legend=dict(title="Chú Thích"),
    )

    # Hiển thị biểu đồ trên Streamlit
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("So sánh biến thể Omicron và Delta")

    # Lọc dữ liệu cho biến chủng Delta (từ tháng 5/2021 đến tháng 12/2021)
    delta_wave = World_data[
        (World_data["date"] >= "2021-05-01") & (World_data["date"] <= "2021-12-31")
    ]

    # Lọc dữ liệu cho biến chủng Omicron (từ tháng 12/2021 đến giữa năm 2022)
    omicron_wave = World_data[
        (World_data["date"] >= "2021-12-01") & (World_data["date"] <= "2022-06-30")
    ]

    # Tạo hai cột song song
    col1, col2 = st.columns(2)

    with col1:
        # Biểu đồ số ca nhiễm
        fig_cases = go.Figure()
        fig_cases.add_trace(
            go.Scatter(
                x=delta_wave["date"],
                y=delta_wave["new_cases"],
                mode="lines",
                name="Delta Wave",
                line=dict(color="blue"),
            )
        )
        fig_cases.add_trace(
            go.Scatter(
                x=omicron_wave["date"],
                y=omicron_wave["new_cases"],
                mode="lines",
                name="Omicron Wave",
                line=dict(color="orange"),
            )
        )
        fig_cases.update_layout(
            title="Số Ca Nhiễm COVID-19 Giữa Delta và Omicron",
            xaxis_title="Ngày",
            yaxis_title="Số ca nhiễm mới",
            legend=dict(x=0.5, y=1.2, orientation="h"),
            xaxis=dict(showgrid=True),
            yaxis=dict(showgrid=True),
        )
        st.plotly_chart(fig_cases, use_container_width=True)

    with col2:
        # Biểu đồ số ca tử vong
        fig_deaths = go.Figure()
        fig_deaths.add_trace(
            go.Scatter(
                x=delta_wave["date"],
                y=delta_wave["new_deaths"],
                mode="lines",
                name="Delta Wave",
                line=dict(color="red"),
            )
        )
        fig_deaths.add_trace(
            go.Scatter(
                x=omicron_wave["date"],
                y=omicron_wave["new_deaths"],
                mode="lines",
                name="Omicron Wave",
                line=dict(color="green"),
            )
        )
        fig_deaths.update_layout(
            title="Số Ca Tử Vong COVID-19 Giữa Delta và Omicron",
            xaxis_title="Ngày",
            yaxis_title="Số ca tử vong mới",
            legend=dict(x=0.5, y=1.2, orientation="h"),
            xaxis=dict(showgrid=True),
            yaxis=dict(showgrid=True),
        )
        st.plotly_chart(fig_deaths, use_container_width=True)

    st.subheader("Phân tích tiêm chủng toàn cầu")
    wc = pd.read_csv(FILE_PATH2)
    wc["date"] = pd.to_datetime(wc["date"])

    # Tính tổng số ca nhiễm và tổng số liều tiêm theo ngày
    time_vaccine_cases = wc.groupby("date").sum()

    # Tạo hai cột song song
    col1, col2 = st.columns(2)

    with col1:
        # Biểu đồ số ca tử vong mới và số liều tiêm
        fig1 = go.Figure()
        fig1.add_trace(
            go.Scatter(
                x=time_vaccine_cases.index,
                y=time_vaccine_cases["new_deaths"],
                mode="lines",
                name="Số ca tử vong mới",
                line=dict(color="red"),
            )
        )
        fig1.add_trace(
            go.Scatter(
                x=time_vaccine_cases.index,
                y=time_vaccine_cases["total_vaccinations"] / 1e6,
                mode="lines",
                name="Tổng số liều tiêm (triệu)",
                line=dict(color="blue"),
                yaxis="y2",
            )
        )
        fig1.update_layout(
            title="Số Ca Tử Vong Mới và Số Liều Tiêm",
            xaxis=dict(title="Ngày"),
            yaxis=dict(
                title="Số ca tử vong mới",
                titlefont=dict(color="red"),
                tickfont=dict(color="red"),
            ),
            yaxis2=dict(
                title="Tổng số liều tiêm (triệu)",
                overlaying="y",
                side="right",
                titlefont=dict(color="blue"),
                tickfont=dict(color="blue"),
            ),
            legend=dict(x=0.5, y=1.15, orientation="h"),
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # Biểu đồ số ca nhiễm mới và số liều tiêm
        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(
                x=time_vaccine_cases.index,
                y=time_vaccine_cases["new_cases"],
                mode="lines",
                name="Số ca nhiễm mới",
                line=dict(color="green"),
            )
        )
        fig2.add_trace(
            go.Scatter(
                x=time_vaccine_cases.index,
                y=time_vaccine_cases["total_vaccinations"] / 1e6,
                mode="lines",
                name="Tổng số liều tiêm (triệu)",
                line=dict(color="blue"),
                yaxis="y2",
            )
        )
        fig2.update_layout(
            title="Số Ca Nhiễm Mới và Số Liều Tiêm",
            xaxis=dict(title="Ngày"),
            yaxis=dict(
                title="Số ca nhiễm mới",
                titlefont=dict(color="red"),
                tickfont=dict(color="red"),
            ),
            yaxis2=dict(
                title="Tổng số liều tiêm (triệu)",
                overlaying="y",
                side="right",
                titlefont=dict(color="blue"),
                tickfont=dict(color="blue"),
            ),
            legend=dict(x=0.5, y=1.15, orientation="h"),
        )
        st.plotly_chart(fig2, use_container_width=True)


with tab4:
    sub_option2 = st.selectbox(
        "Chọn phân tích chi tiết:",
        [
            "Ảnh hưởng của tiêm chủng lên ca nhiễm và tử vong",
            "So sánh các biến thể với nhau",
            "Biện pháp cách ly giản cách xã hội",
            "Dự đoán số ca nhiễm và tử vong trong tương lai",
        ],
    )
    if sub_option2 == "Ảnh hưởng của tiêm chủng lên ca nhiễm và tử vong":

        # Phân nhóm dữ liệu trước và sau khi tiêm vaccine
        vaccination_threshold = st.slider(
            "Chọn tỷ lệ tiêm vaccine để phân chia nhóm", 0, 100, 30
        )
        pre_vaccination = World_data[
            World_data["people_fully_vaccinated_per_hundred"] < vaccination_threshold
        ]
        post_vaccination = World_data[
            World_data["people_fully_vaccinated_per_hundred"] >= vaccination_threshold
        ]

        # Hiển thị thông tin nhóm
        st.subheader(
            f"Nhóm trước khi tiêm vaccine (Dưới {vaccination_threshold}% tiêm vaccine)"
        )
        st.write(pre_vaccination)

        st.subheader(
            f"Nhóm sau khi tiêm vaccine (Trên {vaccination_threshold}% tiêm vaccine)"
        )
        st.write(post_vaccination)

        # T-test: So sánh số ca nhiễm trước và sau khi tiêm vaccine
        t_stat_cases, p_value_cases = ttest_ind(
            pre_vaccination["new_cases"], post_vaccination["new_cases"], equal_var=False
        )
        t_stat_deaths, p_value_deaths = ttest_ind(
            pre_vaccination["new_deaths"],
            post_vaccination["new_deaths"],
            equal_var=False,
        )

        # Hiển thị kết quả T-test
        st.subheader("Kết quả kiểm định T-test")
        st.write(
            f"T-test số ca nhiễm - T-statistic: {t_stat_cases}, P-value: {p_value_cases}"
        )
        st.write(
            f"T-test số ca tử vong - T-statistic: {t_stat_deaths}, P-value: {p_value_deaths}"
        )

        # Đưa ra kết luận
        if p_value_cases < 0.05:
            st.write(
                "➡️Có sự khác biệt có ý nghĩa thống kê về số ca nhiễm trước và sau khi tiêm vaccine."
            )
        else:
            st.write(
                "➡️Không có sự khác biệt có ý nghĩa thống kê về số ca nhiễm trước và sau khi tiêm vaccine."
            )

        if p_value_deaths < 0.05:
            st.write(
                "➡️Có sự khác biệt có ý nghĩa thống kê về số ca tử vong trước và sau khi tiêm vaccine."
            )
        else:
            st.write(
                "➡️Không có sự khác biệt có ý nghĩa thống kê về số ca tử vong trước và sau khi tiêm vaccine."
            )

        # Đánh giá mô hình phân loại (chỉ tính Accuracy)
        # Chọn đặc trưng và nhãn để huấn luyện mô hình (ví dụ, bạn có thể chọn số ca nhiễm hoặc tử vong là nhãn)
        X = World_data[
            ["people_fully_vaccinated_per_hundred", "new_cases", "new_deaths"]
        ]  # Các đặc trưng
        y = (World_data["new_cases"] > 100).astype(
            int
        )  # Giả sử phân loại: 1 nếu số ca nhiễm > 100, 0 nếu không

        # Chia dữ liệu thành tập huấn luyện và kiểm tra
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Huấn luyện mô hình RandomForestClassifier
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        # Dự đoán trên dữ liệu kiểm tra
        y_pred = model.predict(X_test)

        # Tính toán accuracy
        accuracy = accuracy_score(y_test, y_pred)

        # Hiển thị kết quả
        st.subheader("Đánh giá hiệu suất mô hình (Accuracy)")
        st.write(f"Accuracy: {accuracy}")
        # Tính tổng số ca nhiễm và tổng số liều tiêm theo ngày
        # Chuyển đổi cột ngày thành datetime
        wc["date"] = pd.to_datetime(wc["date"])
        # Tính tổng số ca nhiễm và tổng số liều tiêm theo ngày
        time_vaccine_cases = wc.groupby("date").sum()
        # Tạo hai cột song song
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Số Ca Tử Vong Mới và Số Liều Tiêm Theo Thời Gian")
            # Biểu đồ số ca tử vong mới và số liều tiêm
            fig11 = go.Figure()
            fig11.add_trace(
                go.Scatter(
                    x=time_vaccine_cases.index,
                    y=time_vaccine_cases["new_deaths"],
                    mode="lines",
                    name="Số ca tử vong mới",
                    line=dict(color="red"),
                )
            )
            fig11.add_trace(
                go.Scatter(
                    x=time_vaccine_cases.index,
                    y=time_vaccine_cases["total_vaccinations"] / 1e6,
                    mode="lines",
                    name="Tổng số liều tiêm (triệu)",
                    line=dict(color="blue"),
                    yaxis="y2",
                )
            )
            fig11.update_layout(
                title="Số Ca Tử Vong Mới và Số Liều Tiêm",
                xaxis=dict(title="Ngày"),
                yaxis=dict(
                    title="Số ca tử vong mới",
                    titlefont=dict(color="red"),
                    tickfont=dict(color="red"),
                ),
                yaxis2=dict(
                    title="Tổng số liều tiêm (triệu)",
                    overlaying="y",
                    side="right",
                    titlefont=dict(color="blue"),
                    tickfont=dict(color="blue"),
                ),
                legend=dict(x=0.5, y=1.15, orientation="h"),
            )
            st.plotly_chart(fig11, use_container_width=True, key="fig11")

        with col2:
            st.subheader("Số Ca Nhiễm Mới và Số Liều Tiêm Theo Thời Gian")
            # Biểu đồ số ca nhiễm mới và số liều tiêm
            fig21 = go.Figure()
            fig21.add_trace(
                go.Scatter(
                    x=time_vaccine_cases.index,
                    y=time_vaccine_cases["new_cases"],
                    mode="lines",
                    name="Số ca nhiễm mới",
                    line=dict(color="green"),
                )
            )
            fig21.add_trace(
                go.Scatter(
                    x=time_vaccine_cases.index,
                    y=time_vaccine_cases["total_vaccinations"] / 1e6,
                    mode="lines",
                    name="Tổng số liều tiêm (triệu)",
                    line=dict(color="blue"),
                    yaxis="y2",
                )
            )
            fig21.update_layout(
                title="Số Ca Nhiễm Mới và Số Liều Tiêm",
                xaxis=dict(title="Ngày"),
                yaxis=dict(
                    title="Số ca nhiễm mới",
                    titlefont=dict(color="red"),
                    tickfont=dict(color="red"),
                ),
                yaxis2=dict(
                    title="Tổng số liều tiêm (triệu)",
                    overlaying="y",
                    side="right",
                    titlefont=dict(color="blue"),
                    tickfont=dict(color="blue"),
                ),
                legend=dict(x=0.5, y=1.15, orientation="h"),
            )
            st.plotly_chart(fig21, use_container_width=True, key="fig21")
    elif sub_option2 == "So sánh các biến thể với nhau":
        st.title("So sánh biến chủng Delta và Omicron qua số ca nhiễm và tử vong")
        # Lọc dữ liệu cho biến chủng Delta (từ tháng 5/2021 đến tháng 12/2021)
        delta_wave = World_data[
            (World_data["date"] >= "2021-05-01") & (World_data["date"] <= "2021-12-31")
        ]
        # Lọc dữ liệu cho biến chủng Omicron (từ tháng 12/2021 đến giữa năm 2022)
        omicron_wave = World_data[
            (World_data["date"] >= "2021-12-01") & (World_data["date"] <= "2022-06-30")
        ]
        # Thực hiện ANOVA cho số ca nhiễm mới
        anova_cases = f_oneway(delta_wave["new_cases"], omicron_wave["new_cases"])
        # Thực hiện ANOVA cho số ca tử vong mới
        anova_deaths = f_oneway(delta_wave["new_deaths"], omicron_wave["new_deaths"])
        # Hiển thị kết quả ANOVA
        st.subheader("Kết quả kiểm định ANOVA")
        st.write("**ANOVA - Số ca nhiễm mới**:")
        st.write(
            f"F-statistic: {anova_cases.statistic:.2f}, P-value: {anova_cases.pvalue:.4f}"
        )
        if anova_cases.pvalue < 0.05:
            st.write(
                "➡️ Có sự khác biệt có ý nghĩa thống kê về số ca nhiễm mới giữa hai biến chủng."
            )
        else:
            st.write(
                "➡️ Không có sự khác biệt có ý nghĩa thống kê về số ca nhiễm mới giữa hai biến chủng."
            )
        st.write("**ANOVA - Số ca tử vong mới**:")
        st.write(
            f"F-statistic: {anova_deaths.statistic:.2f}, P-value: {anova_deaths.pvalue:.4f}"
        )
        if anova_deaths.pvalue < 0.05:
            st.write(
                "➡️ Có sự khác biệt có ý nghĩa thống kê về số ca tử vong mới giữa hai biến chủng."
            )
        else:
            st.write(
                "➡️ Không có sự khác biệt có ý nghĩa thống kê về số ca tử vong mới giữa hai biến chủng."
            )
        # Lọc dữ liệu cho biến chủng Delta (từ tháng 5/2021 đến tháng 12/2021)
        delta_wave = World_data[
            (World_data["date"] >= "2021-05-01") & (World_data["date"] <= "2021-12-31")
        ]
        # Lọc dữ liệu cho biến chủng Omicron (từ tháng 12/2021 đến giữa năm 2022)
        omicron_wave = World_data[
            (World_data["date"] >= "2021-12-01") & (World_data["date"] <= "2022-06-30")
        ]
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Số Ca Nhiễm COVID-19 Giữa Biến Chủng Delta và Omicron")
            # Biểu đồ số ca nhiễm
            fig_cases1 = go.Figure()
            fig_cases1.add_trace(
                go.Scatter(
                    x=delta_wave["date"],
                    y=delta_wave["new_cases"],
                    mode="lines",
                    name="Delta Wave",
                    line=dict(color="blue"),
                )
            )
            fig_cases1.add_trace(
                go.Scatter(
                    x=omicron_wave["date"],
                    y=omicron_wave["new_cases"],
                    mode="lines",
                    name="Omicron Wave",
                    line=dict(color="orange"),
                )
            )
            fig_cases1.update_layout(
                title="Số Ca Nhiễm COVID-19 Giữa Delta và Omicron",
                xaxis_title="Ngày",
                yaxis_title="Số ca nhiễm mới",
                legend=dict(x=0.5, y=1.2, orientation="h"),
                xaxis=dict(showgrid=True),
                yaxis=dict(showgrid=True),
            )
            st.plotly_chart(fig_cases1, use_container_width=True, key="fig_cases1")

        with col2:
            st.subheader("Số Ca Tử Vong COVID-19 Giữa Biến Chủng Delta và Omicron")
            # Biểu đồ số ca tử vong
            fig_deaths1 = go.Figure()
            fig_deaths1.add_trace(
                go.Scatter(
                    x=delta_wave["date"],
                    y=delta_wave["new_deaths"],
                    mode="lines",
                    name="Delta Wave",
                    line=dict(color="red"),
                )
            )
            fig_deaths1.add_trace(
                go.Scatter(
                    x=omicron_wave["date"],
                    y=omicron_wave["new_deaths"],
                    mode="lines",
                    name="Omicron Wave",
                    line=dict(color="green"),
                )
            )
            fig_deaths1.update_layout(
                title="Số Ca Tử Vong COVID-19 Giữa Delta và Omicron",
                xaxis_title="Ngày",
                yaxis_title="Số ca tử vong mới",
                legend=dict(x=0.5, y=1.2, orientation="h"),
                xaxis=dict(showgrid=True),
                yaxis=dict(showgrid=True),
            )
            st.plotly_chart(fig_deaths1, use_container_width=True, key="fig_deaths1")
    elif sub_option2 == "Biện pháp cách ly giản cách xã hội":
        # Giao diện người dùng Streamlit
        st.title("Phân tích tác động chính sách COVID-19 và đánh giá mô hình")

        # Chọn thời điểm triển khai chính sách
        policy_date = st.date_input(
            "Chọn ngày triển khai chính sách:", value=pd.to_datetime("2020-03-01")
        )

        # Chuyển đổi policy_date về kiểu datetime64[ns] để tương thích với cột 'date'
        policy_date = pd.to_datetime(policy_date)

        # Chia dữ liệu thành hai nhóm: trước và sau chính sách
        before_policy = World_data[World_data["date"] < policy_date]
        after_policy = World_data[World_data["date"] >= policy_date]

        # Đảm bảo hai nhóm có cùng số lượng mẫu
        min_samples = min(len(before_policy), len(after_policy))
        before_cases = before_policy["new_cases"].iloc[:min_samples]
        after_cases = after_policy["new_cases"].iloc[:min_samples]

        before_deaths = before_policy["new_deaths"].iloc[:min_samples]
        after_deaths = after_policy["new_deaths"].iloc[:min_samples]

        # Kiểm định T-Test ghép cặp cho số ca nhiễm mới
        ttest_cases_stat, ttest_cases_p = ttest_rel(before_cases, after_cases)

        # Kiểm định T-Test ghép cặp cho số ca tử vong mới
        ttest_deaths_stat, ttest_deaths_p = ttest_rel(before_deaths, after_deaths)

        # Hiển thị kết quả kiểm định
        st.subheader("Kết quả kiểm định T-Test (Paired Samples)")
        st.write("**Số ca nhiễm mới:**")
        st.write(f"T-statistic: {ttest_cases_stat:.2f}, P-value: {ttest_cases_p:.4f}")
        if ttest_cases_p < 0.05:
            st.write(
                "➡️ Có sự khác biệt có ý nghĩa thống kê về số ca nhiễm mới trước và sau chính sách."
            )
        else:
            st.write(
                "➡️ Không có sự khác biệt có ý nghĩa thống kê về số ca nhiễm mới trước và sau chính sách."
            )

        st.write("**Số ca tử vong mới:**")
        st.write(f"T-statistic: {ttest_deaths_stat:.2f}, P-value: {ttest_deaths_p:.4f}")
        if ttest_deaths_p < 0.05:
            st.write(
                "➡️ Có sự khác biệt có ý nghĩa thống kê về số ca tử vong mới trước và sau chính sách."
            )
        else:
            st.write(
                "➡️ Không có sự khác biệt có ý nghĩa thống kê về số ca tử vong mới trước và sau chính sách."
            )
    elif sub_option2 == "Dự đoán số ca nhiễm và tử vong trong tương lai":
        # Tính số ngày kể từ ngày đầu tiên trong dữ liệu
        World_data["days_since_start"] = (
            World_data["date"] - World_data["date"].min()
        ).dt.days

        # Tạo mô hình hồi quy tuyến tính cho số ca nhiễm
        X = World_data[["days_since_start"]]  # Biến độc lập (số ngày)
        y_cases = World_data["new_cases"]  # Biến phụ (số ca nhiễm)

        regressor_cases = LinearRegression()
        regressor_cases.fit(X, y_cases)

        # Dự đoán số ca nhiễm trong tương lai (365 ngày tới)
        future_days = np.arange(
            X["days_since_start"].max() + 1, X["days_since_start"].max() + 366
        ).reshape(-1, 1)
        future_cases = regressor_cases.predict(future_days)

        # Đảm bảo rằng số ca nhiễm không thể âm
        future_cases = np.maximum(future_cases, 0)

        # Tạo mô hình hồi quy tuyến tính cho số ca tử vong
        y_deaths = World_data["new_deaths"]  # Biến phụ (số ca tử vong)

        regressor_deaths = LinearRegression()
        regressor_deaths.fit(X, y_deaths)

        # Dự đoán số ca tử vong trong tương lai (365 ngày tới)
        future_deaths = regressor_deaths.predict(future_days)

        # Đảm bảo rằng số ca tử vong không thể âm
        future_deaths = np.maximum(future_deaths, 0)

        # Tạo layout cho Streamlit
        st.title("Dự đoán Số Ca Nhiễm và Ca Tử Vong COVID-19")

        # Biểu đồ dự đoán số ca nhiễm
        st.subheader("Dự đoán số ca nhiễm COVID-19 trong 1 năm tới")
        fig_cases = go.Figure()

        # Thêm dữ liệu thực tế
        fig_cases.add_trace(
            go.Scatter(
                x=World_data["date"],
                y=World_data["new_cases"],
                mode="lines",
                name="Số ca nhiễm thực tế",
                line=dict(color="green"),
            )
        )

        # Thêm dữ liệu dự đoán
        fig_cases.add_trace(
            go.Scatter(
                x=pd.to_datetime(World_data["date"].min())
                + pd.to_timedelta(future_days.flatten(), unit="D"),
                y=future_cases,
                mode="lines",
                name="Dự đoán số ca nhiễm",
                line=dict(color="blue", dash="dash"),
            )
        )

        #        Cấu hình biểu đồ
        fig_cases.update_layout(
            title="Dự đoán Số Ca Nhiễm COVID-19",
            xaxis_title="Ngày",
            yaxis_title="Số ca nhiễm mới",
            legend=dict(x=0.5, y=1.2, orientation="h"),
            xaxis=dict(showgrid=True),
            yaxis=dict(showgrid=True),
        )

        st.plotly_chart(fig_cases, use_container_width=True)

        # Biểu đồ dự đoán số ca tử vong
        st.subheader("Dự đoán số ca tử vong COVID-19 trong 1 năm tới")
        fig_deaths = go.Figure()

        # Thêm dữ liệu thực tế
        fig_deaths.add_trace(
            go.Scatter(
                x=World_data["date"],
                y=World_data["new_deaths"],
                mode="lines",
                name="Số ca tử vong thực tế",
                line=dict(color="red"),
            )
        )

        # Thêm dữ liệu dự đoán
        fig_deaths.add_trace(
            go.Scatter(
                x=pd.to_datetime(World_data["date"].min())
                + pd.to_timedelta(future_days.flatten(), unit="D"),
                y=future_deaths,
                mode="lines",
                name="Dự đoán số ca tử vong",
                line=dict(color="orange", dash="dash"),
            )
        )

        # Cấu hình biểu đồ
        fig_deaths.update_layout(
            title="Dự đoán Số Ca Tử Vong COVID-19",
            xaxis_title="Ngày",
            yaxis_title="Số ca tử vong mới",
            legend=dict(x=0.5, y=1.2, orientation="h"),
            xaxis=dict(showgrid=True),
            yaxis=dict(showgrid=True),
        )

        st.plotly_chart(fig_deaths, use_container_width=True)
with tab5:
    st.header("Ảnh hưởng của COVID-19")

    # Các tùy chọn cho phần ảnh hưởng
    sub_option = st.selectbox(
        "Chọn phân tích chi tiết:",
        [
            "Ảnh hưởng kinh tế với các quốc gia lớn (Q2 2020)",
            "Ảnh hưởng đến du lịch (tỉ lệ chuyến đi)",
            "GDP và tỉ lệ thất nghiệp tương quan với số ca nhiễm",
            "Ảnh hưởng tích cực đến môi trường",
        ],
    )

    if sub_option == "Ảnh hưởng kinh tế với các quốc gia lớn (Q2 2020)":
        st.subheader("Ảnh hưởng kinh tế với các quốc gia lớn")
        # Đọc dữ liệu GDP
        data_economic = gdp1[["Entity", "GDP growth from previous year, 2020 Q2"]]
        data_economic = data_economic.sort_values(
            by="GDP growth from previous year, 2020 Q2", ascending=True
        )
        fig = px.bar(
            data_economic,
            x="GDP growth from previous year, 2020 Q2",
            y="Entity",
            orientation="h",
            text="GDP growth from previous year, 2020 Q2",
            labels={
                "Entity": "Khu vực",
                "GDP growth from previous year, 2020 Q2": "Tăng trưởng GDP (%)",
            },
            color="GDP growth from previous year, 2020 Q2",
            color_continuous_scale=px.colors.sequential.Plasma,
            title="Kinh tế các nước bị ảnh hưởng do dịch Covid-19",
        )
        # Tùy chỉnh giao diện biểu đồ
        fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside",
            marker_line_width=1.5,
            marker_line_color="black",
        )
        fig.update_layout(
            height=700,  # Chiều cao biểu đồ
            width=1200,  # Chiều rộng biểu đồ
            xaxis_title="Tăng trưởng GDP (%)",
            yaxis_title="Khu vực",
            title_font=dict(size=18, color="#FF0033"),
            xaxis=dict(showgrid=True, tickfont=dict(size=12, color="#FFD700")),
            yaxis=dict(tickfont=dict(size=12, color="#00EE00")),
            coloraxis_colorbar=dict(title="GDP (%)"),
        )
        st.plotly_chart(fig, use_container_width=True)
    elif sub_option == "Ảnh hưởng đến du lịch (tỉ lệ chuyến đi)":
        st.subheader("Ảnh hưởng đến du lịch")
        # Lọc dữ liệu từ 2018 đến 2022
        df_filtered = travel[(travel["Year"] >= 2018) & (travel["Year"] <= 2022)]
        fig = px.line(
            df_filtered,
            x="Year",
            y="inbound_tourism_by_region",
            color="Entity",
            markers=True,
            labels={
                "Year": "Năm",
                "inbound_tourism_by_region": "Lượng khách du lịch (Inbound)",
                "Entity": "Khu vực",
            },
            title="Lượng Khách Du Lịch Inbound Theo Năm (2018-2022)",
        )
        # Tuỳ chỉnh giao diện
        fig.update_traces(line=dict(width=3), marker=dict(size=10))
        fig.update_layout(
            title_font=dict(size=18, color="white"),
            xaxis_title_font=dict(size=14, color="black"),
            yaxis_title_font=dict(size=14, color="black"),
            legend_title_font=dict(size=12),
            xaxis=dict(tickmode="linear", tick0=2018, dtick=1),
            yaxis=dict(showgrid=True),
            legend=dict(
                title="Khu vực",
                orientation="h",
                yanchor="bottom",
                y=-0.4,  # Đẩy chú thích xuống dưới
                xanchor="center",
                x=0.5,
            ),
        )
        st.title("Ảnh Hưởng Đến Ngành Giao Thông, Du Lịch Dịch Vụ")
        st.plotly_chart(fig, use_container_width=True, height=800)  # Tăng chiều cao
    elif sub_option == "GDP và tỉ lệ thất nghiệp tương quan với số ca nhiễm":
        st.title("Ảnh hưởng COVID-19 đến GDP và Tỉ lệ Thất nghiệp (2021-2022)")

        mapping = {
            "AR": "ARG",
            "AU": "AUS",
            "BR": "BRA",
            "CA": "CAN",
            "CN": "CHN",
            "DE": "DEU",
            "EU": "EU",
            "FR": "FRA",
            "GB": "GBR",
            "ID": "IDN",
            "IN": "IND",
            "IT": "ITA",
            "JP": "JPN",
            "KR": "KOR",
            "MX": "MEX",
            "RU": "RUS",
            "SA": "SAU",
            "TR": "TUR",
            "US": "USA",
            "VN": "VNM",
            "ZA": "ZAF",
        }
        gdp_data["iso_code"] = gdp_data["Country"].map(mapping)

        # Chuẩn bị dữ liệu
        def prepare_data(year):
            gdp_unemployment = gdp_data[gdp_data["Year"] == year][
                ["iso_code", "GDP Growth (%)", "Unemployment Rate (%)"]
            ]
            covid_year = covid_data[covid_data["date"] <= f"{year}-12-31"]
            total_cases = (
                covid_year.groupby("iso_code")["total_cases"]
                .max()
                .reset_index()
                .rename(columns={"total_cases": "Total Cases"})
            )
            merged_data = total_cases.merge(
                gdp_unemployment, on="iso_code", how="inner"
            ).sort_values(by="Total Cases", ascending=False)
            return merged_data

        merged_data_2021 = prepare_data(2021)
        merged_data_2022 = prepare_data(2022)

        def plot_data(data, title):
            fig = go.Figure()

            # Cột tổng số ca nhiễm
            fig.add_trace(
                go.Bar(
                    x=data["iso_code"],
                    y=data["Total Cases"],
                    name="Tổng số ca nhiễm",
                    marker_color="skyblue",
                    yaxis="y1",
                )
            )

            # Đường tăng trưởng GDP
            fig.add_trace(
                go.Scatter(
                    x=data["iso_code"],
                    y=data["GDP Growth (%)"],
                    mode="lines+markers",
                    name="Tăng trưởng GDP (%)",
                    marker=dict(color="red"),
                    yaxis="y2",
                )
            )

            # Đường tỷ lệ thất nghiệp
            fig.add_trace(
                go.Scatter(
                    x=data["iso_code"],
                    y=data["Unemployment Rate (%)"],
                    mode="lines+markers",
                    name="Tỷ lệ thất nghiệp (%)",
                    marker=dict(color="green"),
                    yaxis="y3",
                )
            )

            # Cấu hình giao diện
            fig.update_layout(
                title=title,
                xaxis=dict(title="Quốc gia", tickangle=-45),
                yaxis=dict(
                    title="Tổng số ca nhiễm",
                    titlefont=dict(color="skyblue"),
                    showgrid=False,
                ),
                yaxis2=dict(
                    title="GDP (%)",
                    overlaying="y",
                    side="right",
                    titlefont=dict(color="red"),
                ),
                yaxis3=dict(
                    title="Tỷ lệ thất nghiệp (%)",
                    overlaying="y",
                    side="right",
                    anchor="free",
                    position=0.95,
                    titlefont=dict(color="green"),
                ),
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5
                ),
                template="plotly_white",
            )
            return fig

        st.plotly_chart(
            plot_data(merged_data_2021, "COVID-19 và GDP, Tỉ lệ Thất nghiệp năm 2021"),
            use_container_width=True,
        )
        st.plotly_chart(
            plot_data(merged_data_2022, "COVID-19 và GDP, Tỉ lệ Thất nghiệp năm 2022"),
            use_container_width=True,
        )

    elif sub_option == "Ảnh hưởng tích cực đến môi trường":
        st.header("Ảnh hưởng tích cực đến môi trường")
        # Loại bỏ các entity "International Aviation"
        # Lọc dữ liệu từ năm 2018 đến 2022
        data_co2 = data_co2[~data_co2["Entity"].isin(["International aviation"])]
        data_f = data_co2[(data_co2["Year"] >= 2018) & (data_co2["Year"] <= 2022)]
        regions = data_f["Entity"].unique()
        lst = []
        # Tính toán sự thay đổi phần trăm
        for region in regions:
            region_data = data_f[data_f["Entity"] == region]
            region_data = region_data.sort_values(by="Year")
            region_data["% Change"] = (
                region_data["Annual CO₂ emissions"].pct_change() * 100
            )
            lst.append(region_data)
        # Kết hợp tất cả dữ liệu sau tính toán
        lst = pd.concat(lst)
        # Tạo các năm chuyển đổi
        lst["Change"] = (lst["Year"]).astype(str)
        # Lọc dữ liệu chỉ tính toán năm tiếp theo
        lst = lst.dropna(subset=["% Change"])
        st.title("Biến động phần trăm lượng khí thải CO2 trong giai đoạn 2018-2022")
        # Vẽ biểu đồ
        fig = px.line(
            lst,
            x="Change",
            y="% Change",
            color="Entity",
            markers=True,
            labels={
                "Change": "Năm",
                "% Change": "Thay đổi (%)",
                "Entity": "Khu vực/Quốc gia",
            },
            title="Biến động phần trăm lượng khí thải CO2 trong giai đoạn 2018-2022",
        )

        # Cập nhật giao diện cho biểu đồ
        fig.update_layout(
            height=700,
            width=1200,
            xaxis=dict(showgrid=True, tickangle=45),
            yaxis=dict(showgrid=True),
            title_font=dict(size=16, color="#FF0033"),
            legend_title=dict(font=dict(size=12)),
            hovermode="x unified",
        )

        # Hiển thị biểu đồ trong Streamlit
        st.plotly_chart(fig, use_container_width=True)
with tab6:

    st.title("Kết Luận về Đại Dịch COVID-19")

    text = """
1. Tác động toàn cầu của đại dịch: Dữ liệu cho thấy COVID-19 đã ảnh hưởng nghiêm trọng đến toàn cầu, với số ca nhiễm và tử vong đáng kể trên khắp các châu lục. Một số quốc gia chịu ảnh hưởng nặng nề hơn những nước khác, cho thấy sự khác biệt về khả năng ứng phó với đại dịch.

2. Hiệu quả của tiêm chủng: Phân tích cho thấy tiêm chủng có tác động tích cực trong việc giảm số ca nhiễm và tử vong do COVID-19. Mặc dù không loại bỏ hoàn toàn nguy cơ nhiễm bệnh, vaccine đã giúp làm giảm mức độ nghiêm trọng của bệnh và giảm tải cho hệ thống y tế.

3. Sự khác biệt giữa các biến thể: So sánh giữa các biến thể Delta và Omicron cho thấy sự khác biệt về khả năng lây lan và độc lực. Omicron lây lan nhanh hơn Delta nhưng có vẻ gây bệnh ít nghiêm trọng hơn. Điều này nhấn mạnh tầm quan trọng của việc theo dõi và nghiên cứu các biến thể mới để có biện pháp ứng phó phù hợp.

4. Tác động kinh tế - xã hội: COVID-19 đã gây ra những tác động tiêu cực đến nền kinh tế toàn cầu, ảnh hưởng đến tăng trưởng GDP, tỷ lệ thất nghiệp và ngành du lịch. Các biện pháp giãn cách xã hội, mặc dù cần thiết để kiểm soát dịch bệnh, cũng đã góp phần vào sự suy thoái kinh tế. Mặt khác, đại dịch lại có tác động tích cực đến môi trường, được thể hiện qua việc giảm phát thải CO2 trong giai đoạn giãn cách xã hội.

5. Hạn chế của nghiên cứu: Dự án này có một số hạn chế nhất định, bao gồm:

	Dữ liệu không đầy đủ: Dữ liệu về COVID-19 có thể không chính xác hoặc không đầy đủ ở một số khu vực, đặc biệt là ở các nước đang phát triển.

	Mô hình dự đoán: Mô hình dự đoán số ca nhiễm và tử vong chỉ mang tính chất tham khảo và có thể không chính xác trong thực tế do sự biến đổi phức tạp của đại dịch.

	Các yếu tố khác: Dự án chưa xem xét đầy đủ các yếu tố khác có thể ảnh hưởng đến diễn biến của đại dịch, chẳng hạn như các biện pháp y tế công cộng, điều kiện khí hậu, và các yếu tố xã hội khác.

6. Đề xuất:

	Tăng cường tiêm chủng: Tiếp tục đẩy mạnh chiến dịch tiêm chủng và nghiên cứu phát triển các loại vaccine hiệu quả hơn để đối phó với các biến thể mới.

	Nâng cao năng lực y tế: Đầu tư vào hệ thống y tế, bao gồm cả việc đào tạo nhân viên y tế và cung cấp trang thiết bị y tế cần thiết.

	Giám sát và nghiên cứu: Tiếp tục giám sát chặt chẽ diễn biến của đại dịch và nghiên cứu các biến thể mới để có biện pháp ứng phó kịp thời và hiệu quả.

	Hợp tác quốc tế: Tăng cường hợp tác quốc tế trong việc chia sẻ dữ liệu, nghiên cứu, và phát triển vaccine.

Tổng kết: Dự án này cung cấp một cái nhìn tổng quan về tác động của đại dịch COVID-19 trên nhiều khía cạnh. Các kết quả phân tích có thể hỗ trợ cho việc hoạch định chính sách và đưa ra các biện pháp can thiệp phù hợp để kiểm soát dịch bệnh và giảm thiểu tác động của nó. Tuy nhiên, cần lưu ý những hạn chế của nghiên cứu và tiếp tục cập nhật dữ liệu và phân tích để có cái nhìn toàn diện hơn về tình hình đại dịch.
"""  # Paste your full text here

    st.text(text)
