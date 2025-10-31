import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np

# 设置页面配置
st.set_page_config(
    page_title="学生成绩分析与预测系统",
    page_icon="📊",
    layout="wide"
)

# 读取并预处理数据
try:
    df = pd.read_excel("D:\\streamlit_env\\student_data_adjusted_rounded1.xlsx")
except FileNotFoundError:
    st.error("数据文件未找到，请检查路径")
    st.stop()

# 数据预处理
numeric_cols = ["每周学习时长（小时）", "上课出勤率", "期中考试分数", "作业完成率", "期末考试分数"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 侧边栏导航
with st.sidebar:
    st.title("导航菜单")
    page = st.radio(
        "选择页面",
        ["项目分析", "专业数据介绍", "成绩预测"]
    )

# 项目分析页面
if page == "项目分析":
    st.title("📊 学生成绩分析与预测系统")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 项目概述")
        st.text("本项目是一个基于Streamlit的学生成绩分析平台，通过数据可视化和机器学习技术，")
        st.markdown("**主要特点**")
        st.markdown("- 📊 数据可视化：多维展示学生学业数据")
        st.markdown("- 🧠 智能分析：按专业分类的详细统计分析")
        st.markdown("- 🔮 成绩预测：基于机器学习模型的成绩预测")
        st.markdown("- 💡 学习建议：根据预测结果提供个性化反馈")
        
        st.header("🎯 项目目标")
        a1, a2, a3 = st.columns(3)
        with a1:
            st.subheader("目标一")
            st.markdown("**分析影响因素**")
            st.text("识别关键学习指标")
        with a2:
            st.subheader("目标二")
            st.markdown("**专业对比分析**")
            st.text("学业表现对比")
        with a3:
            st.subheader("目标三")
            st.markdown("**机器学习模型**")
            st.text("预测学生成绩")
    
    with col2:
        st.header("💻 技术架构")
        b1, b2, b3 = st.columns(3)
        with b1:
            st.markdown("**前端框架**")
            st.code("Streamlit", language="python")
        with b2:
            st.markdown("**数据处理**")
            st.code("Pandas\nNumPy", language="python")
        with b3:
            st.markdown("**可视化**")
            st.code("Plotly\nMatplotlib", language="python")
        st.image("images/学生数据分析示意图.png", caption="学生数据分析示意图", width=400)

# 专业数据介绍页面
elif page == "专业数据介绍":
    st.title("各专业数据分布")
    
    # 模块1：各专业男女性别比例（堆叠柱状图）
    with st.container():
        st.subheader("1. 各专业男女性别分布")
        col1, col2 = st.columns([0.7, 0.3])
        with col1:
            gender_profession = df.groupby(["专业", "性别"]).size().unstack(fill_value=0)
            fig_gender = px.bar(
                gender_profession,
                barmode="stack",
                color_discrete_map={"男": "#2B86AB", "女": "#A23B72"},
                labels={"专业": "专业名称", "value": "学生人数"}
            )
            fig_gender.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig_gender, use_container_width=True)
        with col2:
            st.write("性别分布数据")
            st.dataframe(gender_profession, use_container_width=True)
    
    # 模块2：各专业学习指标对比（分组柱状图）
    with st.container():
        st.subheader("2. 各专业核心学习指标对比")
        col3, col4 = st.columns([0.7, 0.3])
        with col3:
            prof_metrics = df.groupby("专业").agg({
                "每周学习时长（小时）": "mean",
                "上课出勤率": "mean",
                "期末考试分数": "mean"
            }).round(2)
            prof_metrics = prof_metrics.reset_index()
            prof_metrics_melt = pd.melt(
                prof_metrics,
                id_vars="专业",
                value_vars=["每周学习时长（小时）", "上课出勤率", "期末考试分数"],
                var_name="指标类型",
                value_name="指标均值"
            )
            fig_metrics = px.bar(
                prof_metrics_melt,
                x="专业",
                y="指标均值",
                color="指标类型",
                barmode="group",
                title="各专业学习指标均值对比（学习时长/出勤率/期末分数）",
                labels={"专业": "专业名称", "指标均值": "指标平均值"}
            )
            fig_metrics.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig_metrics, use_container_width=True)
        with col4:
            st.write("学习指标数据")
            st.dataframe(prof_metrics, use_container_width=True)
    
    # 模块3：各专业出勤率分布（彩色柱状图）
    with st.container():
        st.subheader("3. 各专业上课出勤率分布")
        col5, col6 = st.columns([0.7, 0.3])
        with col5:
            attendance_prof = df.groupby("专业")["上课出勤率"].mean().round(3).reset_index()
            attendance_prof["上课出勤率"] = attendance_prof["上课出勤率"] * 100  # 转为百分比
            fig_attendance = px.bar(
                attendance_prof,
                x="专业",
                y="上课出勤率",
                color="上课出勤率",
                color_continuous_scale=px.colors.sequential.Reds,
                title="各专业上课出勤率（%）",
                labels={"专业": "专业名称", "上课出勤率": "出勤率（%）"}
            )
            fig_attendance.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig_attendance, use_container_width=True)
        with col6:
            st.write("出勤率排名")
            st.dataframe(attendance_prof, use_container_width=True)
    
    # 模块4：大数据管理专业专项分析（横向柱状图）
    with st.container():
        st.subheader("4. 大数据管理专业专项分析")
        bigdata_df = df[df["专业"] == "大数据管理"]
        if not bigdata_df.empty:
            bigdata_metrics = bigdata_df[numeric_cols].mean().round(2).reset_index()
            bigdata_metrics.columns = ["考核指标", "指标均值"]
            fig_bigdata = px.bar(
                bigdata_metrics,
                y="考核指标",
                x="指标均值",
                orientation="h",
                color="考核指标",
                color_discrete_sequence=px.colors.qualitative.Set3,
                title="大数据管理专业各指标均值",
                labels={"指标均值": "指标平均值"}
            )
            st.plotly_chart(fig_bigdata, use_container_width=True)
        else:
            st.warning("暂无大数据管理专业数据")
    
    # 数据下载功能（侧边栏）
    with st.sidebar:
        st.subheader("↓ 数据下载")
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="下载原始数据CSV",
            data=csv,
            file_name="student_data_download.csv",
            mime="text/csv"
        )

# 成绩预测页面
elif page == "成绩预测":
    st.title("期末成绩预测")
    st.write("请输入学生学习信息，系统将预测期末成绩并给出学习建议")
    
    # 用户输入
    col1, col2 = st.columns(2)
    with col1:
        student_id = st.text_input("学号", "12345678")
        gender = st.selectbox("性别", ["男", "女"])
        major_options = df['专业'].unique() if not df.empty else ["无数据"]
        major = st.selectbox("专业", major_options)
    with col2:
        weekly_hours = st.number_input("每周学习时长", min_value=0, max_value=40, value=20)
        attendance = st.number_input("上课出勤率", min_value=0.0, max_value=1.0, value=0.9)
        midterm_score = st.number_input("期中考试分数", min_value=0, max_value=100, value=75)
        assignment_completion = st.number_input("作业完成率", min_value=0.0, max_value=1.0, value=0.8)
    
    # 预测按钮
    if st.button("预测期末成绩"):
        if not df.empty:
            # 简单线性回归模型
            X = df[['每周学习时长（小时）', '期中考试分数', '上课出勤率']]
            y = df['期末考试分数']
            
            model = LinearRegression()
            model.fit(X, y)
            
            # 准备用户输入数据
            user_data = np.array([[weekly_hours, midterm_score, attendance]])
            predicted_score = model.predict(user_data)[0]
            
            st.subheader("预测结果")
            st.write(f"预测期末成绩：{predicted_score:.2f} 分")
            
            if predicted_score >= 60:
                st.success("恭喜你，预测成绩及格！")
                st.image("https://images.unsplash.com/photo-1518676902727-4c7edcad4cfe?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80", 
                         caption="Congratulations!", use_column_width=True)
            else:
                st.warning("预测成绩不及格，请继续努力！")
        else:
            st.error("无数据可用于训练模型，请先导入有效数据！")
