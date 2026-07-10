import streamlit as st
import matplotlib.pyplot as plt
import joblib
import plotly.express as px
import numpy as np
import pandas as pd

# 定义特征名称（中文标签）
feature_names_chinese = [
    "您现在所处的年级",
    "您的家庭亲密度情况如何", 
    "是否因长期远离家人不能团聚而苦恼",
    "您对学习成绩的满意度",
    "您的课余生活是否丰富",
    "您对大学班级的班级风气评价",
    "您对本专业的喜爱程度",
    "您对评奖评优的满意度",
    "您是否规律饮食",
    "您的失眠程度"
]

# 定义特征的选项（中文）
options_dict_chinese = {
    "您现在所处的年级": [("大一", 1), ("大二", 2), ("大三", 3), ("大四", 4), ("大五", 5)],
    "您的家庭亲密度情况如何": [("亲密", 1), ("较亲密", 2), ("一般", 3), ("较冷漠", 4), ("冷漠", 5)],
    "是否因长期远离家人不能团聚而苦恼": [("无", 1), ("较轻", 2), ("适中", 3), ("严重", 4)],
    "您对学习成绩的满意度": [("非常满意", 1), ("较满意", 2), ("适中", 3), ("较不满意", 4), ("非常不满意", 5)],
    "您的课余生活是否丰富": [("非常丰富", 1), ("较丰富", 2), ("一般", 3), ("无", 4)],
    "您对大学班级的班级风气评价": [("积极向上", 1), ("和睦融洽", 2), ("一般", 3), ("懒散懈怠", 4), ("勾心斗角", 5)],
    "您对本专业的喜爱程度": [("非常喜欢", 1), ("较喜欢", 2), ("一般", 3), ("较不喜欢", 4), ("非常不喜欢", 5)],
    "您对评奖评优的满意度": [("非常满意", 1), ("较满意", 2), ("一般", 3), ("较不满意", 4), ("非常不满意", 5)],
    "您是否规律饮食": [("否", 0), ("是", 1)],
    "您的失眠程度": [("无", 1), ("适中", 2), ("较轻", 3), ("严重", 4)]
}

# 设置页面标题（中文）
st.title("大学生抑郁早期风险预测模型")
st.markdown("""
本工具基于机器学习模型，通过社会生态系统理论分析大学生在校期间的导致抑郁的多维度抑郁因素，评估抑郁风险水平。
请根据您的实际情况填写以下信息。
""")

# 加载训练好的模型
@st.cache_resource
def load_model():
    # 请确保您的模型文件命名为"DXSDP.pkl"并放在同一目录下
    try:
        model = joblib.load("DXSDP.pkl")
        return model
    except FileNotFoundError:
        st.error("模型文件未找到，请确保'DXSDP.pkl'存在于当前目录")
        return None

model = load_model()

# 使用 Streamlit 表单，控制运行行为
with st.form("student_depression_form"):
    st.subheader("个人信息填写")
    st.markdown("请根据您的实际情况选择以下选项：")
    
    # 创建两列布局，使界面更紧凑
    col1, col2 = st.columns(2)
    
    inputs = {}
    with col1:
        inputs["您现在所处的年级"] = st.selectbox(
            "您现在所处的年级",
            [option[0] for option in options_dict_chinese["您现在所处的年级"]],
            help="选择您当前所在的年级"
        )
        
        inputs["您的家庭亲密度情况如何"] = st.selectbox(
            "您的家庭亲密度情况如何", 
            [option[0] for option in options_dict_chinese["您的家庭亲密度情况如何"]],
            help="评估您与家庭成员之间的亲密程度"
        )
        
        inputs["是否因长期远离家人不能团聚而苦恼"] = st.selectbox(
            "是否因长期远离家人不能团聚而苦恼",
            [option[0] for option in options_dict_chinese["是否因长期远离家人不能团聚而苦恼"]],
            help="评估因远离家人而产生的情绪困扰程度"
        )
        
        inputs["您对学习成绩的满意度"] = st.selectbox(
            "您对学习成绩的满意度",
            [option[0] for option in options_dict_chinese["您对学习成绩的满意度"]],
            help="对当前学习成绩的满意程度"
        )
        
        inputs["您的课余生活是否丰富"] = st.selectbox(
            "您的课余生活是否丰富",
            [option[0] for option in options_dict_chinese["您的课余生活是否丰富"]],
            help="评估课余生活的丰富程度"
        )
    
    with col2:
        inputs["您对大学班级的班级风气评价"] = st.selectbox(
            "您对大学班级的班级风气评价",
            [option[0] for option in options_dict_chinese["您对大学班级的班级风气评价"]],
            help="对班级整体氛围的评价"
        )
        
        inputs["您对本专业的喜爱程度"] = st.selectbox(
            "您对本专业的喜爱程度",
            [option[0] for option in options_dict_chinese["您对本专业的喜爱程度"]],
            help="对所学专业的兴趣和喜爱程度"
        )
        
        inputs["您对评奖评优的满意度"] = st.selectbox(
            "您对评奖评优的满意度",
            [option[0] for option in options_dict_chinese["您对评奖评优的满意度"]],
            help="对学校评奖评优制度的满意程度"
        )
        
        inputs["您是否规律饮食"] = st.selectbox(
            "您是否规律饮食",
            [option[0] for option in options_dict_chinese["您是否规律饮食"]],
            help="饮食是否规律"
        )
        
        inputs["您的失眠程度"] = st.selectbox(
            "您的失眠程度", 
            [option[0] for option in options_dict_chinese["您的失眠程度"]],
            help="近期失眠情况的严重程度"
        )
    
    # 添加提交按钮（中文）
    submitted = st.form_submit_button("进行抑郁风险评估")

# 只有在点击按钮后才运行以下代码
if submitted and model is not None:
    try:
        # 准备输入数据（将选择的中文标签转换为数字）
        input_data = np.array([
            dict(options_dict_chinese[feature])[inputs[feature]] 
            for feature in feature_names_chinese
        ]).reshape(1, -1)

        # 预测结果和概率
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)

        # 根据新的风险分层进行分类
        prob_depression = probability[0][1]  # 有抑郁风险的概率
        risk_level = ""
        risk_color = ""

        if prob_depression < 0.42:
            risk_level = "无风险"
            risk_color = "green"
        elif 0.42 <= prob_depression < 0.60:
            risk_level = "低风险"
            risk_color = "blue"
        elif 0.60 <= prob_depression < 0.76:
            risk_level = "中风险" 
            risk_color = "orange"
        else:
            risk_level = "高风险"
            risk_color = "red"

        # 显示结果（中文）
        st.subheader("评估结果")
        
        # 使用容器突出显示风险等级
        with st.container():
            st.markdown(f"""
            <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid {risk_color};">
                <h3 style="color: {risk_color}; margin-top: 0;">抑郁风险等级: {risk_level}</h3>
                <p style="font-size: 16px; margin-bottom: 0;"><strong>抑郁风险概率:</strong> {prob_depression:.2%}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # 风险解释和建议
        st.subheader("风险说明与建议")
        if risk_level == "无风险":
            st.success("""
            ✅ **当前状态良好**：您的心理健康状况良好，请继续保持健康的生活和学习习惯。
            **建议**：保持规律作息，积极参与校园活动，建立良好的社交支持系统。
            """)
        elif risk_level == "低风险":
            st.info("""
            ℹ️ **需要关注**：您存在一定的抑郁风险因素，建议适当关注心理健康。
            **建议**：加强自我调节，适当减压，可咨询学校心理咨询中心获取指导。
            """)
        elif risk_level == "中风险":
            st.warning("""
            ⚠️ **中度风险**：您的抑郁风险较高，需要重视心理健康状况。
            **建议**：建议尽快联系学校心理咨询中心进行专业评估，学习压力管理技巧。
            """)
        else:
            st.error("""
            🚨 **高风险**：您的抑郁风险较高，需要立即采取行动。
            **建议**：请立即联系学校心理咨询中心或专业心理医生，寻求专业帮助。同时可告知信任的亲友获取支持。
            """)
        
        # 获取特征重要性（如果模型有此属性）
        if hasattr(model, 'feature_importances_'):
            feature_importances = model.feature_importances_
            
            # 创建DataFrame，使用中文特征名
            importance_df = pd.DataFrame({
                '特征': feature_names_chinese,
                '重要性': feature_importances
            }).sort_values(by='重要性', ascending=True)  # 改为升序，使图表从上到下显示
            
            # 特征重要性图表
            st.subheader("影响因素重要性分析")
            fig_importance = px.bar(
                importance_df,
                x='重要性',
                y='特征',
                orientation='h',
                title='各因素对抑郁风险的影响程度',
                labels={'重要性': '影响重要性', '特征': '影响因素'},
                color='重要性',
                color_continuous_scale='Blues'
            )
            fig_importance.update_layout(showlegend=False)
            st.plotly_chart(fig_importance, use_container_width=True)

        # 风险概率分布图
        st.subheader("风险概率分布")
        risk_ranges = ['无风险(0-0.42)', '低风险(0.42-0.60)', '中风险(0.60-0.76)', '高风险(0.76-1.0)']
        risk_probs = [
            min(prob_depression, 0.42),
            max(0, min(prob_depression - 0.42, 0.18)) if prob_depression > 0.42 else 0,
            max(0, min(prob_depression - 0.60, 0.16)) if prob_depression > 0.60 else 0,
            max(0, prob_depression - 0.76) if prob_depression > 0.76 else 0
        ]
        
        # 确保概率分布正确显示当前风险区间
        current_risk_index = 0 if prob_depression < 0.42 else 1 if prob_depression < 0.60 else 2 if prob_depression < 0.76 else 3
        for i in range(len(risk_probs)):
            if i == current_risk_index:
                risk_probs[i] = prob_depression - (0.42 if i==1 else 0.60 if i==2 else 0.76 if i==3 else 0)
            elif i < current_risk_index:
                risk_probs[i] = 0.42 if i==0 else 0.18 if i==1 else 0.16
            else:
                risk_probs[i] = 0
        
        prob_df = pd.DataFrame({
            '风险等级': risk_ranges,
            '概率区间': risk_probs
        })
        
        fig_probability = px.bar(
            prob_df,
            x='概率区间',
            y='风险等级',
            orientation='h',
            title='风险概率分布',
            labels={'概率区间': '概率值', '风险等级': '风险等级'},
            color='风险等级',
            color_discrete_sequence=['green', 'blue', 'orange', 'red']
        )
        st.plotly_chart(fig_probability, use_container_width=True)
        
        # 显示用户输入数据汇总
        st.subheader("您的输入信息汇总")
        user_input_values = [
            dict(options_dict_chinese[feature])[inputs[feature]]
            for feature in feature_names_chinese
        ]
        user_input_df = pd.DataFrame({
            '评估维度': feature_names_chinese,
            '您的选择': [inputs[feature] for feature in feature_names_chinese],
            '对应数值': user_input_values
        })
        
        st.dataframe(user_input_df, use_container_width=True)
        
        # 免责声明
        st.markdown("---")
        st.caption("""
        **免责声明**：本评估工具基于机器学习模型预测，结果仅供参考，不能替代专业心理医生的诊断。 
        如果您感到情绪困扰，请及时联系学校心理咨询中心或专业医疗机构。
        """)

    except ValueError as e:
        st.error("输入数据格式错误，请检查所有选项是否已正确选择！")
    except Exception as e:
        st.error(f"评估过程中出现错误: {str(e)}")

elif submitted and model is None:
    st.error("模型加载失败，无法进行风险评估。请检查模型文件是否存在。")

# 侧边栏添加额外信息
with st.sidebar:
    st.header("关于本工具")
    st.markdown("""
    本大学生抑郁风险预测模型基于社会生态系统理论分析多维度大学生抑郁风险因素构建，已经过多轮验证，旨在为大学生提供早期心理健康风险评估。
    
    **评估维度包括**：
    - 学业相关因素
    - 家庭支持系统  
    - 校园生活环境
    - 个人生活习惯
    - 心理健康状况
    """)
    
