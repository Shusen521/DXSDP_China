import streamlit as st
import joblib
import xgboost as xgb
import plotly.express as px
import pandas as pd


# ============================================================
# 1. Page configuration
# ============================================================
st.set_page_config(
    page_title="Depressive Symptoms Risk Prediction",
    page_icon="🧠",
    layout="wide"
)


# ============================================================
# 2. Model feature names
# These names must exactly match the training dataset.
# Do not translate or modify them.
# ============================================================
model_feature_names = [
    "X3_grade",
    "X14_family cohesion",
    "X16_anxiety about leaving home",
    "X18_academic satisfaction",
    "X20_after-school activity",
    "X23_class atmosphere",
    "X24_professional preference",
    "X25_award evaluation satisfaction",
    "X27_eat regularly",
    "X37_insomnia condition"
]


# ============================================================
# 3. English display names
# These names are only displayed on the webpage.
# ============================================================
display_names = {
    "X3_grade": "Grade Level",
    "X14_family cohesion": "Family Cohesion",
    "X16_anxiety about leaving home": "Homesickness-Related Distress",
    "X18_academic satisfaction": "Satisfaction with Academic Performance",
    "X20_after-school activity": "Richness of Extracurricular Life",
    "X23_class atmosphere": "Class Atmosphere",
    "X24_professional preference": "Satisfaction with Major",
    "X25_award evaluation satisfaction": "Satisfaction with Awards and Honors",
    "X27_eat regularly": "Regular Diet",
    "X37_insomnia condition": "Insomnia Severity"
}


# ============================================================
# 4. Response options and numerical coding
# The numerical values must match the coding used in training.
# ============================================================
options_dict = {
    "X3_grade": [
        ("Freshman", 1),
        ("Sophomore", 2),
        ("Junior", 3),
        ("Senior", 4),
        ("Fifth-Year Student", 5)
    ],

    "X14_family cohesion": [
        ("Very Close", 1),
        ("Relatively Close", 2),
        ("Neutral", 3),
        ("Relatively Distant", 4),
        ("Very Distant", 5)
    ],

    "X16_anxiety about leaving home": [
        ("None", 1),
        ("Mild", 2),
        ("Moderate", 3),
        ("Severe", 4)
    ],

    "X18_academic satisfaction": [
        ("Very Satisfied", 1),
        ("Satisfied", 2),
        ("Neutral", 3),
        ("Dissatisfied", 4),
        ("Very Dissatisfied", 5)
    ],

    "X20_after-school activity": [
        ("Very Rich", 1),
        ("Relatively Rich", 2),
        ("Average", 3),
        ("Not Rich", 4)
    ],

    "X23_class atmosphere": [
        ("Positive and Motivating", 1),
        ("Harmonious and Supportive", 2),
        ("Average", 3),
        ("Inactive and Unmotivated", 4),
        ("Competitive and Conflict-Ridden", 5)
    ],

    "X24_professional preference": [
        ("Like It Very Much", 1),
        ("Like It", 2),
        ("Neutral", 3),
        ("Dislike It", 4),
        ("Dislike It Very Much", 5)
    ],

    "X25_award evaluation satisfaction": [
        ("Very Satisfied", 1),
        ("Satisfied", 2),
        ("Neutral", 3),
        ("Dissatisfied", 4),
        ("Very Dissatisfied", 5)
    ],

    "X27_eat regularly": [
        ("No", 0),
        ("Yes", 1)
    ],

    "X37_insomnia condition": [
        ("None", 1),
        ("Moderate", 2),
        ("Mild", 3),
        ("Severe", 4)
    ]
}


# ============================================================
# 5. Page title and introduction
# ============================================================
st.title(
    "Early Risk Prediction Model for Depressive Symptoms "
    "Among College Students"
)

st.markdown(
    """
This tool uses a machine learning model developed within a social ecological
systems framework to estimate the risk of depressive symptoms among college
students.

The assessment considers factors related to academic life, family relationships,
campus environment, lifestyle behaviors, and psychological well-being.

Please select the responses that best reflect your current situation.
"""
)


# ============================================================
# 6. Load the trained model
# ============================================================
@st.cache_resource
def load_model():
    try:
        loaded_model = joblib.load("DXSDP.pkl")
        return loaded_model

    except FileNotFoundError:
        st.error(
            "The model file was not found. Please ensure that "
            "'DXSDP.pkl' is located in the same directory as this application."
        )
        return None

    except ModuleNotFoundError as error:
        st.error(
            f"A required Python package is missing: {str(error)}. "
            "Please add the missing package to requirements.txt "
            "and redeploy the application."
        )
        return None

    except Exception as error:
        st.error(
            f"An error occurred while loading the model: {str(error)}"
        )
        return None


model = load_model()


# ============================================================
# 7. Assessment form
# ============================================================
with st.form("student_depressive_symptoms_form"):

    st.subheader("Assessment Information")

    st.markdown(
        "Please select the option that most accurately describes "
        "your current situation."
    )

    col1, col2 = st.columns(2)

    inputs = {}

    with col1:
        inputs["X3_grade"] = st.selectbox(
            "1. What is your current grade level?",
            [item[0] for item in options_dict["X3_grade"]],
            help="Select your current year of university study."
        )

        inputs["X14_family cohesion"] = st.selectbox(
            "2. How would you describe the level of cohesion within your family?",
            [item[0] for item in options_dict["X14_family cohesion"]],
            help=(
                "Consider emotional closeness, communication, and support "
                "among your family members."
            )
        )

        inputs["X16_anxiety about leaving home"] = st.selectbox(
            "3. How distressed are you because you have been away "
            "from your family for an extended period?",
            [
                item[0]
                for item in options_dict["X16_anxiety about leaving home"]
            ],
            help=(
                "Evaluate the emotional distress associated with being "
                "separated from your family."
            )
        )

        inputs["X18_academic satisfaction"] = st.selectbox(
            "4. How satisfied are you with your current academic performance?",
            [
                item[0]
                for item in options_dict["X18_academic satisfaction"]
            ],
            help=(
                "Evaluate your satisfaction with your current "
                "academic performance."
            )
        )

        inputs["X20_after-school activity"] = st.selectbox(
            "5. How rich and varied is your extracurricular life?",
            [
                item[0]
                for item in options_dict["X20_after-school activity"]
            ],
            help=(
                "Consider your participation in recreational, social, "
                "athletic, cultural, or student activities outside class."
            )
        )

    with col2:
        inputs["X23_class atmosphere"] = st.selectbox(
            "6. How would you describe the overall atmosphere in your class?",
            [
                item[0]
                for item in options_dict["X23_class atmosphere"]
            ],
            help=(
                "Evaluate the interpersonal, motivational, and learning "
                "environment in your class."
            )
        )

        inputs["X24_professional preference"] = st.selectbox(
            "7. How much do you like your current academic major?",
            [
                item[0]
                for item in options_dict["X24_professional preference"]
            ],
            help=(
                "Evaluate your interest in and satisfaction with "
                "your academic major."
            )
        )

        inputs["X25_award evaluation satisfaction"] = st.selectbox(
            "8. How satisfied are you with the evaluation and allocation "
            "of awards and honors?",
            [
                item[0]
                for item in options_dict[
                    "X25_award evaluation satisfaction"
                ]
            ],
            help=(
                "Evaluate your satisfaction with the policies and procedures "
                "used to determine awards and honors."
            )
        )

        inputs["X27_eat regularly"] = st.selectbox(
            "9. Do you maintain a regular eating schedule?",
            [
                item[0]
                for item in options_dict["X27_eat regularly"]
            ],
            help=(
                "Select Yes if you generally eat meals at regular times "
                "and maintain consistent eating habits."
            )
        )

        inputs["X37_insomnia condition"] = st.selectbox(
            "10. How severe is your current insomnia?",
            [
                item[0]
                for item in options_dict["X37_insomnia condition"]
            ],
            help=(
                "Evaluate recent difficulties falling asleep, staying asleep, "
                "or obtaining restorative sleep."
            )
        )

    submitted = st.form_submit_button(
        "Assess Depressive Symptoms Risk",
        use_container_width=True
    )


# ============================================================
# 8. Prediction and results
# ============================================================
if submitted and model is not None:

    try:
        # Convert selected English labels to numerical values
        input_values = [
            dict(options_dict[feature])[inputs[feature]]
            for feature in model_feature_names
        ]

        # IMPORTANT:
        # The columns use the exact names and order from the training data.
        input_data = pd.DataFrame(
            [input_values],
            columns=model_feature_names
        )

        # Predict probability
        probability = model.predict_proba(input_data)

        prob_depressive_symptoms = float(probability[0][1])


        # ====================================================
        # 9. Risk stratification
        # ====================================================
        if prob_depressive_symptoms < 0.42:
            risk_level = "No Risk"
            risk_color = "#2E7D32"

        elif prob_depressive_symptoms < 0.60:
            risk_level = "Low Risk"
            risk_color = "#1565C0"

        elif prob_depressive_symptoms < 0.76:
            risk_level = "Moderate Risk"
            risk_color = "#EF6C00"

        else:
            risk_level = "High Risk"
            risk_color = "#C62828"


        # ====================================================
        # 10. Assessment result
        # ====================================================
        st.subheader("Assessment Results")

        result_html = (
            f'<div style="background-color:#f0f2f6;'
            f'padding:20px;'
            f'border-radius:10px;'
            f'border-left:6px solid {risk_color};'
            f'margin-bottom:15px;">'
            f'<h3 style="color:{risk_color};'
            f'margin:0 0 10px 0;">'
            f'Depressive Symptoms Risk Level: {risk_level}'
            f'</h3>'
            f'<p style="font-size:16px;margin:0;">'
            f'<strong>Predicted Risk Probability:</strong> '
            f'{prob_depressive_symptoms:.2%}'
            f'</p>'
            f'</div>'
        )

        st.markdown(
            result_html,
            unsafe_allow_html=True
        )


        # ====================================================
        # 11. Risk interpretation
        # ====================================================
        st.subheader("Risk Interpretation and Recommendations")

        if risk_level == "No Risk":
            st.success(
                """
✅ **Current status: No apparent risk**

The model indicates that your current estimated risk of depressive
symptoms is relatively low.

**Recommendations:** Continue maintaining a regular daily routine,
healthy eating and sleeping habits, active participation in campus
activities, and supportive social relationships.
"""
            )

        elif risk_level == "Low Risk":
            st.info(
                """
ℹ️ **Current status: Some attention may be beneficial**

The model indicates that you may have some factors associated with
depressive symptoms.

**Recommendations:** Pay attention to changes in your mood and daily
functioning, maintain regular sleep and eating habits, and use appropriate
stress-management strategies. Consider contacting your university
counseling center if emotional distress persists.
"""
            )

        elif risk_level == "Moderate Risk":
            st.warning(
                """
⚠️ **Current status: Further assessment is recommended**

The model indicates a moderately elevated estimated risk of depressive
symptoms.

**Recommendations:** Consider contacting your university counseling
center or a qualified mental health professional for a comprehensive
assessment. You may also discuss your situation with a trusted family
member, friend, teacher, or counselor.
"""
            )

        else:
            st.error(
                """
🚨 **Current status: Prompt professional support is recommended**

The model indicates a relatively high estimated risk of depressive
symptoms.

**Recommendations:** Please contact your university counseling center
or a qualified mental health professional as soon as possible for a
comprehensive assessment and appropriate support. You are also encouraged
to inform a trusted family member, friend, teacher, or counselor.
"""
            )


        # ====================================================
        # 12. Global feature importance
        # ====================================================
        if hasattr(model, "feature_importances_"):

            feature_importances = model.feature_importances_

            if len(feature_importances) == len(model_feature_names):

                importance_df = pd.DataFrame(
                    {
                        "Predictive Factor": [
                            display_names[feature]
                            for feature in model_feature_names
                        ],
                        "Importance": feature_importances
                    }
                ).sort_values(
                    by="Importance",
                    ascending=True
                )

                st.subheader("Global Predictor Importance")

                st.caption(
                    "This chart shows the overall importance of each predictor "
                    "in the trained model. It does not represent the contribution "
                    "of each factor to this individual prediction."
                )

                fig_importance = px.bar(
                    importance_df,
                    x="Importance",
                    y="Predictive Factor",
                    orientation="h",
                    title="Overall Importance of Predictors in the Model",
                    labels={
                        "Importance": "Feature Importance",
                        "Predictive Factor": "Predictive Factor"
                    },
                    color="Importance",
                    color_continuous_scale="Blues"
                )

                fig_importance.update_layout(
                    showlegend=False,
                    coloraxis_showscale=False,
                    title_x=0.5,
                    yaxis_title=None,
                    margin=dict(l=20, r=20, t=60, b=20)
                )

                st.plotly_chart(
                    fig_importance,
                    use_container_width=True
                )


        # ====================================================
        # 13. Risk probability position
        # ====================================================
        st.subheader("Risk Probability Position")

        risk_ranges = [
            "No Risk (0.00–0.42)",
            "Low Risk (0.42–0.60)",
            "Moderate Risk (0.60–0.76)",
            "High Risk (0.76–1.00)"
        ]

        interval_values = [
            min(prob_depressive_symptoms, 0.42),

            max(
                0,
                min(prob_depressive_symptoms, 0.60) - 0.42
            ),

            max(
                0,
                min(prob_depressive_symptoms, 0.76) - 0.60
            ),

            max(
                0,
                prob_depressive_symptoms - 0.76
            )
        ]

        probability_df = pd.DataFrame(
            {
                "Risk Category": risk_ranges,
                "Probability Portion": interval_values
            }
        )

        fig_probability = px.bar(
            probability_df,
            x="Probability Portion",
            y="Risk Category",
            orientation="h",
            title="Position of the Predicted Probability Across Risk Intervals",
            labels={
                "Probability Portion": "Probability Portion",
                "Risk Category": "Risk Category"
            },
            color="Risk Category",
            color_discrete_map={
                "No Risk (0.00–0.42)": "green",
                "Low Risk (0.42–0.60)": "blue",
                "Moderate Risk (0.60–0.76)": "orange",
                "High Risk (0.76–1.00)": "red"
            }
        )

        fig_probability.update_layout(
            title_x=0.5,
            showlegend=False,
            yaxis_title=None,
            margin=dict(l=20, r=20, t=60, b=20)
        )

        st.plotly_chart(
            fig_probability,
            use_container_width=True
        )


        # ====================================================
        # 14. Summary of user responses
        # ====================================================
        st.subheader("Summary of Your Responses")

        user_input_df = pd.DataFrame(
            {
                "Assessment Dimension": [
                    display_names[feature]
                    for feature in model_feature_names
                ],
                "Selected Response": [
                    inputs[feature]
                    for feature in model_feature_names
                ],
                "Encoded Value": input_values
            }
        )

        st.dataframe(
            user_input_df,
            use_container_width=True,
            hide_index=True
        )


        # ====================================================
        # 15. Disclaimer
        # ====================================================
        st.markdown("---")

        st.caption(
            """
**Disclaimer:** This assessment tool uses a machine learning model to
estimate the risk of depressive symptoms. The result is intended for
screening and research purposes only. It does not constitute a clinical
diagnosis and cannot replace an assessment by a qualified mental health
professional.

If you are experiencing persistent emotional distress, impaired daily
functioning, thoughts of self-harm, or an immediate mental health crisis,
please seek professional or emergency assistance promptly.
"""
        )


    except ValueError as error:
        st.error(
            "The input data are incorrectly formatted. "
            f"Error details: {str(error)}"
        )

    except Exception as error:
        st.error(
            f"An error occurred during the assessment: {str(error)}"
        )


elif submitted and model is None:
    st.error(
        "The model could not be loaded, and the assessment cannot be "
        "completed. Please verify that the model file and all required "
        "packages are available."
    )


# ============================================================
# 16. Sidebar
# ============================================================
with st.sidebar:

    st.header("About This Tool")

    st.markdown(
        """
This machine-learning-based prediction model was developed within a
social ecological systems framework to estimate the risk of depressive
symptoms among college students.

The tool is intended to support early risk identification and mental
health screening rather than clinical diagnosis.

**Assessment dimensions include:**

- Academic factors
- Family relationships and support
- Campus and class environment
- Lifestyle behaviors
- Sleep and psychological well-being
"""
    )

    st.markdown("---")

    st.subheader("Risk Classification")

    st.markdown(
        """
- **No Risk:** probability below 0.42
- **Low Risk:** probability from 0.42 to below 0.60
- **Moderate Risk:** probability from 0.60 to below 0.76
- **High Risk:** probability of 0.76 or higher
"""
    )

    st.markdown("---")

    st.caption(
        "This tool is intended for research and preliminary "
        "screening purposes only."
    )
