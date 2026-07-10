import streamlit as st
import joblib
import xgboost as xgb
import plotly.express as px
import numpy as np
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
# 2. Feature names
# The order must remain consistent with the training dataset.
# ============================================================
feature_names = [
    "Grade Level",
    "Family Cohesion",
    "Homesickness-Related Distress",
    "Satisfaction with Academic Performance",
    "Richness of Extracurricular Life",
    "Class Atmosphere",
    "Satisfaction with Major",
    "Satisfaction with Awards and Honors",
    "Regular Diet",
    "Insomnia Severity"
]


# ============================================================
# 3. Response options and numerical coding
# The numerical values must remain consistent with model training.
# ============================================================
options_dict = {
    "Grade Level": [
        ("Freshman", 1),
        ("Sophomore", 2),
        ("Junior", 3),
        ("Senior", 4),
        ("Fifth-Year Student", 5)
    ],

    "Family Cohesion": [
        ("Very Close", 1),
        ("Relatively Close", 2),
        ("Neutral", 3),
        ("Relatively Distant", 4),
        ("Very Distant", 5)
    ],

    "Homesickness-Related Distress": [
        ("None", 1),
        ("Mild", 2),
        ("Moderate", 3),
        ("Severe", 4)
    ],

    "Satisfaction with Academic Performance": [
        ("Very Satisfied", 1),
        ("Satisfied", 2),
        ("Neutral", 3),
        ("Dissatisfied", 4),
        ("Very Dissatisfied", 5)
    ],

    "Richness of Extracurricular Life": [
        ("Very Rich", 1),
        ("Relatively Rich", 2),
        ("Average", 3),
        ("Not Rich", 4)
    ],

    "Class Atmosphere": [
        ("Positive and Motivating", 1),
        ("Harmonious and Supportive", 2),
        ("Average", 3),
        ("Inactive and Unmotivated", 4),
        ("Competitive and Conflict-Ridden", 5)
    ],

    "Satisfaction with Major": [
        ("Like It Very Much", 1),
        ("Like It", 2),
        ("Neutral", 3),
        ("Dislike It", 4),
        ("Dislike It Very Much", 5)
    ],

    "Satisfaction with Awards and Honors": [
        ("Very Satisfied", 1),
        ("Satisfied", 2),
        ("Neutral", 3),
        ("Dissatisfied", 4),
        ("Very Dissatisfied", 5)
    ],

    "Regular Diet": [
        ("No", 0),
        ("Yes", 1)
    ],

    "Insomnia Severity": [
        ("None", 1),
        ("Moderate", 2),
        ("Mild", 3),
        ("Severe", 4)
    ]
}


# ============================================================
# 4. Page title and introduction
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
# 5. Load the trained model
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
# 6. Assessment form
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
        inputs["Grade Level"] = st.selectbox(
            "1. What is your current grade level?",
            [option[0] for option in options_dict["Grade Level"]],
            help="Select your current year of university study."
        )

        inputs["Family Cohesion"] = st.selectbox(
            "2. How would you describe the level of cohesion within your family?",
            [option[0] for option in options_dict["Family Cohesion"]],
            help=(
                "Consider the emotional closeness, communication, "
                "and support among your family members."
            )
        )

        inputs["Homesickness-Related Distress"] = st.selectbox(
            "3. How distressed are you because you have been away "
            "from your family for an extended period?",
            [
                option[0]
                for option in options_dict["Homesickness-Related Distress"]
            ],
            help=(
                "Evaluate the emotional distress associated with "
                "being separated from your family."
            )
        )

        inputs["Satisfaction with Academic Performance"] = st.selectbox(
            "4. How satisfied are you with your current academic performance?",
            [
                option[0]
                for option in options_dict[
                    "Satisfaction with Academic Performance"
                ]
            ],
            help=(
                "Evaluate your satisfaction with your current "
                "academic performance."
            )
        )

        inputs["Richness of Extracurricular Life"] = st.selectbox(
            "5. How rich and varied is your extracurricular life?",
            [
                option[0]
                for option in options_dict[
                    "Richness of Extracurricular Life"
                ]
            ],
            help=(
                "Consider your participation in recreational, social, "
                "athletic, cultural, or student activities outside class."
            )
        )

    with col2:
        inputs["Class Atmosphere"] = st.selectbox(
            "6. How would you describe the overall atmosphere in your class?",
            [option[0] for option in options_dict["Class Atmosphere"]],
            help=(
                "Evaluate the interpersonal, motivational, and learning "
                "environment in your class."
            )
        )

        inputs["Satisfaction with Major"] = st.selectbox(
            "7. How much do you like your current academic major?",
            [
                option[0]
                for option in options_dict["Satisfaction with Major"]
            ],
            help=(
                "Evaluate your interest in and satisfaction with "
                "your academic major."
            )
        )

        inputs["Satisfaction with Awards and Honors"] = st.selectbox(
            "8. How satisfied are you with the evaluation and allocation "
            "of awards and honors?",
            [
                option[0]
                for option in options_dict[
                    "Satisfaction with Awards and Honors"
                ]
            ],
            help=(
                "Evaluate your satisfaction with the policies and procedures "
                "used to determine awards and honors."
            )
        )

        inputs["Regular Diet"] = st.selectbox(
            "9. Do you maintain a regular eating schedule?",
            [option[0] for option in options_dict["Regular Diet"]],
            help=(
                "Select Yes if you generally eat meals at regular times "
                "and maintain consistent eating habits."
            )
        )

        inputs["Insomnia Severity"] = st.selectbox(
            "10. How severe is your current insomnia?",
            [
                option[0]
                for option in options_dict["Insomnia Severity"]
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
# 7. Prediction and results
# ============================================================
if submitted and model is not None:

    try:
        # Convert response labels into numerical values
        input_values = [
            dict(options_dict[feature])[inputs[feature]]
            for feature in feature_names
        ]

        # Use a DataFrame to preserve the feature order
        input_data = pd.DataFrame(
            [input_values],
            columns=feature_names
        )

        # Generate predicted probability
        probability = model.predict_proba(input_data)

        prob_depressive_symptoms = float(probability[0][1])

        # Risk stratification
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
        # 8. Assessment result box
        # ====================================================
        st.subheader("Assessment Results")

        # The HTML is written as a continuous string to prevent
        # Streamlit Markdown from displaying the tags as plain text.
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
            f'<p style="font-size:16px;'
            f'margin:0;">'
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
        # 9. Risk interpretation and recommendations
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
        # 10. Global feature importance
        # ====================================================
        if hasattr(model, "feature_importances_"):

            feature_importances = model.feature_importances_

            if len(feature_importances) == len(feature_names):

                importance_df = pd.DataFrame(
                    {
                        "Predictive Factor": feature_names,
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
        # 11. Risk probability position chart
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
        # 12. Summary of user responses
        # ====================================================
        st.subheader("Summary of Your Responses")

        user_input_df = pd.DataFrame(
            {
                "Assessment Dimension": feature_names,
                "Selected Response": [
                    inputs[feature]
                    for feature in feature_names
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
        # 13. Disclaimer
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
# 14. Sidebar information
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
