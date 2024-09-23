
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import xgboost as xgb

# Load the model
xgb_classifier = pickle.load(open('xgb_classifier.pkl', 'rb'))

# Min and Max values for normalization (replace with actual values)
min_max_values = {
    'Greed': (0, 100),  # Placeholder values, replace with real dataset min/max values
    'Respect': (0, 50),
    'Discontentment': (0, 40),
    'Loyalty_History': (0, 30),
    'Financial_Stress': (0, 200000),
    'Morale': (0, 100),
    'Age': (18, 50),  # Age range
    'Years_of_Service': (0, 30),  # Years of service range
    'Rank': (1, 10)  # Rank range
}

# Define normalization function
def normalize(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)

# Define functions to calculate combined scores
def calculate_greed(compensation_requests_count, luxury_spending_score):
    weight1, weight2 = 0.5, 0.5
    greed_index = compensation_requests_count * weight1 + luxury_spending_score * weight2
    return greed_index

def calculate_respect(peer_review, superior_review, warnings_count):
    weight1, weight2, weight3 = 0.4, 0.4, 0.2
    respect_score_raw = peer_review * weight1 + superior_review * weight2 - warnings_count * weight3
    return respect_score_raw

def calculate_discontentment(peer_conflicts, poor_performance, missed_promotions, complaints_filed):
    weight1, weight2, weight3, weight4 = 0.25, 0.25, 0.25, 0.25
    discontentment_score_raw = (peer_conflicts * weight1 +
                                poor_performance * weight2 +
                                missed_promotions * weight3 +
                                complaints_filed * weight4)
    return discontentment_score_raw

def calculate_loyalty(participation_count, family_history, betrayals_count, peer_respect_score):
    weight1, weight2, weight3, weight4 = 0.3, 0.3, 0.2, 0.2
    loyalty_score_raw = (participation_count * weight1 +
                         family_history * weight2 -
                         betrayals_count * weight3 +
                         peer_respect_score * weight4)
    return loyalty_score_raw

def calculate_financial_stress(debt_amount, income, financial_aid_requests_count, dependents_count):
    weight1, weight2, weight3, weight4 = 0.3, -0.3, 0.2, 0.2
    financial_stress_raw = (debt_amount * weight1 +
                            financial_aid_requests_count * weight2 +
                            dependents_count * weight3 -
                            income * weight4)
    return financial_stress_raw

def calculate_morale(job_satisfaction, health_status, peer_relationship, recognition_count):
    weight1, weight2, weight3, weight4 = 0.3, 0.3, 0.2, 0.2
    morale_score_raw = (job_satisfaction * weight1 +
                        health_status * weight2 +
                        peer_relationship * weight3 +
                        recognition_count * weight4)
    return morale_score_raw

# Streamlit UI
st.title("Troop Betrayal Prediction")

st.write("Provide the following inputs to predict troop betrayal:")

# Greed Section
st.header("Greed")
compensation_requests_count = st.number_input('Compensation Requests Count', min_value=0, max_value=10, value=5)
luxury_spending_score = st.number_input('Luxury Spending Score', min_value=0, max_value=100, value=50)

greed_score = calculate_greed(compensation_requests_count, luxury_spending_score)
greed_normalized = normalize(greed_score, *min_max_values['Greed'])
st.write(f"Greed Score: {greed_score:.2f}")
st.write(f"Normalized Greed Score: {greed_normalized:.2f}")
st.markdown("---")

# Respect Section
st.header("Respect")
peer_review = st.number_input('Peer Review', min_value=0, max_value=10, value=5)
superior_review = st.number_input('Superior Review (1-5)', min_value=1, max_value=5, value=3)
warnings_count = st.number_input('Warnings Count', min_value=0, max_value=10, value=2)

respect_score = calculate_respect(peer_review, superior_review, warnings_count)
respect_normalized = normalize(respect_score, *min_max_values['Respect'])
st.write(f"Respect Score: {respect_score:.2f}")
st.write(f"Normalized Respect Score: {respect_normalized:.2f}")
st.markdown("---")

# Discontentment Section
st.header("Discontentment")
peer_conflicts = st.number_input('Peer Conflicts', min_value=0, max_value=10, value=2)
poor_performance = st.number_input('Poor Performance Reviews', min_value=0, max_value=10, value=3)
missed_promotions = st.number_input('Missed Promotions', min_value=0, max_value=10, value=1)
complaints_filed = st.number_input('Complaints Filed', min_value=0, max_value=10, value=1)

discontentment_score = calculate_discontentment(peer_conflicts, poor_performance, missed_promotions, complaints_filed)
discontentment_normalized = normalize(discontentment_score, *min_max_values['Discontentment'])
st.write(f"Discontentment Score: {discontentment_score:.2f}")
st.write(f"Normalized Discontentment Score: {discontentment_normalized:.2f}")
st.markdown("---")

# Loyalty Section
st.header("Loyalty")
participation_count = st.number_input('Participation in Loyalty Programs', min_value=0, max_value=10, value=5)
family_history = st.number_input('Family History (Generations Served)', min_value=0, max_value=5, value=2)
betrayals_count = st.number_input('Betrayals Count', min_value=0, max_value=5, value=0)
peer_respect_score = st.number_input('Peer Respect Score', min_value=0, max_value=10, value=7)

loyalty_score = calculate_loyalty(participation_count, family_history, betrayals_count, peer_respect_score)
loyalty_normalized = normalize(loyalty_score, *min_max_values['Loyalty_History'])
st.write(f"Loyalty Score: {loyalty_score:.2f}")
st.write(f"Normalized Loyalty Score: {loyalty_normalized:.2f}")
st.markdown("---")

# Financial Stress Section
st.header("Financial Stress")
debt_amount = st.number_input('Debt Amount', min_value=0, max_value=200000, value=50000)
income = st.number_input('Income', min_value=0, max_value=200000, value=50000)
financial_aid_requests_count = st.number_input('Financial Aid Requests', min_value=0, max_value=10, value=1)
dependents_count = st.number_input('Dependents Count', min_value=0, max_value=10, value=2)

financial_stress_score = calculate_financial_stress(debt_amount, income, financial_aid_requests_count, dependents_count)
financial_stress_normalized = normalize(financial_stress_score, *min_max_values['Financial_Stress'])
st.write(f"Financial Stress Score: {financial_stress_score:.2f}")
st.write(f"Normalized Financial Stress Score: {financial_stress_normalized:.2f}")
st.markdown("---")

# Morale Section
st.header("Morale")
job_satisfaction = st.number_input('Job Satisfaction (1-5)', min_value=1, max_value=5, value=3)
health_status = st.number_input('Health Status (1-5)', min_value=1, max_value=5, value=4)
peer_relationship = st.number_input('Peer Relationship (1-10)', min_value=1, max_value=10, value=7)
recognition_count = st.number_input('Recognition Count', min_value=0, max_value=10, value=2)

morale_score = calculate_morale(job_satisfaction, health_status, peer_relationship, recognition_count)
morale_normalized = normalize(morale_score, *min_max_values['Morale'])
st.write(f"Morale Score: {morale_score:.2f}")
st.write(f"Normalized Morale Score: {morale_normalized:.2f}")
st.markdown("---")

# New Inputs for Age, Years of Service, and Rank
st.header("Other Factors")
age = st.number_input('Age', min_value=18, max_value=50, value=30)
years_of_service = st.number_input('Years of Service', min_value=0, max_value=30, value=10)
rank = st.number_input('Rank', min_value=1, max_value=10, value=5)

# Normalize inputs for Age, Years of Service, and Rank
age_normalized = normalize(age, *min_max_values['Age'])
years_of_service_normalized = normalize(years_of_service, *min_max_values['Years_of_Service'])
rank_normalized = normalize(rank, *min_max_values['Rank'])

st.write(f"Normalized Age: {age_normalized:.2f}")
st.write(f"Normalized Years of Service: {years_of_service_normalized:.2f}")
st.write(f"Normalized Rank: {rank_normalized:.2f}")

# Collect all features into a dataframe
features = pd.DataFrame({
    'Greed': [greed_normalized],
    'Respect': [respect_normalized],
    'Discontentment': [discontentment_normalized],
    'Loyalty_History': [loyalty_normalized],
    'Rank': [rank_normalized],
    'Age': [age_normalized],
    'Years_of_Service': [years_of_service_normalized],
    'Financial_Stress': [financial_stress_normalized],
    'Morale': [morale_normalized]
})

# Prediction button
if st.button('Predict Betrayal'):
    probability = xgb_classifier.predict_proba(features)[0, 1]
    st.write(f"Probability of Betrayal: {probability*100:.2f}%")
    if probability > 0.5:
        st.write("Troop betrayal is likely.")
    else:
        st.write("Troop betrayal is unlikely.")
