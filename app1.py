import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="centered"
)

# -----------------------------------
# LOAD DATA
# -----------------------------------
data = pd.read_csv("Titanic-Dataset.csv")

# -----------------------------------
# SELECT REQUIRED COLUMNS
# -----------------------------------
data = data[
    [
        'Pclass',
        'Sex',
        'Age',
        'SibSp',
        'Parch',
        'Fare',
        'Embarked',
        'Survived'
    ]
]

# -----------------------------------
# HANDLE MISSING VALUES
# -----------------------------------

# Numerical columns
num_cols = data.select_dtypes(include=['int64', 'float64']).columns

for col in num_cols:
    data[col] = data[col].fillna(data[col].median())

# Categorical columns
cat_cols = data.select_dtypes(include=['object']).columns

for col in cat_cols:
    data[col] = data[col].fillna(data[col].mode()[0])

# -----------------------------------
# ENCODE CATEGORICAL DATA
# -----------------------------------
data = pd.get_dummies(data, drop_first=True)

# Convert all columns to float
data = data.astype(float)

# -----------------------------------
# FEATURES AND TARGET
# -----------------------------------
X = data.drop('Survived', axis=1)
y = data['Survived']

# -----------------------------------
# TRAIN TEST SPLIT
# -----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------------
# TRAIN MODEL
# -----------------------------------
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# -----------------------------------
# MODEL ACCURACY
# -----------------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

# -----------------------------------
# STREAMLIT UI
# -----------------------------------
st.title("🚢 Titanic Survival Prediction")

st.write(
    "Predict whether a passenger survived the Titanic disaster using Logistic Regression."
)

st.write(f"### Model Accuracy: {accuracy:.2f}")

# -----------------------------------
# USER INPUTS
# -----------------------------------

Pclass = st.selectbox(
    "Passenger Class",
    [1, 2, 3]
)

Sex = st.selectbox(
    "Sex",
    ["male", "female"]
)

Age = st.slider(
    "Age",
    1,
    80,
    25
)

SibSp = st.slider(
    "Number of Siblings/Spouses",
    0,
    10,
    0
)

Parch = st.slider(
    "Number of Parents/Children",
    0,
    10,
    0
)

Fare = st.number_input(
    "Fare",
    min_value=0.0,
    value=50.0
)

Embarked = st.selectbox(
    "Embarked Port",
    ["Q", "S", "C"]
)

# -----------------------------------
# CONVERT INPUT TO MODEL FORMAT
# -----------------------------------

Sex_male = 1 if Sex == "male" else 0

Embarked_Q = 1 if Embarked == "Q" else 0
Embarked_S = 1 if Embarked == "S" else 0

input_data = pd.DataFrame({
    'Pclass': [Pclass],
    'Age': [Age],
    'SibSp': [SibSp],
    'Parch': [Parch],
    'Fare': [Fare],
    'Sex_male': [Sex_male],
    'Embarked_Q': [Embarked_Q],
    'Embarked_S': [Embarked_S]
})

# -----------------------------------
# PREDICTION
# -----------------------------------
if st.button("Predict Survival"):

    prediction = model.predict(input_data)

    probability = model.predict_proba(input_data)[0][1]

    if prediction[0] == 1:

        st.success("Passenger is likely to survive ✅")

    else:

        st.error("Passenger is unlikely to survive ❌")

    st.write(f"### Survival Probability: {probability:.2f}")