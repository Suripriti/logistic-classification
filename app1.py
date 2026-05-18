import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# -----------------------------
# LOAD DATA
# -----------------------------
data = pd.read_csv("Titanic-Dataset.csv")

# -----------------------------
# SELECT IMPORTANT COLUMNS
# -----------------------------
data = data[[
    'Pclass',
    'Sex',
    'Age',
    'SibSp',
    'Parch',
    'Fare',
    'Embarked',
    'Survived'
]]

# -----------------------------
# HANDLE MISSING VALUES
# -----------------------------
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# -----------------------------
# ENCODE CATEGORICAL DATA
# -----------------------------
data = pd.get_dummies(data, drop_first=True)

# -----------------------------
# FEATURES AND TARGET
# -----------------------------
X = data.drop('Survived', axis=1)
y = data['Survived']

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# TRAIN MODEL
# -----------------------------
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("Titanic Survival Prediction")

st.write("Predict whether a passenger survived the Titanic disaster.")

# -----------------------------
# USER INPUTS
# -----------------------------
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
    "Siblings/Spouses Aboard",
    0,
    10,
    0
)

Parch = st.slider(
    "Parents/Children Aboard",
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
    "Embarked",
    ["Q", "S", "C"]
)

# -----------------------------
# CONVERT INPUTS
# -----------------------------
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

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Predict Survival"):

    prediction = model.predict(input_data)

    probability = model.predict_proba(input_data)[0][1]

    if prediction[0] == 1:
        st.success("Passenger is likely to survive")

    else:
        st.error("Passenger is unlikely to survive")

    st.write(f"Survival Probability: {probability:.2f}")