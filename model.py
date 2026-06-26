import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import hstack

# Load dataset
df = pd.read_csv("AI-Based Hiring Prediction System.csv")

# Drop unwanted columns
df.drop(["Resume_ID", "Name"], axis=1, inplace=True)

# Target conversion
df["Recruiter Decision"] = df["Recruiter Decision"].map({
    "Hire": 1,
    "Reject": 0
})

# Combine text columns
df["combined_text"] = (
    df["Skills"].astype(str) + " " +
    df["Certifications"].astype(str) + " " +
    df["Job Role"].astype(str)
)

# Text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

df["combined_text"] = df["combined_text"].apply(clean_text)

# TFIDF
tfidf = TfidfVectorizer(max_features=500)
X_text = tfidf.fit_transform(df["combined_text"])

# Encode education
le = LabelEncoder()
df["Education"] = le.fit_transform(df["Education"])

# Numeric features
X_numeric = df[
    ["Experience (Years)", "Salary Expectation ($)",
     "Projects Count", "Education"]
]

y = df["Recruiter Decision"]

# Combine features
X = hstack((X_text, X_numeric))

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler(with_mean=False)
X_train = scaler.fit_transform(X_train)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)


# Prediction function
def predict_candidate(skills, experience, education,
                      certifications, projects, salary):

    text = clean_text(skills + " " + certifications)
    text_vector = tfidf.transform([text])

    edu_encoded = le.transform([education])[0]

    numeric = np.array([[experience, salary, projects, edu_encoded]])

    combined = hstack((text_vector, numeric))
    combined = scaler.transform(combined)

    prediction = model.predict(combined)[0]
    probability = model.predict_proba(combined)[0][1]

    result = "Hire" if prediction == 1 else "Reject"

    return result, round(probability, 2)