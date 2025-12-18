import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("/home/intellect/Documents/Data_Scientist/knn_streamlit_project/Social_Network_Ads.csv")

# Features and target
X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']

# Feature scaling (VERY IMPORTANT for KNN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train KNN model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Save model and scaler
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("✅ KNN model trained and saved successfully")


from sklearn.metrics import accuracy_score

for k in range(1, 15):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f"K={k}, Accuracy={accuracy_score(y_test, preds):.2f}")
