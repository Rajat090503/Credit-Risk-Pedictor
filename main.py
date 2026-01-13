import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



df = pd.read_csv(r"C:\Users\Rajat\Downloads\credit_card_risk_dataset.csv")

print(df.head())
print(df.info())

print("\nRisk Level Count:")
print(df['Risk_Level'].value_counts())

risk_by_age = df.groupby('Age_Group')['Risk_Level'].value_counts().unstack()
print("\nRisk by Age Group:")
print(risk_by_age)

risk_by_age.plot(kind='bar', figsize=(8,5))
plt.title("Risk Level by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Number of Users")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

print("\nAverage Credit Utilization by Age Group:")
print(df.groupby('Age_Group')['Credit_Utilization'].mean())


le = LabelEncoder()
df['Employment_Type'] = le.fit_transform(df['Employment_Type'])
df['Risk_Level'] = le.fit_transform(df['Risk_Level'])  # High=1, Low=0

X = df[['Age', 'Monthly_Income', 'Credit_Limit', 
        'Monthly_Credit_Spend', 'Credit_Utilization', 'Employment_Type']]
y = df['Risk_Level']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

print("Risk_Level mapping:")
print(le.classes_)

import seaborn as sns

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Risk Level Prediction")
plt.show()

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
}).sort_values(by='Coefficient', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

print("\nKey Observations:")
print("- Higher credit utilization increases risk")
print("- Lower income users are more likely to be high risk")
print("- Employment type affects credit behavior")

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("\nRandom Forest Accuracy:", accuracy_score(y_test, rf_pred))
print("\nRandom Forest Confusion Matrix:\n", confusion_matrix(y_test, rf_pred))
print("\nRandom Forest Classification Report:\n", classification_report(y_test, rf_pred))

rf_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nRandom Forest Feature Importance:")
print(rf_importance)


X_reg = df[['Age', 'Monthly_Income', 'Credit_Limit',
            'Monthly_Credit_Spend', 'Credit_Utilization', 'Employment_Type']]


y_reg = df['Repayment_Months']

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

scaler_reg = StandardScaler()
Xr_train = scaler_reg.fit_transform(Xr_train)
Xr_test = scaler_reg.transform(Xr_test)

lr_model = LinearRegression()
lr_model.fit(Xr_train, yr_train)

yr_pred = lr_model.predict(Xr_test)

print("\nREGRESSION MODEL RESULTS")

print("MAE (Mean Absolute Error):", mean_absolute_error(yr_test, yr_pred))
print("MSE (Mean Squared Error):", mean_squared_error(yr_test, yr_pred))
rmse = mean_squared_error(yr_test, yr_pred) ** 0.5
print("RMSE:", rmse)

print("R2 Score:", r2_score(yr_test, yr_pred))

reg_importance = pd.DataFrame({
    'Feature': X_reg.columns,
    'Coefficient': lr_model.coef_
}).sort_values(by='Coefficient', ascending=False)

print("\nRegression Feature Impact:")
print(reg_importance)

import joblib

joblib.dump(model, "risk_model.pkl")
joblib.dump(rf_model, "rf_risk_model.pkl")
joblib.dump(lr_model, "repayment_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(scaler_reg, "scaler_reg.pkl")

print("Models saved successfully!")
