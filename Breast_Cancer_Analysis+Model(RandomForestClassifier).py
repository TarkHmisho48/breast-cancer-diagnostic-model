#Importing libraries:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('wdbc.csv', header=None)
df
#Manually naming columns
df.columns = ['id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst']
df
#Label encoding catagorical columns:
from sklearn.preprocessing import LabelEncoder
catagorical_cols = ['diagnosis']
for col in catagorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

#Assgining dependent and independent vareibles:
df = df.drop(columns=['id'], axis=1)
X = df.drop(columns=['diagnosis'], axis=1)
y = df['diagnosis']

sns.countplot(x=df['diagnosis'].map({0: 'Benigen', 1:'Mlignant'}))
plt.xlabel('Diagnosis')
plt.ylabel('Count')
plt.title('Diagnosis Class Distribution')
plt.show()

plt.figure(figsize=(14, 12))
sns.heatmap(df.corr(), cmap='coolwarm', annot=False)
plt.title('Features Corrlation Map')
plt.show()
sns.pairplot(df[['radius_mean', 'texture_mean', 'area_mean', 'diagnosis']], hue='diagnosis')
plt.show()
#0: Begain         1:Mlignant

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=20, random_state=42)

#Feature scaling the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
Classifier = RandomForestClassifier(n_estimators=250, criterion='entropy', random_state=0)
Classifier.fit(X_train, y_train)

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
y_pred = Classifier.predict(X_test)
CM = confusion_matrix(y_test, y_pred)
Accuracy_Score = accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred, target_names=['Benign', 'Malignant']))
print('-Confusion Matrix:', CM)
print('-Accuracy Score:', Accuracy_Score)

#Predicting if the patient has a cancer or not based on information we have:
patients_info = [[1.20, 2.22, np.nan, 1001.0, 0.1184, 0.2776, 0.3, 0.1471,
                0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 
                0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 
                184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]]
#Filling the missing data
mean_values = X.mean()
patients_info = [[value if not np.isnan(value) else mean_values[i] for i, value in enumerate(patients_info[0])]]
#Scaling the information before predicting:
patients_info = sc.transform(patients_info)
#Predicting the result:
prediction = Classifier.predict(patients_info)
if prediction[0] == 1:
    print("Diagnosis: Malignant (Cancer)")
else:
    print("Diagnosis: Benign (No cancer)")
#Predicting the probabiltiy:
proba = Classifier.predict_proba(patients_info)
print(f"Probability of Benign:    {proba[0][0]:.2f}")
print(f"Probability of Malignant: {proba[0][1]:.2f}")

import joblib
joblib.dump(Classifier, 'Breast Cancer Analysis +Model (RandomForestClassifier).pkl')