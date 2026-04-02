import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("/content/scholarship_programs.csv")

df.shape

df.head()

df.info()

df.describe()

df.columns

print(df.nunique())

df.isnull().sum()

df.corr()['eligible'].sort_values(ascending=False)

df = df.drop([
    'scholarship_id',
    'scheme_name',
    'application_link',
    'documents_required',
    'deadline_date'
], axis=1, errors='ignore')

df.duplicated().sum()

df.select_dtypes(include='object').columns

# Create age_range column from existing min_age and max_age
df['age_range'] = df['max_age'] - df['min_age']

df.drop(columns=['Age'], inplace=True, errors='ignore')

df = pd.get_dummies(df, drop_first=True)

df.drop(columns=['benefit_amount'], inplace=True, errors='ignore')

df['age_range'].hist(bins=20, color='steelblue', edgecolor='black')
plt.title('Age Eligibility Window per Scheme')
plt.xlabel('Age Range (years)')
plt.ylabel('Count')
plt.show()

numeric_cols = ['min_age', 'max_age', 'age_range']
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.show()

df.hist(figsize=(12,10))
plt.suptitle("Feature Distributions",fontsize=16)
plt.show()

df['eligible'] = df['age_range'].apply(
    lambda x: 1 if x > df['age_range'].median() else 0
)
sns.boxplot(x='eligible', y='min_age', data=df)
plt.show()

df.drop(columns=['documents_required', 'deadline_date'], inplace=True, errors='ignore')
print(df.columns)

df.columns = df.columns.str.strip()

y = df['eligible']

X = df[['min_age', 'max_age', 'max_income']]

df.drop(columns=['priority_score'], inplace=True, errors='ignore')

X = df.drop('eligible', axis=1)
y = df['eligible']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
