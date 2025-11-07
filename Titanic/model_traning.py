import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import pickle

# Load the training and test data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')  # test.csv without 'Survived'

def preprocess(df):
    # Fix chained assignment warnings by avoiding inplace=True on column slices:
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    rare_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major',
                   'Rev', 'Sir', 'Jonkheer', 'Dona']
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    df['Title'] = df['Title'].apply(lambda x: 'Rare' if x in rare_titles else x)
    title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}
    df['Title'] = df['Title'].map(title_mapping)
    df['Title'] = df['Title'].fillna(0)
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
    return df

# Preprocess training data
train_df = preprocess(train_df)

features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize', 'Title']
X = train_df[features]
y = train_df['Survived']

# Split train data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate on validation set
y_pred = model.predict(X_val)
print('Validation Accuracy:', accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred))

# Preprocess test set (no Survived column)
test_passenger_ids = test_df['PassengerId']
test_df = preprocess(test_df)
X_test = test_df[features]

# Save the trained model to a pickle file
with open('titanic_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Predict on test set
y_test_pred = model.predict(X_test)

# Use y_test_pred as needed (e.g., for submission)
