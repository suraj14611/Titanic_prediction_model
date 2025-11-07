from flask import Flask, render_template, request
import pickle
import pandas as pd
import re

app = Flask(__name__)

# Load your saved model
with open('titanic_model.pkl', 'rb') as f:
    model = pickle.load(f)

def preprocess_input(form_data):
    # Ensure all numeric fields are scalars, not lists
    form_data['Age'] = str(form_data['Age'])
    form_data['Fare'] = str(form_data['Fare'])
    form_data['Pclass'] = int(form_data['Pclass'])
    form_data['SibSp'] = int(form_data['SibSp'])
    form_data['Parch'] = int(form_data['Parch'])
    # Calculate FamilySize
    form_data['FamilySize'] = form_data['SibSp'] + form_data['Parch'] + 1

    df = pd.DataFrame([form_data])

    # Safe numeric conversions and filling missing values
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df['Age'] = df['Age'].fillna(df['Age'].median())

    df['Fare'] = pd.to_numeric(df['Fare'], errors='coerce')
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    # Extract and map Title
    name = df.loc[0, 'Name']
    title_search = re.search(' ([A-Za-z]+)\.', name)
    title = title_search.group(1) if title_search else ''
    rare_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major',
                   'Rev', 'Sir', 'Jonkheer', 'Dona']
    if title in ['Mlle', 'Ms']:
        title = 'Miss'
    elif title == 'Mme':
        title = 'Mrs'
    elif title not in ['Mr', 'Miss', 'Mrs', 'Master'] + rare_titles:
        title = 'Rare'
    title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}
    df['Title'] = title_mapping.get(title, 0)

    # Encode categorical columns
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize', 'Title']
    return df[features]

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        form_data = {
            'Pclass': request.form['Pclass'],
            'Sex': request.form['Sex'],
            'Age': request.form['Age'],
            'Fare': request.form['Fare'],
            'Embarked': request.form['Embarked'],
            'SibSp': request.form['SibSp'],
            'Parch': request.form['Parch'],
            'Name': request.form['Name'],
        }
        # Preprocess and predict
        input_df = preprocess_input(form_data)
        prediction = model.predict(input_df)[0]
        result = 'Survived' if prediction == 1 else 'Did not survive'
        return render_template('result.html', result=result)

    return render_template('form.html')

if __name__ == '__main__':
    app.run(debug=True)
