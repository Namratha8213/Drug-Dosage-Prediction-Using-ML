from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Load your dataset
data = 'drugdos.csv'
df = pd.read_csv(data)

# Asuming df is your DataFrame and 'column_to_remove' is the column you want to remove
column_to_remove = 'review'  # Replace with the actual column name
df = df.drop(column_to_remove, axis=1)


# Define categorical columns
cat_columns = ['drug', 'bp', 'sugar', 'Sideeffects']

# Filter columns that are present in the DataFrame
cat_columns = [col for col in cat_columns if col in df.columns]

# Split the data into training and testing sets
X = df.drop(['dosage', 'condition', 'rating', 'usefulCount', 'temperature'], axis=1)
Y = df['dosage']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create a column transformer with OneHotEncoder and SimpleImputer
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), cat_columns),
        ('num', SimpleImputer(strategy='mean'), X.select_dtypes(include=['number']).columns)
    ],
    remainder='passthrough'
)

# Create a pipeline with the column transformer and the RandomForestRegressor
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor())
])

# Train the model
pipeline.fit(X_train, Y_train)

# Flask app setup
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_data = {
        'age': int(request.form['age']),
        'sugar': request.form['sugar'],
        'bp': request.form['bp'],
        'drug': request.form['drug'],
        'Sideeffects': request.form['Sideeffects']
    }

    user_data_df = pd.DataFrame([user_data])
    dosage_prediction = pipeline.predict(user_data_df)[0]

    return render_template('result.html', result=f'Predicted Dosage: {dosage_prediction}')

if __name__ == '__main__':
    app.run(debug=True)