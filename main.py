import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load the data from the CSV file
df = pd.read_csv('111.csv')

# Features and target
X = df[['no_of_persons', 'temperature', 'humidity']]
y = df['set_temp']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
score = model.score(X_test, y_test)
print(f'Model R^2 score: {score}')

# Save the model to a file
with open('mlr_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved to mlr_model.pkl")
