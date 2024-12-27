# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Define a sample dataset
data = {
    'Square_Feet': [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500],
    'Bedrooms': [2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
    'Price': [200000, 250000, 300000, 350000, 400000, 450000, 500000, 550000, 600000, 650000],
    'age': [20, 25, 30, 35, 40, 45, 50, 55, 60, 65]
}

# Create a pandas DataFrame from the dataset
df = pd.DataFrame(data)

# Create a correlation heatmap to visualize relationships between features
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Create a scatter plot to visualize the relationship between Square Feet and Price
sns.scatterplot(x='Square_Feet', y='Price', data=df)
plt.title('Square Feet vs Price')
plt.show()

# Define the features (X) and target variable (y)
x = df[['Square_Feet', 'Bedrooms', 'age']]
y = df['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train a linear regression model on the training data
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the testing data
predictions = model.predict(X_test)

# Evaluate the model using mean absolute error
mae = mean_absolute_error(y_test, predictions)
print(f'Mean Absolute Error: {mae}')

# Visualize the actual vs predicted prices
plt.scatter(y_test, predictions)
plt.xlabel('Actual prices')
plt.ylabel('Predicted prices')
plt.title('Actual vs Predicted prices')
plt.show()