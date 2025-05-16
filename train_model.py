import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Flatten
import joblib

# Load the dataset
df = pd.read_csv('user_data.csv')  # Update with the correct path to your dataset

# Features and target variable
X = df[['followers', 'following', 'engagement_rate', 'is_promotional', 'post_frequency', 'sentiment']]
y = df['category']

# One-hot encode the target variable
y = pd.get_dummies(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape the data for CNN (1D Convolutional Layer requires 3D input)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build the CNN model
model = Sequential([
    Conv1D(32, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Dropout(0.25),
    Conv1D(64, kernel_size=2, activation='relu'),
    GlobalMaxPooling1D(),  # Replaced MaxPooling1D with GlobalMaxPooling1D
    Dropout(0.25),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(y_train.shape[1], activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Save the model and scaler to a file
model.save('models/cnn_model.h5')
joblib.dump(scaler, 'models/scaler.pkl')

print("Model training completed and saved!")
