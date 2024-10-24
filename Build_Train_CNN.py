import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Step 1: Load the dataset
def load_data(file_path):
    # Assuming the file is in CSV format
    data = pd.read_csv(file_path)
    return data

# Step 2: Preprocess the data
def preprocess_data(data):
    # Assuming the label is in the first column and features in the rest
    X = data.iloc[:, 1:].values  # Features (all columns except the first)
    y = data.iloc[:, 0].values   # Labels (first column)
    
    # Convert labels to integers using LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    # Convert labels to categorical (for multi-class classification)
    num_classes = len(np.unique(y)) # Infer the number of classes from unique values in y
    y = to_categorical(y, num_classes=num_classes) 
    
    # Normalize/scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Reshape the input data for CNN (add extra dimension for 1D CNN)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    return X, y, num_classes

# Step 3: Build the CNN model
def create_cnn_model(input_shape, num_classes):
    model = Sequential()
    
    # 1st Convolutional Layer
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    
    # 2nd Convolutional Layer
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    
    # Flatten the result before feeding into Dense layers
    model.add(Flatten())
    
    # Fully connected layers
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))  # Dropout to prevent overfitting
    
    # Output Layer
    model.add(Dense(num_classes, activation='softmax'))  # Output layer with the correct number of units
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Step 4: Train and evaluate the model
def train_and_evaluate(model, X_train, X_test, y_train, y_test, epochs=10, batch_size=32):
    # Train the model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2) 
    
    # Evaluate the model on test data
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, first plot
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, second plot
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

# Load, preprocess, and split the data
file_path = '/content/Shapes Dataset.csv'  # <-- Put your file path here

# Step 1: Load the dataset
data = load_data(file_path)

# Step 2: Preprocess the data
X, y, num_classes = preprocess_data(data)

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Build and train the CNN model
input_shape = (X_train.shape[1], 1)
model = create_cnn_model(input_shape, num_classes)

# Step 5: Train and evaluate the model
train_and_evaluate(model, X_train, X_test, y_train, y_test, epochs=10, batch_size=32)
