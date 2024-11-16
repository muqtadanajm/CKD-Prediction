import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix, log_loss
import tensorflow as tf
from keras.models import load_model


# Prepare and preprocess data
def prepare_data():
    # Load the dataset
    df = pd.read_csv('../data/kidney_disease.csv')
    df.head()

    # Preprocessing
    df[['htn', 'dm', 'cad', 'pe', 'ane']] = df[['htn', 'dm', 'cad', 'pe', 'ane']].replace(
        to_replace={'yes': 1, 'no': 0})
    df[['rbc', 'pc']] = df[['rbc', 'pc']].replace(to_replace={'abnormal': 1, 'normal': 0})
    df[['pcc', 'ba']] = df[['pcc', 'ba']].replace(to_replace={'present': 1, 'notpresent': 0})
    df[['appet']] = df[['appet']].replace(to_replace={'good': 1, 'poor': 0, 'no': np.nan})
    df['classification'] = df['classification'].replace(to_replace={'ckd': 1.0, 'ckd\t': 1.0, 'notckd': 0.0, 'no': 0.0})
    df.rename(columns={'classification': 'class'}, inplace=True)
    df.drop('id', axis=1, inplace=True)
    df = df.replace(["?", "	?"], np.nan).fillna(method='ffill').fillna(method='backfill')

    # Convert categorical columns to numeric
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = pd.Categorical(df[column]).codes

    X = df.drop("class", axis=1)
    y = df["class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


# Build and compile the neural network model
def build_model(input_dim):
    model = Sequential()
    model.add(Dense(units=128, kernel_initializer='uniform', activation='relu', input_dim=input_dim))
    model.add(Dropout(0.2))
    model.add(Dense(units=128, kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=64, kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=32, kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Train and save the model
def train_and_save_model():
    X_train, X_test, y_train, y_test = prepare_data()

    model = build_model(X_train.shape[1])

    # Training the model
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

    # Save the model
    model.save('../model/kidney_disease_model.h5')
    print("Model saved to 'model/kidney_disease_model.h5'")


# Evaluate the model
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test).flatten()
    predictions = (predictions > 0.5).astype(int)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    confusion = confusion_matrix(y_test, predictions)
    log_loss_value = log_loss(y_test, predictions)

    print(f"Accuracy: {accuracy:.4%}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Confusion Matrix:\n{confusion}")
    print(f"Log Loss: {log_loss_value:.4f}")


# Execute the training and saving process
train_and_save_model()
