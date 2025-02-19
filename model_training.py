if __name__ == '__main__':
    import pandas as pd  # Import the pandas library for data manipulation
    from sklearn import linear_model  # Import the linear_model module from scikit-learn for linear regression
    from sklearn.model_selection import train_test_split  # Import the train_test_split function for splitting data
    from sklearn.metrics import mean_squared_error  # Import the mean_squared_error function for model evaluation
    import joblib  # Import joblib for saving and loading models

    df = pd.read_excel('frauds.xlsx')  # Read the Excel file into a DataFrame

    # Define mappings for categorical variables
    device_type_map = {'browser': 1, 'mobile': 2, "machine": 3}
    connection_type_map = {'Wifi': 1, 'mobile_data': 2}
    gps_status_map = {'ON': 1, 'OFF': 0}
    know_gps_map = {False: 0, True: 1}
    know_recipient_map = {False: 0, True: 1}

    # Map the categorical variables to numerical values
    df['device_type'] = df['device_type'].map(device_type_map)
    df['connection_type'] = df['connection_type'].map(connection_type_map)
    df['gps_status'] = df['gps_status'].map(gps_status_map)
    df['know_gps'] = df['know_gps'].map(know_gps_map)
    df['know_recipient'] = df['know_recipient'].map(know_recipient_map)

    # Separate features and target variable
    x = df.drop(columns=["is_fraud"])  # Features
    y = df["is_fraud"]  # Target variable

    model = linear_model.LinearRegression()  # Create a Linear Regression model

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2) # Split the data into training and testing sets

    model.fit(x_train, y_train)  # Train the model on the training data

    joblib.dump(model, 'fraud_detection_model.pkl') # Save the model to a file

    y_pred = model.predict(x_test)  # Predict the target variable for the test data

    print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred)}')  # Print the Mean Squared Error (prediction quality)

    # Predict the probability of fraud for a new sample
    predict = model.predict(pd.DataFrame([[1, 2, 1, 0, 0, 178]], columns=["device_type", "connection_type", "gps_status", "know_gps", "know_recipient", "Value"])) 

    print("probability to be a fraud", predict[0])  # Print the prediction result