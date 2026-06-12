from sklearn.ensemble import RandomForestClassifier

def create_and_train_model(X_train, y_train):
    """
    This function is called by the AutoGrader.
    It takes training data as input and must return a TRAINED model.
    """
    print("Training Random Forest model inside student_submission.py...")
    # Initialize the model
    # Try changing n_estimators to 1 or 2 and see how your score drops!
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Return the trained model so the autograder can test it
    return model
