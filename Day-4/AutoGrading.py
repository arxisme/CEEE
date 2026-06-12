import traceback
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# =====================================================================
# MACHINE LEARNING AUTO-GRADER
# This script imports a student's Python file and tests their ML model.
# =====================================================================

def grade_student_submission(student_module_name):
    print(f"--- Grading Submission: {student_module_name}.py ---")
    score = 0
    max_score = 10
    
    try:
        # 1. Attempt to import the student's code dynamically
        student_code = __import__(student_module_name)
        print("✅ Code compiled and imported successfully. (+2 marks)")
        score += 2
        
        # 2. Check if the required function exists
        if not hasattr(student_code, 'create_and_train_model'):
            print("❌ ERROR: Function 'create_and_train_model' not found.")
            return score
            
        print("✅ Correct function signature found. (+2 marks)")
        score += 2
        
        # 3. Generate a HIDDEN dataset the student has never seen
        X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 4. Run the student's function
        # The student's function should take training data and return a trained model
        student_model = student_code.create_and_train_model(X_train, y_train)
        
        # 5. Evaluate the model on hidden test data
        predictions = student_model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        
        print(f"📊 Student Model Accuracy on Hidden Data: {acc*100:.2f}%")
        
        if acc > 0.85:
            print("✅ Model achieved > 85% accuracy. (+6 marks)")
            score += 6
        elif acc > 0.70:
            print("⚠️ Model achieved > 70% accuracy. Partial credit. (+3 marks)")
            score += 3
        else:
            print("❌ Model accuracy too low. (0 marks for performance)")
            
    except Exception as e:
        print("❌ CRITICAL ERROR IN STUDENT CODE:")
        print(traceback.format_exc())
        
    print(f"\n======================================")
    print(f"FINAL SCORE FOR {student_module_name}: {score}/{max_score}")
    print(f"======================================\n")
    return score

if __name__ == "__main__":
    # Example usage: Grade a file named 'student_submission.py'
    # In reality, you would loop over a folder of 100 student files.
    try:
        grade_student_submission("student_submission")
    except ModuleNotFoundError:
        print("Please create a 'student_submission.py' file to test the autograder.")