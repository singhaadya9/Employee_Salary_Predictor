# 💼 Employee Salary Predictor

A simple end-to-end Machine Learning project (Task 3) that predicts an employee's annual salary based on their profile.

Built using **Pandas, NumPy, Scikit-learn, and Streamlit**.

## 📁 Project Structure

```text
employee_salary_predictor/
│
├── generate_dataset.py       # Creates a realistic synthetic dataset
├── train_model.py            # Preprocessing, model training and evaluation
├── app.py                    # Streamlit web interface
├── requirements.txt          # Python dependencies
│
├── data/
│   └── employee_salary.csv   # Generated dataset
│
└── model/
    ├── model_comparison.csv  # Model performance comparison
    └── ...                    # Saved model and preprocessing objects