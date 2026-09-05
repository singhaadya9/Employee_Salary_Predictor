# 💼 Employee Salary Predictor

A simple end-to-end Machine Learning project (Task 3) that predicts an employee's
annual salary based on their profile — built with **Pandas, NumPy, Scikit-learn,
and Streamlit**.

## Project Structure

```
employee_salary_predictor/
├── generate_dataset.py     # Creates a realistic synthetic dataset
├── train_model.py          # Preprocessing + model training + evaluation
├── app.py                  # Streamlit web interface
├── requirements.txt
├── data/
│   └── employee_salary.csv # Generated dataset
└── model/                  # Saved model + preprocessing objects (created after training)
```

## Dataset

Features used to predict **Salary**:

| Feature | Type | Description |
|---|---|---|
| Age | Numeric | Employee age |
| Gender | Categorical | Male / Female |
| Education Level | Categorical | High School, Bachelor's, Master's, PhD |
| Department | Categorical | IT, Sales, HR, Finance, Marketing, Operations |
| Job Title | Categorical | Role within the department |
| Years of Experience | Numeric | Total years worked |
| Location | Categorical | Metro City, Tier-2 City, Remote |

The dataset is **synthetically generated** (`generate_dataset.py`) with realistic
correlations (e.g. salary rises with experience/education, varies by department
and location) plus random noise and a few missing values — so it behaves like a
real-world dataset while keeping the project fully self-contained and
reproducible. You can swap in a real dataset (e.g. a Kaggle salary dataset) by
replacing `data/employee_salary.csv` with the same column names.

## ML Workflow

1. **Preprocessing** — median imputation for missing numeric values, label
   encoding for categorical features, standard scaling for numeric features.
2. **Model training** — three regressors are trained and compared:
   - Linear Regression
   - Random Forest Regressor
   - Gradient Boosting Regressor
3. **Evaluation** — MAE, RMSE, and R² on a held-out 20% test split.
4. **Model selection** — the best model (by R²) is saved automatically for
   use in the Streamlit app.

### Example results (synthetic dataset)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | ~11,775 | ~15,647 | 0.56 |
| Random Forest | ~5,502 | ~7,308 | 0.90 |
| **Gradient Boosting** | **~3,420** | **~4,612** | **0.96** |

(Your exact numbers may vary slightly depending on the random seed / regenerated data.)

## How to Run

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate the dataset**
   ```bash
   python generate_dataset.py
   ```

3. **Train the model** (evaluates 3 models, saves the best one)
   ```bash
   python train_model.py
   ```

4. **Launch the Streamlit app**
   ```bash
   streamlit run app.py
   ```

5. Open the local URL Streamlit prints (usually `http://localhost:8501`),
   fill in the employee details, and click **Predict Salary**.

## Notes / Possible Extensions

- Swap the synthetic dataset for a real one (just keep the column names, or
  update `categorical_cols`/`numeric_cols` in `train_model.py` to match).
- Add more models (e.g. XGBoost) to the comparison in `train_model.py`.
- Add SHAP or feature-importance plots to explain individual predictions.
- Deploy the Streamlit app for free on [Streamlit Community Cloud](https://streamlit.io/cloud).
