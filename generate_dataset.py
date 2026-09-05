

import numpy as np
import pandas as pd

np.random.seed(42)

N = 3000

genders = ["Male", "Female"]
education_levels = ["High School", "Bachelor's", "Master's", "PhD"]
departments = ["IT", "Sales", "HR", "Finance", "Marketing", "Operations"]
job_titles_by_dept = {
    "IT": ["Software Engineer", "Data Analyst", "IT Support", "DevOps Engineer"],
    "Sales": ["Sales Executive", "Sales Manager", "Account Manager"],
    "HR": ["HR Executive", "HR Manager", "Recruiter"],
    "Finance": ["Financial Analyst", "Accountant", "Finance Manager"],
    "Marketing": ["Marketing Executive", "Marketing Manager", "SEO Specialist"],
    "Operations": ["Operations Analyst", "Operations Manager", "Logistics Coordinator"],
}
locations = ["Metro City", "Tier-2 City", "Remote"]

education_base = {"High School": 20000, "Bachelor's": 32000, "Master's": 42000, "PhD": 55000}
education_exp_multiplier = {"High School": 900, "Bachelor's": 1500, "Master's": 2000, "PhD": 2600}
dept_multiplier = {
    "IT": 1.35, "Finance": 1.15, "Marketing": 1.05,
    "Sales": 1.0, "Operations": 0.95, "HR": 0.9,
}
location_multiplier = {"Metro City": 1.25, "Tier-2 City": 1.0, "Remote": 1.05}

title_bump = {
    "Manager": 12000, "Senior": 8000, "Lead": 9000,
}

rows = []
for _ in range(N):
    age = int(np.clip(np.random.normal(35, 9), 21, 65))
    gender = np.random.choice(genders, p=[0.55, 0.45])
    education = np.random.choice(education_levels, p=[0.15, 0.45, 0.30, 0.10])

    max_possible_exp = age - 20
    experience = int(np.clip(np.random.normal(max_possible_exp * 0.55, 4), 0, max(max_possible_exp, 0)))

    department = np.random.choice(departments)
    job_title = np.random.choice(job_titles_by_dept[department])
    location = np.random.choice(locations, p=[0.45, 0.35, 0.20])

    base = education_base[education]
    exp_component = experience * education_exp_multiplier[education]
    salary = (base + exp_component) * dept_multiplier[department] * location_multiplier[location]

   
    for keyword, bump in title_bump.items():
        if keyword in job_title:
            salary += bump


    salary += age * 150
    salary += np.random.normal(0, 3500) 
    salary = max(salary, 15000)  

    rows.append({
        "Age": age,
        "Gender": gender,
        "Education Level": education,
        "Department": department,
        "Job Title": job_title,
        "Years of Experience": experience,
        "Location": location,
        "Salary": round(salary, 2),
    })

df = pd.DataFrame(rows)

for col in ["Age", "Years of Experience"]:
    missing_idx = np.random.choice(df.index, size=int(0.01 * N), replace=False)
    df.loc[missing_idx, col] = np.nan

df.to_csv("data/employee_salary.csv", index=False)
print(f"Saved {len(df)} rows to data/employee_salary.csv")
print(df.head())
print("\nMissing values per column:\n", df.isnull().sum())
