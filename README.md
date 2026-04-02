# Scholarship Eligibility Checker

## About the Project

This project is a simple machine learning-based application that helps users check whether they are eligible for different scholarship schemes based on their profile.

The idea behind this project is to make scholarship filtering easier by using data analysis and basic ML techniques. Instead of manually checking eligibility criteria for each scheme, users can input their details and quickly get a prediction.

---

## What the Project Does

* Takes user inputs like age, income, category, and academic performance
* Processes the data using a trained ML model
* Predicts whether the user is eligible or not
* Displays a confidence score along with the result
* Provides a clean and simple interface using Streamlit

---

## Approach Used

### Data Processing

The dataset was cleaned by:

* Removing unnecessary columns like IDs and links
* Handling missing values
* Converting categorical data into numerical form

---

### Feature Engineering

A new feature called `age_range` was created using:

* `max_age - min_age`

This helped in simplifying the eligibility condition.

---

### Model

* Logistic Regression was used for classification
* Data was scaled using StandardScaler
* Missing values were handled using SimpleImputer

---

### Prediction Logic

The system uses a combination of:

* Model prediction
* A simple scoring mechanism based on user inputs

This helps in giving a more practical and understandable output.

---

## Dataset

The dataset contains around 5000 scholarship records with details such as:

* Age criteria
* Income limit
* Category
* Education level
* Scholarship amount
* State and provider

---

## Tools & Technologies

* Python
* Pandas, NumPy
* Scikit-learn
* Streamlit
* Matplotlib & Seaborn

---

##  How to Run

1. Clone the repository

```bash id="7x9r2p"
git clone https://github.com/your-username/scholarship-recommender.git
cd scholarship-recommender
```

2. Install required libraries

```bash id="n6g8dt"
pip install -r requirements.txt
```

3. Run the app

```bash id="5c6kzp"
streamlit run scheme.py
```

---

##  What I Learned

* How to clean and preprocess real-world datasets
* Importance of feature engineering
* How ML models behave when data is not strong
* Building an end-to-end ML project
* Connecting ML with a frontend using Streamlit

---

##  Future Improvements

* Use real-world verified scholarship data
* Improve prediction accuracy
* Add proper recommendation system
* Enhance UI and add more filters

---

---

##  Note

This project was built as part of my learning in Machine Learning and application development.
