# Survival Prediction Model

### Overview

A Survival Prediction Model for patients with cardiovascular heart disease using a dataset of 300 patients with heart failure collected in 2015 and 12 clinical features

Data Analysis & Machine Learning Tools Used: Python, NumPy, Pandas, Seaborn, Matplotlib, and Scikit-learn

For a step-by-step walkthrough, please refer to the ipynb notebook in this repository. 

### Initial Observations on the data
- Clean dataset - no null values, infinity values
- Dataset had only numeric data types (int64, float64)
- Ranges in the data vary from column to column (i.e., feature scaling is needed)
- Features like age, ejection_fraction, platelets and serum_sodium have a "bell-shaped" or even distributions.

### Data Visualizations
1. What is the distribution of all the numeric features in the dataset?

2. What is the relationship between a patient's serum_creatinine levels and their ejection_fraction? How do these features influence patient survival?
3. What is the pairwise relationship between serum_creatinine and ejection_fraction, and other features?
4. What is the relationship between smoking and ejection_fraction? How does the patient's sex and survival factor into this?
5. What is the relationship between serum_sodium and high_blood_pressure? How do these features influence patient survival?

### ML Training Strategies
- Used the Stratfied Shuffle Split to the split the data into bins based on age
- Compared with the Random Split 
- Transformations performed includes the following:
-   Feature Scaling using the Standard Scaler
-   No Imputer used since the dataset contained no missing data
-   No Encoder used since the dataset contained no categorical data

### ML Models and Analysis
- DecisionTreeClassifier score: 0.774, +/-  0.05
- KNeighborsClassifier score: 0.740, +/-  0.04
- RandomForestClassifier score: 0.849, +/-  0.05
- MLP (Neural Network) Classifier score: 0.727 +/-  0.09

### Fine Tuning:
- Finetuned the RandomForestClassifier using the Randomized Search CV, resulting in an accuracy score of 0.867
- Important Clinical Features in the dataset were shown: Time (0.331), Serum Creatinine Levels (0.139), and Ejection Fraction (0.131)

### ML Model Performance
The performance of the different models were analyzed by comparing the ROC and PRC curves.

The Best ML Model, the Random Forests Classifier model, was analyzed further for its performance using a Confusions Matrix.

### Conclusions
Lessons Learned:
- Preparing Datasets for Training
- Testing and Training ML Models
- Comparing Different Models using ROC and PRC curves
- Identifying the Best Model and Finetuning
- Understanding Model Performance using Confusions Matrix

Challenges and future possibilities:
- Using a larger dataset will help us finetune our models further and would improve the credibility of the accuracy score. 
- We could also try finetuning with more parameters. 
