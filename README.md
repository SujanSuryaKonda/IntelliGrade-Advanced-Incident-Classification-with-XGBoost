# IntelliGrade-Advanced-Incident-Classification-with-XGBoost

# IntelliGrade-Advanced-Incident-Classification-with-XGBoost

## Overview
**IntelliGrade** is an advanced incident classification tool built using machine learning techniques, specifically leveraging the power of **XGBoost** and other machine learning models. This project aims to predict incident grades based on structured and categorical data, assisting organizations in automating threat analysis and decision-making processes.

## Features
- Advanced data preprocessing and feature engineering.
- Label encoding for categorical data.
- Data balancing using `RandomOverSampler`.
- Training using `XGBoostClassifier` with hyperparameter tuning.
- Comprehensive evaluation with metrics like accuracy, precision, recall, and F1 score.
- Visualizations for feature correlations and data distributions.
- Fully documented code for easy replication and adaptation.

## Project Structure
## Overview
This project focuses on classifying incidents into categories such as TP, BP, and FP using various machine learning techniques. The approach ensures robust data preprocessing, model training, and evaluation to achieve optimal performance. The final model aims to aid in the efficient and accurate classification of incidents, which can be beneficial for integration into SOC workflows.

## Approach

### 1. Data Exploration and Understanding
- **Initial Inspection**: Load `train.csv` and perform an initial inspection to understand the data structure, including the number of features, types of variables (categorical, numerical), and distribution of the target variable (TP, BP, FP).
- **Exploratory Data Analysis (EDA)**: Use visualizations (e.g., histograms, bar charts) and statistical summaries to identify patterns, correlations, and potential anomalies. Pay attention to class imbalances for later handling.

### 2. Data Preprocessing
- **Handling Missing Data**: Identify missing values and apply appropriate strategies such as imputation (e.g., mean, median, or mode) or removing rows/columns with excessive missing values.
- **Feature Engineering**: Create or modify features to improve performance (e.g., derive new features from timestamps or combine related features). Normalize numerical variables if needed.
- **Encoding Categorical Variables**: Use techniques such as one-hot encoding, label encoding, or target encoding to convert categorical features into numerical representations.

### 3. Data Splitting
- **Train-Validation Split**: Split `train.csv` into training and validation sets (e.g., 70-30 or 80-20 split) to evaluate model performance before testing on `test.csv`.
- **Stratification**: Apply stratified sampling to maintain similar class distributions in both training and validation sets if the target variable is imbalanced.

### 4. Model Selection and Training
- **Baseline Model**: Start with a simple model like logistic regression or a decision tree to establish a benchmark.
- **Advanced Models**: Experiment with more complex models such as:
  - **Random Forests**
  - **Gradient Boosting Machines (e.g., XGBoost, LightGBM)**
  - **Neural Networks**
- **Cross-Validation**: Use k-fold cross-validation to ensure consistent performance across data subsets and reduce overfitting risks.

### 5. Model Evaluation and Tuning
- **Performance Metrics**: Evaluate the model using the validation set, focusing on macro-F1 score, precision, and recall. Analyze these metrics across classes (TP, BP, FP) for balanced performance.
- **Hyperparameter Tuning**: Use grid search or random search to fine-tune hyperparameters such as learning rate, regularization, tree depth, and the number of estimators.
- **Handling Class Imbalance**:
  - Apply **SMOTE** or **RandomOverSampler** to balance the dataset.
  - Adjust class weights or use ensemble methods to improve the model's handling of minority classes.

### 6. Model Interpretation
- **Feature Importance**: Analyze feature importance using SHAP values, permutation importance, or model-specific methods to understand key contributors to the model's predictions.
- **Error Analysis**: Identify common misclassifications to gain insights for further improvements, such as enhanced feature engineering or model adjustments.

### 7. Final Evaluation on Test Set
- **Testing**: Evaluate the final model on `test.csv` and report metrics including macro-F1 score, precision, and recall to measure generalization to unseen data.
- **Comparison to Baseline**: Compare the final test results to the baseline model and validation metrics to confirm consistency and improvement.

### 8. Documentation and Reporting
- **Model Documentation**: Document the entire process, including:
  - The rationale behind chosen methods.
  - Challenges and how they were addressed.
  - Key findings and performance metrics.
- **Recommendations**: Include recommendations for integrating the model into SOC workflows, areas for future enhancement, and deployment considerations for real-world applications.

## Key Findings and Insights
- The final model, tuned with hyperparameters and evaluated with cross-validation, demonstrates balanced performance across all classes.
- Significant features contributing to predictions were identified, providing valuable insights for incident classification.

## Future Enhancements
- Implement more sophisticated feature selection techniques.
- Integrate real-time prediction capabilities.
- Experiment with other algorithms like **CatBoost** or ensemble methods.
