# IQVIA_technical_exercise

## Contents:
### 1. EDA
### 2. Model evaluation
### 3. Conclusion

# 1. EDA:

Violin plots for individual features against patient labels (1) no cancer, (2) cancer.

![alt text](/images/violin.png)

### Agreggated statistics
#### For all patients:

|       |       Age |       BMI |     Height |    Glucose |   Insulin |      HOMA |    Leptin | Adiponectin |  Resistin |       MCP.1 |
|------:|----------:|----------:|-----------:|-----------:|----------:|----------:|----------:|------------:|----------:|------------:|
| count | 97.000000 | 97.000000 |  97.000000 |  97.000000 | 97.000000 | 97.000000 | 97.000000 |   97.000000 | 97.000000 |   97.000000 |
|  mean | 57.329897 | 27.674812 | 171.556701 |  98.175258 | 10.108206 |  2.749533 | 27.700072 |   10.704966 | 14.590833 |  523.578196 |
|   std | 16.800745 |  5.067532 |  12.550705 |  21.309236 | 10.295882 |  3.823989 | 19.901463 |    7.210423 | 12.592142 |  338.716789 |
|   min | 24.000000 | 18.670000 | 150.000000 |  60.000000 |  2.432000 |  0.467409 |  4.311000 |    1.656020 |  3.270000 |   45.843000 |
|   max | 89.000000 | 38.578759 | 194.000000 | 201.000000 | 58.460000 | 25.050342 | 90.280000 |   38.040000 | 82.100000 | 1698.440000 |

#### For cancer patients:

|       |       Age |       BMI |     Height |   Glucose  |   Insulin |      HOMA |    Leptin | Adiponectin |  Resistin |       MCP.1 |
|------:|----------:|----------:|-----------:|-----------:|----------:|----------:|----------:|------------:|----------:|------------:|
| count | 52.000000 | 52.000000 |  52.000000 | 52.000000  | 52.000000 | 52.000000 | 52.000000 |   52.000000 | 52.000000 |   52.000000 |
|  mean | 56.807692 | 27.167446 | 171.730769 | 106.288462 | 12.564481 |  3.711366 | 27.720650 |   10.615498 | 17.057816 |  563.816288 |
|   std | 14.429054 |  4.583536 |  12.218782 | 24.774541  | 12.789162 |  4.905591 | 20.042640 |    6.467862 | 12.457474 |  380.282997 |
|   min | 34.000000 | 19.560000 | 150.000000 | 74.000000  |  2.432000 |  0.507936 |  6.333900 |    1.656020 |  3.270000 |   90.090000 |
|   max | 86.000000 | 37.109375 | 192.000000 | 201.000000 | 58.460000 | 25.050342 | 90.280000 |   33.750000 | 55.215300 | 1698.440000 | 

#### For non-cancer patients:

|       |       Age |       BMI |     Height |    Glucose |   Insulin |      HOMA |    Leptin | Adiponectin |  Resistin |       MCP.1 |
|------:|----------:|----------:|-----------:|-----------:|----------:|----------:|----------:|------------:|----------:|------------:|
| count | 45.000000 | 45.000000 |  45.000000 |  45.000000 | 45.000000 | 45.000000 | 45.000000 |   45.000000 | 45.000000 |   45.000000 |
|  mean | 57.933333 | 28.261102 | 171.355556 |  88.800000 |  7.269844 |  1.638080 | 27.676293 |   10.808352 | 11.740098 |  477.080844 |
|   std | 19.334848 |  5.569736 |  13.059576 |  10.564951 |  5.131318 |  1.286254 | 19.963276 |    8.057558 | 12.270771 |  280.305574 |
|   min | 24.000000 | 18.670000 | 151.000000 |  60.000000 |  2.707000 |  0.467409 |  4.311000 |    2.194280 |  3.291750 |   45.843000 |
|   max | 89.000000 | 38.578759 | 194.000000 | 118.000000 | 26.211000 |  7.111918 | 83.482100 |   38.040000 | 82.100000 | 1256.083000 |

**Fasting blood levels:**
- Glucose = 130 mg/dL
- Insulin = 12 ulU/ml

High values for glucose, insulin, HOMA and MCP.1 can be associated with a patient having cancer, however the proportion is too low to make conclusive statements. Non-cancer patients show the highest value of glucose at 118 mg/dL, which is below the fasting threshold level.

![alt text](/images/hist_glucose.png)

High blood glucose levels are correlated to higher age groups, the highest records (201.0mg/dL and 196.0mg/dL) are associated with the older patients (86 years) the oldest patient being 89 years. Patients over the age of 50 with high fasting blood levels of glucose (> 130 mg/dL) and insulin (> 12 ulU/ml) all have cancer.

![alt text](/images/hist_age.png)

There is a threshold age range (40-50 years old) above which the chances of a patient getting cancer is much more likely. This is in concurrence with current breast cancer trends.

Plot for patients with cancer only for below 50 years old (0) and above 50 years old (1):

![alt text](/images/hist_BMI_age.png)

Patients with cancer above the age of 50 years have higher BMI scores. For patients without cancer above the age of 50 years, BMI scores fall within the range for patients below the age of 50 and not higher.

### Feature importance:

We predict that Glucose and Age will be important for models in predicting whether a patient has cancer or not.

### Improvement on data:

The objective of classifying the cancer status of a patient is difficult and will likely produce unreliable results because the dataset is small and needs additional features.

- More samples: for lower estimation variance and better predictive performance. Small sample size can lead to overfitting of the model.

- History of breast cancer: A woman who has had breast cancer in one breast is at an increased risk of developing cancer in her other breast.

- Family history of breast cancer: A woman has a higher risk of breast cancer if her relatives have had breast cancer, especially at a young age (before 40). 

- Genetic factors: Women with certain genetic mutations are at higher risk of developing breast cancer during their lifetime.

- Childbearing and menstrual history: The older a woman is when she has her first child, the greater her risk of breast cancer.

- Cancer label: status of the cancer the patients have is unknown. i.e. benign, malignant or in remission. These can determine the survivability of a patient. 


# 2. Model evaluation

### Lasso regularization

|    | Feature Name   |   Alpha = 1.000000 |   Alpha = 0.100000 |
|---:|:---------------|-------------------:|-------------------:|
|  0 | Age            |           -0.35212 |            0       |
|  1 | BMI            |           -0.28823 |            0       |
|  2 | Height         |            0.0154  |            0       |
|  3 | Glucose        |            1.57976 |            0.30405 |
|  4 | Insulin        |            0.2431  |            0       |
|  5 | HOMA           |            0       |            0       |
|  6 | Leptin         |           -0.20162 |            0       |
|  7 | Adiponectin    |            0.26581 |            0       |
|  8 | Resistin       |            0.44358 |            0       |
|  9 | MCP.1          |            0.19835 |            0       |

Lasso regression confirms that glucose has a stronger relevance in determining if a patient has cancer.

## Models

We will assess 5 different types of models based on initial accuracy scores:

- Logistic Regression
- Random Forest
- Decision Tree
- Gradient Boost
- XGBoost

### Change in test_size parameter has a large impact on model scores:

Train-test split proportion graph for Logistic Regression:

![alt text](/images/train_test_split_lgr.png)

Train-test sample size plots for the models show large variations in mean squared error (mse) values - ideally mse should decrease as training set becomes larger.  

### Accuracy scores:

After tuning the parameters of the models the accuracy scores are as follows:

|          | Logistic Regression | Random Forest | Decision Tree | Gradient Boost |  XGBoost   |
|---------:|--------------------:|--------------:|--------------:|---------------:|-----------:|
| Accuracy |        73.33%       |     83.33%    |      80%      |       80%      |   73.33%   |

###  Cross validation results are inconsistent:

Accuracy score for K-fold, times series and blocking time series cross validation for Logistic Regression:

|  Cross validation technique |        |        |        |        |        |
|----------------------------:|-------:|-------:|-------:|-------:|-------:|
|            K-fold           | 42.86% | 64.26% | 53.86% | 76.92% | 84.62% |
|          Time Series        | 72.73% | 36.36% | 72.72% | 72.73% | 45.45% |
|     Blocking Time series    |   100% | 33.33% | 66.75% | 66.67% | 66.67% | 

### Variation in data shows why cross validation is inconsistent:

Partial dependence plots are selected from the most important features from feature importance plots.

![alt text](/images/importance_gbc.png)

Partial dependence and ICE plots show a large variation in the data set - orange dotted line represents the average and blue lines represent individual data points.

![alt text](/images/partial_gbc.png)

### Learning curves show model fits:

![alt text](/images/learn_curve.png)

Learning curves for models Gradient Boost, XGBoost and Decision Tree show the models are overfitting on the training data - this makes it difficult to acccurately assess feature importance. 

### ROC curve comparison:

ROC curve is an important metric for the performance, the bigger the area under the curve the higher the performance.

![alt text](/images/roc_plot.png)

Decision tree model is not performing as well as the other models

### Confusion matrices:

It is more important for a model to correctly predict that a patient has cancer. The models show to be slightly more accurate for predicting cancer patients compared to for when prediciting non-cancer patients.

|                     | Prediction|Prediction| True label |
|--------------------:|----------:|---------:|-----------:|
|                     | No Cancer |  Cancer  |            |
|                                                         |
| Logistic Regression |    67%    |    33%   |  No Cancer |
|                     |    23%    |    79%   |   Cancer   |
|                                                         |
|   Gradient Boost    |    69%    |    31%   |  No Cancer |
|                     |    23%    |    77%   |   Cancer   |
|                                                         |
|       XGBoost       |    69%    |    31%   |  No Cancer |
|                     |    27%    |    73%   |   Cancer   |
|                                                         |
|    Random Forest    |    60%    |    40%   |  No Cancer |
|                     |    38%    |    62%   |   Cancer   |
|                                                         |
|    Decision Tree    |    64%    |    36%   |  No Cancer |
|                     |    23%    |    77%   |   Cancer   |


# 3. Conclusions:

### 1. Sample size is too small:

* This leads to inconsistent cross validation results.

* Variations in the data have a larger effect on models ability to generalise.

* Difficult to assess model evaluation metrics accurately when the results are not consistent.

### 2. Need better features: 

* Glucose is consitently an important feature in predicting if a patient has cancer or not.

* Feature importance assessments from all 5 models are not consistent - we would expect some degree of consensus towards data generalisation. This is important if we want models to perform well on new, unseen data. 

### 3. Models:

* Decision Tree, Gradient Boost and XGBoost are overfitting on the training data - they wont be able to accurately predict on new data.

* Logistic regression might perform better on new data. The confusion matrix shows it is better at predicting if a patient has cancer specifically, it may be more generalised than Decision Tree, Gradient Boost and XGBoost models.




