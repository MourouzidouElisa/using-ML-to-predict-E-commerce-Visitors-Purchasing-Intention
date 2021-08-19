# using-ML-to-predict-E-commerce-Visitors-Purchasing-Intention
Developed machine learning models to predict e-commerce visitors’  purchasing intention.
Classification problem with imbalance output.(Pandas, NumPy, Sklearn, Imblearn)

### Explanatory Data Analysis.
Data Exploration / Univariate Analysis / Bivariate Analysis for both numerical and categorical features. 

### Cleaning the Data.
Missing values / outliers 

### Data pre-processing.
- OneHotEncoder for categorical variables
- Multiple Scalers for the numerical variables (RobustScaler, StandardScaler, MinMaxScaler, PowerTransformer)

### First Model Implementations.
- LogisticRegression
- SupportVectorMachine
- XGBoost
- [x] Performance Evaluation.

### Extra preprocessing.
- Binning
- Feature selection with Multicollinearity

### Second Model Implementations.
- LogisticRegression
- SupportVectorMachine
- XGBoost
- [x] Performance Evaluation.

### Third Model implementations, balancing the data for the models with the extra preprocessing & for the models with the basic preprocessing
In these models, I balanced the data (not perfectly) to notice if balancing influences positively our models since we observed imbalanced data. We used appropriate parameters
for each of our classifiers separately. For LogisticRegression and for SupportVectorMachine i used the class_weight = {0:0.3, 1:0.7} and for the XGBClassifier
i used the scale_pos_weight = 5.
- [x] Performance Evaluation.

### Optimization and Hyperparameters Tuning to the best models.
- RandomizedSearchCV with Cross Validation (RepeatedStratifiedKFold)
