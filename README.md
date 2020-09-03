# Credit-AI

**The goal here is to build an end to end automated Machine Learning solution where a user will be able to predict whether a bank customer should be approved for attaining the credit card or not. The user is only need to give the value of feature variables and the model will able to predict the binary outcome (Approve/ Not Approve). The model will be able take care of all intermediate functionalities like cross validation, hyper parameter tuning, algorithm selection etc.**

# Data set

https://archive.ics.uci.edu/ml/datasets/Credit+Approval

Data Description
The client will send data in multiple sets of files in batches at a given location. Data will contain different classes of Credit Approval and 15 columns of different values.
"Class" column will have two unique values “+’’ & “-”

Apart from training files, we also require a "schema" file from the client, which contains all the relevant information about the training files such as:
 Number of Columns, Name of the Columns, and their datatype.
 
# Data Validation 
In this step, we perform different sets of validation on the given set of training files.  

1. Number of Columns - We validate the number of columns present in the files, and if it doesn't match with the value given in the schema file, then the file is moved to "Rejected folder" else moved to “Validate folder”

      a) For training: training_data_reject,training_data_validate

      b) For predicting: predicting_data_reject,predicting_data_validate

2. Name of Columns - The name of the columns is validated and should be the same as given in the schema file. If not, then the file is moved to "Rejected folder".

3. The datatype of columns - The datatype of columns is given in the schema file. This is validated when we insert the files into Database. If the datatype is wrong, then the file is moved to "Rejected folder".

4. Null values in columns - If any of the columns in a file have all the values as NULL or missing, we discard such a file and move it to "Rejected folder".



# Data Insertion in Database
 
1) Database Creation and connection - Create a database with the given name passed. If the database is already created, open the connection to the database. 
2) Table creation in the database - Table with name - "Train Data", is created in the database for inserting the files in the "Validate Folder" based on given column names and datatype in the schema file. If the table is already present, then the new table is not created and new files are inserted in the already present table as we want training to be done on new as well as old training files.**     
3) Insertion of files in the table - All the files in the "Validate Folder" are inserted in the above-created table. If any file has invalid data type in any of the columns, the file is not loaded in the table and is moved to "Rejected Folder".
 
Model Training 
1) Data Export from Db - The data in a stored database is exported as a CSV file to be used for model training.
2) Data Preprocessing   
   a) Drop columns not useful for training the model. Such columns were selected while doing the EDA.
   
   b) Replace the invalid values(‘?’) with numpy “nan” so we can use imputer on such values.
   
   c) Encode the categorical values
   
   d) Check for null values in the columns. If present, impute the null values using the KNN imputer.
   
   e) Top four feature is selected with Selectkbest & chi2
   
3) Model Selection - After feature selection, we find the best model for each cluster. We are using two algorithms, "Random Forest" and "Xgboost". For each cluster, both the algorithms are passed with the best parameters derived from GridSearch. We calculate the AUC scores for both models and select the model with the best score. Similarly, the model is selected for each cluster. All the models for every cluster are saved for use in prediction. 
 
# Prediction Data Description
 The client will send data in multiple sets of files in batches at a given location. Data will contain different classes of Credit Approval and 15 columns of different values.
Apart from prediction files, we also require a "schema" file from the client, which contains all the relevant information about the training files such as:
 Number of Columns, Name of the Columns, and their datatype.

Data Validation  
In this step, we perform different sets of validation on the given set of training files.  
1. Number of Columns - We validate the number of columns present in the files, and if it doesn't match with the value given in the schema file, then the file is moved to "Rejected folder" else moved to “Validate folder”
  a) For training: training_data_reject,training_data_validate
  b) For predicting: predicting_data_reject,predicting_data_validate

2. Name of Columns - The name of the columns is validated and should be the same as given in the schema file. If not, then the file is moved to "Rejected folder".

3. The datatype of columns - The datatype of columns is given in the schema file. This is validated when we insert the files into Database. If the datatype is wrong, then the file is moved to "Rejected folder".

4. Null values in columns - If any of the columns in a file have all the values  as NULL or missing, we discard such a file and move it to "Rejected folder". 

# Data Insertion in Database 

1) Database Creation and connection - Create a database with the given name passed. If the database is already created, open the connection to the database. 

2) Table creation in the database - Table with name - "Predict Data", is created in the database for inserting the files in the "Validate Folder" based on given column names and datatype in the schema file. If the table is already present, then the new table is not created and new files are inserted in the already present table as we want training to be done on new as well as old training files.     

3) Insertion of files in the table - All the files in the "Validate Folder" are inserted in the above-created table. If any file has invalid data type in any of the columns, the file is not loaded in the table and is moved to "Rejected Folder".


# Prediction 
 
1) Data Export from Db - The data in the stored database is exported as a CSV file to be used for prediction.
2) Data Preprocessing   
   
   a) Drop columns not useful for training the model. Such columns were selected while doing the EDA.
   
   b) Replace the invalid values with numpy “nan” so we can use imputer on such values.
   
   c) Encode the categorical values
   
   d) Check for null values in the columns. If present, impute the null values using the KNN imputer.
   
   e)top four feature is selected with Selectkbest & chi2
3) Prediction -  the best model is loaded and is used to predict the data .
4) Once the prediction is made, the predictions along with the original names before label encoder are saved in a CSV file at a given location and the location is returned to the client.

# Data Profiling
After reading the data, automatically the following details should be shown:

a)	The number of rows

b)	The number of columns

c)	Number of missing values per column and their percentage

d)	Total missing values and it’s percentage

e)	Number of categorical columns and their list

f)	Number of numerical columns and their list

g)	Number of duplicate rows

h)	Number of columns with zero standard deviation and their list

i)	Size occupied in RAM

# EDA

**Statistics Based EDA**
1) VIF

2) Correlation

3) Column contributions/ importance

4) Chi Square test

5) Z test

**Graph Based EDA**

1) Correlation Heatmaps

2) Check for balance/imbalance

3) Count plots

4) Boxplot for outliers

5) Piecharts for categories

6) Line charts for  trends

7) Barplots

8) Area Charts

9) Stacked charts

10) Scatterplot

# Data Transformers( Pre-processing steps)

a) Null value handling

b) Categorical to numerical

c) Imbalanced data set handling

d) Normalisation

e) Outlier detection

f) Data Scaling/ Normalisation

g) Feature Selection: https://scikit-learn.org/stable/auto_examples/index.html#feature-selection

# ML Model Selection

3 Models—Logistic regression, Naive bayes classification, K nearest neighbour classification

**Phase 1**

Model Selection criteria : Accuracy 

**Phase 2**

Model Selection criteria : F1 Score 

# Testing Modules
Divide the training data itself into  train and test sets
Use test data to have tests run on the three best models
Give the test report

a)	Accuracy

b)	Precision

c)	Recall

d)	F1 Score

# Logging

a) Separate Folder for logs

b) Logging of every step

c) Entry to the methods

d) Exit from the methods with success/ failure message

e) Error message Logging

f) Model comparisons

g) Training start and end

h) Prediction start and end

i) Achieve asynchronous logging

j) Options for Logging in DB

h) Options for Log Publish




















