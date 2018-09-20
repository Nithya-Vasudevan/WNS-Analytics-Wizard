# WNS-Analytics-Wizard ML Hackathon
WNS ML Hackathon Solution

Problem Statement: Consider your Client is a Multinational Company and has 7 verticals. Every year promotion is given to certain Employees based on their Performance, Grade, Ratings etc. Create a Machine Learning Model to predict the Employees who will be promoted.

Given a set of 23491 employees, predict the employees who can be promoted.
Training data has 54808 records of employees with below details. 
Employee_ID, Dept, education, gender, recruitment_channel, region, previous_year_rating, kpi metrics>80%, training, avg_score etc.

Among them education and previous_year_rating columns alone have missing data in between.
Missing data could be dealt with Imputer package, by assigning mean or median or most_frequent values to the respective columns for missing records.

Another approach is to neglect the records if any of the column values are missing. But this approach may not support the model as their corresponding y_labels, say is_promoted column is affected adversely as most of their true values belong under this category.
So this has been handled in a different way.
The first step is to predict the missing values of education column using all other values except employee_id and is_promoted column

As employee_id column is an independent variable which won't make sense if we include it for training and prediction

Is_promoted (Y_label) is neglected because this column won't be available in the test dataset.

Similarly, with the same model previous_year_rating is predicted for both train and test datasets.

And finally when all missing data are handled, actual neural network model is created.

#read training datasets
#data preprocessing step
#LabelEncoder
#One Hot Encoding
#Define a Model
#Compile the Model
#Fit the Model 

#Read Test Dataset
#Data Preprocessing Step
#Label Encoder
#One Hot Encoding

#Predict the Test Datasets 

Write the resultant predicted class to a solution.csv file
