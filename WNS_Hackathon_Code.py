import pandas as pd
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout

train_dataset = pd.read_csv("C:/Users/Hi/Documents/WNS/train_LZdllcl.csv")
train_dataset['education'] = train_dataset.education.str.replace("'","")
train_dataset_excludeNan_edu = train_dataset.dropna(how='any', subset=['education'])
train_dataset_excludeNan_rtg = train_dataset.dropna(how='any', subset=['previous_year_rating'])
Edu_NaN_records = train_dataset[~train_dataset.index.isin(train_dataset_excludeNan_edu.index)]
Rtg_NaN_records = train_dataset[~train_dataset.index.isin(train_dataset_excludeNan_rtg.index)]

train_dataset_excludeNan = train_dataset_excludeNan.dropna(how='any', subset=['education','previous_year_rating'])
#Categorical data - transform to numerical form
X    = train_dataset_excludeNan.iloc[:,1:13]
X = X.loc[:, X.columns != 'education']
X = X.loc[:, X.columns != 'previous_year_rating'].values

Y = train_dataset_excludeNan.loc[:,['education','previous_year_rating']].values

for i in range(0,4):
    labelEncoder_X = LabelEncoder()
    X[:,i] = labelEncoder_X.fit_transform(X[:,i])
oneHotEncoder = OneHotEncoder(categorical_features = [0])
X = oneHotEncoder.fit_transform(X).toarray()
oneHotEncoder = OneHotEncoder(categorical_features = [9])
X = oneHotEncoder.fit_transform(X).toarray()
oneHotEncoder = OneHotEncoder(categorical_features = [44])
X = oneHotEncoder.fit_transform(X).toarray()
X = pd.DataFrame(X)
X = X.drop(columns=[8,41,45], axis=1)
X=X.values


#Education and prev_yr_rating as target variables - To identify the NaN (Alternative approach for Imputer)
for i in range(0,1):
    labelEncoder_Y = LabelEncoder()
    Y[:,i] = labelEncoder_Y.fit_transform(Y[:,i])

oneHotEncoderY = OneHotEncoder(categorical_features = 'all')
Y = oneHotEncoderY.fit_transform(Y).toarray()

Y_education = Y[:,0:3]
Y_rating = Y[:,3:8]


def predict_NaN(X,y_param):
    X_train, X_test, y_train, y_test = train_test_split(X, y_param)
    sc = StandardScaler()
    X_train = pd.DataFrame(sc.fit_transform(X_train))
    X_test = pd.DataFrame(sc.transform(X_test))
    #Initialising the ANN
    Nmodel = Sequential()
    #Adding Dense Layers
    Nmodel.add(Dense(units = 50, init = 'uniform', activation = 'relu', input_dim=50))
    Nmodel.add(Dense(units = 42, init = 'uniform', activation = 'relu'))
    Nmodel.add(Dropout(0.1))
    Nmodel.add(Dense(units = 34, init = 'uniform', activation = 'relu'))
    Nmodel.add(Dropout(0.1))
    Nmodel.add(Dense(units = 26, init = 'uniform', activation = 'relu'))
    Nmodel.add(Dropout(0.1))
    Nmodel.add(Dense(units = 18, init = 'uniform', activation = 'relu'))
    Nmodel.add(Dropout(0.1))
    Nmodel.add(Dense(units = 10, init = 'uniform', activation = 'relu'))
    Nmodel.add(Dense(units = y_param.shape[1], init = 'uniform', activation = 'sigmoid'))
    #Nmodel.add(Activation('sigmoid', name='activation'))
    Nmodel.summary()
    #Compile the model
    Nmodel.compile(optimizer= 'rmsprop', loss='binary_crossentropy', metrics = ['accuracy'])
    #Fitting the NN model
    #Nmodel.fit(X_train, y_train, epochs= 200, batch_size=32)
    history = Nmodel.fit(X_train, y_train, epochs= 50, batch_size=100, validation_data=[X_test,y_test])
    import matplotlib.pyplot as plt
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['X_train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    return Nmodel

#Predict the NaN under Education Column
X_pred = Edu_NaN_records.iloc[:,1:13]
X_pred = X_pred.loc[:, X_pred.columns != 'education']
X_pred = X_pred.loc[:, X_pred.columns != 'previous_year_rating'].values
for j in range(0,4):
    labelEncoder_X_pred = LabelEncoder()
    X_pred[:,j] = labelEncoder_X_pred.fit_transform(X_pred[:,j])
oneHotEncoder_X_pred = OneHotEncoder(categorical_features = [0])
X_pred = oneHotEncoder_X_pred.fit_transform(X_pred).toarray()
oneHotEncoder_X_pred = OneHotEncoder(categorical_features = [9])
X_pred = oneHotEncoder_X_pred.fit_transform(X_pred).toarray()
oneHotEncoder_X_pred = OneHotEncoder(categorical_features = [41])
X_pred = oneHotEncoder_X_pred.fit_transform(X_pred).toarray()
sc = StandardScaler()
X_pred = pd.DataFrame(sc.fit_transform(X_pred))
#Prediction of Education NaN records
eduModel = predict_NaN(X,Y_education)
Pred_Edu_NaN = eduModel.predict_classes(X_pred)
Edu_NaN_records = Edu_NaN_records.rename(columns={'education':'edu'})
Edu_NaN_records.insert(column='education', loc=3, value= Pred_Edu_NaN)
Edu_NaN_records = Edu_NaN_records.drop(columns='edu', axis=1)
Edu_NaN_records.education = Edu_NaN_records.education.map({0:'Bachelors',1:'Below Secondary', 2:'Masters & above'})
#Predict the NaN under Previous Year rating column
X_pred_rtg = Rtg_NaN_records.iloc[:,1:13]
X_pred_rtg = X_pred_rtg.loc[:, X_pred_rtg.columns != 'education']
X_pred_rtg = X_pred_rtg.loc[:, X_pred_rtg.columns != 'previous_year_rating'].values
for j in range(0,4):
    labelEncoder_X_pred = LabelEncoder()
    X_pred_rtg[:,j] = labelEncoder_X_pred.fit_transform(X_pred_rtg[:,j])
oneHotEncoder_X_pred = OneHotEncoder(categorical_features = [0])
X_pred_rtg = oneHotEncoder_X_pred.fit_transform(X_pred_rtg).toarray()
oneHotEncoder_X_pred = OneHotEncoder(categorical_features = [9])
X_pred_rtg = oneHotEncoder_X_pred.fit_transform(X_pred_rtg).toarray()
oneHotEncoder_X_pred = OneHotEncoder(categorical_features = [44])
X_pred_rtg = oneHotEncoder_X_pred.fit_transform(X_pred_rtg).toarray()
X_pred_rtg = pd.DataFrame(X_pred_rtg)
X_pred_rtg = X_pred_rtg.drop(columns=[8,41,45], axis=1)
X_pred_rtg=X_pred_rtg.values
sc = StandardScaler()
X_pred_rtg = pd.DataFrame(sc.fit_transform(X_pred_rtg))
#Prediction of Previous NaN records
rtgModel = predict_NaN(X,Y_rating)
Pred_rtg_NaN = rtgModel.predict_classes(X_pred_rtg)
Rtg_NaN_records = Rtg_NaN_records.rename(columns={'previous_year_rating':'rtg'})
Rtg_NaN_records.insert(column='previous_year_rating', loc=8, value= Pred_rtg_NaN)
Rtg_NaN_records = Rtg_NaN_records.drop(columns='rtg', axis=1)
Rtg_NaN_records.previous_year_rating = Rtg_NaN_records.previous_year_rating.map({0:1,1:2,2:3,3:4,4:5})
#Replace NaN
train_dataset1= train_dataset
train_dataset1 = pd.merge(train_dataset1, Rtg_NaN_records[['employee_id','previous_year_rating']], on=['employee_id'], how='outer')
train_dataset1 = pd.merge(train_dataset1, Edu_NaN_records[['employee_id','education']], on=['employee_id'], how='outer')
train_dataset1.insert(column='previous_year_rating', loc = 8, value=None)
train_dataset1['previous_year_rating_x'] = train_dataset1['previous_year_rating_x'].fillna(0)
train_dataset1['previous_year_rating_y'] = train_dataset1['previous_year_rating_y'].fillna(0)
train_dataset1['previous_year_rating'] = train_dataset1.previous_year_rating_x + train_dataset1.previous_year_rating_y
train_dataset1 = train_dataset1.drop(columns=['previous_year_rating_x','previous_year_rating_y'], axis=1)
train_dataset1.insert(column='education', loc = 8, value=None)
train_dataset1['education_x'] = train_dataset1['education_x'].fillna("")
train_dataset1['education_y'] = train_dataset1['education_y'].fillna("")
train_dataset1['education'] = train_dataset1.education_x + train_dataset1.education_y
train_dataset1 = train_dataset1.drop(columns=['education_x','education_y'], axis=1)
test_dataset = pd.read_csv("C:/Users/Hi/Documents/WNS/test_2umaH9m.csv")
test_dataset['education'] = test_dataset.education.str.replace("'","")
test_dataset_excludeNan_edu = test_dataset.dropna(how='any', subset=['education'])
test_dataset_excludeNan_rtg = test_dataset.dropna(how='any', subset=['previous_year_rating'])
Edu_tst_NaN_records = test_dataset[~test_dataset.index.isin(test_dataset_excludeNan_edu.index)]
Rtg_tst_NaN_records = test_dataset[~test_dataset.index.isin(test_dataset_excludeNan_rtg.index)]
test_dataset_excludeNan = test_dataset.dropna(how='any', subset=['education','previous_year_rating'])
#Categorical data - transform to numerical form
X_val    = test_dataset_excludeNan.iloc[:,1:13]
X_val = X_val.loc[:, X_val.columns != 'education']
X_val = X_val.loc[:, X_val.columns != 'previous_year_rating'].values
Y_val = test_dataset_excludeNan.loc[:,['education','previous_year_rating']].values
for i in range(0,4):
    labelEncoder_X = LabelEncoder()
    X_val[:,i] = labelEncoder_X.fit_transform(X_val[:,i])
oneHotEncoder = OneHotEncoder(categorical_features = [0])
X_val = oneHotEncoder.fit_transform(X_val).toarray()
oneHotEncoder = OneHotEncoder(categorical_features = [9])
X_val = oneHotEncoder.fit_transform(X_val).toarray()
oneHotEncoder = OneHotEncoder(categorical_features = [44])
X_val = oneHotEncoder.fit_transform(X_val).toarray()
X_val = pd.DataFrame(X_val)
X_val = X_val.drop(columns=[8,41,45], axis=1)
X_val=X_val.values
#Education and prev_yr_rating as target variables - To identify the NaN (Alternative approach for Imputer)
for i in range(0,2):
    labelEncoder_Y = LabelEncoder()
    Y_val[:,i] = labelEncoder_Y.fit_transform(Y_val[:,i])
oneHotEncoderY = OneHotEncoder(categorical_features = 'all')
Y_val = oneHotEncoderY.fit_transform(Y_val).toarray()
Y_val_education = Y_val[:,0:3]
Y_val_rating = Y_val[:,3:8]
#Validation set :- Predict the NaN under Education Column
X_val_pred = Edu_tst_NaN_records.iloc[:,1:13]
X_val_pred = X_val_pred.loc[:, X_val_pred.columns != 'education']
X_val_pred = X_val_pred.loc[:, X_val_pred.columns != 'previous_year_rating'].values
for j in range(0,4):
    labelEncoder_X_pred = LabelEncoder()
    X_val_pred[:,j] = labelEncoder_X_pred.fit_transform(X_val_pred[:,j])
oneHotEncoder_X_pred = OneHotEncoder(categorical_features = [0])
X_val_pred = oneHotEncoder_X_pred.fit_transform(X_val_pred).toarray()
oneHotEncoder_X_pred = OneHotEncoder(categorical_features = [9])
X_val_pred = oneHotEncoder_X_pred.fit_transform(X_val_pred).toarray()
oneHotEncoder_X_pred = OneHotEncoder(categorical_features = [41])
X_val_pred = oneHotEncoder_X_pred.fit_transform(X_val_pred).toarray()
sc = StandardScaler()
X_val_pred = pd.DataFrame(sc.fit_transform(X_val_pred))
X_val_pred = X_val_pred.drop(columns=[0], axis=1)
X_val_pred=X_val_pred.values
#Predict
#Prediction of Education NaN records
tsteduModel = predict_NaN(X_val,Y_val_education)
Pred_tst_Edu_NaN = tsteduModel.predict_classes(X_val_pred)
Edu_tst_NaN_records = Edu_tst_NaN_records.rename(columns={'education':'edu'})
Edu_tst_NaN_records.insert(column='education', loc=3, value= Pred_tst_Edu_NaN)
Edu_tst_NaN_records = Edu_tst_NaN_records.drop(columns='edu', axis=1)
Edu_tst_NaN_records.education = Edu_tst_NaN_records.education.map({0:'Bachelors',1:'Below Secondary', 2:'Masters & above'})
#Validation set - Predict the NaN under Previous Year rating column
X_val_pred_rtg = Rtg_tst_NaN_records.iloc[:,1:13]
X_val_pred_rtg = X_val_pred_rtg.loc[:, X_val_pred_rtg.columns != 'education']
X_val_pred_rtg = X_val_pred_rtg.loc[:, X_val_pred_rtg.columns != 'previous_year_rating'].values
for j in range(0,4):
    labelEncoder_X_pred = LabelEncoder()
    X_val_pred_rtg[:,j] = labelEncoder_X_pred.fit_transform(X_val_pred_rtg[:,j])
oneHotEncoder_X_pred = OneHotEncoder(categorical_features = [0])
X_val_pred_rtg = oneHotEncoder_X_pred.fit_transform(X_val_pred_rtg).toarray()
oneHotEncoder_X_pred = OneHotEncoder(categorical_features = [9])
X_val_pred_rtg = oneHotEncoder_X_pred.fit_transform(X_val_pred_rtg).toarray()
oneHotEncoder_X_pred = OneHotEncoder(categorical_features = [44])
X_val_pred_rtg = oneHotEncoder_X_pred.fit_transform(X_val_pred_rtg).toarray()
X_val_pred_rtg = pd.DataFrame(X_val_pred_rtg)
X_val_pred_rtg = X_val_pred_rtg.drop(columns=[8,41,45], axis=1)
X_val_pred_rtg=X_val_pred_rtg.values
sc = StandardScaler()
X_val_pred_rtg = pd.DataFrame(sc.fit_transform(X_val_pred_rtg))
#Prediction of Previous NaN records
valrtgModel = predict_NaN(X_val,Y_val_rating)
valPred_rtg_NaN = valrtgModel.predict_classes(X_val_pred_rtg)
Rtg_tst_NaN_records = Rtg_tst_NaN_records.rename(columns={'previous_year_rating':'rtg'})
Rtg_tst_NaN_records.insert(column='previous_year_rating', loc=8, value= valPred_rtg_NaN)
Rtg_tst_NaN_records = Rtg_tst_NaN_records.drop(columns='rtg', axis=1)
Rtg_tst_NaN_records.previous_year_rating = Rtg_tst_NaN_records.previous_year_rating.map({0:1,1:2,2:3,3:4,4:5})
#Replace NaN
test_dataset1= test_dataset
test_dataset1 = pd.merge(test_dataset1, Rtg_tst_NaN_records[['employee_id','previous_year_rating']], on=['employee_id'], how='outer')
test_dataset1 = pd.merge(test_dataset1, Edu_tst_NaN_records[['employee_id','education']], on=['employee_id'], how='outer')
test_dataset1.insert(column='previous_year_rating', loc = 8, value=None)
test_dataset1['previous_year_rating_x'] = test_dataset1['previous_year_rating_x'].fillna(0)
test_dataset1['previous_year_rating_y'] = test_dataset1['previous_year_rating_y'].fillna(0)
test_dataset1['previous_year_rating'] = test_dataset1.previous_year_rating_x + test_dataset1.previous_year_rating_y
test_dataset1 = test_dataset1.drop(columns=['previous_year_rating_x','previous_year_rating_y'], axis=1)
test_dataset1.insert(column='education', loc = 8, value=None)
test_dataset1['education_x'] = test_dataset1['education_x'].fillna("")
test_dataset1['education_y'] = test_dataset1['education_y'].fillna("")
test_dataset1['education'] = test_dataset1.education_x + test_dataset1.education_y
test_dataset1 = test_dataset1.drop(columns=['education_x','education_y'], axis=1)
#Actual coding part
train_dataset1 = train_dataset1.reindex_axis(['employee_id', 'department', 'region',	'education', 'gender', 'recruitment_channel', 'no_of_trainings',	'age',	'previous_year_rating',	'length_of_service',	'KPIs_met >80%',	'awards_won?', 'avg_training_score','is_promoted'], axis=1)
test_dataset1 = test_dataset1.reindex_axis(['employee_id', 'department', 'region',	'education', 'gender', 'recruitment_channel', 'no_of_trainings',	'age',	'previous_year_rating',	'length_of_service',	'KPIs_met >80%',	'awards_won?', 'avg_training_score'], axis=1)
#Categorical data - transform to numerical form
X_src    = train_dataset1.iloc[:,1:13].values
Y_src = train_dataset1.iloc[:,13].values
for i in range(0,5):
    labelEncoder_X = LabelEncoder()
    X_src[:,i] = labelEncoder_X.fit_transform(X_src[:,i])
oneHotEncoder = OneHotEncoder(categorical_features = [0])
X_src = oneHotEncoder.fit_transform(X_src).toarray()
oneHotEncoder = OneHotEncoder(categorical_features = [9])
X_src = oneHotEncoder.fit_transform(X_src).toarray()
oneHotEncoder = OneHotEncoder(categorical_features = [43])
X_src = oneHotEncoder.fit_transform(X_src).toarray()
X_src = pd.DataFrame(X_src)
X_src = X_src.drop(columns=[0,41,45], axis=1)
X_src=X_src.values
#oneHotEncoderY = OneHotEncoder(categorical_features = [0])
Y_src = Y_src.reshape(54808,1)
#Y_src = oneHotEncoderY.fit_transform(Y_src).toarray()
#Education and prev_yr_rating as target variables - To identify the NaN (Alternative approach for Imputer)
"""for i in range(0,1):
    labelEncoder_Y = LabelEncoder()
    Y[:,i] = labelEncoder_Y.fit_transform(Y[:,i])
oneHotEncoderY = OneHotEncoder(categorical_features = [0])
Y_src = oneHotEncoderY.fit_transform(Y_src).toarray()"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_src, Y_src)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = pd.DataFrame(sc.fit_transform(X_train))
X_test = pd.DataFrame(sc.transform(X_test))
import keras
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
    #Initialising the ANN
Nmodel = Sequential()
    #Adding Dense Layers
Nmodel.add(Dense(units = X_train.shape[1], init = 'uniform', activation = 'relu', input_dim=X_train.shape[1]))
Nmodel.add(Dense(units = 45, init = 'uniform', activation = 'relu'))
Nmodel.add(Dropout(0.1))
Nmodel.add(Dense(units = 34, init = 'uniform', activation = 'relu'))
Nmodel.add(Dropout(0.1))
Nmodel.add(Dense(units = 26, init = 'uniform', activation = 'relu'))
Nmodel.add(Dropout(0.1))
Nmodel.add(Dense(units = 18, init = 'uniform', activation = 'relu'))
Nmodel.add(Dropout(0.1))
Nmodel.add(Dense(units = 10, init = 'uniform', activation = 'relu'))
Nmodel.add(Dense(units = Y_src.shape[1], init = 'uniform', activation = 'sigmoid'))
    #Nmodel.add(Activation('sigmoid', name='activation'))
Nmodel.summary()
    #Compile the model
Nmodel.compile(optimizer= 'rmsprop', loss='binary_crossentropy', metrics = ['accuracy'])
    #Fitting the NN model
    #Nmodel.fit(X_train, y_train, epochs= 200, batch_size=32)
history = Nmodel.fit(X_train, y_train, epochs= 50, batch_size=100, validation_data=[X_test,y_test])
    # summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['X_train', 'test'], loc='upper left')
plt.show()
    # summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#PREDICTING ACTUAL VALIDATION SET
X_test_    = test_dataset1.iloc[:,1:13].values
#Y_xcludeNaN = train_dataset_excludeNan.iloc[:,13].values
for i in range(0,5):
    labelEncoder_X = LabelEncoder()
    X_test_[:,i] = labelEncoder_X.fit_transform(X_test_[:,i])
oneHotEncoder = OneHotEncoder(categorical_features = [0])
X_test_ = oneHotEncoder.fit_transform(X_test_).toarray()
oneHotEncoder = OneHotEncoder(categorical_features = [9])
X_test_ = oneHotEncoder.fit_transform(X_test_).toarray()
oneHotEncoder = OneHotEncoder(categorical_features = [43])
X_test_ = oneHotEncoder.fit_transform(X_test_).toarray()
X_test_ = pd.DataFrame(X_test_)
X_test_ = X_test_.drop(columns=[0,41,45], axis=1)
X_test_=X_test_.values
sc = StandardScaler()
X_test_ = sc.fit_transform(X_test_)
X_test_ = pd.DataFrame(X_test_)
#Predict X_test_
is_promotable_prediction = Nmodel.predict(X_test_)
is_promotable_pred = (is_promotable_prediction > 0.50)
Solution_frame = pd.DataFrame(is_promotable_pred, columns= ['is_promoted'])
Solution_frame.is_promoted = Solution_frame.is_promoted.astype(int)
Solution_frame.insert(loc= 0, column='employee_id', value=test_dataset1['employee_id'])
Solution_frame.to_csv("Solution.csv")
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)
