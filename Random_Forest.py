import logging

logger = logging.getLogger()

logging.basicConfig(filename='Results.log' , format='%(asctime)s | %(levelname)s: %(message)s', level=logging.NOTSET)



try:
    import pandas as pd
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score,confusion_matrix
except Exception as e:
    print(e)
else:
    pass

try:
    def data_load(path):
        data = pd.read_csv(path)
        logger.info(data.head())
        #print(data.head())
        logger.info(data.isna().any().sum())
        #print(data.isna().any().sum())
        data = pd.get_dummies(data, columns = ["3D_available", "Genre"], drop_first = True)
        predictors = data.loc[:,data.columns != "Start_Tech_Oscar"]
        target = data["Start_Tech_Oscar"]
        x_train,x_test,y_train,y_test =train_test_split(predictors,target,test_size = 0.2,random_state = 0)
        return x_train,x_test,y_train,y_test
except Exception as e:
    print(e)
else:
    pass
try:

    def model(x_train,x_test,y_train,y_test):
        rf_clf = RandomForestClassifier(n_estimators=500,n_jobs=1,random_state=42)
        rf_clf.fit(x_train,y_train)
        logger.info(confusion_matrix(y_test, rf_clf.predict(x_test)))
        #print(confusion_matrix(y_test, rf_clf.predict(x_test)))
        logger.info(accuracy_score(y_test, rf_clf.predict(x_test)))
        #print(accuracy_score(y_test, rf_clf.predict(x_test)))

# Evaluation on Training Data

        #print(confusion_matrix(y_train, rf_clf.predict(x_train)))
        logger.info(confusion_matrix(y_train, rf_clf.predict(x_train)))
        #print(accuracy_score(y_train, rf_clf.predict(x_train)))
        logger.info(accuracy_score(y_train, rf_clf.predict(x_train)))
except Exception as e:
    print(e)
else:
    pass

try:   
    def hyper_tunning(x_train,y_train):
        rf_clf_grid = RandomForestClassifier(n_estimators=500, n_jobs=1, random_state=42)

        param_grid = {"max_features": [4, 5, 6, 7, 8, 9, 10], "min_samples_split": [2, 3, 10]}
        grid_search = GridSearchCV(rf_clf_grid,param_grid,n_jobs = -1, cv = 5, scoring = 'accuracy')
        grid_search.fit(x_train,y_train)
        logger.info(f'{grid_search.best_params_},\n{grid_search.best_estimator_}')
        #print(grid_search.best_params_)
        cv_rf_clf = grid_search.best_estimator_
        return cv_rf_clf
except Exception as e:
    print(e)
else:
    pass

try:
    def final_model(cv_rf_clf,x_test,y_test):
        logger.info(confusion_matrix(y_test, cv_rf_clf.predict(x_test)))
        #print(confusion_matrix(y_test, cv_rf_clf.predict(x_test)))
        #print(accuracy_score(y_test, cv_rf_clf.predict(x_test)))
        logger.info(accuracy_score(y_test, cv_rf_clf.predict(x_test)))
except Exception as e:
    print(e)
else:
    pass


path = "E:\\Ensemble\\Datasets_EnsembleTechniques\\movies_classification.csv"
x_train,x_test,y_train,y_test = data_load(path)
model(x_train,x_test,y_train,y_test)

hyper_tune = hyper_tunning(x_train,y_train)
final_model(hyper_tune,x_test,y_test)