import pandas as pd 
import numpy as np
import time 

from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

class ML_Helpers(object):
    """Helpers for feature selection and model selection."""

    def __init__(self, df, y):
        self.df = df.dropna()
        self.y = y

    def select_features(self, drop_cols=[], cv=5):
        """
        Uses recursive feature elemination to identify the best features,
        Returns a two element list containing a list and df.

        drop_cols = list of column names.
        """
        
        # Exclude any non numeric columns and create X and y dfs.
        df = self.df.select_dtypes([np.number]).dropna().drop(drop_cols, axis=1)
        X = df.drop(self.y, axis=1)
        y = df[self.y]
        
        # RFECV requires a classifier with feature importances such as RandomForest
        clf = RandomForestClassifier() 
        
        # Check average feature importance
        fi_list = []
        for c in range(cv):
            cv_X, _, cv_y, _ = train_test_split(X, y, test_size = 1/cv)
            clf.fit(cv_X, cv_y)
            fi_list.append(clf.feature_importances_)
        
        # Create list of importances standarised by the maximum average value across cv iterations.
        cv_fi = np.mean(fi_list, axis=0)  
        df_fi = pd.DataFrame(cv_fi, index=X.columns, columns=['feat_imp'])/cv_fi.max()
        df_fi = df_fi.sort_values(by='feat_imp', ascending=False)
        
        # Fit RFECV on RandomForest object and pull out best features.
        selector = RFECV(clf, cv=cv)
        selector.fit(X, y)
        best_features = X.columns[selector.support_]
        
        return [best_features, df_fi]    

    
    def select_model(self, model_list, features):
        """
        Performs GridSearch for model_list and selects the best estimator from model list,
        Returns a dictionary.

        model_list = a list of dictionaries.
        features = list of column names
        """
        X = self.df[features]
        y = self.df[self.y]

        model_list = model_list
        
        best_score = -1
        for model in model_list:
            start = time.time()
            print('Model: {}'.format(model['name']))
            grid = GridSearchCV(
                model['estimator'], param_grid=model['hyperparameters']
            )
            grid.fit(X, y)
            model['best_params'] = grid.best_params_
            model['best_score'] = grid.best_score_
            model['best_estimator'] = grid.best_estimator_
            end = time.time()
            
            print('Score: {}'.format(model['best_score']))
            print('Time: {} seconds'.format(end-start), '\n')
            
            if model['best_score'] > best_score:
                best_score, best_model = model['best_score'], model

        print('Best Model: {}, Best Score: {}.'.format(best_model['name'], best_score)) 

        return best_model