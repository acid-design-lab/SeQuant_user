from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from lazypredict.Supervised import LazyRegressor
from lazypredict.Supervised import LazyClassifier


class NovaPredictTools:

    def Lazyregressor_vae(self, features, target,
                      size_of_test=0.2,
                      scaler = MinMaxScaler(),
                      random_st = 0
                       ):
        X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size = size_of_test, random_state = random_st)
        
        scaler.fit(X_train)
        x_train = scaler.transform(X_train)
        x_test = scaler.transform(X_test)

        clf = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
        train, test = clf.fit(x_train, x_test, Y_train, Y_test)
        test_mod = test.iloc[:-1, :]
        return test_mod



    def LazyClass_vae(self, features, target,
                      size_of_test=0.2,
                      scaler = MinMaxScaler(),
                      use_scaler = True,
                      random_st = 0
                       ):
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = size_of_test, random_state = random_st)
        if use_scaler:
            scaler.fit(X_train)
            sc_x_train = scaler.transform(X_train)
            scaled_test_x_ = scaler.transform(X_test)

            clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
            models, predictions = clf.fit(sc_x_train, scaled_test_x_, y_train, y_test)
            return models
        
        else:
            clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
            models, predictions = clf.fit(X_train, X_test, y_train, y_test)
            
            return models
        
