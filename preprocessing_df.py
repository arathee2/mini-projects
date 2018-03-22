import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn_pandas import DataFrameMapper  
from category_encoders import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn_pandas import CategoricalImputer

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
adult = pd.read_csv(url, names=['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'class'])

# eda
adult.head()
adult.info()
adult.isnull().any()

# divide numeric and categorical variables
num = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
cat = [name for name in adult.columns if name not in num + ['class']]

#adult.workclass = adult.workclass.str.strip()
#adult.education = adult.education.str.strip()
    
preprocess = DataFrameMapper([
        (['age'], [Imputer(), StandardScaler()]),
        (['fnlwgt'], [Imputer(), StandardScaler()]),
        (['education_num'], [Imputer(), StandardScaler()]),
        (['capital_gain'], [Imputer(), StandardScaler()]),
        (['capital_loss'], [Imputer(), StandardScaler()]),
        (['hours_per_week'], [Imputer(), StandardScaler()]),
        (['workclass'], [CategoricalImputer(), LabelEncoder()]),
        (['education'], [CategoricalImputer(), LabelEncoder()]),
        (['marital_status'], [CategoricalImputer(), LabelEncoder()]),
        (['occupation'], [CategoricalImputer(), LabelEncoder()]),
        (['relationship'], [CategoricalImputer(), LabelEncoder()]),
        (['race'], [CategoricalImputer(), LabelEncoder()]),
        (['sex'], [CategoricalImputer(), LabelEncoder()]),
        (['native_country'], [CategoricalImputer(), LabelEncoder()])
        ],
    df_out=True)
df = preprocess.fit_transform(adult) 
df.apply(lambda x: x.astype(float))
df.info()
df.shape
one_hot_encode = DataFrameMapper([
        (['age'], None),
        (['fnlwgt'], None),
        (['education_num'], None),
        (['capital_gain'], None),
        (['capital_loss'], None),
        (['hours_per_week'], None),
        (['workclass'], OneHotEncoder()),
        (['education'], OneHotEncoder()),
        (['marital_status'], OneHotEncoder()),
        (['occupation'], OneHotEncoder()),
        (['relationship'], OneHotEncoder()),
        (['race'], OneHotEncoder()),
        (['sex'], OneHotEncoder()),
        (['native_country'], OneHotEncoder())
        ],
    df_out=True)
df = one_hot_encode.fit_transform(df)
df = df.toarray()
df.shape
df.info()

X = df
y = adult['class'].values
X.shape
y.shape

# divide data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 4, stratify = y)

# logistic regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
model.score(X_test, y_test)

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names]

class CustomImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy='mean',filler='NA'):
       self.strategy = strategy
       self.fill = filler

    def fit(self, X, y=None):
       if self.strategy in ['mean','median']:
           if not all(X.dtypes == np.number):
               raise ValueError('dtypes mismatch np.number dtype is \
                                 required for '+ self.strategy)
       if self.strategy == 'mean':
           self.fill = X.mean()
       elif self.strategy == 'median':
           self.fill = X.median()
       elif self.strategy == 'mode':
           self.fill = X.mode().iloc[0]
       elif self.strategy == 'fill':
           if type(self.fill) is list and type(X) is pd.DataFrame:
               self.fill = dict([(cname, v) for cname,v in zip(X.columns, self.fill)])
       return self

    def transform(self, X, y=None):
       return X.fillna(self.fill)

num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num)),
        ('imputer', Imputer()),
        ('scaler', StandardScaler())
        ])
 
cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat)),
        ('imputer', CustomImputer(strategy='mode')),
        ('label_encoder', OrdinalEncoder()),
        ('one_hot_encoder', OneHotEncoder())
    ])

full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline)
    ])

full_pipeline.fit(adult)
df2 = full_pipeline.transform(adult)
df2.shape
df2.toarray()

X = df2
y = adult['class'].values
X.shape
y.shape

# divide data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 4, stratify = y)

# logistic regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
model.score(X_test, y_test)






