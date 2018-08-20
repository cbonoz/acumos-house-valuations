
# coding: utf-8

# <img src="assets/assistant_logo.png"/>
# 
# # Acumos Property Assistant
# 
# What's my Home Worth?
# 
# Many real estate agents follow a notion of comps (or comparables) when pricing a home. Problem here is that this is often subject to an individual realtors' biases and often only looks at 2-3 properties in the area as a comparison.
# 
# What's my home worth is a machine learning model deployed on the Acumos store for pricing your home based on fundamental property characteristics.
# 
# More specifically, we survey the intrinsic properties of a home, such as number of bathrooms, number of bedrooms, square footage, and location, as sampled from the Redfin website (www.redfin.com), and allow users to price their home using a large dataset of collected properties.
# 
# By no means is this an exhaustive model, but should give users as idea of what their home (or someone else's home) is at worth in the current market.
# 
# 
# 

# In[18]:


import os
import shutil

import numpy as np
import scipy as sp
import matplotlib as mp
mp.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
# import xgboost as xgb
import gc

from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model
from scipy import stats
# from skimage import color

from sklearn.linear_model import LinearRegression, LogisticRegression, SGDRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

pw = os.environ['ACUMOS_PASSWORD']
user = os.environ['ACUMOS_USERNAME']

MODEL_PATH = "Acumos Property Assistant"

pd.set_option('display.float_format', lambda x: '%.2f' % x)
mp.use('TkAgg')

from acumos.modeling import Model, List, Dict, create_namedtuple, create_dataframe
from acumos.session import AcumosSession
# Session Endpoints for Acumos Challenge model upload and auth.
# session = AcumosSession(push_api="https://acumos-challenge.org/onboarding-app/v2/models",
#                         auth_api="https://acumos-challenge.org/onboarding-app/v2/auth")
session = AcumosSession()

print('starting demo')


# In[19]:


REDFIN_TRAIN_CSV = os.path.join("assets","redfin_2018_8_boston.csv")
REDFIN_TEST_CSV = os.path.join("assets", "redfin_2018_active_boston.csv")


# In[20]:


def rename_col(x):
    return x.lower().replace(' ', '_').replace('$', 'cost').replace('/', '_per_')
class RedfinAcumosModel:
    URL_COL = 'URL (SEE http://www.redfin.com/buy-a-home/comparative-market-analysis FOR INFO ON PRICING)'
    DROP_COLS = list(map(rename_col, [URL_COL, 'NEXT OPEN HOUSE START TIME', 'FAVORITE', 'INTERESTED', 'ZIP', 'SALE TYPE',
                 'NEXT OPEN HOUSE END TIME','STATUS', 'SOLD DATE', 'MLS#', '$/SQUARE_FEET',
                 'SOURCE', 'ADDRESS','LATITUDE','LONGITUDE'])) # 'LISTING ID'
    ENCODED_COLS = list(map(rename_col, ['CITY', 'STATE', 'PROPERTY TYPE', 'LOCATION']))

    def __init__(self):
        # self.train = self.process_train_data(self.data)
        self.X = None
        self.y = None
        self.df = None
        return
        

    def get_formatted_test_cols(self, data):
        data_cols = list(data.columns.values)
        model_cols = list(filter(lambda x: x not in set(self.DROP_COLS), data_cols))
        
        df_num = data.select_dtypes(exclude=[np.number])
        raw_cols = list(df_num.columns.values)
        
        model_cols = list(map(rename_col, model_cols))
        model_cols.remove('price')
        
        dtypes = list(map(lambda x: "str" if x in raw_cols else "%s" % str(data[x].dtype)[:-2], model_cols))
        zipped_cols = list(zip(model_cols, dtypes))
        res = str(zipped_cols).replace("'List", "List").replace("]'", "]")
        return res

    
    def encode_onehot(self, df, cols):
        """
        One-hot encoding is applied to columns specified in a pandas DataFrame.

        Modified from: https://gist.github.com/kljensen/5452382

        Details:

        http://en.wikipedia.org/wiki/One-hot
        http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html

        @param df pandas DataFrame
        @param cols a list of columns to encode
        @return a DataFrame with one-hot encoding
        """
        vec = DictVectorizer()

        vec_data = pd.DataFrame(vec.fit_transform(df[cols].to_dict(orient='records')).toarray())
        vec_data.columns = vec.get_feature_names()
        vec_data.index = df.index

        df = df.drop(cols, axis=1)
        df = df.join(vec_data)
        self.vec_data_cols = vec_data.columns.values
                     
        return df
    
    
    def process_data(self, data, training=False):
        # First interpolate on missing values.
        if training:
            data = data.interpolate()
            
        # Remove any remaining NaN and standardize string fields.
        data['lot_size'] = data['lot_size'].replace(np.NaN, 0)
        data['city'] = data['city'].astype(str).apply(lambda x: x.lower())
        data['property_type'] = data['property_type'].astype(str).apply(lambda x: x.lower())

        # Convert string columns to one hot encoding.
        data = self.encode_onehot(data, RedfinAcumosModel.ENCODED_COLS)

        # Drop the original location value (after being encoded).
        if 'location' in data.columns.values:
            data = data.drop('location', axis=1)
            
        # Drop unique/unknown location cols.
        location_cols = [x for x in data.columns.values if 'location' in x]
        for c in location_cols:
            if sum(data[c]) < 2:
                data = data.drop(c, axis=1)

        # print('processed data columns: %s' % data.columns.values)
        if training:
            self.df = data
            self.X = self.df.drop('price', axis=1)
            self.y = self.df['price']
        return data

    def get_encodings(self, data):
        self.encoders = {}
        for c in RedfinAcumosModel.ENCODED_COLS:
            le = LabelEncoder()
            data[c] = data[c].apply(lambda x: x if not pd.isnull(x) else 'Null')
            le.fit(data[c])
            data[c] = le.transform(data[c])
            self.encoders[c] = le
    
    def transform(self, data):
        for c in RedfinAcumosModel.ENCODED_COLS:
            data[c] = data[c].apply(lambda x: x if not pd.isnull(x) else 'Null')
            data[c] = self.encoders[c].transform(data[c])
        return data
    
    def inverse_transform(self, data):
        for c in RedfinAcumosModel.ENCODED_COLS:
            data[c] = self.encoders[c].inverse_transform(data[c])
        return data
    
    def cat_correlation(self, category, transform_func=None):
        cat = self.df[category]
        if transform_func:
            cat = cat.apply(lambda x: transform_func(x))
        cat_df = pd.get_dummies(cat).join(self.df['price']).astype(int)
        if 'nan' in cat_df.columns.values:
            del cat_df['nan']
        levels =  len(cat_df.columns.values)
        if levels > 60:
            print('%s: too many unique categories %d' % (category, levels))
            return cat_df
        corr = cat_df.corr()
        plt.figure(figsize=(16, 16))
        sns.heatmap(corr, vmax=1, square=True)
        return cat_df
    
    def match_train(self, test_df):
        # Merge test cols with train_cols
        test_cols = set(test_df.columns.values)
        train_cols = set(self.X.columns.values)
        # print("\n=== Merging Encoded Columns for Training ===\n")
        needed_cols = train_cols - test_cols
        extra_cols = test_cols - train_cols
        # print('Train data needs columns: %s' % needed_cols)
        # print('\nTrain data should remove columns: %s' % extra_cols)
        if extra_cols:
            test_df = test_df.drop(extra_cols, axis=1)
        if needed_cols: # add the necessary columns to match test set and fill with zeroes
            test_df = test_df.reindex(columns=self.X.columns.values, fill_value=0)
        
        return test_df
    
    def train_model(self):
        # self.model = Pipeline([('poly', PolynomialFeatures(degree=2)), ('linear', LinearRegression())])
        self.model = LinearRegression()
        # model = LinearRegression()
        # print('Final cols: %s' % ', '.join(self.X.columns.values))
        self.model.fit(self.X, self.y)

    def predict_xgb(self, data, d=4, est=200):
        self.clf = xgb.XGBRegressor(max_depth=d, n_estimators=est)
        self.clf.fit(self.X, self.y)
        pred = self.clf.predict(data)
        return pred
    
# Utility functions
def print_loc_cols(df):
    cols = [x for x in df.columns.values if 'location' in x]
    for c in cols:
        print("%40s: %d" % (c, sum(df[c])))

def convert_int(x):
    try:
        return int(x)
    except:
        return float('nan')

def reformat_train_data(data):
    data = data.rename(rename_col, axis='columns')
    data = data.drop(RedfinAcumosModel.DROP_COLS, axis=1)
    data = data[(data['price'] < 1.5*10**6) & (data['price'] > 10**5) & (data['property_type'] != 'Vacant Land')]
    return data

def reformat_test_data(data):
    data = data.rename(rename_col, axis='columns')
    data = data.drop(RedfinAcumosModel.DROP_COLS, axis=1)
    data = data.interpolate()
    return data


train = reformat_train_data(pd.read_csv(REDFIN_TRAIN_CSV))
train_cols = train.columns.values
print(train_cols)
np.set_printoptions(precision=2) #2 decimal places
np.set_printoptions(suppress=True) #remove scientific notation



# This is the list of core property types as used by Redfin

# In[23]:


set(train['property_type'])


# In[24]:


redfin = RedfinAcumosModel()

model_cols = redfin.get_formatted_test_cols(train)
print(model_cols)
# print(len(model_cols))


# ### These are the required dataframe input arguments for the model.

# In[25]:

# items is the model_cols list from the previous slide
items = [('baths', float), ('beds', float), ('square_feet', float), ('property_type', str), ('year_built', float), 
         ('lot_size', float), ('hoa_per_month', float), ('days_on_market', float), ('location', str), ('state', str), ('city', str)]

HouseDataFrame = create_namedtuple('HouseDataFrame', items)

# here, an appropriate NamedTuple type is inferred from a pandas DataFrame
# HouseDataFrame = create_dataframe('HouseDataFrame', X_df)
print(HouseDataFrame.__doc__)

# In[26]:


df = redfin.process_data(train, True)
redfin.df = df
redfin.df.info()

# These are the list of actual encoded columns to be used in the model


raw_test_df = pd.read_csv(REDFIN_TEST_CSV)
test_data = reformat_test_data(raw_test_df)
print(test_data.info())


# In[32]:


# raw_test_df[raw_cols] = raw_test_df[raw_cols].astype(str)


# In[ ]:





# In[33]:

def predict(data):
    test_df = redfin.process_data(data)
    # Train by merging locations/columns found in the train dataframe.
    test_df = redfin.match_train(test_df)
    redfin.train_model()
    # print('Using %d Calculated Features for Valuation' % len(test_df.columns.values))
    return redfin.model.predict(test_df)

def appraise_multiple(data: List[HouseDataFrame]) -> List[float]:
    res = pd.DataFrame(data, columns=HouseDataFrame._fields)
    return predict(res)

def appraise(data: HouseDataFrame) -> List[float]:
    return appraise_multiple([data])

# In[34]:


# results = predict(test_data)
acumos_test_data = pd.read_csv('assets/test_data.csv')
acumos_test_data = acumos_test_data.interpolate()
print(acumos_test_data.columns.values)
results = predict(acumos_test_data)


# In[35]:


result_df = pd.DataFrame()
result_df['address'] = raw_test_df['ADDRESS']
result_df['city'] = raw_test_df['CITY']
result_df['state'] = raw_test_df['STATE']
result_df['listing_price'] = raw_test_df['PRICE']
result_df['prediction'] = results

print("First 20 Predictions of Active Listings\n===")
print(result_df[:20])

print("\nMean Estimated Price of ALL current listings: $%.2f" % np.mean(results))

result_df.to_csv('assets/active_predictions.csv')
print('write predictions to csv')


# In[36]:


raw_test_df['predicted'] = results


# In[37]:


print(model_cols)


# In[38]:


# print(redfin.X.columns.values)
print(results)


# In[39]:


cols = list(filter(lambda x: 'property' not in x and 'location' not in x and 'city' not in x, redfin.X.columns.values))

# print(test_df[:10])
print(cols)


# In[40]:


acumos_model = Model(appraise=appraise, appraise_multiple=appraise_multiple)
if os.path.isdir(MODEL_PATH):
    shutil.rmtree(MODEL_PATH)
session.dump(acumos_model, MODEL_PATH, '.')
# session.push(model, MODEL_PATH) # usable with active credentials
print('Acumos %s saved' % MODEL_PATH)


# ### Sanity Check
# 
# As a sanity check, we test Run the model on the existing training data (should be similar estimates for price).

# In[41]:


redfin.train_model()
results = redfin.model.predict(redfin.X)
n = 50
zipped_data = list(zip(list(redfin.y[:n]), results[:n]))
for pair in zipped_data:
    print("Actual: $%.2f, Predicted: $%.2f" % (pair[0], pair[1]))


# In[42]:


from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(redfin.y, results))
rms

# Reasonable, considering many homes can be in the 1M+ range.

# In[43]:


# Example appraise usage externally, invoke the acumos_model object from your server.

df = HouseDataFrame(1, 1, 1000, 'Other', 2000, 1000, 
                    1000, 10, 'Malden', 'MA', 'Boston')

res = acumos_model.appraise.inner(df)
print("$%s" % res)

# In[44]:

# Example appraise multiple usage externally, invoke the acumos_model object from your server.

df = HouseDataFrame(1, 1, 1000, 'Other', 2000, 1000, 
                    1000, 10, 'Malden', 'MA', 'Boston')

df2 = HouseDataFrame(1, 3, 1400, 'Other', 2000, 1000, 
                    1000, 10, 'Malden', 'MA', 'Boston')
ex_data = [df, df2]

# In[45]:


res = acumos_model.appraise_multiple.inner(ex_data)
print("$%s" % res)

