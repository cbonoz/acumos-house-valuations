import os
import shutil

import numpy as np
import scipy as sp
import matplotlib as mp
import matplotlib.pyplot as plt
import scipy.linalg.interpolative as sli
import seaborn as sns
import pandas as pd
import math
# import xgboost as xgb
import gc

from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
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

from acumos.modeling import Model, List, Dict, create_namedtuple, create_dataframe
from acumos.session import AcumosSession
# Session Endpoints for Acumos Challenge model upload and auth.
# session = AcumosSession(push_api="https://acumos-challenge.org/onboarding-app/v2/models",
#                         auth_api="https://acumos-challenge.org/onboarding-app/v2/auth")
session = AcumosSession()

print('done')


# In[145]:


REDFIN_TRAIN_CSV = os.path.join("assets","redfin_2018_8_boston.csv")
REDFIN_TEST_CSV = os.path.join("assets", "redfin_2018_active_boston.csv")


# In[146]:


def rename_col(x):
    return x.lower().replace(' ', '_').replace('$', 'cost').replace('/', '_per_')


# ### Main Redfin Acumos Model
#
# Using recently sold properties to predict current property values, includes qualitative (encoded) and quantitative metrics.
class RedfinAcumosModel:
    URL_COL = 'URL (SEE http://www.redfin.com/buy-a-home/comparative-market-analysis FOR INFO ON PRICING)'
    DROP_COLS = list(map(rename_col, [URL_COL, 'NEXT OPEN HOUSE START TIME', 'FAVORITE', 'INTERESTED', 'ZIP', 'SALE TYPE',
                 'NEXT OPEN HOUSE END TIME','STATUS', 'SOLD DATE', 'MLS#',
                 'SOURCE', 'ADDRESS','LATITUDE','LONGITUDE'])) # 'LISTING ID'
    ENCODED_COLS = list(map(rename_col, ['CITY', 'STATE', 'PROPERTY TYPE', 'LOCATION']))

    def __init__(self):
        # self.train = self.process_train_data(self.data)
        self.X = None
        self.y = None
        self.df = None
        return
        

    def get_formatted_test_cols(self, data):
        data_cols = set(data.columns.values)
        model_cols = list(data_cols - set(self.DROP_COLS))
        
        df_num = data.select_dtypes(exclude=[np.number])
        raw_cols = list(df_num.columns.values)
        
        model_cols = list(map(rename_col, model_cols))
        model_cols.remove('price')
        
        dtypes = list(map(lambda x: "List[str]" if x in raw_cols else "List[%s]" % str(data[x].dtype)[:-2], data))
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
        print("\n=== Merging Encoded Columns for Training ===\n")
        needed_cols = train_cols - test_cols
        extra_cols = test_cols - train_cols
#         print('Train data needs columns: %s' % needed_cols)
#         print('\nTrain data should remove columns: %s' % extra_cols)
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
    return data


train = reformat_train_data(pd.read_csv(REDFIN_TRAIN_CSV))
train_cols = train.columns.values
print(train_cols)
np.set_printoptions(precision=2) #2 decimal places
np.set_printoptions(suppress=True) #remove scientific notation

redfin = RedfinAcumosModel()

model_cols = redfin.get_formatted_test_cols(train)
print(model_cols)
# print(len(model_cols))

# items is the model_cols list from the previous slide
items = [('cost_per_square_feet', List[str]), ('baths', List[str]), ('beds', List[str]), ('square_feet', List[float]), ('property_type', List[float]), ('year_built', List[float]), ('lot_size', List[str]), ('hoa_per_month', List[float]), ('days_on_market', List[float]), ('location', List[float]), ('state', List[float]), ('city', List[float])]

HouseDataFrame = create_namedtuple('HouseDataFrame', items)


# here, an appropriate NamedTuple type is inferred from a pandas DataFrame
# HouseDataFrame = create_dataframe('HouseDataFrame', X_df)
print(HouseDataFrame.__doc__)

df = redfin.process_data(train, True)
redfin.df = df
redfin.df.info()

def appraise(data: HouseDataFrame) -> List[float]:
    res = pd.DataFrame([data], columns=HouseDataFrame._fields)
    return predict(res)

def predict(data):
    test_df = redfin.process_data(data)
    # Train by merging locations/columns found in the train dataframe.
    test_df = redfin.match_train(test_df)
    redfin.train_model()
    # print('Using %d Calculated Features for Valuation' % len(test_df.columns.values))
    return redfin.model.predict(test_df)

acumos_model = Model(appraise=appraise)
# session.push(model, MODEL_PATH) # usable with active credentials
print('Acumos %s created' % MODEL_PATH)

import os
from flask import request, jsonify, Flask
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)

PORT = 3001

@app.route('/hello')
def hello_world():
    return 'Hello, World!'

# Listen for conversation state change events.
@app.route('/predict', methods=['POST'])
def events():
    raw_data = request.data
    data = json.loads(raw_data)
    
    house_data = HouseDataFrame(data['cpsf'], data['baths'], data['beds'], data['sf'], data['prop_type'], data['year_built'],
                        data['lot_size'], data['hoa'], data['dom'], data['location'], data['state'], data['city'])
    result = acumos_model.appraise.inner(house_data)[0]
    print('received payload', house_data, 'prediction', result)
    return jsonify({'prediction': '$%.2f' % result})

# df = HouseDataFrame(100, 1, 2, 2000, 'Other', 2000, 1000, 1000, 10, 'Malden', 'MA', 'Boston')
sample_data = HouseDataFrame(100, 1, 1, 2700, 'Other', 2000, 1000, 
                    1000, 10, 'Malden', 'MA', 'Boston')
res1 = acumos_model.appraise.inner(sample_data)[0]
print('test prediction $%.2f' % res1)

if __name__ == '__main__':
      app.run(port=PORT)
