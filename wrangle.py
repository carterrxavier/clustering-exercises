import numpy as np
import pandas as pd
from env import host, user ,password
from sklearn.model_selection import train_test_split
import os


def get_connection(db, user = user, host = host, password = password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'
    
    
def get_telco_tenure():
    '''
    This function gets the tenure information from the telco data set for customers with 2 year contracts
    '''
    file_name = 'telco_tenure.csv'
    if os.path.isfile(file_name):
        return pd.read_csv(file_name)
    
    else:
        query =  '''
        select customer_id, monthly_charges, tenure, total_charges
        from customers
        where contract_type_id = 3
        '''
    df = pd.read_sql(query, get_connection('telco_churn'))  
    
    #replace white space with nulls
    df = df.replace(r'^\s*$', np.NaN, regex=True)
    
    df.to_csv(file_name, index = False)
    return df

def clean_telco_tenure(df):
    '''
    cleans telco tenure data
    
    '''
    #fill total charges with monthly charges
    df['total_charges'].fillna(df['monthly_charges'], inplace = True)
    
    #convert total_charges object type into a float
    df['total_charges'] = pd.to_numeric(df['total_charges'],errors='coerce')
    
    #change tenure from zero to one
    df.loc[df['tenure'] == 0, 'tenure'] = 1
    
    return df
    

def get_zillow_data():
    '''
    This function gets the zillow data needed to predict single unit properities.
    '''
    file_name = 'zillow.csv'
    if os.path.isfile(file_name):
        return pd.read_csv(file_name)
    
    else:
        query =  '''
        select bedroomcnt,bathroomcnt,calculatedfinishedsquarefeet,taxvaluedollarcnt,yearbuilt,taxamount, fips, propertylandusetypeid from properties_2017
        where propertylandusetypeid = 261
        '''
    df = pd.read_sql(query, get_connection('zillow'))  
    
    #replace white space with nulls
    df = df.replace(r'^\s*$', np.NaN, regex=True)
    
    df.to_csv(file_name, index = False)
    return df



def get_all_zillow_data():
    '''
    This function gets the zillow data needed to predict single unit properities.
    '''
    file_name = 'zillow.csv'
    if os.path.isfile(file_name):
        return pd.read_csv(file_name)
    
    else:
        query =  '''
          select *
from properties_2017
join (select parcelid, logerror, max(transactiondate) as transactiondate
FROM predictions_2017 group by parcelid, logerror) as pred_2017 using(parcelid)
#left join predictions_2017 on properties_2017.parcelid = predictions_2017.parcelid
left join airconditioningtype using(airconditioningtypeid)
left join architecturalstyletype using(architecturalstyletypeid)
left join buildingclasstype using(buildingclasstypeid)
left join heatingorsystemtype using(heatingorsystemtypeid)
left join propertylandusetype using(propertylandusetypeid)
left join storytype using(storytypeid)
left join typeconstructiontype using(typeconstructiontypeid)
where properties_2017.latitude is not null
and properties_2017.longitude is not null;
     
        '''
    df = pd.read_sql(query, get_connection('zillow'))  
    
    df.set_index('parcelid', inplace=True)
    
    #replace white space with nulls
    df = df.replace(r'^\s*$', np.NaN, regex=True)
    
    df.to_csv(file_name, index = False)
    return df



###############################################

def handle_outliers(df , col, lquan, upquan):
    q1 = df[col].quantile(lquan)
    q3 = df[col].quantile(upquan)
    iqr = q3-q1 #Interquartile range
    lower_bound  = q1-1.5*iqr
    upper_bound = q3+1.5*iqr
    if lower_bound < 0:
        lower_bound = 0
    if upper_bound > df[col].max():
        upper_bound = df[col].max()
    df_out = df.loc[(df[col] > lower_bound) & (df[col] < upper_bound)]
    return df_out



###############################################


def clean_zillow_data(zillow_df):
    '''
    this function cleans data for zillow data 
    '''
    zillow_df = zillow_df.dropna(axis=0, subset=['bedroomcnt'])
    zillow_df = zillow_df.dropna(how='all')
    zillow_df = zillow_df.dropna(axis=0, subset=['calculatedfinishedsquarefeet'])
    zillow_df = zillow_df.dropna(axis=0, subset=['taxrate'])
    zillow_df['taxvaluedollarcnt'].fillna(zillow_df['taxvaluedollarcnt'].mean(), inplace = True)
    zillow_df['taxamount'].fillna(zillow_df['taxamount'].mean(), inplace = True)
    mode = zillow_df[(zillow_df['yearbuilt'] > 1947) & (zillow_df['yearbuilt'] <= 1957)].yearbuilt.mode()
    zillow_df['yearbuilt'].fillna(value=mode[0], inplace = True)
    
    return zillow_df
    
  
    
def split_for_model(df):
    '''
    This function take in the telco_churn data acquired,
    performs a split into 3 dataframes. one for train, one for validating and one for testing 
    Returns train, validate, and test dfs.
    '''
    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=765)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=231)
    
    print('train{},validate{},test{}'.format(train.shape, validate.shape, test.shape))
    return train, validate, test


def get_mall_customers():
    '''
    This function gets the tenure information from the telco data set for customers with 2 year contracts
    '''
    file_name = 'mall_customers.csv'
    if os.path.isfile(file_name):
        return pd.read_csv(file_name)
    
    else:
        query =  '''
        select *
        from customers
        '''
    df = pd.read_sql(query, get_connection('mall_customers'))  
    
    #replace white space with nulls
    df = df.replace(r'^\s*$', np.NaN, regex=True)
    
    df.to_csv(file_name, index = False)

    return df

def get_mallcustomer_data():
    df = pd.read_sql('SELECT * FROM customers;', get_connection('mall_customers'))

def summerize_df(df):
    print('-----Head-------')
    print(df.head(3))
    print('-----shape------')
    print('{} rows and {} columns'.format(df.shape[0], df.shape[1]))
    print('---info---')
    print(df.info())
    print(df.describe())
    print('----Catagorical Variables----')
    print(df.select_dtypes(include='object').columns.tolist())
    print('----Continous  Variables----')
    print(df.select_dtypes(exclude='object').columns.tolist())
    
    print('--nulls--')
    df = df.replace(r'^\s*$', np.NaN, regex=True)
    print(df.isna().sum())
    
    num_cols = [col for col in df.columns if df[col].dtype != 'O']
    cat_cols = [col for col in df.columns if col not in num_cols]
    
    print('----Value Counts-----')
    for col in df.columns:
        if col in cat_cols:
            print(df[col].value_counts())
            print('-----------------------')
        else:
            print('-----------------------')
            print(df[col].value_counts(bins=10, sort=False))
    


#remove nulls and columns based on %
##############################################################
##############################################################
##############################################################
def nulls_by_col(df):
    num_missing = df.isnull().sum()
    print(type(num_missing))
    rows = df.shape[0]
    prcnt_miss = num_missing / rows * 100
    cols_missing = pd.DataFrame({'num_rows_missing': num_missing, 'percent_rows_missing': prcnt_miss})
    return cols_missing

def nulls_by_row(df):
    num_missing = df.isnull().sum(axis=1)
    prcnt_miss = num_missing / df.shape[1] * 100
    rows_missing = pd.DataFrame({'num_cols_missing': num_missing, 'percent_cols_missing': prcnt_miss})\
    .reset_index()\
    .groupby(['num_cols_missing', 'percent_cols_missing']).count()\
    .rename(index=str, columns={'index': 'num_rows'}).reset_index()
    return rows_missing



def get_nulls(df):
    col =  nulls_by_col(df)
    row =  nulls_by_row(df)
    
    return col, row


def drop_null_columns(df , null_min , col_missing = 'percent_rows_missing'):
    cols = get_nulls(df)[0]
    for i in range(len(cols)):
        if cols[col_missing][i] >= null_min:
            df = df.drop(columns = cols.index[i])
        
    return df

def drop_null_rows(df , percentage):
    min_count = int(((100-percentage)/100)*df.shape[1] + 1)
    df = df.dropna(axis=0, thresh = min_count)
        
    return df

def drop_nulls(df, axis, percentage):
    if axis == 0:
        df = drop_null_rows(df, percentage)   
    else:
        df = drop_null_columns(df, percentage)
    return df
##############################################################
##############################################################
##############################################################

