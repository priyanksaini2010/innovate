import sagemaker
import boto3
from sagemaker import get_execution_role

region = boto3.Session().region_name

session = sagemaker.Session()
bucket = session.default_bucket()
prefix = 'sagemaker/autopilot-dm'

role = get_execution_role()

sm = boto3.Session().client(service_name='sagemaker',region_name=region)
!wget https://innovatein48autobankload.s3.us-east-2.amazonaws.com/credit_train.csv
    
local_data = "./credit_train.csv"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows',500)

df = pd.read_csv(local_data)
df.head(2)

df.info()

df.isnull().sum()

len(df.index)

df['Credit Score'].fillna(df['Credit Score'].median(),inplace=True) 
df['Annual Income'].fillna(df['Annual Income'].median(),inplace=True) 
#f = df[df['Credit Score'].notna()]

len(df.index)
df['Current Loan Amount'].unique()
df['Current Loan Amount'].fillna(df['Current Loan Amount'].median(),inplace=True)

df.isnull().sum()

df['Months since last delinquent'].fillna(df['Months since last delinquent'].median(),inplace=True) 


df['Bankruptcies'].fillna(df['Bankruptcies'].median(),inplace=True) 

df['Years in current job'].fillna('0 Years',inplace=True)

df['Tax Liens'].fillna(df['Tax Liens'].median(),inplace=True) 
df['Maximum Open Credit'].fillna(df['Maximum Open Credit'].median(),inplace=True) 

df['Loan Status'].unique()

mask = df['Loan Status'] ==  'Fully Paid'
column_name = 'Loan Status'
df.loc[mask, column_name] = 1
mask_charged_off = df['Loan Status'] ==  'Charged Off'
column_name = 'Loan Status'
df.loc[mask_charged_off, column_name] = 0

mask_0 = df['Loan Status'] ==  '0'
column_name = 'Loan Status'
df.loc[mask, column_name] = 'Fully Paid'

df['Loan Status'].unique()

df = df[df['Loan Status'].notna()]

df.isnull().sum()

df.to_csv('clean_input_onlynum.csv')

import re, sys, math, json, os, sagemaker, urllib.request

boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'clean_input_onlynum.csv')).upload_file('clean_input_onlynum.csv')