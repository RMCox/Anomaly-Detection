
# Dashboard Overview
# 3 tabs
# Tab 1: General:

    # options:
    # Use Recommended
        # The user is shown a series of graphs with anomalies, and options to define what is flagged as anomalous.
        # User inputs are: Display date from, display date to (for non-neural network based anomaly detection, what data range should be displayed)
        #                : Train date from; train date to (for neural networks, train date range, note that indices in the "disregard" list, are excluded from training)
        #                : Test date from; test date to (for neural networks, what date range should the neural network predict)
        # Anomalies can be download
    # Customise
        # User can choose any input and any method for anomaly detection.
        # Anomalies can be downloaded

# Tab 2: General - manage disregard indices
        # The user can add or remove anomalous indices so that they can be excluded from the TRAINING of neural networks
        # Note that the presence of an index in the disregard list will, for prudence, exclude any time bucket interval which contains that index from training. e.g. if 12:03:34 is excluded, and 5min time buckted used, then 12:00-12:05 will be excluded

# Tab 3: Targeted (not yet finished):
        # The user can look for specific malicious exploits within the data, and can simulate them to see how the detection would work.
        # current not finished, a dataframe is always returned but not yet filtered well
        # could expand on this and use clustering or some other measure to classify

# Tab 4: Tageted - manage bad IP addresses
        # The user can add known IP addresees so that, if they have appeared in days prior to the day of checking, they will still be flagged as a new IP

        
import streamlit as st

import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#from datetime import datetime
import datetime
import re

import PIL
from PIL import Image

import pickle
import random 
import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from cryptography.fernet import Fernet
from datetime import timedelta
import string
from tqdm import tqdm_notebook
from mpl_toolkits.mplot3d import Axes3D

import warnings
warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:20,.2f}'.format

from sklearn.externals.six import StringIO 
from IPython.display import Image
import pydot
from sklearn.ensemble import RandomForestClassifier
import sklearn.tree as tree
from sklearn.model_selection import train_test_split

from scipy.stats import norm

sns.set_style('white')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Formatting
# ----------------------------------------------------------------------------    

st.write('# Anomaly Detection in DNS Logs')



def load_data_original():
    df = pd.read_table('C:\\Users\\roder\\Projects\\CyberSecurity\\dns.log', header=None, names=['ts', 'uid', 'id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p', 'proto', 'port', 'query', 'qclass', 'qclass_name', 'qtype', 'qtype_name', 'rcode', 'rcode_name', 'QR', 'AA', 'TC', 'RD', 'Z', 'answers', 'TTLs', 'rejected'])
    df.drop_duplicates(subset=None, keep='first', inplace=True)
    return(df)
         
def load_data_with_engineered_features():
    df = pd.read_csv('C:\\Users\\roder\\Projects\\CyberSecurity\\df_adapted.csv')
    df.set_index('dt', inplace=True)
    df.index = pd.to_datetime(df.index)
    return(df)


    
    
df_original = load_data_with_engineered_features()

folder_path = 'C:\\Users\\roder\\Projects\\CyberSecurity\\'




load_new_data_button = st.sidebar.button("load new data: dns.log")
    
tabs = st.sidebar.radio(          
    "Display a Tab:",
    ("Outlier Detection", "Outlier Detection - Manage Disregard Indices", "Exploit Detection", "Exploit Detection - Manage Infected Cache IPs"),
    key='tabs'
)

def get_top_level_domain(query):
    '''returns the value after the last ".", if there is none, it returns "internal"
    Parameters:
        Query
    Returns:
        string for top level domain
    '''
    if (not '.' in query) and (len(query) > 0):
        return('internal')
    tld = str.split(query, sep='.')
    if len(tld) > 1:
        return(str.lower(tld[-1]))
    else:
        return('')
        

def get_subdomain_length(df):
    '''get the length prior to the last full stop, **is this useful?**'''
    if df['top_level_domain'] == 'internal':
        return(len(df['query']))
    else:
        if len(re.findall('.*\.', str(df['query']))) != 0:
            return(len(re.findall('.*\.', str(df['query']))[-1][:-1]))
        else:
            return(len(df['query']))
        

def convert_TTL_to_float(TTL_vector):
    TTL = np.array(str.split(TTL_vector, ',')).astype('float')
    return(TTL)
  

def get_answer_vector(answer):
    return(str.split(answer, sep=','))

def get_proportion_special_chars(string):
    return(len(re.findall('[^a-z]', string))/len(string))
  
def get_shannon_entropy(string):
    '''returns shannon entropy in bits'''
    total_chars_in_string = len(string)
    
    unique_chars_in_string = list(set(string))
    
    appearance_count_list = []
    for letter in unique_chars_in_string:
        # get how many times a letter appears in the string
        appearance_count = 0
        for i in range(len(string)):
            if string[i] == letter:
                appearance_count += 1
        appearance_count_list.append(appearance_count)
    
    shannon_entropy = 0
    for appearances in appearance_count_list:
        shannon_entropy += -(appearances/total_chars_in_string) * np.log2((appearances/total_chars_in_string))
    
    return(shannon_entropy)
    
# this is a list of the top level and second level domains
# second level domains are labelled as "country codes"
# can use this dataframe to determine the "co" in "foo.co.uk" as second level domain
# not a parent domain        
iana_df = pd.read_excel("C:\\Users\\roder\\Projects\\CyberSecurity\\iana_domains.xlsx")
country_2nd_level_domains = iana_df[iana_df['TYPE'] == 'country-code']['DOMAIN'].unique().tolist()
country_2nd_level_domains = [str.replace(country_2nd_level_domains[i], '.', '') for i in range(len(country_2nd_level_domains))]

def get_parent_domain(query):
    '''Looks at the query from the last "." backwards. If what it encounters isn't in the second level domain list
    then it returns the string, if it is then it continues to look backwards until it finds something not in the
    second level domain list.
    If it can't find anything not in the second level domain list before it gets to the end of the string, it will return the very first
    part of the query, before the first "." e.g. "ac.co.uk" would return "ac"'''
    if (not '.' in query) and (len(query) > 0):
        return('internal')
    tld = str.split(query, sep='.')
    if len(tld) > 1:
        for i in range(1, len(tld)):
            if ~(str.lower(tld[-(i+1)]) in country_2nd_level_domains):
                return(str.lower(tld[-(i+1)]))
        print('Entire string in', query,"made up of TLD and second level domains - returning the first second level domain:" + str.lower(tld[0]))
        return(str.lower(tld[0]))
        
def get_subquery(query):
    '''get the value of the subquery. E.g. somestudent.qub.ac.uk = somestudent.
    Find the parent domain in the string and get the text immediately before it, provided the index before is a "."
    if there is no index before the parent domain, then there is no subquery, and it will return a blank string'''
    parent_domain = get_parent_domain(query)
    if parent_domain == None:
        print(query)
    parent_domain_start_index = query.find(parent_domain)

    try:
        # try to get the string up to the index before where the parent domain starts, which should be a "."
        return(query[:parent_domain_start_index-1])
    except:
        # if there is no index prior to that, then just return blank
        return('')


def apply_feature_engineering(df):
    '''add the engineered features to the dataframe.
    this is used for simulation. In the simulation, only features from the original dns logs database with no feature engineering is loaded.
    After simulation, the engineered features are added'''
    
    df.drop_duplicates(subset=None, keep='first', inplace=True)
    df['dt'] = df['ts'].apply(lambda x: datetime.datetime.fromtimestamp(x))
    df['ID'] = range(len(df))
    df = df.set_index('dt')
    df['top_level_domain'] = df['query'].apply(get_top_level_domain)
    df['query_length'] = df['query'].apply(len)
    df['number_of_dots_in_query'] = df['query'].apply(lambda x: x.count('.'))
    df['subdomain_length'] = df.apply(get_subdomain_length, axis=1)
    df.loc[df['TTLs'] == '-', 'TTLs'] = 0
    df.loc[df['TTLs'] == '0.000000,0.000000', 'TTLs'] = 0
    df['TTLs'] = df['TTLs'].astype('str')
    df['TTL_float_vector'] = df['TTLs'].apply(convert_TTL_to_float)
    df['TTL_max'] = df['TTL_float_vector'].apply(max)
    df['TTL_min'] = df['TTL_float_vector'].apply(min)
    df['TTL_mean'] = df['TTL_float_vector'].apply(np.mean)
    df['TTL_std'] = df['TTL_float_vector'].apply(np.std)
    df['TTL_coeff_var'] = df['TTL_std'] / df['TTL_mean']
    df['TTL_len'] = df['TTL_float_vector'].apply(len)
    df['answer_vector_length'] = df['answers'].apply(lambda x: str.split(x, sep=','))
    df['answer_vector_len'] = df['answer_vector_length'].apply(len)
    df['query_information_level'] = df['query'].apply(get_shannon_entropy)
    df['nonblank_answers'] = df['answers'].apply(lambda x: 1 if x != "-" else 0)
    df['parent_domain'] = df['query'].apply(get_parent_domain)
    df['subquery'] = df['query'].apply(get_subquery)
    df['subquery'] = df['subquery'].apply(str)
    df['subquery_information_level'] = df['subquery'].apply(get_shannon_entropy)
    df['no_response'] = df['QR'] == 'F'
    return(df)

def add_parent_randomness_level_LSTM_feature(df):
    '''adds parent domain feature to dataframe by convering to tensors and using a pretrained LSTM.
    this is used for simulation. In the simulation, only features from the original dns logs database with no feature engineering is loaded.
    Seperate to apply_feature_engineering_targeted as it is so computationally expensive'''
    
    # get the max length url for the LSTM classifier
    with open("C:\\Users\\roder\\Projects\\CyberSecurity\\max_length_url.txt", "rb") as fp:   # Unpickling
        max_length_url = pickle.load(fp)
    st.code('max_length_url used in parent randomness level feature:' + str(max_length_url))

    # input_size is the length of each row, here it is 66
    # hidden_size is the number of nodes in each hidden layer
    # num_layers is the number of hidden layers to use
    # num_classes is the number of classes and therefore the size of the output        
    class LSTM_Classifier(nn.Module):
        def __init__(self, input_size = 66*max_length_url, hidden_size = 100, num_layers = 3, num_classes = 2):
            super(LSTM_Classifier, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            # bidirectional means that the state of a row will depend on previous states and states after the row
            self.bidirectional = True
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=self.bidirectional)
            # batch_first=True means: # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size
            if self.bidirectional:
                # since bidirectional we need twice as many hidden layers and weights
                self.fc = nn.Linear(hidden_size*2, num_classes)
            else:
                self.fc = nn.Linear(hidden_size, num_classes)
            self.dropout = nn.Dropout(p=0.2)
        
        def forward(self, x):
            # making x a rolled out 2d image
            x = x.view(-1, 1, 66*max_length_url).float().to(device) 
            # setting initial weight and cell states
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
            if self.bidirectional:
                # since bidirectional we need twice as many hidden layers and weights
                h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device) 
                c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device) 
            
            
            # Forward propagate LSTM
            out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
            
            # Decode the hidden state of the last time step
            out = self.fc(out[:, -1, :])
            return F.softmax(out, dim=1) # actually changing softmax JUST FOR THE USE - fine to train on log softmax
    
    # load the model
    model = LSTM_Classifier()
    model.load_state_dict(torch.load("C:\\Users\\roder\\Projects\\CyberSecurity\\LSTM_Classifier"))
    
    
    # update the max length if the new dataframe contains a longer max_length_url
    df['parent_domain'] = df['parent_domain'].apply(str)
    if max(df['parent_domain'].apply(len)) > max_length_url:
        max_length_url = max(df['parent_domain'].apply(len))   
        st.code(max_length_url)
    
    # convert letters to tensors
    all_letters = string.ascii_letters + "-._~" + "0123456789" # all allowed in urls
    n_letters = len(all_letters)
    
    def line_to_tensor(line):
        tensor = torch.zeros(max_length_url, 1, n_letters) # slight change here to ensure padding
        for li, letter in enumerate(line):
            letter_index = all_letters.find(letter)
            tensor[li][0][letter_index] = 1
        return tensor
    
    # apply line to tensor
    df['parent_domain_tensor'] = df['parent_domain'].apply(line_to_tensor)
    
    # use model to get feature
    df['parent_domain_randomness_level'] = df['parent_domain_tensor'].apply(lambda x: model(x)[0][1].item())     
    
    return(df)

if load_new_data_button:
    df = pd.read_table(folder_path + 'dns.log', header=None, names=['ts', 'uid', 'id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p', 'proto', 'port', 'query', 'qclass', 'qclass_name', 'qtype', 'qtype_name', 'rcode', 'rcode_name', 'QR', 'AA', 'TC', 'RD', 'Z', 'answers', 'TTLs', 'rejected'])
    df = apply_feature_engineering(df)
    df = add_parent_randomness_level_LSTM_feature(df)
    df = df.drop('parent_domain_tensor', axis=1)
    df.to_csv(folder_path + 'df_adapted.csv')
    df = pd.read_csv(folder_path + 'df_adapted.csv')
    st.success('new data loaded, saved as "df_adapted.csv"')
    
# General and General - Manage Disregard Indices
# ----------------------------------------------------------------------------
# ****************************************************************************
# ----------------------------------------------------------------------------
    # disregard indices functions are only used for the neural networks, no use in the other detection methods

def add_indices_to_disregard_list(df, start_time, end_time):
    '''Add all date indices in a range from a dataframe to a disregard list
       Only unique indices are kept
    '''
    new_indices = df.loc[start_time:end_time].index.unique()
    with open('C:\\Users\\roder\\Projects\\CyberSecurity\\bad_indices.pkl', 'rb') as f:
        bad_indices = pickle.load(f)
    bad_indices = bad_indices.append(new_indices)
    bad_indices = bad_indices.unique()
    with open('C:\\Users\\roder\\Projects\\CyberSecurity\\bad_indices.pkl', 'wb') as f:
        pickle.dump(bad_indices, f)
    print('added indices to list')

def add_indices_to_disregard_list_from_dataframe(df):
    ''' takes a dataframe input with a column called "dt", and extracts for the disregard list'''
    
    new_indices = df.index.unique()
    with open('C:\\Users\\roder\\Projects\\CyberSecurity\\bad_indices.pkl', 'rb') as f:
        bad_indices = pickle.load(f)
    bad_indices = bad_indices.append(new_indices)
    bad_indices = bad_indices.unique()
    with open('C:\\Users\\roder\\Projects\\CyberSecurity\\bad_indices.pkl', 'wb') as f:
        pickle.dump(bad_indices, f)
    print('added indices to list')


def remove_indices_from_disregard_list(start_time=None, end_time=None, remove_all=False):
    '''remove all date indices in a range from the disregard list'''
    if remove_all:
        with open('C:\\Users\\roder\\Projects\\CyberSecurity\\bad_indices.pkl', 'rb') as f:
            bad_indices = pickle.load(f)
        bad_indices = pd.to_datetime([])
        with open('C:\\Users\\roder\\Projects\\CyberSecurity\\bad_indices.pkl', 'wb') as f:
            pickle.dump(bad_indices, f)
        print('removed all indices from list')
        return
    indices_to_remove = df.loc[start_time:end_time].index
    with open('C:\\Users\\roder\\Projects\\CyberSecurity\\bad_indices.pkl', 'rb') as f:
        bad_indices = pickle.load(f)
    bad_indices = bad_indices[~bad_indices.isin(indices_to_remove)]
    with open('C:\\Users\\roder\\Projects\\CyberSecurity\\bad_indices.pkl', 'wb') as f:
        pickle.dump(bad_indices, f)
    print('removed indices from list')


def remove_indices_from_disregard_list_from_dataframe(df):
    '''remove all date indices in a dataframe index from the disregard list'''
    indices_to_remove = df.index.tolist()
    with open('C:\\Users\\roder\\Projects\\CyberSecurity\\bad_indices.pkl', 'rb') as f:
        bad_indices = pickle.load(f)
    bad_indices = bad_indices[~bad_indices.isin(indices_to_remove)]
    with open('C:\\Users\\roder\\Projects\\CyberSecurity\\bad_indices.pkl', 'wb') as f:
        pickle.dump(bad_indices, f)
    print('removed indices from list')
    

def get_disregard_indices():
    '''return the date indices that are in the disregard list'''
    with open('C:\\Users\\roder\\Projects\\CyberSecurity\\bad_indices.pkl', 'rb') as f:
        bad_indices = pickle.load(f)
    return(bad_indices)


def disregard_indices(df):
    ''' return a dataframe without any indices from the disregard list (unbucketed dataframe)'''
    with open('C:\\Users\\roder\\Projects\\CyberSecurity\\bad_indices.pkl', 'rb') as f:
        bad_indices = pickle.load(f)
    df = df[~df.index.isin(bad_indices)]
    return(df)


def get_disregard_indices_intervals(frequency_argument):
    ''' return the time buckets for all indices that will be disregarded, so that they can be disregarded from a bucketed dataframe'''
    indices_counts_per_interval = pd.DataFrame(data={'index': get_disregard_indices(), 'dummy':1}).set_index('index').groupby([pd.Grouper(freq=frequency_argument)]).agg('sum')
    grouped_indices = indices_counts_per_interval[indices_counts_per_interval['dummy'] >= 1].index
    return(grouped_indices)


def disregard_indices_intervals(df_bucketed, grouped_indices):
    ''' return the bucketed dataframe, with any time buckets that contain a value from the disregard list filled with a nan value'''
    df_bucketed.iloc[df_bucketed.index.isin(grouped_indices)] = np.nan
    return(df_bucketed)


def add_some_trial_indices(df):
    'add some indices that are visibly aomalous, for the sake of trialling the functions'
    add_indices_to_disregard_list(df, start_time='2012-03-17 16:34', end_time='2012-03-17 16:36')
    add_indices_to_disregard_list(df, start_time='2012-03-16 18:10', end_time='2012-03-16 18:12')
    add_indices_to_disregard_list(df, start_time='2012-03-17 16:35', end_time='2012-03-17 16:36')
    add_indices_to_disregard_list(df, start_time='2012-03-17 17:03', end_time='2012-03-17 17:04')
    add_indices_to_disregard_list(df, start_time='2012-03-16 18:03', end_time='2012-03-16 18:05')
    add_indices_to_disregard_list(df, start_time='2012-03-16 18:08', end_time='2012-03-16 18:09')

# adding some to trial the effectiveness of the functions
#add_some_trial_indices(df=df_original)


# next few functions keep track of ip addresses for cache poisoning
    
def add_ip_to_bad_ip_list(ip_address):
    with open('C:\\Users\\roder\\Projects\\CyberSecurity\\bad_ips.pkl', 'rb') as f:
        bad_ips = pickle.load(f)
    if bad_ips == None:
        bad_ips = []
    bad_ips.append(ip_address)   
    if bad_ips != []:
        bad_ips = list(set(bad_ips))
    with open('C:\\Users\\roder\\Projects\\CyberSecurity\\bad_ips.pkl', 'wb') as f:
        pickle.dump(bad_ips, f)
    print('added bad ip to list')        

def add_ips_to_bad_IP_list_from_dataframe(df):
    ''' takes a dataframe input with a column called "IP", and extracts for the bad IP list'''
    new_ips = df.index.unique().tolist()
    with open('C:\\Users\\roder\\Projects\\CyberSecurity\\bad_ips.pkl', 'rb') as f:
        bad_ips = pickle.load(f)
    bad_ips = bad_ips + new_ips
    bad_ips = list(set(bad_ips))
    st.code(bad_ips)
    with open('C:\\Users\\roder\\Projects\\CyberSecurity\\bad_ips.pkl', 'wb') as f:
        pickle.dump(bad_ips, f)
    print('added IPs to list')


def remove_ip_from_bad_ips(ip_address=None, remove_all=False):
    if remove_all:
        with open('C:\\Users\\roder\\Projects\\CyberSecurity\\bad_ips.pkl', 'rb') as f:
            bad_ips = pickle.load(f)
        bad_ips = []
        with open('C:\\Users\\roder\\Projects\\CyberSecurity\\bad_ips.pkl', 'wb') as f:
            pickle.dump(bad_ips, f)
        print('removed all ips from list')
        return
    with open('C:\\Users\\roder\\Projects\\CyberSecurity\\bad_ips.pkl', 'rb') as f:
        bad_ips = pickle.load(f)
    bad_ips.remove(ip_address)
    
    with open('C:\\Users\\roder\\Projects\\CyberSecurity\\bad_ips.pkl', 'wb') as f:
        pickle.dump(bad_ips, f)
    print('removed indices from list')
    return
    
def get_bad_ips():
    with open('C:\\Users\\roder\\Projects\\CyberSecurity\\bad_ips.pkl', 'rb') as f:
        bad_ips = pickle.load(f)
    return(bad_ips)    
    
def remove_ips_from_bad_IPs_list_from_dataframe(df):
    '''remove all IPs in dataframe index from the bad list'''
    ips_to_remove = df.index.tolist()
    with open('C:\\Users\\roder\\Projects\\CyberSecurity\\bad_ips.pkl', 'rb') as f:
        bad_ips = pickle.load(f)
    bad_ips = [ip for ip in bad_ips if ip not in ips_to_remove]
    with open('C:\\Users\\roder\\Projects\\CyberSecurity\\bad_ips.pkl', 'wb') as f:
        pickle.dump(bad_ips, f)
    print('removed ips from list')
    

# General 
# ----------------------------------------------------------------------------
# ****************************************************************************
# ----------------------------------------------------------------------------

if tabs=="Outlier Detection":

# Bucketing and indices
# ----------------------------------------------------------------------------    
    st.write('## Outlier Detection')
    @st.cache
    def bucket_dataframe(df, interval_unit, interval_size, aggregation='count', target_column='ID', date_from=None, date_to=None, disregard_custom_indices_intervals=False, remove_nans=False):
        '''
        takes a dataframe, aggregates on a target column, applies a time bucket
        For the training of neural networks:
            - the argument "disregard_custom_indices_intervals" should be used, as it will replace any time buckets deemed to be anomalous with a nan value, which is later dealt with.
            - the argument remove_nans should be used, which willstop NAN value infecting any training data
        For testing, neither should be used, NANs in the testing should be flagged, captured and viewed. This is not relevant when using "count", but "mean" can produce nan values
        
        Parameters:
            df: Dataframe to use
            interval_unit: time unit {'s' : 'second', 'min' : 'minute', 'h': 'hour', 'd': 'day', 'm':'month', 'y': 'year'}
            interval_size: number of time units e.g. 30
            aggregation: type of aggregation to perform on target column e.g. 'count', 'mean'
            target_column: column on which to perform aggregation
            date_from: date from which the dataframe selected
            date_to: date to which the dataframe is selected
            disregard_custom_indices_intervals: replace the value for any time bucket that contains an date index in the disregard list with a NAN
            remove_nans: remove all NAN time bucketes from the dataframe, note that this will also apply to the indices in the disregard list
    
        Returns:
            df_bucketed: dataframe after bucketing
        '''
        # guides user, error showing how to enter units

        valid_interval_units = {'s', 'min', 'h', 'd', 'm', 'y'}
        
        interval_dictionary = {'s' : 'second', 'min' : 'minute', 'h': 'hour', 'd': 'day', 'm':'month', 'y': 'year'}
        if interval_unit not in valid_interval_units:
            raise ValueError("interval_unit must be one of %r." % valid_interval_units)
        
        if not isinstance(interval_size, int):
            raise ValueError("interval_size must be an integer")
        
        # group the data by the frequency argument
        freq_argument = str(interval_size) + interval_unit
        df_bucketed = pd.DataFrame(df[date_from:date_to].groupby([pd.Grouper(freq=freq_argument)]).agg(aggregation)[target_column])

        # get a sensible name for the column
        if target_column == 'ID':
            bucketed_target_col_name = str(aggregation) + " of " + "all queries"
        else:
            bucketed_target_col_name = str(aggregation) + " of " + str(target_column)
        df_bucketed.columns = [bucketed_target_col_name]
        
        # replace any time buckets that contain any indices from the disregard list with nan values
        if disregard_custom_indices_intervals:
            grouped_indices = get_disregard_indices_intervals(frequency_argument=freq_argument)
            df_bucketed = disregard_indices_intervals(df_bucketed=df_bucketed, grouped_indices=grouped_indices)     
        
        # remove all nan values (including any from disregard list if that option is selected)    
        if remove_nans:
            df_bucketed = df_bucketed[~df_bucketed.isna().any(axis=1)]
    
        return(df_bucketed)

# Defining Functions 
# -----------------------------------------------------------------------------        

    def compare_interval_to_last_5(df_bucketed, interval_unit, interval_size, aggregation='count', target_column='ID', plot=False, upper_threshold_factor=3):
        
        '''
        Computes the proportion of the last 5 time intervals' average that is made up by the current time interval.
        Can enter any feature/aggregation combination.
        Anomalies are identified by an upper threshold (anomalousness will vary by feature)
            (e.g. 3 = time bucket value must be greater than 3 times the proportion of the average of the last 5 time intervals)
        A limit can be placed on the proportion of the data that is returned as anomalies - the most anomalous is retained
        
        Parameters:
            df_bucketed: Dataframe bucketed by time
            interval_unit: time unit {'s' : 'second', 'min' : 'minute', 'h': 'hour', 'd': 'day', 'm':'month', 'y': 'year'}
            interval_size: number of time units e.g. 30
            aggregation: type of aggregation to perform on target column e.g. 'count', 'mean'
            target_column: column on which to perform aggregation
            plot: plot the time series and scatter anomalies
            upper_threshold_factor: how much greater that the previous 5 time intervals will the value need to be to be classified as anomalous
    
        Returns:
            df_bucketed: dataframe after operations have been performed
            anomalies: dataframe of time buckets identified as anomalous
            nan_indices: dataframe of time buckets that are nan_values (no information available) 
        '''
        bucketed_target_col_name = df_bucketed.columns[0]
        
        # keep a record of anomalous indices before removing. Easier for other functions, but early identification is important for computation here
        nan_anomalies =  pd.DataFrame(df_bucketed[df_bucketed.isna().any(axis=1)].index).set_index('dt')
        
        # note that this means that, if there is a null value, the next value's proportional change will be calculated from the previous non-null value (will skip over)
        df_bucketed = df_bucketed[~df_bucketed.isnull().any(axis=1)]
        
        # perform computations
        df_bucketed['lagged_rolling_5_interval'] = pd.DataFrame(df_bucketed[bucketed_target_col_name]).rolling(5, min_periods=1).agg('mean').shift(1)
        df_bucketed['pc_of_last_5_interval'] = df_bucketed[bucketed_target_col_name]/df_bucketed['lagged_rolling_5_interval']
        df_bucketed = df_bucketed.drop(['lagged_rolling_5_interval'], axis=1)
        df_bucketed = df_bucketed.sort_index()
    
        # get anomalies by flagging values where the pc difference is greater than the threshold set for anomalous flag
        anomalies = df_bucketed[df_bucketed['pc_of_last_5_interval'] >= upper_threshold_factor]

        # if plotting is required, plot the original data on one axis, and the proportional change and anomalies on another
        if plot:                
            fig, axs = plt.subplots(2, figsize=(15,15))
            df_bucketed[bucketed_target_col_name].plot(ax=axs[0])
            axs[0].scatter(x=nan_anomalies.index, y=[0]*len(nan_anomalies), s=5, c='k')
            axs[0].set_xlabel('Time ' + "(" + str(interval_size) + " " + str(interval_unit) + " intervals)")
            axs[0].set_ylabel(bucketed_target_col_name)
            
            df_bucketed['pc_of_last_5_interval'].plot(ax=axs[1])
            plt.title(str(interval_size) + interval_unit + " " + aggregation + " of " + target_column + " as proportion of last 5 " + str(interval_size) + interval_unit + " intervals")
            axs[1].scatter(x=anomalies.index, y=anomalies['pc_of_last_5_interval'], c='r')
            axs[1].scatter(x=nan_anomalies.index, y=[0]*len(nan_anomalies), s=5, c='k')
            axs[1].set_xlabel('Time ' + "(" + str(interval_size) + " " + str(interval_unit) + " intervals)")
            axs[1].set_ylabel('Proportional Change: ' + bucketed_target_col_name)
            st.pyplot()
                
        # state proportion of data anomalous and nan_anomalous
        st.code('proportion of anomalous data :' + str(round(len(anomalies)/(len(df_bucketed) + len(nan_anomalies)), 5)))
        st.code('proportion of na anomalous data :' + str(round(len(nan_anomalies)/(len(df_bucketed) + len(nan_anomalies)), 5)))   
        
        return(df_bucketed, anomalies, nan_anomalies)


    def bucket_and_compare_to_last_5(df, interval_unit, interval_size, aggregation, target_column, plot=False, upper_threshold_factor=3, remove_nans=False, disregard_custom_indices_intervals=False):
        '''shortcut function'''
        
        df_bucketed = bucket_dataframe(df, interval_unit=interval_unit, interval_size=interval_size, aggregation=aggregation, target_column=target_column, remove_nans=remove_nans, disregard_custom_indices_intervals=disregard_custom_indices_intervals)
        
        return(compare_interval_to_last_5(df_bucketed=df_bucketed,
                                   interval_unit=interval_unit,
                                   interval_size=interval_size,
                                   aggregation=aggregation,
                                   target_column=target_column,
                                   plot=plot,
                                   upper_threshold_factor=upper_threshold_factor))



    def get_standard_dev(df_bucketed, interval_unit, interval_size, aggregation='count', target_column='ID', sd_threshold=2):
        '''
        Identifies anomalies by checking whether the values for each time bucket deviate further from the mean more than a user specified multiple of the standard deviation of the data
        
        Parameters:
            df_bucketed: Dataframe bucketed by time
            interval_unit: time unit {'s' : 'second', 'min' : 'minute', 'h': 'hour', 'd': 'day', 'm':'month', 'y': 'year'}
            interval_size: number of time units e.g. 30
            aggregation: type of aggregation to perform on target column e.g. 'count', 'mean'
            target_column: column on which to perform aggregation
            sd_threshold: the number of standard deviations from the mean that a bucket's value must exceed to be classified as anomalous
    
        Returns:
            df_bucketed_std: dataframe after operations have been performed
            anomalies: dataframe of anomalies
            nan_anomalies: dataframe of nan anomalies
        '''
        # get column name
        bucketed_target_col_name = df_bucketed.columns[0]
        
        # make copy of the dataframe
        df_bucketed_std = df_bucketed.copy()
        
        # computation
        df_bucketed_std['mean over period'] = np.mean(df_bucketed_std[bucketed_target_col_name])
        df_bucketed_std['std over period'] = np.std(df_bucketed_std[bucketed_target_col_name])      

        # include this column to show the user the most anomalous buckets        
        df_bucketed_std['SDs_out'] = df_bucketed_std[bucketed_target_col_name]/df_bucketed_std['std over period']
        
        # get thresholds
        df_bucketed_std['upper_threshold'] =  df_bucketed_std['mean over period'] + sd_threshold*df_bucketed_std['std over period']
        df_bucketed_std['lower_threshold'] =  df_bucketed_std['mean over period'] - sd_threshold*df_bucketed_std['std over period']
        
        # anomalies
        anomalies = df_bucketed_std[(df_bucketed_std[bucketed_target_col_name] < df_bucketed_std['lower_threshold'])|(df_bucketed_std[bucketed_target_col_name] > df_bucketed_std['upper_threshold'])]
        nan_anomalies = df_bucketed_std[df_bucketed_std[bucketed_target_col_name].isna()]
        
        # plotting the original data on one axis, and the mean, sd thresholds and anomalies on another
        fig, axs = plt.subplots(2, figsize=(12,12))
        df_bucketed.plot(ax=axs[0])
        axs[0].legend([])
        axs[0].set_xlabel('Time ' + "(" + str(interval_size) + " " + str(interval_unit) + " intervals)")
        axs[0].set_ylabel(bucketed_target_col_name)
        
        df_bucketed_std[[bucketed_target_col_name, 'mean over period']].plot(ax=axs[1])
        axs[1].scatter(x=anomalies.index, y=anomalies[bucketed_target_col_name], s=10,c='r')
        axs[1].scatter(x=nan_anomalies.index, y=[0]*len(nan_anomalies), s=10, c='k')
        axs[1].fill_between(df_bucketed_std.index, df_bucketed_std['lower_threshold'], df_bucketed_std['upper_threshold'], alpha=0.2, color='grey')
        axs[1].legend(['value', 'mean over period', 'outliers', 'NA outliers', 'outlier detection tolerance'], loc='lower right')
        axs[1].set_xlabel('Time ' + "(" + str(interval_size) + " " + str(interval_unit) + " intervals)")
        axs[1].set_ylabel(bucketed_target_col_name)
        axs[1].set_title('Outlier Detection - Standard Deviation')
        fig.tight_layout()
        st.pyplot()
    
        # state proportion of data anomalous and nan_anomalous
        # no need to add length of nan anomalies as before as there is no removal of anomalies here
        st.code('proportion of anomalous data :' + str(round(len(anomalies)/len(df_bucketed_std), 5)))
        st.code('proportion of na anomalous data :' + str(round(len(nan_anomalies)/len(df_bucketed_std), 5)))        
        
        return(df_bucketed_std, anomalies, nan_anomalies)


    def bucket_and_get_standard_dev(df, interval_unit, interval_size, aggregation, target_column, sd_threshold=2, remove_nans=False, disregard_custom_indices_intervals=False):
        
        df_bucketed = bucket_dataframe(df, interval_unit=interval_unit, interval_size=interval_size, aggregation=aggregation, target_column=target_column, remove_nans=remove_nans, disregard_custom_indices_intervals=disregard_custom_indices_intervals)
        return(get_standard_dev(df_bucketed=df_bucketed,
                                interval_unit=interval_unit,
                                interval_size=interval_size,
                                aggregation=aggregation,
                                target_column=target_column,
                                sd_threshold=sd_threshold))


    def get_rolling_standard_dev(df_bucketed, interval_unit, interval_size, aggregation='count', target_column='ID', sd_threshold=2):
        '''
        Identifies anomalies by checking whether the value for each time bucket deviates further from the 1 hour rolling mean more than a user specified multiple of the 1 hour rolling standard deviation.

        Parameters:
            df_bucketed: Dataframe bucketed by time
            interval_unit: time unit {'s' : 'second', 'min' : 'minute', 'h': 'hour', 'd': 'day', 'm':'month', 'y': 'year'}
            interval_size: number of time units e.g. 30
            aggregation: type of aggregation to perform on target column e.g. 'count', 'mean'
            target_column: column on which to perform aggregation
            sd_threshold: the number of rolling standard deviations from the rolling mean that a bucket's value must exceed to be classified as anomalous
    
        Returns:
            df_bucketed_std: dataframe after operations have been performed
            anomalies: dataframe of anomalies
            nan_anomalies: dataframe of nan anomalies
        '''
        
        # get column name
        bucketed_target_col_name = df_bucketed.columns[0]
        
        # make copy of the dataframe
        df_bucketed_std = df_bucketed.copy()
        
        # computation
        df_bucketed_std['rolling_mean'] = df_bucketed_std.rolling('1h', min_periods=1)[bucketed_target_col_name].agg('mean').shift(1)
        df_bucketed_std['rolling_std'] = df_bucketed_std.rolling('1h', min_periods=1)[bucketed_target_col_name].agg('std').shift(1)
        
        # include this column to show the user the most anomalous buckets        
        df_bucketed_std['SDs_out'] = df_bucketed_std[bucketed_target_col_name]/df_bucketed_std['rolling_std']
        
        # get thresholds
        df_bucketed_std['rolling_upper_threshold'] =  df_bucketed_std['rolling_mean'] + sd_threshold*df_bucketed_std['rolling_std']
        df_bucketed_std['rolling_lower_threshold'] =  df_bucketed_std['rolling_mean'] - sd_threshold*df_bucketed_std['rolling_std']
        
        # anomalies
        anomalies = df_bucketed_std[(df_bucketed_std[bucketed_target_col_name] < df_bucketed_std['rolling_lower_threshold'])|(df_bucketed_std[bucketed_target_col_name] > df_bucketed_std['rolling_upper_threshold'])]
        nan_anomalies = df_bucketed_std[df_bucketed_std[bucketed_target_col_name].isna()]
        
        # plotting the original data on one axis, and the mean, sd thresholds and anomalies on another
        fig, axs = plt.subplots(2, figsize=(15,15))
        df_bucketed.plot(ax=axs[0])
        axs[0].legend([])
        axs[0].set_xlabel('Time ' + "(" + str(interval_size) + " " + str(interval_unit) + " intervals)")
        axs[0].set_ylabel(bucketed_target_col_name)        
        
        
        df_bucketed_std[[bucketed_target_col_name, 'rolling_mean']].plot(ax=axs[1])
        axs[1].scatter(x=anomalies.index, y=anomalies[bucketed_target_col_name], s=10,c='r')
        axs[1].scatter(x=nan_anomalies.index, y=[0]*len(nan_anomalies), s=10, c='k')
        axs[1].fill_between(df_bucketed_std.index, df_bucketed_std['rolling_lower_threshold'], df_bucketed_std['rolling_upper_threshold'], alpha=0.2, color='grey')
        axs[1].set_xlabel('Time ' + "(" + str(interval_size) + " " + str(interval_unit) + " intervals)")
        axs[1].set_ylabel(bucketed_target_col_name)           
        axs[1].legend(['value', 'rolling mean', 'outliers', 'NA outliers', 'outlier detection tolerance'], loc='lower right')
        fig.tight_layout()
        st.pyplot()
    
        # state proportion of data anomalous and nan_anomalous
        # no need to add length of nan anomalies as before as there is no removal of anomalies here
        st.code('proportion of anomalous data :' + str(round(len(anomalies)/len(df_bucketed_std), 5)))
        st.code('proportion of na anomalous data :' + str(round(len(nan_anomalies)/len(df_bucketed_std), 5)))        
        return(df_bucketed_std, anomalies, nan_anomalies)


    def bucket_and_get_rolling_standard_dev(df, interval_unit, interval_size, aggregation, target_column, sd_threshold=2, remove_nans=False, disregard_custom_indices_intervals=False):
        '''shortcut function'''
        
        df_bucketed = bucket_dataframe(df, interval_unit=interval_unit, interval_size=interval_size, aggregation=aggregation, target_column=target_column, remove_nans=remove_nans, disregard_custom_indices_intervals=disregard_custom_indices_intervals)
        return(get_rolling_standard_dev(df_bucketed=df_bucketed,
                                interval_unit=interval_unit,
                                interval_size=interval_size,
                                aggregation=aggregation,
                                target_column=target_column,
                                sd_threshold=sd_threshold))

    def get_MAD(df_bucketed, interval_unit, interval_size, aggregation='count', target_column='ID', MAD_threshold=2):
        '''
        Identifies anomalies by checking whether the value for each time bucket deviates further from the 1 hour median more than a user specified multiple of the median absolute deviation 

        Parameters:
            df_bucketed: Dataframe bucketed by time
            interval_unit: time unit {'s' : 'second', 'min' : 'minute', 'h': 'hour', 'd': 'day', 'm':'month', 'y': 'year'}
            interval_size: number of time units e.g. 30
            aggregation: type of aggregation to perform on target column e.g. 'count', 'mean'
            target_column: column on which to perform aggregation
            MAD_threshold: the number of median absolute deviations from the median that a bucket's value must exceed to be classified as anomalous
    
        Returns:
            df_bucketed_std: dataframe after operations have been performed
            anomalies: dataframe of anomalies
            nan_anomalies: dataframe of nan anomalies
        '''
        # get column name
        bucketed_target_col_name = df_bucketed.columns[0]
        
        # make copy of the dataframe
        df_bucketed_MAD = df_bucketed.copy()
        
        # computation
        df_bucketed_MAD['median over period'] = np.nanmedian(df_bucketed_MAD[bucketed_target_col_name],)
        df_bucketed_MAD['abs_differences'] = abs(df_bucketed_MAD[bucketed_target_col_name] - df_bucketed_MAD['median over period'])
        df_bucketed_MAD['MAD over period'] = np.nanmedian(df_bucketed_MAD['abs_differences'])
        
        # include this column to show the user the most anomalous buckets      
        df_bucketed_MAD['MADs_out'] = df_bucketed_MAD[bucketed_target_col_name]/df_bucketed_MAD['MAD over period'] 
        
        # get thresholds
        df_bucketed_MAD['upper_threshold'] =  df_bucketed_MAD['median over period'] + MAD_threshold*df_bucketed_MAD['MAD over period']
        df_bucketed_MAD['lower_threshold'] =  df_bucketed_MAD['median over period'] - MAD_threshold*df_bucketed_MAD['MAD over period']
        
        # anomalies
        anomalies = df_bucketed_MAD[(df_bucketed_MAD[bucketed_target_col_name] < df_bucketed_MAD['lower_threshold'])|(df_bucketed_MAD[bucketed_target_col_name] > df_bucketed_MAD['upper_threshold'])]
        nan_anomalies = df_bucketed_MAD[df_bucketed_MAD[bucketed_target_col_name].isna()]
        
        # plotting the original data on one axis, and the median, MAD thresholds and anomalies on another
        fig, axs = plt.subplots(2, figsize=(12, 12))
        df_bucketed.plot(ax=axs[0])
        axs[0].legend([]) 
        axs[0].set_xlabel('Time ' + "(" + str(interval_size) + " " + str(interval_unit) + " intervals)")    
        axs[0].set_ylabel(bucketed_target_col_name) 
        
        df_bucketed_MAD[[bucketed_target_col_name, 'median over period']].plot(ax=axs[1])
        axs[1].scatter(x=anomalies.index, y=anomalies[bucketed_target_col_name], s=10,c='r')
        axs[1].scatter(x=nan_anomalies.index, y=[0]*len(nan_anomalies), s=10, c='k')
        axs[1].fill_between(df_bucketed_MAD.index, df_bucketed_MAD['lower_threshold'], df_bucketed_MAD['upper_threshold'], alpha=0.2, color='grey')
        axs[1].legend(['value', 'median over period', 'outliers', 'NA outliers', 'outlier detection tolerance'], loc='lower right') 
        axs[1].set_xlabel('Time ' + "(" + str(interval_size) + " " + str(interval_unit) + " intervals)")    
        axs[1].set_ylabel(bucketed_target_col_name)    
        
        axs[1].set_title('Outlier Detection - Median Absolute Deviation (MAD)')
        fig.tight_layout()
        st.pyplot()
    
        # state proportion of data anomalous and nan_anomalous
        # no need to add length of nan anomalies as before as there is no removal of anomalies here
        st.code('proportion of anomalous data :' + str(round(len(anomalies)/len(df_bucketed_MAD), 5)))
        st.code('proportion of na anomalous data :' + str(round(len(nan_anomalies)/len(df_bucketed_MAD), 5)))        
        
        return(df_bucketed_MAD, anomalies, nan_anomalies)

    def bucket_and_get_MAD(df, interval_unit, interval_size, aggregation, target_column, MAD_threshold=2, remove_nans=False, disregard_custom_indices_intervals=False):
        '''shortcut function'''
        df_bucketed = bucket_dataframe(df, interval_unit=interval_unit, interval_size=interval_size, aggregation=aggregation, target_column=target_column, remove_nans=remove_nans, disregard_custom_indices_intervals=disregard_custom_indices_intervals)
        
        return(get_MAD(df_bucketed=df_bucketed, interval_unit=interval_unit, interval_size=interval_size, aggregation=aggregation, target_column=target_column, MAD_threshold=MAD_threshold))

    def get_rolling_MAD(df_bucketed, interval_unit, interval_size, aggregation='count', target_column='ID', MAD_threshold=2):
        '''
        Identifies anomalies by checking whether the values for each time bucket deviates further from the 1 hour rolling median more than a user specified multiple of the 1 hour rolling median absolute deviation 

        Parameters:
            df_bucketed: Dataframe bucketed by time
            interval_unit: time unit {'s' : 'second', 'min' : 'minute', 'h': 'hour', 'd': 'day', 'm':'month', 'y': 'year'}
            interval_size: number of time units e.g. 30
            aggregation: type of aggregation to perform on target column e.g. 'count', 'mean'
            target_column: column on which to perform aggregation
            MAD_threshold: the number of rolling median absolute deviations from the rolling median that a bucket's value must exceed to be classified as anomalous
    
        Returns:
            df_bucketed_std: dataframe after operations have been performed
            anomalies: dataframe of anomalies
            nan_anomalies: dataframe of nan anomalies
        '''
        # get column name
        bucketed_target_col_name = df_bucketed.columns[0]
        
        # make copy of the dataframe
        df_bucketed_MAD = df_bucketed.copy()
        
        # computation
        df_bucketed_MAD['rolling_median'] = df_bucketed_MAD.rolling('1h', min_periods=1)[bucketed_target_col_name].agg('median').shift(1)
        df_bucketed_MAD['abs_differences_rolling'] = abs(df_bucketed_MAD[bucketed_target_col_name] - df_bucketed_MAD['rolling_median'])
        df_bucketed_MAD['rolling_MAD'] = df_bucketed_MAD.rolling('1h', min_periods=1)['abs_differences_rolling'].agg('median').shift(1)
        
        # include this column to show the user the most anomalous buckets      
        df_bucketed_MAD['MADs_out'] = df_bucketed_MAD[bucketed_target_col_name]/df_bucketed_MAD['rolling_MAD']            
        
        # get thresholds
        df_bucketed_MAD['rolling_upper_threshold'] =  df_bucketed_MAD['rolling_median'] + MAD_threshold*df_bucketed_MAD['rolling_MAD']
        df_bucketed_MAD['rolling_lower_threshold'] =  df_bucketed_MAD['rolling_median'] - MAD_threshold*df_bucketed_MAD['rolling_MAD']
        
        # anomalies
        anomalies = df_bucketed_MAD[(df_bucketed_MAD[bucketed_target_col_name] < df_bucketed_MAD['rolling_lower_threshold'])|(df_bucketed_MAD[bucketed_target_col_name] > df_bucketed_MAD['rolling_upper_threshold'])]
        nan_anomalies = df_bucketed_MAD[df_bucketed_MAD[bucketed_target_col_name].isna()]
        
        # plotting the original data on one axis, and the rolling median, rolling MAD thresholds and anomalies on another
        fig, axs = plt.subplots(2, figsize=(15,15))
        df_bucketed.plot(ax=axs[0])
        axs[0].legend([])
        axs[0].set_xlabel('Time ' + "(" + str(interval_size) + " " + str(interval_unit) + " intervals)")
        axs[0].set_ylabel(bucketed_target_col_name)   
        
        df_bucketed_MAD[[bucketed_target_col_name, 'rolling_median']].plot(ax=axs[1])
        axs[1].scatter(x=anomalies.index, y=anomalies[bucketed_target_col_name], s=10,c='r')
        axs[1].scatter(x=nan_anomalies.index, y=[0]*len(nan_anomalies), s=10, c='k')
        axs[1].fill_between(df_bucketed_MAD.index, df_bucketed_MAD['rolling_lower_threshold'], df_bucketed_MAD['rolling_upper_threshold'], alpha=0.2, color='grey')
        axs[1].set_xlabel('Time ' + "(" + str(interval_size) + " " + str(interval_unit) + " intervals)")
        axs[1].set_ylabel(bucketed_target_col_name)           
        axs[1].legend(['value', 'rolling median', 'outliers', 'NA outliers', 'outlier detection tolerance'], loc='lower right')
        fig.tight_layout()
        st.pyplot()
    
        # state proportion of data anomalous and nan_anomalous
        # no need to add length of nan anomalies as before as there is no removal of anomalies here
        st.code('proportion of anomalous data :' + str(round(len(anomalies)/len(df_bucketed_MAD), 5)))
        st.code('proportion of na anomalous data :' + str(round(len(nan_anomalies)/len(df_bucketed_MAD), 5)))        
        
        return(df_bucketed_MAD, anomalies, nan_anomalies)
     
    def bucket_and_get_rolling_MAD(df, interval_unit, interval_size, aggregation, target_column, MAD_threshold=2, remove_nans=False, disregard_custom_indices_intervals=False):
        '''shortcut function'''
        df_bucketed = bucket_dataframe(df, interval_unit=interval_unit, interval_size=interval_size, aggregation=aggregation, target_column=target_column, remove_nans=remove_nans, disregard_custom_indices_intervals=disregard_custom_indices_intervals)
        
        return(get_rolling_MAD(df_bucketed=df_bucketed, interval_unit=interval_unit, interval_size=interval_size, aggregation=aggregation, target_column=target_column, MAD_threshold=MAD_threshold))


    def get_dataset_for_NN(df_bucketed, steps, remove_initial_nans=False):
        ''' 
        Takes a bucketed dataframe and creates a dataset that could be used used in a pytorch dataloader for a feed forward neural network.
        The number of features to be created is equal to "steps", which is the number of previous values which will be used to predict the next value.
        format is list of list of 2 elements, where the first element is a (features x 1) sized numpy array and the second element is the value.
        For training, a bucketed dataframe with nan values and disregard indices REMOVED should be used. Therefore the dataframe will use the "n" previous non-nan values, as the nan values have been removed (skipping over nans)
        For testing, nan values should NOT be disregarded, and should be captured with flags.
        When creating the dataset, the first few values equal to the number of steps will be nan as a result of the shift, these can be removed with "remove_initial_nans"
        
        
        Parameters:
        df_bucketed: Dataframe bucketed by time
        steps: number of previous time intervals used to predict the current time interval (will skip over nan values)
        remove_initial_nans: Determines whether the first few nan values created due to the shift should be removed
        
        Returns:
        dataset: dataset of lagged values which predict the next value, and the actual next value, ready for use in a pytorch dataloader
        df_bucketed: dataframe with lagged values and predicted values
        nan_indices: indices of time buckets that contained a nan value
        '''
        
        # setting seeds for reproducibility
        np.random.seed(42)
        
        # get column name
        bucketed_target_col_name = df_bucketed.columns[0]

        # keep a record of where there are any nan values in the dataframe. These will be the nan indices. Note that even 1 nan value will cause a nan for the number of steps around it
        nan_indices =  df_bucketed[df_bucketed.isna().any(axis=1)].index

        # remove the nans.
        # for training this means the FFNN will function
        # for testing, we have a record of any nan values already
        df_bucketed = df_bucketed[~df_bucketed.isna().any(axis=1)]


        # create lagged variables in dataframe, with names stating their lagged positions
        # each row of the dataframe is now an actual value and all previous values that can be used to predict it, note than nan value have been removed and it will look for the last n avalable non nan values
        for n in range(1, steps+1):
            df_bucketed['lag' + str(n)] = df_bucketed[bucketed_target_col_name].shift(n)
        
        # using "shift" will result in the first few values being nan values, can remove if desired
        if remove_initial_nans:
            df_bucketed = df_bucketed.iloc[steps:]
        
        # creating a list of arrays for the dataset
        all_lagged_values = []
        for i in range(len(df_bucketed)):
            lagged_values_on_bucket = []
            for n in range(1, steps+1):
                lagged_values_on_bucket.append(df_bucketed['lag' + str(n)][i])
            lagged_array = np.reshape(np.array(lagged_values_on_bucket), (steps,1))
            all_lagged_values.append(lagged_array)
    
        # list of lists, first item is the array, second item is the value
        dataset = [[all_lagged_values[i], df_bucketed[bucketed_target_col_name][i]] for i in range(len(df_bucketed))]
        
        return(dataset, df_bucketed, nan_indices)
      
    def bucket_and_get_dataset_for_NN(df, interval_unit, interval_size, aggregation, target_column, steps,
                                      date_from,
                                      date_to,
                                      remove_nans=False,
                                      remove_initial_nans=False,
                                      disregard_custom_indices_intervals=False):
        '''shortcut function'''
        
        df_bucketed = bucket_dataframe(df, interval_unit=interval_unit, interval_size=interval_size, aggregation=aggregation, target_column=target_column, date_from=date_from, date_to=date_to, remove_nans=remove_nans, disregard_custom_indices_intervals=disregard_custom_indices_intervals)
    
        
        return(get_dataset_for_NN(df_bucketed=df_bucketed, steps=steps,
                                  remove_initial_nans=remove_initial_nans))

    def predict_with_NN(NN_prepared_dataset, model, steps, df_bucketed, nan_indices, get_anomalies=False, plot=True, proportional_difference_threshold=1.5, save_name=None, rmse_upper_threshold=1000):
        '''use a NN prepared dataset and a model to predict values. Anomalies are identifed by the proportional difference from the predicted value and the actual value. Predicted vs Actual with scattered anomalies can be plotted.
        
        Parameters:
        NN_prepared_dataset: dataset from get_dataset_for_NN
        model: model to use for prediction
        steps: number of features to use for prediction, must match NN_prepared_dataset features
        df_bucketed: Dataframe bucketed by time, same as output from get_dataset_for_NN
        nan_indices: record of any nan_indices identifed in get_dataset_for_NN, these are seperated and then removed so that the NN can function
        get_anomalies: choose whether anomalies will be identified
        plot: choose whether plots will be made
        proportional_difference_threshold: the proportion of the predicted value that the actual value must exceed to be flagged as anomalous
        
        Returns:
        predictions_dataframe: dataframe of predictions, actual values, MSE and thresholds
        anomalies: dataframe of anomalies
        nan_anomalies: dataframe of nan anomalies
        '''
        bucketed_target_col_name = df_bucketed.columns[0]
        
        # get the preditive part of the dataset
        all_x_data = torch.tensor([NN_prepared_dataset[i][0].tolist() for i in range(len(NN_prepared_dataset))])
        
        # load the model, will be named something specific in training
        model.load_state_dict(torch.load(folder_path + str(save_name) + "_" + model.__class__.__name__ + "_" + str(steps) + "_step"))
        
        # get model output
        model_output = model(torch.tensor(all_x_data))
        
        # make into list, these are the predictions
        output = [model_output[i].item() for i in range(len(model_output))]
        
        # make dataframe from preditive values and predictions
        # note that the indices will match exactly, as long as the same df_buckted returned from the "get_dataset_for_NN" function is used
        predictions_dataframe = pd.DataFrame(data={'predictions':output, 'actual':df_bucketed.iloc[:,0], 'dt':df_bucketed.index}).set_index('dt')
        
        # correction, assume same prediction for the first few
        predictions_dataframe['predictions'][0:steps] = predictions_dataframe['predictions'][steps]
        
        # get MSE for predictions vs actual
#        predictions_dataframe['MSE'] = (predictions_dataframe['predictions'] - predictions_dataframe['actual'])**2
        predictions_dataframe['RMSE'] = predictions_dataframe.apply(lambda x: (np.sqrt((x['actual'] - x['predictions'])**2)), axis=1)
        
        if get_anomalies:
            st.code('RMSE of Predictions ' + str(round(sum(predictions_dataframe['RMSE']), 2)))
            predictions_dataframe['rolling_upper_threshold'] = rmse_upper_threshold
    
            # get anomalies
            predictions_dataframe['anomalous_flag'] = predictions_dataframe.apply(lambda x: 1 if x['RMSE'] > x['rolling_upper_threshold'] else 0, axis=1)
    
            # on one axis, plot the predicted vs actual values, on the other plot the mse and scatter the anomalies
            if plot:
                fig2, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12,8))
                predictions_dataframe['actual'].plot(ax=ax1);
                predictions_dataframe['predictions'].plot(ax=ax1);
                ax1.scatter(x=predictions_dataframe[predictions_dataframe['anomalous_flag'] == 1].index.tolist(), y=predictions_dataframe[predictions_dataframe['anomalous_flag'] == 1]['actual'], s=10,c='r')
#                ax1.scatter(x=nan_indices, y=[0]*len(nan_indices), s=10, c='k')
                ax1.set_xlabel('Time ' + "(" + str(interval_size) + " " + str(interval_unit) + " intervals)")
                ax1.set_ylabel(bucketed_target_col_name)
                
                ax1.legend(['actual value', 'predicted value', 'outliers'], loc='upper right')
                ax1.set_title("Outlier Detection - Feed Forward Neural Network")
                predictions_dataframe['RMSE'].plot(ax=ax2, c='green')
                ax2.legend()
                ax2.fill_between(predictions_dataframe.index, 0, predictions_dataframe['rolling_upper_threshold'], alpha=0.2, color='grey')
                ax2.scatter(x=nan_indices, y=[0]*len(nan_indices), s=10, c='k')
                ax2.set_title("RMSE of Feed Forward Neural Network Predictions")
                ax2.legend(['value', 'outlier detection tolerance'], loc='upper right')
                ax2.set_xlabel('Time ' + "(" + str(interval_size) + " " + str(interval_unit) + " intervals)")   
                ax2.set_ylabel('RMSE')
                fig2.tight_layout()    
                st.pyplot()

            # put anomalies into dataframe
            anomalies = predictions_dataframe[predictions_dataframe['anomalous_flag'] == 1]
            # actually done nothing with the anomalous indices here, these are just carried though from the get_dataset_for_NN function, and returned again here for consistency with other functions
            nan_anomalies = pd.DataFrame(nan_indices)
            nan_anomalies.columns = ['dt']
            nan_anomalies = nan_anomalies.set_index('dt')
                   
            return(predictions_dataframe, anomalies, nan_anomalies)
            
        return(predictions_dataframe) 
    
    def predict_with_NN_backup(NN_prepared_dataset, model, steps, df_bucketed, nan_indices, get_anomalies=False, plot=True, proportional_difference_threshold=1.5, save_name=None):
        '''use a NN prepared dataset and a model to predict values. Anomalies are identifed by the proportional difference from the predicted value and the actual value. Predicted vs Actual with scattered anomalies can be plotted.
        
        Parameters:
        NN_prepared_dataset: dataset from get_dataset_for_NN
        model: model to use for prediction
        steps: number of features to use for prediction, must match NN_prepared_dataset features
        df_bucketed: Dataframe bucketed by time, same as output from get_dataset_for_NN
        nan_indices: record of any nan_indices identifed in get_dataset_for_NN, these are seperated and then removed so that the NN can function
        get_anomalies: choose whether anomalies will be identified
        plot: choose whether plots will be made
        proportional_difference_threshold: the proportion of the predicted value that the actual value must exceed to be flagged as anomalous
        
        Returns:
        predictions_dataframe: dataframe of predictions, actual values, MSE and thresholds
        anomalies: dataframe of anomalies
        nan_anomalies: dataframe of nan anomalies
        '''
        bucketed_target_col_name = df_bucketed.columns[0]
        
        # get the preditive part of the dataset
        all_x_data = torch.tensor([NN_prepared_dataset[i][0].tolist() for i in range(len(NN_prepared_dataset))])
        
        # load the model, will be named something specific in training
        model.load_state_dict(torch.load(folder_path + str(save_name) + "_" + model.__class__.__name__ + "_" + str(steps) + "_step"))
        
        # get model output
        model_output = model(torch.tensor(all_x_data))
        
        # make into list, these are the predictions
        output = [model_output[i].item() for i in range(len(model_output))]
        
        # make dataframe from preditive values and predictions
        # note that the indices will match exactly, as long as the same df_buckted returned from the "get_dataset_for_NN" function is used
        predictions_dataframe = pd.DataFrame(data={'predictions':output, 'actual':df_bucketed.iloc[:,0], 'dt':df_bucketed.index}).set_index('dt')
        
        # correction, assume same prediction for the first few
        predictions_dataframe['predictions'][0:steps] = predictions_dataframe['predictions'][steps]
        
        # get MSE for predictions vs actual
#        predictions_dataframe['MSE'] = (predictions_dataframe['predictions'] - predictions_dataframe['actual'])**2
        predictions_dataframe['RMSE'] = predictions_dataframe.apply(lambda x: (np.sqrt((x['actual'] - x['predictions'])**2)), axis=1)
        
        st.code('RMSE of predictions ' + str(round(sum(predictions_dataframe['RMSE']), 2)))
        
        # anomalies
        if get_anomalies:
            # get thresholds
            predictions_dataframe['rolling_upper_threshold'] =  proportional_difference_threshold*predictions_dataframe['predictions']
            predictions_dataframe['rolling_lower_threshold'] =  (1/proportional_difference_threshold)*predictions_dataframe['predictions']
    
            # get anomalies using thresholds
            predictions_dataframe['anomalous_flag'] = predictions_dataframe.apply(lambda x: 1 if (x['actual'] < x['rolling_lower_threshold']) or
                                                         (x['actual'] > x['rolling_upper_threshold']) else 0, axis=1)
    
            # plot predictions vs actual on one axis, and predictions with thresholds and scattered anomalies on another
            if plot:
                fig2, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12,12))
                predictions_dataframe['actual'].plot(figsize=(12,12), ax=ax1)
                ax1.set_xlabel('Time ' + "(" + str(interval_size) + " " + str(interval_unit) + " intervals)") 
                
                predictions_dataframe[['actual', 'predictions']].plot(ax=ax2);
#                predictions_dataframe['predictions'].plot(figsize=(12,12), ax=ax2)
                plt.legend()
                ax2.scatter(x=predictions_dataframe[predictions_dataframe['anomalous_flag'] == 1].index.tolist(), y=predictions_dataframe[predictions_dataframe['anomalous_flag'] == 1]['actual'], s=10,c='r')
#                ax2.scatter(x=nan_indices, y=[0]*len(nan_indices), s=5, c='k')
                ax2.fill_between(predictions_dataframe.index, predictions_dataframe['rolling_lower_threshold'], predictions_dataframe['rolling_upper_threshold'], alpha=0.2, color='grey')
                ax2.legend(['actual value', 'predicted value', 'outliers', "outlier detection threshold"], loc='best')
                ax2.scatter(x=nan_indices, y=[0]*len(nan_indices), s=5, c='k')
                ax2.set_title('Outlier Detection - Feed Forward Neural Network (FFNN)')
                ax2.set_xlabel('Time ' + "(" + str(interval_size) + " " + str(interval_unit) + " intervals)")
                fig2.tight_layout()
                st.pyplot()
                            
            # put anomalies into dataframe
            anomalies = predictions_dataframe[predictions_dataframe['anomalous_flag'] == 1]
            # actually done nothing with the anomalous indices here, these are just carried though from the get_dataset_for_NN function, and returned again here for consistency with other functions
            nan_anomalies = pd.DataFrame(nan_indices)
            nan_anomalies.columns = ['dt']
            nan_anomalies = nan_anomalies.set_index('dt')
            
            # state proportion of data anomalous and nan_anomalous
            # adding the nan anomalies to the predictions dataframe here to get the length since the nans were removed intially...
            st.code('proportion of anomalous data :' + str(round(len(anomalies)/(len(nan_anomalies) + len(predictions_dataframe)), 5)))
            st.code('proportion of na anomalous data :' + str(round(len(nan_anomalies)/(len(nan_anomalies) + len(predictions_dataframe)), 5)))        
        
            return(predictions_dataframe, anomalies, nan_anomalies)
        
        # plot predictions vs actual
        if plot:
            predictions_dataframe[['predictions', 'actual']].plot(figsize=(15,5)); 
            
        return(predictions_dataframe)    
    

    def bucket_and_predict_with_NN(df, model, steps, interval_unit, interval_size, predict_date_from, predict_date_to, aggregation='count', target_column='ID', remove_nans=False, remove_initial_nans=False, disregard_custom_indices_intervals=False, plot=True, get_anomalies=False, proportional_difference_threshold=1.5, save_name=None, rmse_upper_threshold=1000):
        '''shortcut function'''
        
        df_bucketed = bucket_dataframe(df=df, interval_unit=interval_unit, interval_size=interval_size, aggregation=aggregation, target_column=target_column, date_from=predict_date_from, date_to=predict_date_to, remove_nans=remove_nans, disregard_custom_indices_intervals=disregard_custom_indices_intervals)
        NN_prepared_dataset, df_bucketed, nan_indices = get_dataset_for_NN(df_bucketed=df_bucketed, steps=steps, remove_initial_nans=remove_initial_nans)
        predictions_dataframe = predict_with_NN(NN_prepared_dataset=NN_prepared_dataset, model=model, steps=steps, df_bucketed=df_bucketed, nan_indices=nan_indices, get_anomalies=get_anomalies, plot=plot, rmse_upper_threshold=rmse_upper_threshold, save_name=save_name)
        
        return(predictions_dataframe)
        
        
    # define classes for FFNN
    # 10 features used, (10 lagged variables to predict the next)
    n_features_ffnn = 10
    
    # each feature is just a number, so 1 dimension
    feature_size_ffnn = 1
    
    # 1 output node, the predicted value
    output_nodes_ffnn = 1
    
    class linear_net_regression(nn.Module):
        def __init__(self):
            super(linear_net_regression, self).__init__()
            self.layer1 = nn.Linear(n_features_ffnn * feature_size_ffnn, 3)
            self.layer2 = nn.Linear(3, output_nodes_ffnn)
            
            self.dropout = nn.Dropout(p=0) # too few weights/ features to justify dropout
        def forward(self, x):
            x = x.float().to(device)
            x = x.view(-1, n_features_ffnn * feature_size_ffnn)
            x = self.dropout(F.relu(self.layer1(x)))     
            x = self.layer2(x)
            return(x) # no need for any softmax, that only makes sense for classification!
    
    # function to get MSE, works with the "train" function

    def get_mse(model, data_to_test, criterion, return_loss = True, return_predictions=False):
        '''works alongside "train" function, computes mean squared error for a dataloader input, can return the predicted values, losses and mean squared errors'''
        
        
        # get the information desired and return
        with torch.no_grad():
            # this isn't a loop, there is only 1 iteration
            for _, (X_testing, y_testing) in enumerate(data_to_test):
                # use float tensors for continuous
                y_testing = y_testing.type(torch.FloatTensor)
                # get predictions
                predictions_testing = model(X_testing)
                if return_loss:
                    test_loss = criterion(predictions_testing, y_testing)
                test_mse = mean_squared_error(y_testing, predictions_testing)
                
        if return_predictions:
            if return_loss:
                return predictions_testing, y_testing, test_loss, test_mse
            return predictions_testing, y_testing, test_mse
        
        if return_loss:
            return test_loss, test_mse
        
        return test_mse
    

    def train(dataset, steps, train_test_split, model, learning_rate = 0.05, EPOCHS = 10, batch_size = 32,
              show_plots = True, print_accuracies = True, weight_decay = 0.0001, retrain=False, save_name=None):
        '''train a neural network model. Save the weights of the model at the point where it achieves the best accuracy (MSE) on all the testing data
        can show the plots of the training, print training metrics
        
        Parameters:
            dataset: Prepared dataset from get_NN_prepared_dataset or get_ae_prepared_dataset
            steps: number of features (lagged variables). Must match that used in get_NN_prepared_dataset
            model: model to use, e.g. linear_net_regression()
            learning_rate
            EPOCHS
            batch_size
            show_plots: show loss and MSE graphs
            print_accuracies: print accuracy on each batch iteration
            weight_decay
            retrain: choose whether to retrain the model or use saved weights (saved weights must exist)
        
        Returns
            saved_MSE_testing: The best MSE achieved on the testing data. The weights of the network will have been saved at the point at which the network achieved that accuracy
        '''
        
        # set seeds for reproducibility
        
        np.random.seed(42)
        random.seed(42)
        
        # shuffle dataset to ensure random train/test split
        random.shuffle(dataset) # no need for reassigment, shuffle is applied
    
        # get train/test split
        train_ds = dataset[0:int(train_test_split*len(dataset))]
        test_ds = dataset[int(train_test_split*len(dataset)):]
        
        # put into dataloaders
        train_data_full = DataLoader(train_ds, batch_size=len(train_ds))
        test_data_full = DataLoader(test_ds, batch_size=len(test_ds))
        all_data = DataLoader(dataset, batch_size=len(dataset))

        if retrain:
            train_iter = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, num_workers=0, shuffle=True)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay = weight_decay)
            iteration_list, train_losses, test_losses, mse_list_training, mse_list_testing = [], [], [], [], []
    
            print('approx number of batches', len(train_ds)*EPOCHS/batch_size)
            n = 0
            for epoch in range(EPOCHS):
                for i, (X_training, y_training) in enumerate(train_iter):
                    y_training = y_training.type(torch.FloatTensor)
                    n += 1
                    train_loss = 0
                    test_loss = 0
                    # getting outputs for each batch
                    output = model(X_training)
                    # computing cross_entropy_loss
                    train_loss = criterion(output, y_training)
                    # backpropogation
                    train_loss.backward()
                    optimizer.step()
                    # reset gradients for the next batch
                    optimizer.zero_grad()
    
                    # get the accuracy (MSE) for the entire training data
                    mse_training = get_mse(model = model, data_to_test= train_data_full, criterion = criterion,
                                                                return_loss = False)
                    
                    # get the accuracy (MSE) and predictions for the entire testing data
                    predictions_testing, y_testing, test_loss, mse_testing = get_mse(model = model, data_to_test=test_data_full,
                                                                                    criterion = criterion, return_predictions = True)
    
                    # append iterations, accuracies and losses to lists
                    iteration_list.append(n)
                    mse_list_training.append(mse_training)
                    mse_list_testing.append(mse_testing)
                    train_losses.append(train_loss.item())
                    test_losses.append(test_loss.item())
    
                    # can print accuracies on each batch if required
                    if print_accuracies:
                        print("Batch:", n, "mse_training:", mse_training, "mse_list_testing:", mse_testing)
                    
                    
                    # save at the iteration which achieves the best accuracy
                    if mse_testing == min(mse_list_testing):
                        saved_iteration = n
                        saved_mse_testing = mse_testing
                        saved_mse_training = mse_training
                        saved_loss_testing = test_loss.item()
                        saved_loss_training = train_loss.item()
                        
                        # save the model
                        torch.save(model.state_dict(), folder_path + str(save_name) + "_" + model.__class__.__name__ + "_" + str(steps) + "_step")
    
                        # pickle all information to that point for reference and visualisation
                        with open(folder_path + str(save_name) + "_" + model.__class__.__name__ + "_" + str(steps) + "_step" + "_" + "saved_mse_testing.txt", "wb") as fp:   #Pickling
                            pickle.dump(saved_mse_testing, fp)  
                        with open(folder_path + str(save_name) + "_" + model.__class__.__name__ + "_" + str(steps) + "_step" + "_" + "iteration_list.txt", "wb") as fp:   #Pickling
                            pickle.dump(iteration_list, fp)        
                        with open(folder_path + str(save_name) + "_" + model.__class__.__name__ + "_" + str(steps) + "_step" + "_" + "mse_list_training.txt", "wb") as fp:   #Pickling
                            pickle.dump(mse_list_training, fp)
                        with open(folder_path + str(save_name) + "_" + model.__class__.__name__ + "_" + str(steps) + "_step" + "_" + "mse_list_testing.txt", "wb") as fp:   #Pickling
                            pickle.dump(mse_list_testing, fp)
                        with open(folder_path + str(save_name) + "_" + model.__class__.__name__ + "_" + str(steps) + "_step" + "_" + "train_losses.txt", "wb") as fp:   #Pickling
                            pickle.dump(train_losses, fp)
                        with open(folder_path + str(save_name) + "_" + model.__class__.__name__ + "_" + str(steps) + "_step" + "_" + "test_losses.txt", "wb") as fp:   #Pickling
                            pickle.dump(test_losses, fp)
                        with open(folder_path + str(save_name) + "_" + model.__class__.__name__ + "_" + str(steps) + "_step" + "_" + "y_testing.txt", "wb") as fp:   #Pickling
                            pickle.dump(y_testing, fp)           
                        with open(folder_path + str(save_name) + "_" + model.__class__.__name__ + "_" + str(steps) + "_step" + "_" + "last_predictions.txt", "wb") as fp:   #Pickling
                            pickle.dump(predictions_testing, fp)
    
        # after retraining, or if not retrained at all, load all the info
        with open(folder_path + str(save_name) + "_" + model.__class__.__name__ + "_" + str(steps) + "_step" + "_" + "saved_mse_testing.txt", "rb") as fp:   # Unpickling
            saved_mse_testing = pickle.load(fp)
        with open(folder_path + str(save_name) + "_" + model.__class__.__name__ + "_" + str(steps) + "_step" + "_" + "iteration_list.txt", "rb") as fp:   # Unpickling
            iteration_list = pickle.load(fp)
        with open(folder_path + str(save_name) + "_" + model.__class__.__name__ + "_" + str(steps) + "_step" + "_" + "mse_list_training.txt", "rb") as fp:   # Unpickling
            mse_list_training = pickle.load(fp)
        with open(folder_path + str(save_name) + "_" + model.__class__.__name__ + "_" + str(steps) + "_step" + "_" + "mse_list_testing.txt", "rb") as fp:   # Unpickling
            mse_list_testing = pickle.load(fp)
        with open(folder_path + str(save_name) + "_" + model.__class__.__name__ + "_" + str(steps) + "_step" + "_" + "train_losses.txt", "rb") as fp:   # Unpickling
            train_losses = pickle.load(fp)
        with open(folder_path + str(save_name) + "_" + model.__class__.__name__ + "_" + str(steps) + "_step" + "_" + "test_losses.txt", "rb") as fp:   # Unpickling
            test_losses = pickle.load(fp)
        with open(folder_path + str(save_name) + "_" + model.__class__.__name__ + "_" + str(steps) + "_step" + "_" + "y_testing.txt", "rb") as fp:   # Unpickling
            y_testing = pickle.load(fp)        
        with open(folder_path + str(save_name) + "_" + model.__class__.__name__ + "_" + str(steps) + "_step" + "_" + "last_predictions.txt", "rb") as fp:   # Unpickling
            predictions_testing = pickle.load(fp)
        
        # plot losses on one axis and accuracy on another, for both training and testing        
        if show_plots:
            # loss graph
            fig = plt.figure(figsize=(15,12))
            ax1 = fig.add_subplot(2,1,1)
            ax1.set_title('Training and Testing Losses over Batch Iterations')
            ax1.plot(iteration_list, train_losses, label = 'Training Losses')
            ax1.plot(iteration_list, test_losses, label = 'Testing Losses')
            
            plt.legend()

            
            # accuracy graph
            ax2 = fig.add_subplot(2,1,2)
            ax2.set_title('Train MSE and Test MSE over Batch Iterations')
            ax2.plot(iteration_list, mse_list_training, label = 'Training MSE')
            ax2.plot(iteration_list, mse_list_testing, label = 'Testing MSE')
            
            plt.legend()
            st.pyplot()
    
        # print some useful information and return saved value for the accuracy of the model on the testing data, whenever it occurred
        if retrain:
            st.write("Lowest test MSE:", saved_mse_testing)
            st.write("Lowest MSE on iteration:", saved_iteration)
            st.write("training MSE on that iteration:", saved_mse_training)
            st.write("testing loss on that iteration:", saved_loss_testing)
            st.write("training loss on that iteration:", saved_loss_training)
            return(saved_mse_testing)
        else:
            st.write("Lowest test MSE:", saved_mse_testing)
            return(saved_mse_testing)


    # define classes for FFNN
    
    # 10 features since 10 lagged variables predict
    # for autoencoder the network attempts to predict its own input, no no need for output, as it is the input
    n_features_ae = 10
    
    # feature size is 1 as each predictive variable is just a valu
    feature_size_ae = 1
    
    class autoencoder(nn.Module):
        def __init__(self):
            super(autoencoder, self).__init__()
            self.layer1 = nn.Linear(n_features_ae * feature_size_ae, 4)
            self.layer2 = nn.Linear(4, 3)
            self.layer3 = nn.Linear(3, 2)        
            self.layer4 = nn.Linear(2, 3)
            self.layer5 = nn.Linear(3, 4)        
            self.layer6 = nn.Linear(4, n_features_ae * feature_size_ae)
            
            self.dropout = nn.Dropout(p=0) # too few weights/ features to justify dropout
        def forward(self, x):
            x = x.float().to(device)
            x = x.view(-1, n_features_ae * feature_size_ae)
            x = self.dropout(F.relu(self.layer1(x)))
            x = self.dropout(F.relu(self.layer2(x)))
            x = self.dropout(F.relu(self.layer3(x)))
            x = self.dropout(F.relu(self.layer4(x)))
            x = self.dropout(F.relu(self.layer5(x)))        
            reconstructed = self.layer6(x)
            return(reconstructed) # no need for any softmax, that only makes sense for classification!
            

    def get_dataset_for_AutoEncoder(df_bucketed, steps, remove_initial_nans=False):
        
        '''
        Exact same as get_dataset_for_NN, but now the response variable is the same as the explanatory variable
        Takes a bucketed dataframe and creates a dataset that could be used used in a pytorch dataloader for an autoencoder.
        The number of features to be created is equal to "steps", which is the number of values will be compressed and decompressed, where the autoencoder attempts to predict its own input.
        format is list of list of 2 elements, where the first element is a (features x 1) sized numpy array and the second element is the value.
        For training, a bucketed dataframe with nan values and disregard indices REMOVED should be used. Therefore the dataframe will put each set of n values together, "skipping over" nan values and using the next available non-nan value.
        For testing, nan values should NOT be disregarded, and should be captured with flags.
        When creating the dataset, the first few values equal to the number of steps will be nan as a result of the shift, these can be removed with "remove_initial_nans"
        
        
        Parameters:
        df_bucketed: Dataframe bucketed by time
        steps: number of previous time intervals used to predict the current time interval (will skip over nan values)
        remove_initial_nans: Determines whether the first few nan values created due to the shift should be removed
        
        Returns:
        dataset: dataset of lagged values, ready for use in a pytorch dataloader and subsequently an autoencoder
        df_bucketed: dataframe with input and output lists for the autoencoder
        nan_indices: indices of time buckets that contained a nan value
        '''
    
        # setting seeds for reproducibility
        np.random.seed(42)
        n_features = 1
        
        # get column name
        bucketed_target_col_name = df_bucketed.columns[0]
        
        # save nan indices for later, then remove
        nan_indices =  df_bucketed[df_bucketed.isna().any(axis=1)].index
        df_bucketed = df_bucketed[~df_bucketed.isnull().any(axis=1)]
        
        
        for n in range(1, steps+1):
            df_bucketed['lag' + str(n)] = df_bucketed[bucketed_target_col_name].shift(n)
        
        # using "shift" will result in the first few values being nan values, can remove if desired
        if remove_initial_nans:
            df_bucketed = df_bucketed.iloc[steps:]

        # creating a list of arrays for the dataset     
        all_lagged_values = []
        for i in range(len(df_bucketed)):
            lagged_values_on_minute = []
            for n in range(1, steps+1):
                lagged_values_on_minute.append(df_bucketed['lag' + str(n)][i])
            lagged_array = np.reshape(np.array(lagged_values_on_minute), (steps,1))
            all_lagged_values.append(lagged_array)
            
        # For the autoeconder, the target values are the explanatory variable values
        dataset = [[all_lagged_values[i], all_lagged_values[i]] for i in range(len(df_bucketed))]
        
        return(dataset, df_bucketed, nan_indices)
        
    def bucket_and_get_dataset_for_AutoEncoder(df, interval_unit, interval_size, aggregation, target_column, steps,
                                      date_from,
                                      date_to,
                                      remove_nans=False,
                                      remove_initial_nans=False,
                                      disregard_custom_indices_intervals=False):

        '''shortcut function'''
        df_bucketed = bucket_dataframe(df=df, interval_unit=interval_unit, interval_size=interval_size,aggregation=aggregation, target_column=target_column, date_from=date_from, date_to=date_to, remove_nans=remove_nans, disregard_custom_indices_intervals=disregard_custom_indices_intervals)
    
        
        return(get_dataset_for_AutoEncoder(df_bucketed=df_bucketed, steps=steps,
                                  remove_initial_nans=remove_initial_nans))

    def predict_with_autoencoder(AE_prepared_dataset, model, steps, df_bucketed, nan_indices, plot=True, get_anomalies=False, rmse_upper_threshold=100000, save_name=None, prediction_aggregation='mean'):
        
        '''use a autoencoder prepared dataset and a model to predict values. 
        The lists that the autoencoder generates are seperated out and the mean of every prediction for each time interval is taken, that mean value is then the final prediction for that time interval.
        Anomalies are identifed by whether the MSE for the prediction exceeds a certain value. Predicted vs Actual with scattered anomalies can be plotted.
        
        Parameters:
        AE_prepared_dataset: dataset from get_dataset_for_autoencoder
        model: model to use for prediction
        steps: number of features to use for prediction, must match AE_prepared_dataset features
        df_bucketed: Dataframe bucketed by time, same as output from get_dataset_for_autoencoder
        nan_indices: record of any nan_indices identifed in get_dataset_for_autoencoder, these are seperated and then removed so that the autoencoder can function
        plot: choose  whether plots will be made
        get_anomalies: choose whether anomalies will be identified
        rmse_upper_threshold: the proportion of the predicted value that the actual value must exceed to be flagged as anomalous
        
        Returns:
        predictions_dataframe: dataframe of predictions, actual values, MSE and thresholds
        anomalies: dataframe of anomalies
        nan_anomalies: dataframe of nan anomalies
        '''
        
        bucketed_target_col_name = df_bucketed.columns[0]
        # load the model weights
        model.load_state_dict(torch.load(folder_path + str(save_name) + "_" + model.__class__.__name__ + "_" + str(steps) + "_step"))
        
        # get x data
        all_x_data = torch.tensor([AE_prepared_dataset[i][0].tolist() for i in range(len(AE_prepared_dataset))])

        # assumption here, the first n values contain at least 1 nan value, just assume that they were all equal the the n+1 step
        for i in range(steps):
            all_x_data[i] = all_x_data[steps]

        # get input of the autoencoder as lists (don't include the first few values (number=steps), as they are not the correct length)
        original_input = [all_x_data[i].view(1,-1).tolist()[0] for i in range(len(all_x_data))]
        
        # get the model output, convert to lists
        model_output = model(torch.tensor(all_x_data))
        autoencoded_output = [model_output[i].tolist() for i in range(len(model_output))]    
        
        # getting first few values, using whatever data is available
        # getting the rest of the values, allowing for end point which will vary in length
        new_big_list = []
        for i in range(len(autoencoded_output)):
            new_small_list = []
        #     for j in range(len(big_list[i]))
            for n in range(steps):
                try:
                    new_small_list.append(autoencoded_output[i+n][-(n+1)])
                except:
                    pass
            new_big_list.append(new_small_list)
        
        predictions_vector_list = new_big_list
        
        
        if prediction_aggregation == 'mean':
            agg_predictions = [np.mean(predictions_vector) for predictions_vector in predictions_vector_list]
        if prediction_aggregation == 'median':
            agg_predictions = [np.median(predictions_vector) for predictions_vector in predictions_vector_list]
        if prediction_aggregation == 'min':
            agg_predictions = [np.min(predictions_vector) for predictions_vector in predictions_vector_list]
        if prediction_aggregation == 'max':
            agg_predictions = [np.max(predictions_vector) for predictions_vector in predictions_vector_list]
        
    
        predictions_dataframe = pd.DataFrame(data={'actual': df_bucketed.iloc[:,0], 'predictions':agg_predictions, 'dt':df_bucketed.index[:]}).set_index('dt')
        
        
        predictions_dataframe['RMSE'] = predictions_dataframe.apply(lambda x: (np.sqrt((x['actual'] - x['predictions'])**2)), axis=1)
        
        if get_anomalies:
            st.code('RMSE of Predictions after aggregation ' + str(round(sum(predictions_dataframe['RMSE']), 2)))
            predictions_dataframe['rolling_upper_threshold'] = rmse_upper_threshold
    
            # get anomalies
            predictions_dataframe['anomalous_flag'] = predictions_dataframe.apply(lambda x: 1 if x['RMSE'] > x['rolling_upper_threshold'] else 0, axis=1)
    
            # on one axis, plot the predicted vs actual values, on the other plot the mse and scatter the anomalies
            if plot:
                fig2, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12,8))
                fig2.tight_layout()
                predictions_dataframe['actual'].plot(ax=ax1);
                predictions_dataframe['predictions'].plot(ax=ax1);
                ax1.scatter(x=predictions_dataframe[predictions_dataframe['anomalous_flag'] == 1].index.tolist(), y=predictions_dataframe[predictions_dataframe['anomalous_flag'] == 1]['actual'], s=10,c='r')
                ax1.scatter(x=nan_indices, y=[0]*len(nan_indices), s=10, c='k')
                ax1.legend(['actual value', 'predicted value', 'outliers'], loc='upper right')
                ax1.set_title("Outlier Detection - Autoencoder")
                ax1.set_xlabel('Time ' + "(" + str(interval_size) + " " + str(interval_unit) + " intervals)")  
                ax1.set_ylabel(bucketed_target_col_name)
                
                predictions_dataframe['RMSE'].plot(ax=ax2, c='green')
                ax2.fill_between(predictions_dataframe.index, 0, predictions_dataframe['rolling_upper_threshold'], alpha=0.2, color='grey')
                ax2.scatter(x=nan_indices, y=[0]*len(nan_indices), s=10, c='k')
                ax2.legend(['value', 'outlier detection tolerance'], loc='upper right')
                ax2.set_xlabel('Time ' + "(" + str(interval_size) + " " + str(interval_unit) + " intervals)") 
                ax2.set_ylabel('RMSE')
                ax2.set_title("RMSE of Autoencoder Predictions")
                fig2.tight_layout()   
                st.pyplot()



            # put anomalies into dataframe
            anomalies = predictions_dataframe[predictions_dataframe['anomalous_flag'] == 1]
            # actually done nothing with the anomalous indices here, these are just carried though from the get_dataset_for_NN function, and returned again here for consistency with other functions
            nan_anomalies = pd.DataFrame(nan_indices)
            nan_anomalies.columns = ['dt']
            nan_anomalies = nan_anomalies.set_index('dt')
                    
            return(predictions_dataframe, anomalies, nan_anomalies)            
        # plot actual values one axis, and mse on another
            
        return(predictions_dataframe) 
        
    def bucket_and_predict_with_autoencoder(df, model, interval_unit, interval_size, steps, predict_date_from, predict_date_to, aggregation='count', target_column='ID', remove_nans=False, remove_initial_nans=False, plot=True, disregard_custom_indices_intervals=False, get_anomalies=False, rmse_upper_threshold=100000, save_name=None, prediction_aggregation='mean'):
        '''shortcut function'''
        
        
        df_bucketed = bucket_dataframe(df=df, interval_unit=interval_unit, interval_size=interval_size, aggregation=aggregation, target_column=target_column, date_from=predict_date_from, date_to=predict_date_to, remove_nans=remove_nans, disregard_custom_indices_intervals=disregard_custom_indices_intervals)
        AE_prepared_dataset, df_bucketed, nan_indices = get_dataset_for_AutoEncoder(df_bucketed=df_bucketed, steps=steps, remove_initial_nans=remove_initial_nans)
        predictions_dataframe = predict_with_autoencoder(AE_prepared_dataset=AE_prepared_dataset, model=model, df_bucketed=df_bucketed, nan_indices=nan_indices, steps=steps, plot=plot, get_anomalies=get_anomalies, rmse_upper_threshold=rmse_upper_threshold, save_name=save_name, prediction_aggregation=prediction_aggregation)
        
        return(predictions_dataframe)

    def get_mse_autoencoder(model, data_to_test, criterion, return_loss = True, return_predictions=False):
        
        '''works alongside "train" function, computes mean squared error for a dataloader input, can return the predicted values, losses and mean squared errors
        only difference between get_mse_autoencoder and get_mse is that "view" is required for the autoencoder mse
        '''
        
        
        # get the information desired and return
        with torch.no_grad():
            # this isn't a loop, there is only 1 iteration
            for _, (X_testing, y_testing) in enumerate(data_to_test):
                y_testing = y_testing.type(torch.FloatTensor)
                predictions_testing = model(X_testing)
                if return_loss:
                    test_loss = criterion(predictions_testing.view(1, -1), y_testing.view(1, -1))
                test_mse = mean_squared_error(y_testing.view(1, -1), predictions_testing.view(1,-1))
        if return_predictions:
            if return_loss:
                return predictions_testing, y_testing, test_loss, test_mse
            return predictions_testing, y_testing, test_mse
        
        if return_loss:
            return test_loss, test_mse
        
        return test_mse
    
    def train_autoencoder(dataset, steps, train_test_split,  model, learning_rate = 0.005, EPOCHS = 10, batch_size = 64,
              show_plots = True, print_accuracies = True, weight_decay = 0.0001, retrain=False, save_name=None):
        
        '''train an autoencoder model. Save the weights of the model at the point where it achieves the best accuracy (MSE) on all the testing data
        can show the plots of the training, print training metrics
        only difference between "train_autoencoder" and "train" is that "view" is required for handling the autoencoder output
        
        Parameters:
            dataset: Prepared dataset from get_AE_prepared_dataset or get_ae_prepared_dataset
            steps: number of features (lagged variables). Must match that used in get_AE_prepared_dataset
            model: model to use, e.g. autoencoder()
            learning_rate
            EPOCHS
            batch_size
            show_plots: show loss and MSE graphs
            print_accuracies: print accuracy on each batch iteration
            weight_decay
            retrain: choose whether to retrain the model or use saved weights (saved weights must exist)
        
        Returns
            saved_MSE_testing: The best MSE achieved on the testing data. The weights of the network will have been saved at the point at which the network achieved that accuracy
        '''
        # set seed for reproducibility        
        np.random.seed(42)
        
        # shuffle to ensure random train test split
        random.shuffle(dataset)
        
        # get train test split
        train_ds = dataset[0:int(train_test_split*len(dataset))]
        test_ds = dataset[int(train_test_split*len(dataset)):]
        
        # get dataloader
        train_data_full = DataLoader(train_ds, batch_size=len(train_ds))
        test_data_full = DataLoader(test_ds, batch_size=len(test_ds))
        all_data = DataLoader(dataset, batch_size=len(dataset))
        
        if retrain:
            train_iter = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, num_workers=0, shuffle=True)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay = weight_decay)
            iteration_list, train_losses, test_losses, mse_list_training, mse_list_testing = [], [], [], [], []
    
            print('approx number of batches', len(train_ds)*EPOCHS/batch_size)
            n = 0
            for epoch in range(EPOCHS):
                for i, (X_training, y_training) in enumerate(train_iter):                 
   
                    y_training = y_training.type(torch.FloatTensor)            
                    n += 1
                    train_loss = 0
                    test_loss = 0
                    # getting outputs for each batch
                    output = model(X_training)
                    # computing loss
                    train_loss = criterion(output.view(1, -1), y_training.view(1, -1))
                    # backpropogation
                    train_loss.backward()
                    
                    optimizer.step()
                    # reset gradients for the next batch
                    optimizer.zero_grad()
    
                    # get the accuracy (mse) for the entire training data
                    mse_training = get_mse_autoencoder(model = model, data_to_test= train_data_full, criterion = criterion,
                                                                return_loss = False)
                    # get the accuracy (mse) and predictions and y labels for the entire testing data
                    predictions_testing, y_testing, test_loss, mse_testing = get_mse_autoencoder(model = model, data_to_test= test_data_full,
                                                                                    criterion = criterion, return_predictions = True)
    
                    # append iterations, accuracies and losses to lists
                    iteration_list.append(n)
                    mse_list_training.append(mse_training)
                    mse_list_testing.append(mse_testing)
                    train_losses.append(train_loss.item())
                    test_losses.append(test_loss.item())
    
                    if print_accuracies:
                        print("Batch:", n, "mse_training:", mse_training, "mse_list_testing:", mse_testing)
                    
                    # save at the iteration which achieves the best accuracy
                    if mse_testing == min(mse_list_testing):
                        saved_iteration = n
                        saved_mse_testing = mse_testing
                        saved_mse_training = mse_training
                        saved_loss_testing = test_loss.item()
                        saved_loss_training = train_loss.item()
                        
                        # save model and metrics at the point where the highest accuracy is achieved
                        torch.save(model.state_dict(), folder_path + str(save_name) + "_" + model.__class__.__name__ + "_" + str(steps) + "_step")
                        
                        with open(folder_path + str(save_name) + "_" + model.__class__.__name__ + "_" + str(steps) + "_step" + "_" + "saved_mse_testing.txt", "wb") as fp:   #Pickling
                            pickle.dump(saved_mse_testing, fp)                 
                        with open(folder_path + str(save_name) + "_" + model.__class__.__name__ + "_" + str(steps) + "_step" + "_" + "iteration_list.txt", "wb") as fp:   #Pickling
                            pickle.dump(iteration_list, fp)        
                        with open(folder_path + str(save_name) + "_" + model.__class__.__name__ + "_" + str(steps) + "_step" + "_" + "mse_list_training.txt", "wb") as fp:   #Pickling
                            pickle.dump(mse_list_training, fp)
                        with open(folder_path + str(save_name) + "_" + model.__class__.__name__ + "_" + str(steps) + "_step" + "_" + "mse_list_testing.txt", "wb") as fp:   #Pickling
                            pickle.dump(mse_list_testing, fp)
                        with open(folder_path + str(save_name) + "_" + model.__class__.__name__ + "_" + str(steps) + "_step" + "_" + "train_losses.txt", "wb") as fp:   #Pickling
                            pickle.dump(train_losses, fp)
                        with open(folder_path + str(save_name) + "_" + model.__class__.__name__ + "_" + str(steps) + "_step" + "_" + "test_losses.txt", "wb") as fp:   #Pickling
                            pickle.dump(test_losses, fp)
                        with open(folder_path + str(save_name) + "_" + model.__class__.__name__ + "_" + str(steps) + "_step" + "_" + "y_testing.txt", "wb") as fp:   #Pickling
                            pickle.dump(y_testing, fp)           
                        with open(folder_path + str(save_name) + "_" + model.__class__.__name__ + "_" + str(steps) + "_step" + "_" + "last_predictions.txt", "wb") as fp:   #Pickling
                            pickle.dump(predictions_testing, fp)
    
        # once retrained, or if retraining was not required, load metrics
        with open(folder_path + str(save_name) + "_" + model.__class__.__name__ + "_" + str(steps) + "_step" + "_" + "saved_mse_testing.txt", "rb") as fp:   # Unpickling
            saved_mse_testing = pickle.load(fp)
        with open(folder_path + str(save_name) + "_" + model.__class__.__name__ + "_" + str(steps) + "_step" + "_" + "iteration_list.txt", "rb") as fp:   # Unpickling
            iteration_list = pickle.load(fp)
        with open(folder_path + str(save_name) + "_" + model.__class__.__name__ + "_" + str(steps) + "_step" + "_" + "mse_list_training.txt", "rb") as fp:   # Unpickling
            mse_list_training = pickle.load(fp)
        with open(folder_path + str(save_name) + "_" + model.__class__.__name__ + "_" + str(steps) + "_step" + "_" + "mse_list_testing.txt", "rb") as fp:   # Unpickling
            mse_list_testing = pickle.load(fp)
        with open(folder_path + str(save_name) + "_" + model.__class__.__name__ + "_" + str(steps) + "_step" + "_" + "train_losses.txt", "rb") as fp:   # Unpickling
            train_losses = pickle.load(fp)
        with open(folder_path + str(save_name) + "_" + model.__class__.__name__ + "_" + str(steps) + "_step" + "_" + "test_losses.txt", "rb") as fp:   # Unpickling
            test_losses = pickle.load(fp)
        with open(folder_path + str(save_name) + "_" + model.__class__.__name__ + "_" + str(steps) + "_step" + "_" + "y_testing.txt", "rb") as fp:   # Unpickling
            y_testing = pickle.load(fp)        
        with open(folder_path + str(save_name) + "_" + model.__class__.__name__ + "_" + str(steps) + "_step" + "_" + "last_predictions.txt", "rb") as fp:   # Unpickling
            predictions_testing = pickle.load(fp)
                
        
        # plot loss and accuracy
        if show_plots:
            # loss graph
            fig = plt.figure(figsize=(15,12))
            ax1 = fig.add_subplot(2,1,1)
            ax1.set_title('Training and Testing Losses over Batch Iterations')
            ax1.plot(iteration_list, train_losses, label = 'Training Losses')
            ax1.plot(iteration_list, test_losses, label = 'Testing Losses')
            
            plt.legend()
            
            # accuracy graph
            ax2 = fig.add_subplot(2,1,2)
            ax2.set_title('Train MSE and Test MSE over Batch Iterations')
            ax2.plot(iteration_list, mse_list_training, label = 'Training MSE')
            ax2.plot(iteration_list, mse_list_testing, label = 'Testing MSE')
            
            plt.legend()
            st.pyplot()
    
        # display key metrics and return the saved value of the accuracy of the model on the testing data, whenever it occurred
        if retrain:
            st.write("Lowest test MSE:", saved_mse_testing)
            st.write("Lowest MSE on iteration:", saved_iteration)
            st.write("training MSE on that iteration:", saved_mse_training)
            st.write("testing loss on that iteration:", saved_loss_testing)
            st.write("training loss on that iteration:", saved_loss_training)
            return(saved_mse_testing)
        else:
            st.write("Lowest test MSE:", saved_mse_testing)
            return(saved_mse_testing)

    def get_upper_threshold(predictions_dataframe, rmse_upper_confidence_threshold):
        '''Given a proportion of data, e.g. 0.05, find the highest value of MSE for which 1-proportion of data falls below
        
        Parameters:
            predictions_dataframe: a dataframe of actual and predicted values, and the MSE of the prediction. Typcially this should be previous data, so that the MSE upper threshold can be applied to the current data
            proportion_data_to_flag: what proportion of the previous data should be considered anomalous
        '''
        predictions_dataframe = predictions_dataframe.sort_values(by='RMSE')
        for value in predictions_dataframe['RMSE']:
            if len(predictions_dataframe[predictions_dataframe['RMSE'] > value])/len(predictions_dataframe) < (1-rmse_upper_confidence_threshold):
                upper_threshold = value
                return(upper_threshold)

    # using transformation - can see value is close to actual value
    def get_upper_threshold_tranformed(predictions_dataframe, rmse_upper_confidence_threshold, plot=False):
        '''Given a proportion of data, e.g. 0.05, find the highest value of MSE for which 1-proportion of data falls below
    
        Parameters:
            predictions_dataframe: a dataframe of actual and predicted values, and the MSE of the prediction. Typcially this should be previous data, so that the MSE upper threshold can be applied to the current data
            proportion_data_to_flag: what proportion of the previous data should be considered anomalous
        '''
        predictions_dataframe = predictions_dataframe.sort_values(by='RMSE')
        predictions_dataframe['transformed_RMSE'] = predictions_dataframe['RMSE'].apply(np.cbrt)
        plt.title('Distribution of RMSE on Training Data After Cube Root Transformation')
        plt.xlabel('Transformed RMSE')
        plt.ylabel('Count')

        plt.hist(predictions_dataframe['transformed_RMSE'], bins=100)
        
        if plot:
            st.pyplot()
        # assume normal distribution approximation
        sample_mean = np.mean(predictions_dataframe['transformed_RMSE'])
        sample_std = np.std(predictions_dataframe['transformed_RMSE'])
    
        upper_confidence = sample_mean + norm.ppf(rmse_upper_confidence_threshold)*sample_std
        st.write("Transformed RMSE using Normal Approximation that corresponds to " + str(round(rmse_upper_confidence_threshold,2)) + " confidence: " + str(round(upper_confidence,2)))
        st.write("New Upper Threshold: " + str(round(upper_confidence**3, 2)))
        st.write("This corresponds to a " + str(round(len(predictions_dataframe[predictions_dataframe['RMSE'] < upper_confidence**3])/len(predictions_dataframe),2)) + " confidence interval for the original data")
        return(round(upper_confidence**3, 2))
        
        
    def get_upper_threshold_reverse(predictions_dataframe, upper_threshold):
        '''Given a proportion of data, e.g. 0.05, find the highest value of MSE for which 1-proportion of data falls below
        
        Parameters:
            predictions_dataframe: a dataframe of actual and predicted values, and the MSE of the prediction. Typcially this should be previous data, so that the MSE upper threshold can be applied to the current data
            proportion_data_to_flag: what proportion of the previous data should be considered anomalous
        '''
        
        if upper_threshold != None:
            proportion_df_below_threshold= len(predictions_dataframe[predictions_dataframe['RMSE'] < upper_threshold])/len(predictions_dataframe)
#            st.write(proportion_df_below_threshold)
            return(proportion_df_below_threshold)

    def get_elbow_points(df, training_date_from, training_date_to, predicting_date_from, predicting_date_to, interval_unit, interval_size, target_column, aggregation):
        
        '''Takes a series of proportions of the previous data, and gets an upper limit for each of them. A plot is made showing the proportion of data selected for the training data vs the resulting proportion of data that would be flagged as anomalous if the upper threshold that corresponded to that proportion were selected for the current data
        The aim is to advise the user as to which proportion of data to choose for the previous data, to get an MSE upper limit to apply to the current data.
        If there is a clear seperation between very high values and the remaining values in the current data that would result from choosing a certain proportion of data from the previous data, then that proportion and resulting rmse_upper_threshold may be advised.
        
        Parameters:
            df: dataframe of dns logs (when bucketed, indices in the disregard list are set to be disregarded)
            training_date_from:
            training_date_to:
            predicting_date_from:
            predicting_date_to:
            interval_unit:
            interval_size:
            target_column:
            aggregation:
                
        Returns: threshold_dataframe: Dataframe of the proportions of data, the corresponding mse upper threshold for the old data, and the corresponding proportion of the new data that would be flagged as anomalous by using that mse upper threshold
        
        '''
        # get a lot of proportions of data to mark as anomalous from the previous data        
        upper_limits_to_try = np.linspace(1, 30, 30)/100
        corresponding_thresholds = []
        anomalous_predictions_list = []
        # use the model to predict on the old data
        predictions_dataframe_training_no_anoms = bucket_and_predict_with_autoencoder(df=df, steps=10, model=autoencoder(), interval_unit=interval_unit, interval_size=interval_size, predict_date_from=training_date_from, predict_date_to=training_date_to, target_column=target_column, aggregation=aggregation, disregard_custom_indices_intervals=True, get_anomalies=False, plot=False)
        
        # get the mse upper threshold from the old data for each of the proportions to try
        # use that upper threshold on the current data to find what proportion of current data would be flagged
        # plot that relationship with the aim of finding an elbow point
        for i in upper_limits_to_try:
            # get the mse thresholds
            rmse_upper_threshold = get_upper_threshold(predictions_dataframe_training_no_anoms, i)
            corresponding_thresholds.append(rmse_upper_threshold)
            predictions, _, _ = bucket_and_predict_with_autoencoder(df=df, steps=10, model=autoencoder(), interval_unit=interval_unit, interval_size=interval_size, predict_date_from=predicting_date_from, predict_date_to=predicting_date_to, target_column=target_column, aggregation=aggregation, get_anomalies=True, rmse_upper_threshold=rmse_upper_threshold, plot=False)
            proportion_of_predictions_anomalous = len(predictions[predictions['anomalous_flag'] == 1])/len(predictions)
            anomalous_predictions_list.append(proportion_of_predictions_anomalous)
        
        # get dataframe
        threshold_dataframe = pd.DataFrame(data={'Upper Limits': upper_limits_to_try.tolist(), 'Anomalous Proportions': anomalous_predictions_list, 'Corresponding MSE':corresponding_thresholds})
        return(threshold_dataframe)
            

    def plot_elbow(threshold_dataframe):
        
        '''get a plot of the elbow points from the threshold dataframe
        Parameters: threshold dataframe
        '''
        
        
        fig1 = plt.figure(figsize=(15,15))
        plt.plot(threshold_dataframe['Upper Limits'], threshold_dataframe['Anomalous Proportions']);
        plt.title('Proportion of Training Data Identified as Anomalies by MSE Upper Limit, up to 0.3 threshold')
        plt.xlabel('Upper Limit Proportions')
        plt.ylabel('Anomalous Proportions')
        st.pyplot()
        
        fig2 = plt.figure(figsize=(15,15))
        plt.plot(threshold_dataframe['Upper Limits'][0:10], threshold_dataframe['Anomalous Proportions'][0:10]);
        plt.title('Proportion of Training Data Identified as Anomalies by MSE Upper Limit, up to 0.1 threshold')
        plt.xlabel('Upper Limit Proportions')
        plt.ylabel('Anomalous Proportions')
        st.pyplot()

    def plot_rmse_distributions(predictions_dataframe):
        not_null_predictions_df = predictions_dataframe[~predictions_dataframe.isna().any(axis=1)]
        sns.boxplot(not_null_predictions_df['RMSE'])
        st.pyplot()
        
        # upper whisker is 1.5 times the third percentile
        upper_whisker = np.percentile(not_null_predictions_df['RMSE'], 75) + 1.5*(np.percentile(not_null_predictions_df['RMSE'], 75) - np.percentile(not_null_predictions_df['RMSE'], 25))
        st.write("Upper whisker RMSE (third quartile + 1.5 x interquartile range): " + str(round(upper_whisker, 2)))
        upper_confidence = get_upper_threshold_reverse(not_null_predictions_df, upper_whisker)
        st.write("Upper whisker Corresponding confidence threshold: " + str(round(upper_confidence, 2)))
        
#        sns.distplot(not_null_predictions_df['RMSE'], hist=True)
        
#        st.pyplot()
        
        return(upper_confidence)


    def save_anomalies(button_name, anomalies, nan_anomalies):
        '''saves bucketed anomalies and nan anomalies as a specific name, according to the button on the dashboard that retrieves them.
        
        Parameters:
            button_name: string of the related button for the anomalies
            anomalies:
            nan_anomalies:
        '''
        anomalies_path = "C:\\Users\\roder\\Projects\\CyberSecurity\\" + str(button_name) + "_anomalies.csv"
        nan_anomalies_path = "C:\\Users\\roder\\Projects\\CyberSecurity\\" + str(button_name) + "_nan_anomalies.csv"
        anomalies.to_csv(anomalies_path)
        pd.DataFrame(nan_anomalies).to_csv(nan_anomalies_path)
        st.success("anomalies saved as " + str(button_name) + "'_anomalies.csv'")
        st.success("nan_anomalies saved as " + str(button_name) + "'_nan_anomalies.csv'")


    def save_anomalies_individual_indices(button_name, anomalies, nan_anomalies, interval_size, interval_unit):
        '''saves all observations in the dataframe that fall into the time buckets of the bucketed anomalies and nan anomalies, saves as a specific name according to the button on the dashboard that retrieves them.
        Parameters:
            button_name: string of the related button for the anomalies
            anomalies:
            nan_anomalies:
            interval_size: 1, 20 etc, used to get the correct individual indices
            interval_unit: "min", "sec" etc, used to get the correct individual indices
        '''
        # being sure to use the orignal here, rather than df. If filters are applied we still want to get all individual indices from the entire original dataframe
        df_temp = df_original.copy()
        df_temp['floor_time_bucket'] = pd.Series(df_temp.index).dt.floor(freq=str(interval_size) + str(interval_unit)).tolist()
        nan_anomalies_df = pd.DataFrame(nan_anomalies)
        individual_anomalies = df_temp[(df_temp['floor_time_bucket'].isin(anomalies.index))]
        individual_nan_anomalies = df_temp[df_temp['floor_time_bucket'].isin(nan_anomalies_df.index)]
        individual_anomalies_path = "C:\\Users\\roder\\Projects\\CyberSecurity\\" + str(button_name) + "_individual_anomalies.csv"
        individual_nan_anomalies_path = "C:\\Users\\roder\\Projects\\CyberSecurity\\" + str(button_name) + "_individual_nan_anomalies.csv"
        individual_anomalies.to_csv(individual_anomalies_path)
        individual_nan_anomalies.to_csv(individual_nan_anomalies_path)
        st.success("anomalies saved as " + str(button_name) + "'_individual_anomalies.csv'")
        st.success("nan_anomalies saved as " + str(button_name) + "'_individual_nan_anomalies.csv'")
            
            
# initial user input, date inputs
# -----------------------------------------------------------------------------            
    

        
    date_from = st.sidebar.date_input(      
        label = "Choose a Display Date from:",
        value= min(df_original.index),
        min_value = min(df_original.index),
        max_value = max(df_original.index),
        key='date_from'
    )
    date_to = st.sidebar.date_input(      
        label = "Choose a Display Date from:",
        value = max(df_original.index),
        min_value = min(df_original.index),
        max_value = max(df_original.index),
        key='date_to'
    ) 

    if date_from > date_to:
        st.warning("'Date from' is set as greater than 'Date to', please ammend date selection")
    
    training_date_from = st.sidebar.date_input(      
        label = "Choose a Training Date from: (Neural Net and Autoencoder Only)",
        value = datetime.date(2012,3,16),
        min_value = min(df_original.index),
        max_value = max(df_original.index),
        key='training_date_from'
    )
    training_date_to = st.sidebar.date_input(      
        label = "Choose a Training Date to: (Neural Net and Autoencoder Only)",
        value = datetime.date(2012,3,16),
        min_value = min(df_original.index),
        max_value = max(df_original.index),
        key='training_date_to'
    )     
    
    if training_date_from > training_date_to:
        st.warning("'Training Date from' is set as greater than 'Training Date to', please ammend date selection")
    
    predicting_date_from = st.sidebar.date_input(      
        label = "Choose a Predicting Date from: (Neural Net and Autoencoder Only)",
        value = datetime.date(2012,3,17),
        min_value = min(df_original.index),
        max_value = max(df_original.index),
        key='predicting_date_from'
    )
    predicting_date_to = st.sidebar.date_input(      
        label = "Choose a Predicting Date to: (Neural Net and Autoencoder Only)",
        value = datetime.date(2012,3,17),
        min_value = min(df_original.index),
        max_value = max(df_original.index),
        key='predicting_date_to'
    )     
    
    if predicting_date_from > predicting_date_to:
        st.warning("'Predicting Date from' is set as greater than 'Predicting Date to', please ammend date selection")
      
    # reformatting, want format yyyy-m-d
    date_from = str(date_from.year) + "-" + str(date_from.month) + "-" + str(date_from.day)
    date_to = str(date_to.year) + "-" + str(date_to.month) + "-" + str(date_to.day)        
    training_date_from = str(training_date_from.year) + "-" + str(training_date_from.month) + "-" + str(training_date_from.day)
    training_date_to = str(training_date_to.year) + "-" + str(training_date_to.month) + "-" + str(training_date_to.day)
    predicting_date_from = str(predicting_date_from.year) + "-" + str(predicting_date_from.month) + "-" + str(predicting_date_from.day)
    predicting_date_to = str(predicting_date_to.year) + "-" + str(predicting_date_to.month) + "-" + str(predicting_date_to.day)


# use recommended
# ----------------------------------------------------------------------------
    use_recommended = st.sidebar.radio(
            "Use Recommended Outlier Detection Methods?:",
            ("Use Recommended", "Customise"),
            key='use_recommended'
            )
    
    if use_recommended == "Use Recommended":

        st.write('### Recommended')
        # no user inputs, so no user custom filtering, so can just grab the original dataframe already loaded        
        df = df_original
        
        interval_size = 1
        interval_unit = 'min'
        
        recommended_checks = st.sidebar.radio(
        "Choose a Check to Perform:",
        ("Anomalous DNS Traffic Counts: Autoencoder", "Anomalous DNS Traffic Counts of Uncommon Queries (not A or AAAA): Autoencoder", "Anomalous Counts of Uncommon responses (not NOERROR): Autoencoder", "Anomalous Mean Subquery Information Level: Median Absolute Deviation", "Anomalous Mean Parent Randomness Level: Median Absolute Deviation"),
        key='recommended_checks'
        )
        # 1: autoencoder 1 minute traffic counts
        # ---------------------------------------------------------------------

        if recommended_checks == "Anomalous DNS Traffic Counts: Autoencoder":
            st.write('Anomalous DNS Traffic Counts: Autoencoder')
            # get previous data for training
            data_training_ae, _, _ = bucket_and_get_dataset_for_AutoEncoder(df, interval_unit='min', interval_size=1, aggregation='count',
                                  target_column='ID',
                                  steps=10,
                                  date_from=training_date_from, date_to=training_date_to,
                                  remove_nans=True,
                                  remove_initial_nans=True,
                                  disregard_custom_indices_intervals=True)
            
            # option to retrain an autoencoder
            retrain_all_traffic_ae_on_inputs = st.button("Retrain All Traffic AutoEncoder on Inputs", key='retrain_all_traffic_ae_on_inputs')
            # train on previous data
            if retrain_all_traffic_ae_on_inputs:
                train_autoencoder(dataset=data_training_ae, steps=10, train_test_split=0.8, model=autoencoder(), retrain=True, EPOCHS=100, learning_rate=0.01, print_accuracies=False, show_plots=True, save_name="all_traffic_counts")    
                st.success("Autoencoder for All Traffic retrained on User inputs")
            
            # load saved training model
            train_autoencoder(dataset=data_training_ae, steps=10, train_test_split=0.8, model=autoencoder(), retrain=False, EPOCHS=100, learning_rate=0.01, print_accuracies=False, show_plots=False, save_name="all_traffic_counts")
            # predict back on the training data, so we can get a good upper threshold (previous data is all assumed to be benign)
            predictions_dataframe_training_AE = bucket_and_predict_with_autoencoder(df=df, steps=10, model=autoencoder(), interval_unit='min', interval_size=1, predict_date_from=training_date_from, predict_date_to=training_date_to, target_column='ID', aggregation='count', get_anomalies=False, plot=False, save_name="all_traffic_counts")
                
            # choose an appropriate proportion of data from the previous data to be marked as anomalous
            rmse_upper_confidence_threshold = st.slider('Choose a confidence interval for anomalies using the traninig data (95% would mean that 95% of values in the training data would not have been classified as anomalous). The corresponding MSE then applied to the testing data', min_value=0.50, max_value=0.99, value=0.95, step = 0.01, key='rmse_upper_confidence_threshold')
            upper_threshold = get_upper_threshold_tranformed(predictions_dataframe=predictions_dataframe_training_AE, rmse_upper_confidence_threshold=rmse_upper_confidence_threshold)
            
            # use that upper threshold to identify anomalies in the new data
            predictions_dataframe_testing_AE, anomalies, nan_anomalies = bucket_and_predict_with_autoencoder(df=df, steps=10, model=autoencoder(), interval_unit='min', interval_size=1, predict_date_from=predicting_date_from, predict_date_to=predicting_date_to, target_column='ID', aggregation='count', get_anomalies=True, rmse_upper_threshold=upper_threshold, save_name="all_traffic_counts")
    
            # print proportion of anomalies        
            proportion_data_flagged_anomalous = round(len(anomalies)/len(predictions_dataframe_testing_AE), 2)
            proportion_data_flagged_nan_anomalous = round(len(nan_anomalies)/len(predictions_dataframe_testing_AE),2)
            st.write("Proportion of data flagged as anomalous:", proportion_data_flagged_anomalous)
            st.write("Proportion of data flagged as NAN anomalous:", proportion_data_flagged_nan_anomalous)            
        
            # download anomalies
            download_anomalies_traffic_count_ae = st.button("Download Anomalies: Traffic Counts Autoencoder", key='download_anomalies_traffic_count_ae')
            if download_anomalies_traffic_count_ae:        
                save_anomalies(button_name="download_anomalies_traffic_count_ae", anomalies=anomalies, nan_anomalies=nan_anomalies)
    
            # download individual anomalous indices    
            download_individual_anomalous_indicies_traffic_count_ae = st.button("Download Individual Anomalous Indices: Traffic Counts Autoencoder", key='download_individual_anomalous_indicies_traffic_count_ae')
            if download_individual_anomalous_indicies_traffic_count_ae:        
                save_anomalies_individual_indices(button_name='download_individual_anomalous_indicies_traffic_count_ae', anomalies=anomalies, nan_anomalies=nan_anomalies, interval_unit='min', interval_size=1)
            
        # 2 Autoencoder, Uncommon Requests (not A or AAAA)
        # ---------------------------------------------------------------------
        
        if recommended_checks == "Anomalous DNS Traffic Counts of Uncommon Queries (not A or AAAA): Autoencoder":
            st.write('Anomalous DNS Traffic Counts of Uncommon Query Types: Autoencoder')
            # get previous data for training
            data_training_ae, _, _ = bucket_and_get_dataset_for_AutoEncoder(df[~df['qtype_name'].isin(['A', 'AAAA'])], interval_unit='min', interval_size=1, aggregation='count',
                                  target_column='ID',
                                  steps=10,
                                  date_from=training_date_from, date_to=training_date_to,
                                  remove_nans=True,
                                  remove_initial_nans=True,
                                  disregard_custom_indices_intervals=True)
            
            # option to retrain an autoencoder
            retrain_uncommon_query_traffic_ae_on_inputs = st.button("Retrain Uncommon Query Traffic AutoEncoder on Inputs", key='retrain_uncommon_query_traffic_ae_on_inputs')
            # train on previous data
            if retrain_uncommon_query_traffic_ae_on_inputs:
                train_autoencoder(dataset=data_training_ae, steps=10, train_test_split=0.8, model=autoencoder(), retrain=True, EPOCHS=100, learning_rate=0.01, print_accuracies=False, show_plots=True, save_name="traffic_counts_uncommon_query_types")    
                st.success("Autoencoder for All Traffic retrained on User inputs")
            
            # load saved training model
            train_autoencoder(dataset=data_training_ae, steps=10, train_test_split=0.8, model=autoencoder(), retrain=False, EPOCHS=100, learning_rate=0.01, print_accuracies=False, show_plots=False, save_name="traffic_counts_uncommon_query_types")
            # predict back on the training data, so we can get a good upper threshold (previous data is all assumed to be benign)
            predictions_dataframe_training_AE = bucket_and_predict_with_autoencoder(df=df[~df['qtype_name'].isin(['A', 'AAAA'])], steps=10, model=autoencoder(), interval_unit='min', interval_size=1, predict_date_from=training_date_from, predict_date_to=training_date_to, target_column='ID', aggregation='count', get_anomalies=False, plot=False, save_name="traffic_counts_uncommon_query_types")
            
            # choose an appropriate proportion of data from the previous data to be marked as anomalous
            rmse_upper_confidence_threshold = st.slider('Choose a confidence interval for anomalies using the traninig data (95% would mean that 95% of values in the training data would not have been classified as anomalous). The corresponding MSE then applied to the testing data', min_value=0.50, max_value=0.99, value=0.95, step = 0.01, key='rmse_upper_confidence_threshold')
            
            # that proportion will therefore translate into an mse upper tupper_threshold = get_upper_threshold(predictions_dataframe=predictions_dataframe_training_AE, rmse_upper_confidence_threshold=rmse_upper_confidence_threshold)hreshold
            
            upper_threshold = get_upper_threshold_tranformed(predictions_dataframe=predictions_dataframe_training_AE, rmse_upper_confidence_threshold=rmse_upper_confidence_threshold)
            
            # use that upper threshold to identify anomalies in the new data
            predictions_dataframe_testing_AE, anomalies, nan_anomalies = bucket_and_predict_with_autoencoder(df=df[~df['qtype_name'].isin(['A', 'AAAA'])], steps=10, model=autoencoder(), interval_unit='min', interval_size=1, predict_date_from=predicting_date_from, predict_date_to=predicting_date_to, target_column='ID', aggregation='count', get_anomalies=True, rmse_upper_threshold=upper_threshold, save_name="traffic_counts_uncommon_query_types")
    
            # print proportion of anomalies        
            proportion_data_flagged_anomalous = round(len(anomalies)/len(predictions_dataframe_testing_AE), 2)
            proportion_data_flagged_nan_anomalous = round(len(nan_anomalies)/len(predictions_dataframe_testing_AE),2)
            st.write("Proportion of data flagged as anomalous:", proportion_data_flagged_anomalous)
            st.write("Proportion of data flagged as NAN anomalous:", proportion_data_flagged_nan_anomalous)            
        
            # download anomalies
            download_anomalies_traffic_count_ae_uncommon_queries = st.button("Download Anomalies: Traffic Counts of Uncommon Queries Autoencoder", key='download_anomalies_traffic_count_ae_uncommon_queries')
            if download_anomalies_traffic_count_ae_uncommon_queries:        
                save_anomalies(button_name="download_anomalies_traffic_count_ae_uncommon_queries", anomalies=anomalies, nan_anomalies=nan_anomalies)
    
            # download individual anomalous indices    
            download_individual_anomalous_indicies_traffic_count_ae_uncommon_queries = st.button("Download Individual Anomalous Indices: Traffic Counts Autoencoder", key='download_individual_anomalous_indicies_traffic_count_ae_uncommon_queries')
            if download_individual_anomalous_indicies_traffic_count_ae_uncommon_queries:        
                save_anomalies_individual_indices(button_name='download_individual_anomalous_indicies_traffic_count_ae_uncommon_queries', anomalies=anomalies, nan_anomalies=nan_anomalies, interval_unit='min', interval_size=1)
        
        # 3 Autoencoder anomalous responses (not NOERROR)
        # ---------------------------------------------------------------------
        # using rolling mad here again as it is better than the autoencoder for less available data points
        # in this dataset, since the server was down for most of the project, the traffic has majority no-responses
        # but we can't expect that of real data! so using MAD
        
        if recommended_checks == "Anomalous Counts of Uncommon responses (not NOERROR): Autoencoder":
            st.write("Anomalous Counts of Uncommon Response Types (not NOERROR): Autoencoder")
            # get previous data for training
            data_training_ae, _, _ = bucket_and_get_dataset_for_AutoEncoder(df=df[~df['rcode_name'].isin(['NOERROR'])], interval_unit='min', interval_size=1, aggregation='count',
                                  target_column='ID',
                                  steps=10,
                                  date_from=training_date_from, date_to=training_date_to,
                                  remove_nans=True,
                                  remove_initial_nans=True,
                                  disregard_custom_indices_intervals=True)
            
            # option to retrain an autoencoder
            retrain_uncommon_response_traffic_ae_on_inputs = st.button("Retrain Uncommon Response Traffic AutoEncoder on Inputs", key='retrain_uncommon_response_traffic_ae_on_inputs')
            # train on previous data
            if retrain_uncommon_response_traffic_ae_on_inputs:
                train_autoencoder(dataset=data_training_ae, steps=10, train_test_split=0.8, model=autoencoder(), retrain=True, EPOCHS=100, learning_rate=0.01, print_accuracies=False, show_plots=True, save_name="traffic_counts_uncommon_response_types")    
                st.success("Autoencoder for All Traffic retrained on User inputs")
            
            # load saved training model
            train_autoencoder(dataset=data_training_ae, steps=10, train_test_split=0.8, model=autoencoder(), retrain=False, EPOCHS=100, learning_rate=0.01, print_accuracies=False, show_plots=False, save_name="traffic_counts_uncommon_response_types")
            # predict back on the training data, so we can get a good upper threshold (previous data is all assumed to be benign)
            predictions_dataframe_training_AE = bucket_and_predict_with_autoencoder(df=df[~df['rcode_name'].isin(['NOERROR'])], steps=10, model=autoencoder(), interval_unit='min', interval_size=1, predict_date_from=training_date_from, predict_date_to=training_date_to, target_column='ID', aggregation='count', get_anomalies=False, plot=False, save_name="traffic_counts_uncommon_response_types")
                
            # choose an appropriate proportion of data from the previous data to be marked as anomalous
            rmse_upper_confidence_threshold = st.slider('Choose a confidence interval for anomalies using the traninig data (95% would mean that 95% of values in the training data would not have been classified as anomalous). The corresponding MSE then applied to the testing data', min_value=0.50, max_value=0.99, value=0.95, step = 0.01, key='rmse_upper_confidence_threshold')
            
            # that proportion will therefore translate into an mse upper tupper_threshold = get_upper_threshold(predictions_dataframe=predictions_dataframe_training_AE, rmse_upper_confidence_threshold=rmse_upper_confidence_threshold)hreshold
            
            upper_threshold = get_upper_threshold_tranformed(predictions_dataframe=predictions_dataframe_training_AE, rmse_upper_confidence_threshold=rmse_upper_confidence_threshold)
            
            # use that upper threshold to identify anomalies in the new data
            predictions_dataframe_testing_AE, anomalies, nan_anomalies = bucket_and_predict_with_autoencoder(df=df[~df['rcode_name'].isin(['NOERROR'])], steps=10, model=autoencoder(), interval_unit='min', interval_size=1, predict_date_from=predicting_date_from, predict_date_to=predicting_date_to, target_column='ID', aggregation='count', get_anomalies=True, rmse_upper_threshold=upper_threshold, save_name="traffic_counts_uncommon_response_types")
    
            # print proportion of anomalies        
            proportion_data_flagged_anomalous = round(len(anomalies)/len(predictions_dataframe_testing_AE), 2)
            proportion_data_flagged_nan_anomalous = round(len(nan_anomalies)/len(predictions_dataframe_testing_AE),2)
            st.write("Proportion of data flagged as anomalous:", proportion_data_flagged_anomalous)
            st.write("Proportion of data flagged as NAN anomalous:", proportion_data_flagged_nan_anomalous)            
        
            # download anomalies
            download_anomalies_traffic_count_ae_uncommon_responses = st.button("Download Anomalies: Traffic Counts of Uncommon Queries Autoencoder", key='download_anomalies_traffic_count_ae_uncommon_responses')
            if download_anomalies_traffic_count_ae_uncommon_responses:        
                save_anomalies(button_name="download_anomalies_traffic_count_ae_uncommon_responses", anomalies=anomalies, nan_anomalies=nan_anomalies)
    
            # download individual anomalous indices    
            download_individual_anomalous_indicies_traffic_count_ae_uncommon_responses = st.button("Download Individual Anomalous Indices: Traffic Counts Autoencoder", key='download_individual_anomalous_indicies_traffic_count_ae_uncommon_responses')
            if download_individual_anomalous_indicies_traffic_count_ae_uncommon_responses:        
                save_anomalies_individual_indices(button_name='download_individual_anomalous_indicies_traffic_count_ae_uncommon_responses', anomalies=anomalies, nan_anomalies=nan_anomalies, interval_unit='min', interval_size=1)
        
        
     
        # 4 static MAD query information level
        # ---------------------------------------------------------------------
        if recommended_checks == "Anomalous Mean Subquery Information Level: Median Absolute Deviation":
            st.write('Anomalous Mean Subquery Information Level: Median Absolute Deviation')
    
            MAD_threshold_mean_query_information_level = st.number_input("choose the proportion of the median absolute deviation that a time interval value must exceed to be flagged as potentially anomalous",
                    min_value=1.00,
                    max_value=10.00,
                    value=2.50,
                    step=0.50,
                    key="MAD_threshold_mean_subquery_information_level"
                    )
            
            # get MAD
            data, anomalies, nan_anomalies = bucket_and_get_MAD(df=df, interval_unit='min', interval_size=1, aggregation='mean', target_column='subquery_information_level', MAD_threshold=MAD_threshold_mean_query_information_level)
    
    
            # download anomalies
            download_anomalies_mean_subquery_information_level_MAD = st.button("Download Anomalies: Mean Query Information Level MAD", key='download_anomalies_mean_subquery_information_level_MAD')
            if download_anomalies_mean_subquery_information_level_MAD:
                save_anomalies(button_name="download_anomalies_mean_subquery_information_level_MAD", anomalies=anomalies, nan_anomalies=nan_anomalies)
    
            # download indiviual anomalous indices
            download_individual_anomalous_indicies_mean_subquery_encrypytion_level_MAD = st.button("Download Individual Anomalous Indices", key='download_individual_anomalous_indicies_mean_subbquery_encrypytion_level_MAD')
            if download_individual_anomalous_indicies_mean_subquery_encrypytion_level_MAD:
                save_anomalies_individual_indices(button_name='download_individual_anomalous_indicies_mean_subquery_encrypytion_level_MAD', anomalies=anomalies, nan_anomalies=nan_anomalies, interval_unit='min', interval_size=1)
     
        # 5 static MAD parent domain information level LSTM
        # ---------------------------------------------------------------------
        if recommended_checks == "Anomalous Mean Parent Randomness Level: Median Absolute Deviation":
            st.write('Anomalous Mean Parent Randomness Level: Median Absolute Deviation')
    
            MAD_threshold_mean_parent_randomness_level = st.number_input("choose the proportion of the median absolute deviation that a time interval value must exceed to be flagged as potentially anomalous",
                    min_value=1.00,
                    max_value=10.00,
                    value=2.50,
                    step=0.50,
                    key="MAD_threshold_mean_parent_randomness_level"
                    )
            
            # get MAD
            data, anomalies, nan_anomalies = bucket_and_get_MAD(df=df, interval_unit='min', interval_size=1, aggregation='mean', target_column='parent_domain_randomness_level', MAD_threshold=MAD_threshold_mean_parent_randomness_level)
    
            # download anomalies
            download_anomalies_mean_parent_randomness_level_MAD = st.button("Download Anomalies: Mean Parent Randomness Level MAD", key='download_anomalies_mean_parent_randomness_level_MAD')
            if download_anomalies_mean_parent_randomness_level_MAD:
                save_anomalies(button_name="download_anomalies_mean_parent_randomness_level_MAD", anomalies=anomalies, nan_anomalies=nan_anomalies)
    
            # download indiviual anomalous indices
            download_individual_anomalous_indicies_mean_parent_encrypytion_level_MAD = st.button("Download Individual Anomalous Indices", key='download_individual_anomalous_indicies_mean_parent_encrypytion_level_MAD')
            if download_individual_anomalous_indicies_mean_parent_encrypytion_level_MAD:
                save_anomalies_individual_indices(button_name='download_individual_anomalous_indicies_mean_parent_encrypytion_level_MAD', anomalies=anomalies, nan_anomalies=nan_anomalies, interval_unit='min', interval_size=1)

    if use_recommended == "Customise":

# ----------------------------------------------------------------------------    
        st.write('### Customised')
        interval_unit = st.sidebar.radio(      
            "Choose an Interval Unit",
            ("s", "min"),
            index=1,
            key='interval_unit'
        )
        
        interval_size = st.sidebar.slider(      
            "Choose an Interval Size",
            min_value=1,
            max_value=60,
            value = 1,
            key='interval_unit'
        )
        aggregation = st.sidebar.radio(      
            "Choose an Aggregation",
            ("count", "sum", "mean", "max", "min"), # could add more later
            key='aggregation'
        )
        target_column = st.sidebar.radio(      
            "Choose a Target Feature",
            df_original.columns.tolist(),
            index=23,
#            ("ID", "query_length", "query_encryption_level", "parent_encryption_level_LSTM"), # could add more later, and make "ID" clear
            key='target_column'
        )
        
        filter_column = st.sidebar.selectbox(      
            label="Choose a column to filter (optional)",
            options= ["Do not Filter"] + df_original.columns.tolist(), # could add more later, and make "ID" clear
            index=0,
            key='filter_column'
        )
        
        if filter_column != "Do not Filter":
            filter_values = st.sidebar.multiselect(      
                label="Choose the value of the filtered column (optional)",
                options= df_original[filter_column].unique().tolist(), # could add more later, and make "ID" clear
                key='filter_values'
            )

        if filter_column != "Do not Filter":
            df = df_original[df_original[filter_column].isin(filter_values)]
    
        else:
            # if no filters are applied, can use the original dataframe
            df = df_original        
        
# Seperate Approaches
# ----------------------------------------------------------------------------
        approaches = st.radio(
                "Choose an Approach:",
                ("Proportional Change from Mean of Last 5 Intervals", "Standard Deviation", "Rolling Standard Deviation", "Median Absolute Deviation (MAD)", "Rolling Median Absolute Deviation (MAD)", "Feed Forward Neural Network", "AutoEncoder"),
                key='approaches'
                )
    # approach 1
    # ----------------------------------------------------------------------------
        if approaches=="Proportional Change from Mean of Last 5 Intervals":
            
            st.write("The data is bucketed into equal time intervals. The proportional size of the current time interval against the mean of the last 5 time intervals is calculated.")

            proportional_upper_threshold = st.number_input("choose the minimum proportion of the previous 5 time intervals that a time interval must exceed to be flagged as a potential anomaly",
                    min_value=1.00,
                    max_value=10.00,
                    value=2.50,
                    step=0.50,
                    key="proportional_upper_threshold"
            )
            
            # get anomalies
            data, anomalies, nan_anomalies = bucket_and_compare_to_last_5(df, interval_unit=interval_unit, interval_size=interval_size, aggregation=aggregation, target_column=target_column, plot=True, upper_threshold_factor=proportional_upper_threshold)
    
            # download anomalies
            download_anomalies_prop_change = st.button("Download Anomalies", key='download_anomalies_prop_change')
            if download_anomalies_prop_change:        
                save_anomalies(button_name='download_anomalies_prop_change', anomalies=anomalies, nan_anomalies=nan_anomalies)
                
            # download individual anomalous indices
            download_individual_anomalous_indicies_prop_change = st.button("Download Individual Anomalous Indices", key='download_individual_anomalous_indicies_prop_change')
            if download_individual_anomalous_indicies_prop_change:        
                save_anomalies_individual_indices(button_name='download_individual_anomalous_indicies_prop_change', anomalies=anomalies, nan_anomalies=nan_anomalies, interval_size=interval_size, interval_unit=interval_unit)
    
    
    # approach 2 - standard deviation
    # ----------------------------------------------------------------------------    
           
        if approaches=="Standard Deviation":
            st.write("The data is bucketed into equal time intervals. The mean and standard deviation are calculated. If the standard deviation of the value at a certain time interval exceeds a proportional threshold, it is marked as a potential anomaly")
    
            sd_threshold = st.number_input("choose the proportion of the standard deviation that a time interval value must exceed to be flagged as potentially anomalous",
                    min_value=1.00,
                    max_value=10.00,
                    value=2.00,
                    step=0.50,
                    key="sd_threshold"
            )
            
            # get anomalies
            data, anomalies, nan_anomalies = bucket_and_get_standard_dev(df, interval_unit=interval_unit, interval_size=interval_size, aggregation=aggregation, target_column=target_column, sd_threshold=sd_threshold)
    
            # download anomalies
            download_anomalies_SD = st.button("Download Anomalies", key='download_anomalies_SD')
            if download_anomalies_SD: 
                save_anomalies(button_name='download_anomalies_SD', anomalies=anomalies, nan_anomalies=nan_anomalies)
            
            # download individual anomalous indices
            download_individual_anomalous_indicies_SD = st.button("Download Individual Anomalous Indices", key='download_individual_anomalous_indicies_SD')
            if download_individual_anomalous_indicies_SD:        
                save_anomalies_individual_indices(button_name='download_individual_anomalous_indicies_SD', anomalies=anomalies, nan_anomalies=nan_anomalies, interval_size=interval_size, interval_unit=interval_unit)
    
    
    # approach 3 - rolling standard deviation
    # ----------------------------------------------------------------------------    
           
        if approaches=="Rolling Standard Deviation":
    
            st.write("The data is bucketed into equal time intervals. The 1 hour rolling mean and rolling standard deviation is calculated. If the standard deviation of the value at a certain time interval exceeds a proportional threshold, it is marked as a potential anomaly")
    
            # get anomalies
            sd_threshold = st.number_input("choose the proportion of the standard deviation that a time interval value must exceed to be flagged as potentially anomalous",
                    min_value=1.00,
                    max_value=10.00,
                    value=2.00,
                    step=0.50,
                    key="sd_threshold"
            )
            data, anomalies, nan_anomalies = bucket_and_get_rolling_standard_dev(df, interval_unit=interval_unit, interval_size=interval_size, aggregation=aggregation, target_column=target_column, sd_threshold=sd_threshold)
    
            # download anomalies
            download_anomalies_rolling_SD = st.button("Download Anomalies", key='download_anomalies_rolling_SD')
            if download_anomalies_rolling_SD:        
                save_anomalies(button_name='download_anomalies_rolling_SD', anomalies=anomalies, nan_anomalies=nan_anomalies)
            
            # download individual anomalous indices
            download_individual_anomalous_indicies_rolling_SD = st.button("Download Individual Anomalous Indices", key='download_individual_anomalous_indicies_rolling_SD')
            if download_individual_anomalous_indicies_rolling_SD:        
                save_anomalies_individual_indices(button_name='download_individual_anomalous_indicies_rolling_SD', anomalies=anomalies, nan_anomalies=nan_anomalies, interval_size=interval_size, interval_unit=interval_unit)

                
    
    # approach 4 - Median Absolute Deviation ----------------------------------------------------------------------------    
    #
    
        if approaches=="Median Absolute Deviation (MAD)":
    
            st.write("The data is bucketed into equal time intervals. The median and median absolute deviation is calculated. If the median absolute deviation of the value at a certain time interval exceeds a proportional threshold, it is marked as a potential anomaly")
    
            
            MAD_threshold = st.number_input("choose the proportion of the median absolute deviation that a time interval value must exceed to be flagged as potentially anomalous",
                    min_value=1.00,
                    max_value=10.00,
                    value=2.00,
                    step=0.50,
                    key="MAD_threshold"
                    )
            
            # get anomalies
            data, anomalies, nan_anomalies = bucket_and_get_MAD(df=df, interval_unit=interval_unit, interval_size=interval_size, aggregation=aggregation, target_column=target_column, MAD_threshold=MAD_threshold)
            
            # download anomalies
            download_anomalies_MAD = st.button("Download Anomalies", key='download_anomalies_MAD')
            if download_anomalies_MAD:       
                save_anomalies(button_name='download_anomalies_MAD', anomalies=anomalies, nan_anomalies=nan_anomalies)
    
            # download individual anomalous indices
            download_individual_anomalous_indicies_MAD = st.button("Download Individual Anomalous Indices", key='download_individual_anomalous_indicies_MAD')
            if download_individual_anomalous_indicies_MAD:        
                save_anomalies_individual_indices(button_name='download_individual_anomalous_indicies_MAD', anomalies=anomalies, nan_anomalies=nan_anomalies, interval_size=interval_size, interval_unit=interval_unit)

    
    
    # approach 5 - Rolling Median Absolute Deviation ----------------------------------------------------------------------------    
    
        if approaches=="Rolling Median Absolute Deviation (MAD)":
    
            st.write("The data is bucketed into equal time intervals. The 1 hour rolling median and rolling median absolute deviation is calculated. If the median absolute deviation of the value at a certain time interval exceeds a proportional threshold, it is marked as a potential anomaly")
    
            
            MAD_threshold = st.number_input("choose the proportion of the median absolute deviation that a time interval value must exceed to be flagged as potentially anomalous",
                    min_value=1.00,
                    max_value=10.00,
                    value=2.50,
                    step=0.50,
                    key="MAD_threshold"
                    )

            # get anomalies            
            data, anomalies, nan_anomalies = bucket_and_get_rolling_MAD(df=df, interval_unit=interval_unit, interval_size=interval_size, aggregation=aggregation, target_column=target_column, MAD_threshold=MAD_threshold)

            # download anomalies    
            download_anomalies_rolling_MAD = st.button("Download Anomalies", key='download_anomalies_rolling_MAD')
            if download_anomalies_rolling_MAD:
                save_anomalies(button_name='download_anomalies_rolling_MAD', anomalies=anomalies, nan_anomalies=nan_anomalies)
    
            # download individual anomalous indices
            download_individual_anomalous_indicies_rolling_MAD = st.button("Download Individual Anomalous Indices", key='download_individual_anomalous_indicies_rolling_MAD')
            if download_individual_anomalous_indicies_rolling_MAD:
                save_anomalies_individual_indices(button_name='download_individual_anomalous_indicies_rolling_MAD', anomalies=anomalies, nan_anomalies=nan_anomalies, interval_size=interval_size, interval_unit=interval_unit)
    
    # approach 6 - Feed Forward Neural Network
    # ----------------------------------------------------------------------------    
        if approaches=="Feed Forward Neural Network":
    
            st.write("The data is bucketed into equal time intervals. A Feed Forward Neural Network is trained on the training data selected. The previous 12 time intervals are used to predict the next time interval. The specified data is then predicted. If the true value differs from the predicted value by more than a proportional threshold, then it is marked as a potential anomaly")
                
            # get dataset for old data
            # note training on data disregarding custom indices intervals, to ensure clean
            
            data_training_nn, _, _ = bucket_and_get_dataset_for_NN(df, interval_unit=interval_unit, interval_size=interval_size, aggregation=aggregation,
                                                      target_column=target_column,
                                                      steps=10,
                                                      date_from=training_date_from, date_to=training_date_to,
                                                      remove_nans=True,
                                                      remove_initial_nans=True,
                                                      disregard_custom_indices_intervals=True)
        
        

            # option to retrain an autoencoder
            retrain_custom_nn_on_inputs = st.button("Retrain Custom Feed Forward Neural Network on Inputs", key='retrain_custom_nn_on_inputs')
            # train on previous data
            if retrain_custom_nn_on_inputs:
                train(dataset=data_training_nn, steps=10, train_test_split=0.8, model=linear_net_regression(), retrain=True, EPOCHS=100, learning_rate=0.1, print_accuracies=False, show_plots=True, save_name="custom")    
                st.success("Custom Feed Forward Neural Network retrained on User inputs")
            
            # train on old data
            train(dataset=data_training_nn, steps=10, train_test_split=0.8, model=linear_net_regression(), retrain=False, EPOCHS=100, learning_rate=0.01, print_accuracies=False, show_plots=False, save_name="custom")    
            # predict back on the old data, this is to get MSE and help choose a good proportion of the old data to consider anomalous and get a corresponding upper threshold
            st.write('Predictions on training Data')
            predictions_dataframe_training_nn = bucket_and_predict_with_NN(df=df, model=linear_net_regression(), steps=10, interval_unit=interval_unit, interval_size=interval_size, predict_date_from=training_date_from, predict_date_to=training_date_to, target_column=target_column, aggregation=aggregation, get_anomalies=False, save_name="custom")
            # show distributions of MSE
            plt.hist(predictions_dataframe_training_nn['RMSE'], bins=100)
            plt.title("Distribution of RMSE on training data")
            plt.xlabel('RMSE')
            plt.ylabel('count')
            st.pyplot()

            _ = plot_rmse_distributions(predictions_dataframe_training_nn)
            rmse_converter = st.text_input('convert RMSE limit to confidence threshold', value=0, key='rmse_converter')
            get_upper_threshold_reverse(predictions_dataframe_training_nn, int(rmse_converter))
                
            # choose an appropriate proportion of data from the previous data to be marked as anomalous
            rmse_upper_confidence_threshold = st.slider('Choose a confidence interval for anomalies using the traninig data (95% would mean that 95% of values in the training data would not have been classified as anomalous). The corresponding RMSE then applied to the testing data', min_value=0.50, max_value=0.99, value=0.95, step = 0.01, key='rmse_upper_confidence_threshold')
            
            # that proportion will therefore translate into an mse upper threshold
            upper_threshold = get_upper_threshold_tranformed(predictions_dataframe=predictions_dataframe_training_nn, rmse_upper_confidence_threshold=rmse_upper_confidence_threshold, plot=True)
            
            st.write("Selected upper confidence threshold:", str(round(rmse_upper_confidence_threshold, 2)), "Corresponding RMSE Upper threshold:", str(round(upper_threshold, 2)))
                        
            # now get predictions, anomalies and nan_anomalies on the new data
            st.write('Predictions on New Data')

#            predictions_dataframe_testing_nn, anomalies, nan_anomalies = bucket_and_predict_with_NN(df=df['2012-03-17 16':'2012-03-17 17'], steps=10, model=linear_net_regression(), interval_unit=interval_unit, interval_size=interval_size, predict_date_from=predicting_date_from, predict_date_to=predicting_date_to, target_column=target_column, aggregation=aggregation, get_anomalies=True, rmse_upper_threshold=upper_threshold, save_name="custom")
            predictions_dataframe_testing_nn, anomalies, nan_anomalies = bucket_and_predict_with_NN(df=df['2012-03-17 16':'2012-03-17 17'], steps=10, model=linear_net_regression(), interval_unit=interval_unit, interval_size=interval_size, predict_date_from=predicting_date_from, predict_date_to=predicting_date_to, target_column=target_column, aggregation=aggregation, get_anomalies=True, rmse_upper_threshold=upper_threshold, save_name="custom")
            
            # state the proportion of anomalous data
            proportion_data_flagged_anomalous = len(anomalies)/len(predictions_dataframe_testing_nn)
            proportion_data_flagged_nan_anomalous = len(nan_anomalies)/len(predictions_dataframe_testing_nn)
            st.write("Proportion of data flagged as anomalous:", proportion_data_flagged_anomalous)
            st.write("Proportion of data flagged as NAN anomalous:", proportion_data_flagged_nan_anomalous)
            
            # download anomalies
            download_anomalies_nn = st.button("Download Anomalies", key='download_anomalies_nn')
            if download_anomalies_nn:        
                save_anomalies(button_name="download_anomalies_nn", anomalies=anomalies, nan_anomalies=nan_anomalies)
            # download individual anomalous indices
            download_individual_anomalous_indicies_nn = st.button("Download Individual Anomalous Indices", key='download_individual_anomalous_indicies_nn')
            if download_individual_anomalous_indicies_nn:        
                save_anomalies_individual_indices(button_name='download_individual_anomalous_indicies_nn', anomalies=anomalies, nan_anomalies=nan_anomalies, interval_unit=interval_unit, interval_size=interval_size)
    
    # approach 7 - AutoEncoder
    # ----------------------------------------------------------------------------    
            
        if approaches=="AutoEncoder":
    
            st.write("The data is bucketed into equal time intervals. An autoencoder is trained on the training data selected, where it compresses a vector of 10 time intervals into a low dimensional representation, and attempts to reconstruct the vector. The reconstructed values are then aggregated and used as the predicted values. The distribution of RMSEs is transformed using a cube root transformation, and is subject to a normal approximation. The transformed RMSE that corresponds to the confidence level set by the user is untransformed, and that value is used a an upper threshold. If the RMSE of a prediction exceeds that threshold, then it is marked as a potential anomaly")
                
            # get dataset for old data
            # note training on data disregarding custom indices intervals, to ensure clean
            data_training_ae, _, _ = bucket_and_get_dataset_for_AutoEncoder(df, interval_unit=interval_unit, interval_size=interval_size, aggregation=aggregation,
                                          target_column=target_column,
                                          steps=10,
                                          date_from=training_date_from, date_to=training_date_to,
                                          remove_nans=True,
                                          remove_initial_nans=True,
                                          disregard_custom_indices_intervals=True)
        
        

            prediction_aggregation = st.radio('Choose a method with which to aggregate autoencoder predictions. Mean is the default, min will be more conservative with large values, max will be more conservative will small values', ('mean', 'min', 'max'), key='prediction_aggregation')

            # option to retrain an autoencoder
            retrain_custom_ae_on_inputs = st.button("Retrain Custom AutoEncoder on Inputs", key='retrain_custom_ae_on_inputs')
            # train on previous data
            if retrain_custom_ae_on_inputs:
                train_autoencoder(dataset=data_training_ae, steps=10, train_test_split=0.8, model=autoencoder(), retrain=True, EPOCHS=100, learning_rate=0.01, print_accuracies=False, show_plots=True, save_name="custom")    
                st.success("Custom Autoencoder retrained on User inputs")
            
            # train on old data
            train_autoencoder(dataset=data_training_ae, steps=10, train_test_split=0.8, model=autoencoder(), retrain=False, EPOCHS=100, learning_rate=0.01, print_accuracies=False, show_plots=False, save_name="custom")    
            # predict back on the old data, this is to get MSE and help choose a good proportion of the old data to consider anomalous and get a corresponding upper threshold
            st.write('Predictions on training Data')
            predictions_dataframe_training_ae = bucket_and_predict_with_autoencoder(df=df, steps=10, model=autoencoder(), interval_unit=interval_unit, interval_size=interval_size, predict_date_from=training_date_from, predict_date_to=training_date_to, target_column=target_column, aggregation=aggregation, get_anomalies=False, save_name="custom", prediction_aggregation=prediction_aggregation)
            # show distributions of MSE
            plt.hist(predictions_dataframe_training_ae['RMSE'], bins=100)
            plt.title("Distribution of RMSE on Training Data")
            plt.xlabel('RMSE')
            plt.ylabel('Count')
            st.pyplot()

            _ = plot_rmse_distributions(predictions_dataframe_training_ae)
            rmse_converter = st.text_input('convert RMSE limit to confidence threshold', value=0, key='rmse_converter')
            get_upper_threshold_reverse(predictions_dataframe_training_ae, int(rmse_converter))
                
            # choose an appropriate proportion of data from the previous data to be marked as anomalous
            rmse_upper_confidence_threshold = st.slider('Choose a confidence interval for anomalies using the traninig data (95% would mean that 95% of values in the training data would not have been classified as anomalous). The corresponding RMSE then applied to the testing data', min_value=0.50, max_value=0.99, value=0.95, step = 0.01, key='rmse_upper_confidence_threshold')
            
            upper_threshold = get_upper_threshold_tranformed(predictions_dataframe=predictions_dataframe_training_ae, rmse_upper_confidence_threshold=rmse_upper_confidence_threshold, plot=True)
            
            st.write("Selected upper confidence threshold:", str(round(rmse_upper_confidence_threshold, 2)), "Corresponding RMSE Upper threshold:", str(round(upper_threshold, 2)))
                        
            # now get predictions, anomalies and nan_anomalies on the new data
            st.write('Predictions on New Data')

            predictions_dataframe_testing_ae, anomalies, nan_anomalies = bucket_and_predict_with_autoencoder(df=df['2012-03-17 16':'2012-03-17 17'], steps=10, model=autoencoder(), interval_unit=interval_unit, interval_size=interval_size, predict_date_from=predicting_date_from, predict_date_to=predicting_date_to, target_column=target_column, aggregation=aggregation, get_anomalies=True, rmse_upper_threshold=upper_threshold, save_name="custom", prediction_aggregation=prediction_aggregation)
            
            # state the proportion of anomalous data
            proportion_data_flagged_anomalous = len(anomalies)/len(predictions_dataframe_testing_ae)
            proportion_data_flagged_nan_anomalous = len(nan_anomalies)/len(predictions_dataframe_testing_ae)
            st.write("Proportion of data flagged as anomalous:", proportion_data_flagged_anomalous)
            st.write("Proportion of data flagged as NAN anomalous:", proportion_data_flagged_nan_anomalous)
            
            # download anomalies
            download_anomalies_autoencoder = st.button("Download Anomalies", key='download_anomalies_autoencoder')
            if download_anomalies_autoencoder:        
                save_anomalies(button_name="download_anomalies_autoencoder", anomalies=anomalies, nan_anomalies=nan_anomalies)
            # download individual anomalous indices
            download_individual_anomalous_indicies_autoencoder = st.button("Download Individual Anomalous Indices", key='download_individual_anomalous_indicies_autoencoder')
            if download_individual_anomalous_indicies_autoencoder:        
                save_anomalies_individual_indices(button_name='download_individual_anomalous_indicies_autoencoder', anomalies=anomalies, nan_anomalies=nan_anomalies, interval_unit=interval_unit, interval_size=interval_size)
            
#    

if tabs == "Outlier Detection - Manage Disregard Indices":
    st.write('## Outlier Detection - Managing Disregarded Indices')
    # not yet  complete
    # tabs to amend indices, expects input of 1 columns csv with column titles"dt", and format the save as the exports from the dashboard
    # can just download anomalies and keep the datetimes the user wants to add/remove to/from the disregard list
    
    amend_disregard_indices = st.sidebar.radio(
            "Amend Indices in Disregard List?:",
            ("No", "Add custom to list (import dataframe of indices 'new_indices_to_disregard.xlsx')", "remove custom from list (import dataframe of indices 'indices_to_remove_from_disregard_list.xlsx'", "See list of disregard indices", "Clear list of disregard indices"),
            key='amend_disregard_indices'
            )
            
    if amend_disregard_indices == "Add custom to list (import dataframe of indices 'new_indices_to_disregard.xlsx')":
        new_disregard_indices_df = pd.read_excel("C:\\Users\\roder\\Projects\\CyberSecurity\\new_indices_to_disregard.xlsx")
        new_disregard_indices_df.set_index('dt', inplace=True)
        add_indices_to_disregard_list_from_dataframe(new_disregard_indices_df)
        st.success('indices added to disregard list')
        st.code("Number of unique indices to disregard:" + str(len(get_disregard_indices())))
    if amend_disregard_indices == "remove custom from list (import dataframe of indices 'indices_to_remove_from_disregard_list.xlsx'":
        indices_to_remove_df = pd.read_excel("C:\\Users\\roder\\Projects\\CyberSecurity\\indices_to_remove_from_disregard_list.xlsx")
        indices_to_remove_df.set_index('dt', inplace=True)    
        remove_indices_from_disregard_list_from_dataframe(indices_to_remove_df)
        st.success('indices removed from disregard list')
        st.code("Number of unique indices to disregard:" + str(len(get_disregard_indices())))
    if amend_disregard_indices == "See list of disregard indices":
        st.code("Number of unique indices to disregard:" + str(len(get_disregard_indices())))
        st.table(get_disregard_indices())
    if amend_disregard_indices == "Clear list of disregard indices":
        remove_indices_from_disregard_list(remove_all=True)
        st.success('disregard list has been cleared')
        st.code("Number of unique indices to disregard:" + str(len(get_disregard_indices())))        

# Targeted
# ----------------------------------------------------------------------------
# ****************************************************************************
# ----------------------------------------------------------------------------
if tabs=="Exploit Detection":
    st.write('## Exploit Detection')
    # choose date from and date to
    
    st.sidebar.text('Choose a Date and Time. Checks will be performed on the 24 hours worth of data prior to the Datetime')
    date_to = st.sidebar.date_input(      
        label = "Choose a Date:",
        value = max(df_original.index),
        min_value = min(df_original.index),
        max_value = max(df_original.index),
        key='date_to'
    ) 
    
    time_to = st.sidebar.text_input('Input Hour and Minute in the form HH MM', value = '12 00')

    
    # dates to correct format
    
    date_from = str(date_to.year) + "-" + str(date_to.month) + "-" + str(date_to.day - 1) + " " + time_to[0:2] + "-" + time_to[3:5]
    date_to = str(date_to.year) + "-" + str(date_to.month) + "-" + str(date_to.day) + " " + time_to[0:2] + "-" + time_to[3:5]

    from datetime import datetime
    
# LOAD DATA
# ----------------------------------------------------------------------------
    # main data, with feature engineering already applied
    df = load_data_with_engineered_features()
    df = df[date_from:date_to]

    # warn if not enough data
    if (max(df.index) - min(df.index)).days < 1:
        st.error("'Insufficient Data - 24 hours worth of data is required prior to the selection Datetime")
    
    # also load dataframe with no feature engineering, this is just to get the column so that simulation can mimick the format of the original DNS logs
    # feature engineering is then applied to the simulated data and it is concatenated to the dataframe with feature engineering applied
    df_no_new_features = load_data_original()
    df_no_new_features['dt'] = df_no_new_features['ts'].apply(lambda x: datetime.fromtimestamp(x))
    df_no_new_features['dt'] =  df_no_new_features['dt'].apply(lambda x: datetime(x.year, x.month, x.day, x.hour, x.minute, x.second, microsecond=0))
    df_no_new_features.set_index(['dt'], inplace=True)
    df_no_new_features = df_no_new_features[date_from:date_to]
    df_no_new_features = df_no_new_features.reset_index()
    

    attack_type = st.radio(
            "Check for Attack Type:",
            ("Heartbeat", "Infected Cache", "DNS Tunnelling", "DNS DDoS Attack", "Amplified Reflection DNS DDoS Attack", "Heartbeat Fast Flux", "DNS Tunnelling Fast Flux"),
            key='attack_type'
            )

# Defining functions for targeted section
# ----------------------------------------------------------------------------

    def apply_feature_engineering_targeted(df):
        '''add the engineered features to the dataframe.
        this is used for simulation. In the simulation, only features from the original dns logs database with no feature engineering is loaded.
        After simulation, the engineered features are added'''
        
        df.drop_duplicates(subset=None, keep='first', inplace=True)
        df['ID'] = range(len(df))
        df = df.set_index('dt')
        df['top_level_domain'] = df['query'].apply(get_top_level_domain)
        df['query_length'] = df['query'].apply(len)
        df['number_of_dots_in_query'] = df['query'].apply(lambda x: x.count('.'))
        df['subdomain_length'] = df.apply(get_subdomain_length, axis=1)
        df.loc[df['TTLs'] == '-', 'TTLs'] = 0
        df.loc[df['TTLs'] == '0.000000,0.000000', 'TTLs'] = 0
        df['TTLs'] = df['TTLs'].astype('str')
        df['TTL_float_vector'] = df['TTLs'].apply(convert_TTL_to_float)
        df['TTL_max'] = df['TTL_float_vector'].apply(max)
        df['TTL_min'] = df['TTL_float_vector'].apply(min)
        df['TTL_mean'] = df['TTL_float_vector'].apply(np.mean)
        df['TTL_std'] = df['TTL_float_vector'].apply(np.std)
        df['TTL_coeff_var'] = df['TTL_std'] / df['TTL_mean']
        df['TTL_len'] = df['TTL_float_vector'].apply(len)
        df['answer_vector_length'] = df['answers'].apply(lambda x: str.split(x, sep=','))
        df['answer_vector_len'] = df['answer_vector_length'].apply(len)
        df['query_information_level'] = df['query'].apply(get_shannon_entropy)
        df['nonblank_answers'] = df['answers'].apply(lambda x: 1 if x != "-" else 0)
        df['parent_domain'] = df['query'].apply(get_parent_domain)
        df['subquery'] = df['query'].apply(get_subquery)
        df['subquery'] = df['subquery'].apply(str)
        df['subquery_information_level'] = df['subquery'].apply(get_shannon_entropy)
        df['no_response'] = df['QR'] == 'F'
        return(df)
    
    def get_infected_cache_df(df, date_to):
        
        '''for every IP address in the dataframe, find out if it has been seen before the "date_to"..
        Return a dataframe of parent domains with new IP addresses, sorted by the highest to lowest by the minimum TTL for the IP address'''
        
        # get ip addresses as prior to 17th
        df = df.reset_index()
        
        # assume that every IP address prior to the day of testing is OK (this check should be run for every date)
        # get the isins by the time that every IP address prior to the day of testing appeared, as good a measure as any, could have taken mean or anything
        ip_first_appearance_store = pd.DataFrame(df[df['dt'] < date_to].groupby(['top_level_domain', 'parent_domain', 'id.resp_h'], as_index=False)['dt'].agg('min'))
        
        # remove any of the known bad ips, we don't want them to be on record as good IPs, so remove them from the appearance store
        bad_ips = get_bad_ips()
        ip_first_appearance_store = ip_first_appearance_store[~ip_first_appearance_store['id.resp_h'].isin(bad_ips)]

        # the new ips are any ips in the dataframe that weren't present prior to the date_to
        new_ips = pd.DataFrame(df[~df['id.resp_h'].isin(ip_first_appearance_store['id.resp_h'])].groupby(['top_level_domain', 'parent_domain', 'id.resp_h'], as_index=False)['dt'].agg('min'))
        
        new_ip_list = new_ips['id.resp_h'].unique()
        
        # get any new ips with a very high minimum TTL
        new_isin_ttls = df[df['id.resp_h'].isin(new_ip_list)].groupby(['top_level_domain', 'parent_domain', 'id.resp_h'], as_index=False)['TTL_min'].agg('mean').set_index(['top_level_domain', 'parent_domain'])
                
        return(new_isin_ttls.sort_values('TTL_min', ascending=False))
            
    
    def get_heartbeat_df(df, add_simulated_labels=False, classify=False):
        '''create some new features that are used to spot out how regularly an exact query occurs,
        identify heartbeat by regularity of exact query, query information level, low number of unqiue requests to parent domain,
        return dataframe of queries with identifying features
        "add_simulated_labels" used to add the malicious flag if you are using simulated data'''
        # IP can't be spoofed here, the client needs direction! IP address is actually a good tracker here
    
        
        # add some new features for the heartbeat
        df = df.reset_index()
        df = df.sort_values(by='dt')
        df['previous_query_request_time'] = df.groupby('query')['dt'].shift()
    
        df['seconds_from_last_exact_query'] = df['dt'] - df['previous_query_request_time']
        df['seconds_from_last_exact_query'] = df['seconds_from_last_exact_query'].apply(lambda x: x.seconds)

            
        # required for heartbeat
        parent_domain_unique_value_counts = df.groupby('parent_domain')['query'].nunique()
    
        # Expected number of seconds between queries
        heartbeat_df = pd.DataFrame(df.groupby('query')['seconds_from_last_exact_query'].agg('mean'))        
    
        # Consistent number of seconds between queries
        heartbeat_df['std_of_seconds_from_last_exact_query'] = df.groupby('query')['seconds_from_last_exact_query'].agg('std')
    
        # at least a certain number of queries
        heartbeat_df['query_count'] = df.groupby('query')['ts'].agg('count')
    
        # mean query information level
        heartbeat_df['mean_query_information_level'] = df.groupby('query')['subquery_information_level'].agg('mean')
    
        # expect very few requests to this domain - perhaps only child
        # get the number of unique queries for each parent domain
        heartbeat_df['parent_domain'] = heartbeat_df.reset_index()['query'].apply(get_parent_domain).tolist()
        parent_unique_value_counts = []
        for i in range(len(heartbeat_df['parent_domain'])):
            try:
                parent_unique_value_counts.append(parent_domain_unique_value_counts[heartbeat_df['parent_domain']][i])
            except:
                print(heartbeat_df['parent_domain'][i])
        # add it to the dataframe#
        heartbeat_df['parent_domain_unique_value_counts'] = parent_unique_value_counts
        
        # rename columns
        heartbeat_df.columns = ['mean_seconds_from_last_exact_query', 'std_of_seconds_from_last_exact_query', 'query_count', 'mean_subquery_information_level', 'parent_domain', 'parent_domain_unique_value_counts']
        
        # number of unique IP addresses making query
        heartbeat_df['n_unique_IP_addresses_making_exact_query'] = df.groupby('query')['id.orig_h'].nunique()
    
        # number of unique IP addresses using parent domain
        parent_domain_ip_counts = df.groupby('parent_domain')['id.orig_h'].nunique()
        heartbeat_df = heartbeat_df.reset_index().merge(parent_domain_ip_counts, left_on='parent_domain', right_on=parent_domain_ip_counts.index).set_index('query')
    
        # rename columns again
        heartbeat_df.rename(columns={heartbeat_df.columns[-1]: "n_unique_IP_addresses_querying_parent_domain"}, inplace = True)
    
        # remove any nans
        heartbeat_df = heartbeat_df[~heartbeat_df.isna().any(axis=1)]
            
        if add_simulated_labels:
            heartbeat_df['malicious_label'] = df.groupby('query')['malicious_label'].agg('min')
            if classify:
                heartbeat_tree_model = pickle.load(open(folder_path + 'heartbeat_classifier', 'rb'))
                heartbeat_df['classification'] = heartbeat_tree_model.predict(heartbeat_df.drop(['parent_domain', 'malicious_label'], axis=1))
                
        else:
            if classify:
                heartbeat_tree_model = pickle.load(open(folder_path + 'heartbeat_classifier', 'rb'))
                heartbeat_df['classification'] = heartbeat_tree_model.predict(heartbeat_df.drop(['parent_domain'], axis=1))
    
    
        return(heartbeat_df)

        
    def perform_pca(exploit_df, exploit_df_numeric_only, using_simulated_labels=False):
        '''perform PCA on the heartbeat dataframe, adding principle components as columns, and visualising'''
        
        if using_simulated_labels:
            x_pca = exploit_df_numeric_only.drop('malicious_label', axis=1).values
        
        # Standardizing the features
        x_pca = StandardScaler().fit_transform(x_pca)
        
        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(x_pca)
        
        exploit_df['principle_component_1'] = [pc[0] for pc in principalComponents]
        exploit_df['principle_component_2'] = [pc[1] for pc in principalComponents]
        
        
        st.code(pca.explained_variance_ratio_)
        
        if using_simulated_labels:
            plt.scatter(x=exploit_df['principle_component_1'][exploit_df['malicious_label'] == 1], y=exploit_df['principle_component_2'][exploit_df['malicious_label'] == 1], c='red', alpha=1)
            plt.scatter(x=exploit_df['principle_component_1'][exploit_df['malicious_label'] == 0], y=exploit_df['principle_component_2'][exploit_df['malicious_label'] == 0], c='blue', alpha=0.1)
        else:
            plt.scatter(x=exploit_df['principle_component_1'], y=exploit_df['principle_component_2'][exploit_df['malicious_label'] == 1], alpha=1)

        
        st.pyplot()
        
        pca = PCA(n_components=3)
        principalComponents = pca.fit_transform(x_pca)
        exploit_df['principle_component_3d_1'] = [pc[0] for pc in principalComponents]
        exploit_df['principle_component_3d_2'] = [pc[1] for pc in principalComponents]
        exploit_df['principle_component_3d_3'] = [pc[2] for pc in principalComponents]

        st.code(pca.explained_variance_ratio_)
        
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111, projection='3d')

        if using_simulated_labels:
#            ax.scatter(exploit_df['principle_component_3d_1'],exploit_df['principle_component_3d_2'],exploit_df['principle_component_3d_3'], c=exploit_df['malicious_label'], cmap=plt.get_cmap("jet"), alpha=0.5);
            ax.scatter(exploit_df['principle_component_3d_1'][exploit_df['malicious_label'] == 1],exploit_df['principle_component_3d_2'][exploit_df['malicious_label'] == 1],exploit_df['principle_component_3d_3'][exploit_df['malicious_label'] == 1], c='red', alpha=1);
            ax.scatter(exploit_df['principle_component_3d_1'][exploit_df['malicious_label'] == 0],exploit_df['principle_component_3d_2'][exploit_df['malicious_label'] == 0],exploit_df['principle_component_3d_3'][exploit_df['malicious_label'] == 0], c='blue', alpha=0.1);
        else:
            ax.scatter(exploit_df['principle_component_3d_1'],exploit_df['principle_component_3d_2'],exploit_df['principle_component_3d_3'], alpha=0.5);
            
        ax.set_xlabel('Principle Component 1');
        ax.set_ylabel('Principle Component 2');
        ax.set_zlabel('Principle Component 3');
        plt.title('PCA: 3 Principle Components');
        st.pyplot()
        
        return(exploit_df)
        
    def get_tunnelling_df(df, add_simulated_labels=False, classify=False):
    
        '''identify tunnelling by the proportion of queries to a parent domain that are unique, the number of unique IPs requesting a parent domain,
        the mean query information level to a a parent domain, 
        the proportion of queries to the parent domain that are NXDOMAIN (but should not reply on this), 
        the number of UIDs that request this parent domain (but can't rely on this since easy to open new applications to request from different ports),
        return dataframe of parent domains with identifying features
        "add_simulated_labels" used to add the malicious flag if you are using simulated data'''
    
        # required for tunnelling
        parent_domain_unique_value_counts = df.groupby('parent_domain')['query'].nunique()
        
        # get the label for the query (if it has been simulated)
        tunnelling_df = pd.DataFrame(df.groupby(['parent_domain' ,'top_level_domain'])['ID'].agg('count'))  
        tunnelling_df.columns = ['query_count_in_parent_domain']
    
        # Number of unique queries to domain
        tunnelling_df['unique_query_count_in_parent_domain'] = pd.DataFrame(df.groupby(['parent_domain' ,'top_level_domain'])['query'].nunique())
        
        # Query information Level
        tunnelling_df['mean_subquery_information_level_in_parent_domain'] = df.groupby(['parent_domain' ,'top_level_domain'])['subquery_information_level'].agg('mean')
    
        tunnelling_df['proportion_queries_unique_in_parent_domain'] = tunnelling_df['unique_query_count_in_parent_domain']/tunnelling_df['query_count_in_parent_domain']
    
        if add_simulated_labels:
            tunnelling_df['malicious_label'] = df.groupby(['parent_domain' ,'top_level_domain'])['malicious_label'].agg('min')
            if classify:
                tunnelling_tree_model = pickle.load(open(folder_path + 'tunnelling_classifier', 'rb'))
                tunnelling_df['classification'] = tunnelling_tree_model.predict(tunnelling_df.drop(['malicious_label'], axis=1))
        else:
            if classify:
                tunnelling_tree_model = pickle.load(open(folder_path + 'tunnelling_classifier', 'rb'))
                tunnelling_df['classification'] = tunnelling_tree_model.predict(tunnelling_df)
            
        return(tunnelling_df)

    

    def get_DNS_DDoS_Attack_df(df, add_simulated_labels=False, classify=False):
    
        '''
        identifies DNS DDoS attack by time bucket, looking at large numbers of unique queries,
        large proportion of queries being unique,
        large number of NXDOMAIN responses from parent domain and/or large number of no responses from parent domain
        returns a dataframe of time buckets, ordered by the total number of unique queries to a parent domain.
        "add_simulated_labels" used to add the malicious flag if you are using simulated data'''
    
        # currently appears bad as dataset is full of nxdomains - server getting up and running
    
        # add some features to the dataframe
        df['NXDOMAIN_or_blank'] = df['rcode_name'].isin(["NXDOMAIN", ""])
        df['ANY_or_SOA'] = df['qtype_name'].isin(['*', 'SOA']) 
        df['no_response'] = df['QR'] == 'F'
        # number of requests from each uid to parent domain in each minute
    
        # expect at least a few more queries to parent domain than usual in 1 minute
        DNS_DDoS_Attack_df = pd.DataFrame(df.groupby(['parent_domain', 'top_level_domain', pd.Grouper(freq='1min')]).agg('count')['ID'])
        DNS_DDoS_Attack_df.columns = ['minute_counts_to_parent_domain']        
    
        # dont include arpa! used to check names
        DNS_DDoS_Attack_df = DNS_DDoS_Attack_df.loc[(DNS_DDoS_Attack_df.index.get_level_values('top_level_domain') != 'arpa')]
        # expect most of those queries to be unique, so as to never be caches
        DNS_DDoS_Attack_df['minute_total_unique_queries_to_parent_domain'] = pd.DataFrame(df.groupby(['parent_domain', 'top_level_domain', pd.Grouper(freq='1min')])['query'].nunique())
    
        # expect a good proportion of queries to be unique (maybe, if its a popular site this could just add noise)
        DNS_DDoS_Attack_df['minute_proportion_of_unique_query_counts'] = DNS_DDoS_Attack_df['minute_total_unique_queries_to_parent_domain']/DNS_DDoS_Attack_df['minute_counts_to_parent_domain']
    
        # expect either NXDOMAINS to be much higher as meaningless requests would be made
        DNS_DDoS_Attack_df['minute_total_nxdomain_or_blank_responses_from_parent_domain'] = pd.DataFrame(df.groupby(['parent_domain', 'top_level_domain', pd.Grouper(freq='1min')]).agg('sum')['NXDOMAIN_or_blank'])

        # expect either NXDOMAINS to be much higher as meaningless requests would be made
        DNS_DDoS_Attack_df['minute_proportion_nxdomain_or_blank_responses_from_parent_domain'] = DNS_DDoS_Attack_df['minute_total_nxdomain_or_blank_responses_from_parent_domain']/DNS_DDoS_Attack_df['minute_counts_to_parent_domain'] 
    

        if add_simulated_labels:
            DNS_DDoS_Attack_df['malicious_label'] = df.groupby(['parent_domain', 'top_level_domain', pd.Grouper(freq='1min')])['malicious_label'].agg('min')
            if classify:
                dns_ddos_tree_model = pickle.load(open('dns_ddos_classifier', 'rb'))
                DNS_DDoS_Attack_df['classification'] = dns_ddos_tree_model.predict(DNS_DDoS_Attack_df.drop(['malicious_label'], axis=1))
        else:
            if classify:
                dns_ddos_tree_model = pickle.load(open('dns_ddos_classifier', 'rb'))
                DNS_DDoS_Attack_df['classification'] = dns_ddos_tree_model.predict(DNS_DDoS_Attack_df)
                
        return(DNS_DDoS_Attack_df.sort_values('minute_total_unique_queries_to_parent_domain', ascending=False))
    
    def get_amp_ref_DNS_DDoS_Attack_df(df, add_simulated_labels=False, classify=False):
    
        '''
        identifies an amplified reflective DNS DDoS attack by IP address and time bucket, 
        looking at large numbers of queries from a specific IP address,
        large number of queries from specifc IPs being unique,
        large proportion of queries from specifc IP being unique,
        Large number of no responses from parent domain
        returns a dataframe of IP addresses by time buckets, ordered by the number of unique queries from that IP address in that time bucketed.
        "add_simulated_labels" used to add the malicious flag if you are using simulated data
        '''
    
        # add some features to the dataframe
        df['ANY_or_SOA'] = df['qtype_name'].isin(['*', 'SOA']) 
        df['no_response'] = df['QR'] == 'F'
    
        # get 1 minute counts from unique IP addresses
        amp_ref_DNS_DDoS_Attack_df = pd.DataFrame(df.groupby(['id.orig_h', pd.Grouper(freq='1min')]).agg('count')['ID'])
        amp_ref_DNS_DDoS_Attack_df.columns = ['minute_counts_from_unique_ip']        
    
        # expect most of those queries to be unique, so as to never be caches
        amp_ref_DNS_DDoS_Attack_df['minute_total_unique_queries_from_unique_ip'] = pd.DataFrame(df.groupby(['id.orig_h', pd.Grouper(freq='1min')])['query'].nunique())
    
        # expect a good proportion of queries to be unique (maybe, if its a popular site this could just add noise)
        amp_ref_DNS_DDoS_Attack_df['minute_total_proportion_of_unique_query_counts'] = amp_ref_DNS_DDoS_Attack_df['minute_total_unique_queries_from_unique_ip']/amp_ref_DNS_DDoS_Attack_df['minute_counts_from_unique_ip']
    
        # expect either NXDOMAINS to be much higher as meaningless requests would be made
        amp_ref_DNS_DDoS_Attack_df['minute_total_ANY_or_SOA_queries_from_unique_ip'] = pd.DataFrame(df.groupby(['id.orig_h', pd.Grouper(freq='1min')]).agg('sum')['ANY_or_SOA'])

        # expect either NXDOMAINS to be much higher as meaningless requests would be made
        amp_ref_DNS_DDoS_Attack_df['minute_proportion_ANY_or_SOA_queries_from_unique_ip'] = amp_ref_DNS_DDoS_Attack_df['minute_total_ANY_or_SOA_queries_from_unique_ip']/amp_ref_DNS_DDoS_Attack_df['minute_total_unique_queries_from_unique_ip']
            
        # or no response if the IP has been spoofed or if the bandwidth has been exhausted
        amp_ref_DNS_DDoS_Attack_df['minute_total_no_responses_from_unique_ip'] = pd.DataFrame(df.groupby(['id.orig_h', pd.Grouper(freq='1min')]).agg('sum')['no_response'])
    
        # or no response if the IP has been spoofed or if the bandwidth has been exhausted
        amp_ref_DNS_DDoS_Attack_df['minute_proportion_no_responses_from_unique_ip'] = amp_ref_DNS_DDoS_Attack_df['minute_total_no_responses_from_unique_ip']/amp_ref_DNS_DDoS_Attack_df['minute_counts_from_unique_ip']

        if add_simulated_labels:
            amp_ref_DNS_DDoS_Attack_df['malicious_label'] = df.groupby(['id.orig_h', pd.Grouper(freq='1min')])['malicious_label'].agg('min')
            if classify:
                amp_ref_dns_ddos_tree_model = pickle.load(open('amp_ref_dns_ddos_classifier', 'rb'))
                amp_ref_DNS_DDoS_Attack_df['classification'] = amp_ref_dns_ddos_tree_model.predict(amp_ref_DNS_DDoS_Attack_df.drop(['malicious_label'], axis=1))
        else:
            if classify:
                amp_ref_dns_ddos_tree_model = pickle.load(open('amp_ref_dns_ddos_classifier', 'rb'))
                amp_ref_DNS_DDoS_Attack_df['classification'] = amp_ref_dns_ddos_tree_model.predict(amp_ref_DNS_DDoS_Attack_df)
                
        return(amp_ref_DNS_DDoS_Attack_df.sort_values('minute_total_unique_queries_from_unique_ip', ascending=False))    

    def get_heartbeat_fast_flux_df(df, add_simulated_labels=False, classify=False):
    
        '''creates some new features that are used to spot out how regularly an exact subquery occurs,
        identify heartbeat by regularity of exact SUBquery, and SUBquery information level,
        since fast flux, there is no parent domain to track, so instead use exact subqueries and the mean parent domain randomness level
        return dataframe of subqueries with identifying features, ordered by subquery information level
        "add_simulated_labels" used to add the malicious flag if you are using simulated data'''
    
        # IP can't be spoofed here, the client needs direction! IP address is actually a good tracker here
    
        # add some extra features to the dataframe
        df = df.reset_index()
        df = df.sort_values(by='dt')
        df['previous_subquery_request_time'] = df.groupby('subquery')['dt'].shift()
        df['seconds_from_last_exact_subquery'] = df['dt'] - df['previous_subquery_request_time']
        df['seconds_from_last_exact_subquery'] = df['seconds_from_last_exact_subquery'].apply(lambda x: x.seconds)
    
        # required for heartbeat
        parent_domain_unique_value_counts = df.groupby('parent_domain')['query'].nunique()
        
        # Expected number of seconds between exact subqueries (subqueries rather than queries this time)
        heartbeat_fast_flux_df = pd.DataFrame(df.groupby(['subquery'])['seconds_from_last_exact_subquery'].agg(np.nanmean))
        # Consistent number of seconds between queries
        heartbeat_fast_flux_df['std'] = df.groupby(['subquery'])['seconds_from_last_exact_subquery'].agg(np.nanstd)
    
        # at least a certain number of queries
        heartbeat_fast_flux_df['subquery_count'] = df.groupby(['subquery'])['ts'].agg('count')
    
        # expecting certain level of information
        heartbeat_fast_flux_df['subquery_information_level'] = df.groupby('subquery')['subquery_information_level'].agg('mean')
        
        # FAST FLUX NEW FEATURE
        # mean parent domain information level
        heartbeat_fast_flux_df['mean_parent_randomness_level_for_subquery'] = df.groupby('subquery')['parent_domain_randomness_level'].agg('mean')
        
        heartbeat_fast_flux_df.columns = ['mean_seconds_from_last_exact_subquery', 'std_seconds_from_last_exact_subquery', 'subquery_count', 'subquery_information_level', 'mean_parent_randomness_level_for_subquery']
    
        # number of unique IP addresses making subquery
        heartbeat_fast_flux_df['n_unique_IP_addresses_making_exact_subquery'] = df.groupby(['subquery'])['id.orig_h'].nunique()
    
    
        heartbeat_fast_flux_df = heartbeat_fast_flux_df[~heartbeat_fast_flux_df.isna().any(axis=1)]
        if add_simulated_labels:
            heartbeat_fast_flux_df['malicious_label'] = df.groupby(['subquery'])['malicious_label'].agg('min')
            if classify:
                heartbeat_fast_flux_tree_model = pickle.load(open(folder_path + 'heartbeat_fast_flux_classifier', 'rb'))
                heartbeat_fast_flux_df['classification'] = heartbeat_fast_flux_tree_model.predict(heartbeat_fast_flux_df.drop(['malicious_label'], axis=1))
        else:
            if classify:
                heartbeat_fast_flux_tree_model = pickle.load(open(folder_path + 'heartbeat_fast_flux_classifier', 'rb'))
                heartbeat_fast_flux_df['classification'] = heartbeat_fast_flux_tree_model.predict(heartbeat_fast_flux_df)            
    
        return(heartbeat_fast_flux_df)

        
    def get_tunnelling_fast_flux_df(df, add_simulated_labels=False, classify=False):
        '''identify fast-flux tunnelling by the proportion of queries to a parent domain that are unique, the number of unique IPs requesting a parent domain,
        the mean query information level to a a parent domain, 
        the proportion of queries to the parent domain that are NXDOMAIN (but should not reply on this), 
        the number of UIDs that request this parent domain (but can't rely on this since easy to open new applications to request from different ports),
        and the parent domain randomness level (new for fast_flux)
        return dataframe of parent domains with identifying features
        "add_simulated_labels" used to add the malicious flag if you are using simulated data'''
    
        # Number of requests to domain
        tunnelling_fast_flux_df = pd.DataFrame(df.groupby(['parent_domain' ,'top_level_domain'])['ID'].agg('count'))
        tunnelling_fast_flux_df.columns = ['query_count_in_parent_domain']

        # required for tunnelling
        parent_domain_unique_value_counts = df.groupby('parent_domain')['query'].nunique()
        
        # Number of unique queries to domain
        tunnelling_fast_flux_df['unique_query_count_in_parent_domain'] = pd.DataFrame(df.groupby(['parent_domain' ,'top_level_domain'])['query'].nunique())
    
        # Query information Level
        tunnelling_fast_flux_df['mean_subquery_information_level_in_parent_domain'] = df.groupby(['parent_domain' ,'top_level_domain'])['subquery_information_level'].agg('mean')
    
    
        tunnelling_fast_flux_df['proportion_queries_unique_in_parent_domain'] = tunnelling_fast_flux_df['unique_query_count_in_parent_domain']/tunnelling_fast_flux_df['query_count_in_parent_domain']
    
        # FAST FLUX NEW FEATURE
        # mean parent domain information level
        tunnelling_fast_flux_df['parent_randomness_level'] = df.groupby(['parent_domain' ,'top_level_domain'])['parent_domain_randomness_level'].agg('mean')
    
        tunnelling_fast_flux_df = tunnelling_fast_flux_df.sort_values('parent_randomness_level', ascending=False)
    
        if add_simulated_labels:
            tunnelling_fast_flux_df['malicious_label'] = df.groupby(['parent_domain' ,'top_level_domain'])['malicious_label'].agg('min')        
            if classify:
                tunnelling_fast_flux_tree_model = pickle.load(open('tunnelling_fast_flux_classifier', 'rb'))
                tunnelling_fast_flux_df['classification'] = tunnelling_fast_flux_tree_model.predict(tunnelling_fast_flux_df.drop(['malicious_label'], axis=1))
        else:
            if classify:
                tunnelling_fast_flux_tree_model = pickle.load(open('tunnelling_fast_flux_classifier', 'rb'))
                tunnelling_fast_flux_df['classification'] = tunnelling_fast_flux_tree_model.predict(tunnelling_fast_flux_df)
                
        return(tunnelling_fast_flux_df)
    
    def simulate_heartbeat(df_no_new_features, df, num_simulations):
        from datetime import datetime
        '''simulates heartbeat values, should be used to make a small dataframe of simulated values, with the same columns as the original dns logs, before applying feature engineering and concatenating onto the main dataframe'''
    
        # regular exact queries, short time apart, low ttls, same parent domain
        
        latest_date_in_df = max(df.index).to_pydatetime().replace(microsecond=0)
        first_date_in_df = min(df.index).to_pydatetime().replace(microsecond=0)
        latest_possible_start = latest_date_in_df - timedelta(hours=12)
    
        latest_date_in_df = datetime.strptime(str(latest_date_in_df), '%Y-%m-%d %H:%M:%S') 
        latest_date_in_df = latest_date_in_df.strftime('%Y-%m-%d %H:%M:%S')
    
        first_date_in_df = datetime.strptime(str(first_date_in_df), '%Y-%m-%d %H:%M:%S') 
        first_date_in_df = first_date_in_df.strftime('%Y-%m-%d %H:%M:%S')
    
        latest_possible_start = datetime.strptime(str(latest_possible_start), '%Y-%m-%d %H:%M:%S') 
        latest_possible_start = latest_possible_start.strftime('%Y-%m-%d %H:%M:%S')    
        
        start_times = df[str(first_date_in_df): str(latest_possible_start)].reset_index()['dt']
    
        heartbeat_values = []
        for simulation_number in tqdm_notebook(range(num_simulations)):
    
            # allow for the possibility of multiple heartbeats for parent domains
            domain_number = random.randint(0,int(num_simulations/4))
            
            letters = string.ascii_lowercase
    
            reasonably_low_TTLs = df['TTL_min'][(df['TTL_min'] != 0) & (df['TTL_min'] < 1000)] # looking at distplot, most ttls are less than 1000
            
            datetime = start_times[np.random.choice(len(start_times))]
            first_datetime = datetime
    
            # random spacing wouldn't be more than 1 hour
            random_interal_spacing = np.random.choice([20, 30, 40, 50, 60]) # in minutes
    
            # 4-12 beats
            num_beats = np.random.choice([4, 5, 6, 7, 8, 9, 10, 11, 12])
    
            random_length_from_original_distribution = len(str(np.random.choice(df['query'])))
            random_letters = ''.join(random.choice(letters) for i in range(20))
            outgoing_message = random_letters.encode()
            key = Fernet.generate_key()
            f = Fernet(key)
            encrypted_outgoing_message = f.encrypt(outgoing_message)
            encrypted_outgoing_message = str.lower(str(encrypted_outgoing_message)[2:])
            encrypted_outgoing_message = encrypted_outgoing_message.translate(str.maketrans('', '', '=-_'))
            start_index = np.random.choice([3, 5, 10, 15, 20, 25, 30])
            end_index = start_index + random_length_from_original_distribution
            encrypted_outgoing_message = encrypted_outgoing_message[start_index:end_index]
    
            for i in range(num_beats):
                # encrypt a random string of a similar length to the dataframe
                datetime = datetime + timedelta(minutes=int(random_interal_spacing)) + timedelta(seconds=int(np.random.choice([-100,-50,-30,30,50,100]))) # some randomness but mostly evenly spaced
                ts = np.random.choice(df['ts'])
                uid = np.random.choice(df['uid'])
                id_orig_h = np.random.choice(df['id.orig_h']) # outgoing ip can change with session
                id_orig_p = np.random.choice(df['id.orig_p']) # outgoing ip can change with session
                id_resp_h = np.random.choice(df['id.resp_h']) # response ip can be made up by attacker
                id_resp_p = np.random.choice(df['id.resp_p']) # response ip can be made up by attacker
                proto = "UDP"
                port = np.random.choice(df['port']) # any port
                query = encrypted_outgoing_message + ".ATTACKER_DOMAIN" + "_" + str(domain_number) + ".com"
                qclass = np.random.choice(df['qclass']) # any class
                qclass_name = np.random.choice(df['qclass_name']) # any qclassname
                qtype = np.random.choice(df['qtype']) # any qtype
                qtype_name = np.random.choice(df['qtype_name']) # any qtype_name
                rcode = np.random.choice(df['rcode']) # any rcode
                rcode_name = np.random.choice(df['rcode_name']) # any rcode_name
                QR = np.random.choice(df['QR']) # any QR
                AA = np.random.choice(df['AA']) # any AA
                TC = np.random.choice(df['TC']) # any TC
                RD = np.random.choice(df['RD']) # any RD
                Z = np.random.choice(df['Z']) # any Z
                answers = np.random.choice(df['answers']) # any answer - can be simulated. Likely also be encrypted message but query is focus
                TTLs = np.random.choice(reasonably_low_TTLs) # at least one TTL should be low, just assume 1 here but doesn't matter since we look at min
                rejected = np.random.choice(df['rejected'])
                ID = 'HEARTBEAT'
                heartbeat_values.append([datetime, ts, uid, id_orig_h, id_orig_p, id_resp_h, id_resp_p, proto, port, query, qclass, qclass_name, qtype, qtype_name, rcode, rcode_name, QR, AA, TC, RD, Z, answers, TTLs, rejected])
    
        malicious_df = pd.DataFrame(data=heartbeat_values, columns=df_no_new_features.columns)
    
    #         print("last heartbeat occurs at:" + str(datetime))
        return(malicious_df)


    def simulate_infected_cache_func(df_no_new_features, df):       
        '''simulates infected cache, should be used to make a small dataframe of simulated values, with the same columns as the original dns logs, before applying feature engineering and concatenating onto the main dataframe'''
        
        infected_cache_values = []
        
        reasonably_high_TTLs = df['TTL_min'][df['TTL_min'] > 86400] # longer than 1 day
        
        letters = string.ascii_lowercase
        
        compromised_parent_domain = np.random.choice(df['parent_domain'][df.index < '2012-03-17'].unique())
        relevant_top_level_domain_cache = df['top_level_domain'][df['parent_domain']==compromised_parent_domain][0]

        start_times = df_no_new_features[(df_no_new_features['dt'] > '2012-03-17 12:00:00') & (df_no_new_features['dt'] < '2012-03-17 14:00:00')]['dt'].reset_index()['dt']
        datetime = start_times[np.random.choice(len(start_times))]
        first_datetime = datetime
        
        random_interal_spacing = np.random.choice([20, 30, 40, 50, 60]) # in minutes
        
        num_misdirections = np.random.choice([50, 60, 70, 80, 90, 100])
        
        st.code("compromised parent domain domain:" + str(compromised_parent_domain))
        st.code("number of misdirections:" + str(num_misdirections))
        st.code("last infected cache misdirection occurs at:" + str(first_datetime))

        
        for i in range(num_misdirections):
            datetime = datetime + timedelta(minutes=int(random_interal_spacing)) + timedelta(minutes=int(np.random.choice([0, 10, 20])))
            ts = np.random.choice(df['ts']) # cache infection can be any time, maybe it went unnoticed
            uid = np.random.choice(df['uid'])
            id_orig_h = np.random.choice(df['id.orig_h'])
            id_orig_p = np.random.choice(df['id.orig_p'])
            id_resp_h = '999.999.99.999'
            id_resp_p = np.random.choice(df['id.resp_p'])
            proto = np.random.choice(df['proto'])
            port = np.random.choice(df['port'])
            query = random_subdomain = ''.join(random.choice(letters) for i in range(5)) + "." + compromised_parent_domain + "." + relevant_top_level_domain_cache # a specifc parent domain
            print(query)
            qclass = np.random.choice(df['qclass'])
            qclass_name = np.random.choice(df['qclass_name'])
            qtype = np.random.choice(df['qtype'])
            qtype_name = np.random.choice(df['qtype_name'])
            rcode = np.random.choice(df['rcode'])
            rcode_name = np.random.choice(df['rcode_name'])
            QR = np.random.choice(df['QR'])
            AA = np.random.choice(df['AA'])
            TC = np.random.choice(df['TC'])
            RD = np.random.choice(df['RD'])
            Z = np.random.choice(df['Z'])
            answers = '123.45.6.789'
            TTLs = np.random.choice(reasonably_high_TTLs)
            rejected = np.random.choice(df['rejected'])
            ID = 'INFECTED CACHE'
            
            infected_cache_values.append([datetime, ts, uid, id_orig_h, id_orig_p, id_resp_h, id_resp_p, proto, port, query, qclass, qclass_name, qtype, qtype_name, rcode, rcode_name, QR, AA, TC, RD, Z, answers, TTLs, rejected])        
                    
        malicious_df = pd.DataFrame(data=infected_cache_values, columns=df_no_new_features.columns)
        
        st.code("last infected cache misdirection occurs at:" + str(datetime))
        return(malicious_df)

    def simulate_DNS_tunnelling(df_no_new_features, df, num_simulations):
        from datetime import datetime
        '''simulates heartbeat values, should be used to make a small dataframe of simulated values, with the same columns as the original dns logs, before applying feature engineering and concatenating onto the main dataframe'''
    
        # different subquery each time, since tunnnelling info, same parent domain
        # short time apart for subqueries - malware has to actually do something
        
        latest_date_in_df = max(df.index).to_pydatetime().replace(microsecond=0)
        first_date_in_df = min(df.index).to_pydatetime().replace(microsecond=0)
        latest_possible_start = latest_date_in_df - timedelta(hours=12)
    
        latest_date_in_df = datetime.strptime(str(latest_date_in_df), '%Y-%m-%d %H:%M:%S') 
        latest_date_in_df = latest_date_in_df.strftime('%Y-%m-%d %H:%M:%S')
    
        first_date_in_df = datetime.strptime(str(first_date_in_df), '%Y-%m-%d %H:%M:%S') 
        first_date_in_df = first_date_in_df.strftime('%Y-%m-%d %H:%M:%S')
    
        latest_possible_start = datetime.strptime(str(latest_possible_start), '%Y-%m-%d %H:%M:%S') 
        latest_possible_start = latest_possible_start.strftime('%Y-%m-%d %H:%M:%S')    
        
        start_times = df[str(first_date_in_df): str(latest_possible_start)].reset_index()['dt']
    
        dns_tunnelling_values = []
        for simulation_number in tqdm_notebook(range(num_simulations)):
    
            # allow for the possibility of multiple compromised pcs tunnelling to the same domain
            domain_number = random.randint(0,int(num_simulations/4))
            
            letters = string.ascii_lowercase
            
            datetime = start_times[np.random.choice(len(start_times))]
            first_datetime = datetime
    
            num_instances = np.random.choice([50,100,150,200,250])
    
            for i in range(num_instances):
                # encrypt  a random string of a similar length to the dataframe
                random_length_from_original_distribution = len(str(np.random.choice(df['query'])))
                random_letters = ''.join(random.choice(letters) for i in range(20))
                outgoing_message = random_letters.encode()
                key = Fernet.generate_key()
                f = Fernet(key)
                encrypted_outgoing_message = f.encrypt(outgoing_message)
                encrypted_outgoing_message = str.lower(str(encrypted_outgoing_message)[2:])
                encrypted_outgoing_message = encrypted_outgoing_message.translate(str.maketrans('', '', '=-_'))
                start_index = np.random.choice([3, 5, 10, 15, 20, 25, 30])
                end_index = start_index + random_length_from_original_distribution
                encrypted_outgoing_message = encrypted_outgoing_message[start_index:end_index]
            
                random_interal_spacing = np.random.choice([1, 2, 5, 10]) # in seconds
    
                datetime = datetime + timedelta(seconds=int(random_interal_spacing)) + timedelta(seconds=int(np.random.choice([-5,-3,-1,1,3,5]))) # some randomness but mostly evenly spaced
                ts = np.random.choice(df['ts'])
                uid = np.random.choice(df['uid'])
                id_orig_h = np.random.choice(df['id.orig_h'])
                id_orig_p = np.random.choice(df['id.orig_p'])
                id_resp_h = np.random.choice(df['id.resp_h'])
                id_resp_p = np.random.choice(df['id.resp_p'])
                proto = np.random.choice(df['proto'])
                port = np.random.choice(df['port'])
                query = encrypted_outgoing_message + ".ATTACKER_DOMAIN" + "_" + str(domain_number) + ".com"
                qclass = np.random.choice(df['qclass'])
                qclass_name = np.random.choice(df['qclass_name'])
                qtype = np.random.choice(df['qtype'])
                qtype_name = np.random.choice(df['qtype_name'])
                rcode = np.random.choice(df['rcode'])
                rcode_name = np.random.choice(df['rcode_name'])
                QR = np.random.choice(df['QR'])
                AA = np.random.choice(df['AA'])
                TC = np.random.choice(df['TC'])
                RD = np.random.choice(df['RD'])
                Z = np.random.choice(df['Z'])
                answers = np.random.choice(df['answers']) # could be anything, attacker may have simulated
                TTLs = np.random.choice(df['TTLs'])
                rejected = np.random.choice(df['rejected'])
                ID = 'DNS_TUNNELLING'
    
                dns_tunnelling_values.append([datetime, ts, uid, id_orig_h, id_orig_p, id_resp_h, id_resp_p, proto, port, query, qclass, qclass_name, qtype, qtype_name, rcode, rcode_name, QR, AA, TC, RD, Z, answers, TTLs, rejected]) 
    
        malicious_df = pd.DataFrame(data=dns_tunnelling_values, columns=df_no_new_features.columns)
    
        return(malicious_df)


    def simulate_DNS_DDoS_Attack(df_no_new_features, df, num_simulations):
        from datetime import datetime
        '''simulates DNS DDoS attack, should be used to make a small dataframe of simulated values, with the same columns as the original dns logs, before applying feature engineering and concatenating onto the main dataframe'''
    
        # many instances of queries (especially unique queries) to a specific domain in a time interval
        # responses should be NXDOMAIN or blank
        
        latest_date_in_df = max(df.index).to_pydatetime().replace(microsecond=0)
        first_date_in_df = min(df.index).to_pydatetime().replace(microsecond=0)
        latest_possible_start = latest_date_in_df - timedelta(hours=12)
    
        latest_date_in_df = datetime.strptime(str(latest_date_in_df), '%Y-%m-%d %H:%M:%S') 
        latest_date_in_df = latest_date_in_df.strftime('%Y-%m-%d %H:%M:%S')
    
        first_date_in_df = datetime.strptime(str(first_date_in_df), '%Y-%m-%d %H:%M:%S') 
        first_date_in_df = first_date_in_df.strftime('%Y-%m-%d %H:%M:%S')
    
        latest_possible_start = datetime.strptime(str(latest_possible_start), '%Y-%m-%d %H:%M:%S') 
        latest_possible_start = latest_possible_start.strftime('%Y-%m-%d %H:%M:%S')    
    
        start_times = df[str(first_date_in_df): str(latest_possible_start)].reset_index()['dt']
    
        DNS_DDoS_attack_values = []
        for simulation_number in tqdm_notebook(range(num_simulations)):
            letters = string.ascii_lowercase
            domain_number = random.randint(0,int(num_simulations/4))
            datetime = start_times[np.random.choice(len(start_times))]
            first_datetime = datetime
    
    
            target_parent_domain_ddos = ".VICTIM_DOMAIN" + "_" + str(domain_number) + ".com"
    
            num_instances = np.random.choice([1000,1100,1200,1300,1400])
            print("Victim Domain:" + target_parent_domain_ddos)
            print("number of DDoS requests made:" + str(num_instances))
            print("first instance occurs at:" + str(first_datetime))
    
            for i in range(num_instances):
    
                datetime = datetime + timedelta(milliseconds=int(np.random.choice([50, 100, 200]))) # some randomness but mostly evenly spaced
                ts = np.random.choice(df['ts'])
                uid =  np.random.choice(df['uid'])
                id_orig_h = np.random.choice(df['id.orig_h'])
                id_orig_p = np.random.choice(df['id.orig_p'])
                id_resp_h = np.random.choice(df['id.resp_h'])
                id_resp_p = np.random.choice(df['id.resp_p'])
                proto = np.random.choice(df['proto'])
                port = np.random.choice(df['port'])
                query = ''.join(random.choice(letters) for i in range(5)) + "." + target_parent_domain_ddos
                qclass = np.random.choice(df['qclass'])
                qclass_name = np.random.choice(df['qclass_name'])
                qtype = np.random.choice(df['qtype'])
                qtype_name = np.random.choice(df['qtype_name'])
                rcode = 3  # many responses will be NXDOMAIN, perhaps some work by chance
                rcode_name = np.random.choice(['NXDOMAIN', ""]) # many responses will be NXDOMAIN or blank, perhaps some work by chance
                QR = np.random.choice(df['QR'])
                AA = np.random.choice(df['AA'])
                TC = np.random.choice(df['TC'])
                RD = np.random.choice(df['RD'])
                Z = np.random.choice(df['Z'])
                answers = np.random.choice(df['answers'])
                TTLs = np.random.choice(df['TTLs'])
                rejected = np.random.choice(df['rejected'])
                ID = 'DNS_DDoS_ATTACK'
    
                DNS_DDoS_attack_values.append([datetime, ts, uid, id_orig_h, id_orig_p, id_resp_h, id_resp_p, proto, port, query, qclass, qclass_name, qtype, qtype_name, rcode, rcode_name, QR, AA, TC, RD, Z, answers, TTLs, rejected]) 
    
        malicious_df = pd.DataFrame(data=DNS_DDoS_attack_values, columns=df_no_new_features.columns)
        return(malicious_df)



    def simulate_amp_ref_DNS_DDoS_Attack(df_no_new_features, df, num_simulations):
        '''simulates amplified reflective DNS DDoS attack, should be used to make a small dataframe of simulated values, with the same columns as the original dns logs, before applying feature engineering and concatenating onto the main dataframe'''
    
        # many requests from 1 (spoofed) IP in a short period of time
        # don't need to be unique
        # probably to the same domain
        
        from datetime import datetime
        
        letters = string.ascii_lowercase
    
        latest_date_in_df = max(df.index).to_pydatetime().replace(microsecond=0)
        first_date_in_df = min(df.index).to_pydatetime().replace(microsecond=0)
        latest_possible_start = latest_date_in_df - timedelta(hours=12)
    
        latest_date_in_df = datetime.strptime(str(latest_date_in_df), '%Y-%m-%d %H:%M:%S') 
        latest_date_in_df = latest_date_in_df.strftime('%Y-%m-%d %H:%M:%S')
    
        first_date_in_df = datetime.strptime(str(first_date_in_df), '%Y-%m-%d %H:%M:%S') 
        first_date_in_df = first_date_in_df.strftime('%Y-%m-%d %H:%M:%S')
    
        latest_possible_start = datetime.strptime(str(latest_possible_start), '%Y-%m-%d %H:%M:%S') 
        latest_possible_start = latest_possible_start.strftime('%Y-%m-%d %H:%M:%S')    
        
        start_times = df[str(first_date_in_df): str(latest_possible_start)].reset_index()['dt']
    
        amp_refl_DNS_DDoS_attack_values = []
        for simulation_number in tqdm_notebook(range(num_simulations)):   
            
            domain_number = random.randint(0,int(num_simulations/4))
            
            large_domain_amp_ref = np.random.choice(df['parent_domain'].unique())
            relevant_top_level_domain_amp_ref = ".com"
    
            target_ip = "SIMULATED_IP_" + str(domain_number)
    
            num_instances = np.random.choice([1000,1100,1200,1300,1400])
    
            datetime = start_times[np.random.choice(len(start_times))]
            for i in range(num_instances):
                random_interal_spacing = np.random.choice([1, 2, 3, 5]) # in seconds
                
                
                datetime = datetime + timedelta(milliseconds=int(np.random.choice([50, 100, 200]))) # some randomness but mostly evenly spaced
                ts = np.random.choice(df['ts'])
                uid =  np.random.choice(df['uid'])
                id_orig_h = target_ip # the source ip will be the target ip!
                id_orig_p = np.random.choice(df['id.orig_p'])
                id_resp_h = np.random.choice(df['id.resp_h'])
                id_resp_p = np.random.choice(df['id.resp_p'])
                proto = np.random.choice(df['proto'])
                port = np.random.choice(df['port'])
                query = ''.join(random.choice(letters) for i in range(5)) + "." + large_domain_amp_ref + "." + relevant_top_level_domain_amp_ref # a specifc parent domain
                qclass = np.random.choice(df['qclass'])
                qclass_name = np.random.choice(df['qclass_name'])
                qtype = np.random.choice(df['qtype'])
                qtype_name = np.random.choice(df['qtype_name'])
                rcode = '-' # no response since spoofed IP
                rcode_name = '-' # no response since spoofed IP
                QR = 'T' # still a query since no response
                AA = np.random.choice(df['AA'])
                TC = np.random.choice(df['TC'])
                RD = np.random.choice(df['RD'])
                Z = np.random.choice(df['Z'])
                answers = np.random.choice(df['answers'])
                TTLs = np.random.choice(df['TTLs'])
                rejected = np.random.choice(df['rejected'])
                ID = 'AMP_REFL_DNS_DDOS_ATTACK'
    
                amp_refl_DNS_DDoS_attack_values.append([datetime, ts, uid, id_orig_h, id_orig_p, id_resp_h, id_resp_p, proto, port, query, qclass, qclass_name, qtype, qtype_name, rcode, rcode_name, QR, AA, TC, RD, Z, answers, TTLs, rejected]) 
        
        malicious_df = pd.DataFrame(data=amp_refl_DNS_DDoS_attack_values, columns=df_no_new_features.columns)
        return(malicious_df)


    def simulate_heartbeat_fast_flux(df_no_new_features, df, num_simulations):
        '''simulates heartbeat fast flux values, should be used to make a small dataframe of simulated values, with the same columns as the original dns logs, before applying feature engineering and concatenating onto the main dataframe'''
        from datetime import datetime
        
        letters = string.ascii_lowercase
    
        latest_date_in_df = max(df.index).to_pydatetime().replace(microsecond=0)
        first_date_in_df = min(df.index).to_pydatetime().replace(microsecond=0)
        latest_possible_start = latest_date_in_df - timedelta(hours=12)
    
        latest_date_in_df = datetime.strptime(str(latest_date_in_df), '%Y-%m-%d %H:%M:%S') 
        latest_date_in_df = latest_date_in_df.strftime('%Y-%m-%d %H:%M:%S')
    
        first_date_in_df = datetime.strptime(str(first_date_in_df), '%Y-%m-%d %H:%M:%S') 
        first_date_in_df = first_date_in_df.strftime('%Y-%m-%d %H:%M:%S')
    
        latest_possible_start = datetime.strptime(str(latest_possible_start), '%Y-%m-%d %H:%M:%S') 
        latest_possible_start = latest_possible_start.strftime('%Y-%m-%d %H:%M:%S')    
        
        start_times = df[str(first_date_in_df): str(latest_possible_start)].reset_index()['dt']
        heartbeat_values = []    
        for simulation_number in tqdm_notebook(range(num_simulations)): 
    
            # generate encrypted query
            outgoing_message = "botnet heartbeat".encode()
            key = Fernet.generate_key() # Store this key or get if you already have it
            f = Fernet(key)
            encrypted_outgoing_message = f.encrypt(outgoing_message)
    
            reasonably_low_TTLs = df['TTL_min'][(df['TTL_min'] != 0) & (df['TTL_min'] < 1000)] # looking at distplot, most ttls are less than 1000
    
            # the hearbeat would usually go throughout the first 2 days, but too obvious since the server goes down
            # have it start on the second day, sometime between 12:30 and 12:40 will give time for an interation as most every hour, 8 times
            datetime = start_times[np.random.choice(len(start_times))]
            first_datetime = datetime
            
            random_interal_spacing = np.random.choice([20, 30, 40, 50, 60]) # in minutes
    
            num_beats_per_fast_flux = np.random.choice([6, 7, 8, 9, 10, 11, 12])
    
    
            random_length_from_original_distribution = len(str(np.random.choice(df['query'])))
            random_letters = ''.join(random.choice(letters) for i in range(20))
            outgoing_message = random_letters.encode()
            key = Fernet.generate_key()
            f = Fernet(key)
            encrypted_outgoing_message = f.encrypt(outgoing_message)
            encrypted_outgoing_message = str.lower(str(encrypted_outgoing_message)[2:])
            encrypted_outgoing_message = encrypted_outgoing_message.translate(str.maketrans('', '', '=-_'))
            start_index = np.random.choice([3, 5, 10, 15, 20, 25, 30])
            end_index = start_index + random_length_from_original_distribution
            encrypted_outgoing_message = encrypted_outgoing_message[start_index:end_index]
            
            for i in range(5):
                random_length_from_original_distribution = len(str(np.random.choice(df['parent_domain'])))
                random_letters = ''.join(random.choice(letters) for i in range(20))
                parent_domain_encoded = random_letters.encode()
                encrypted_parent_domain = f.encrypt(parent_domain_encoded)
                encrypted_parent_domain = str.lower(str(encrypted_parent_domain)[2:])
                encrypted_parent_domain = encrypted_parent_domain.translate(str.maketrans('', '', '=-_'))
    
                start_index = np.random.choice([3, 5, 10, 15, 20, 25, 30])
                end_index = start_index + random_length_from_original_distribution
                encrypted_parent_domain = encrypted_parent_domain[start_index:end_index]
                
            query = encrypted_outgoing_message + "." + encrypted_parent_domain +".com"
    
            for beat in range(num_beats_per_fast_flux):
                datetime = datetime + timedelta(minutes=int(random_interal_spacing)) + timedelta(seconds=int(np.random.choice([-100,-50,-30,30,50,100]))) # some randomness but mostly evenly spaced
                ts = np.random.choice(df['ts'])
                uid = np.random.choice(df['uid'])
                id_orig_h = np.random.choice(df['id.orig_h']) # outgoing ip can change with session
                id_orig_p = np.random.choice(df['id.orig_p']) # outgoing ip can change with session
                id_resp_h = np.random.choice(df['id.resp_h']) # response ip can be made up by attacker
                id_resp_p = np.random.choice(df['id.resp_p']) # response ip can be made up by attacker
                proto = "UDP"
                port = np.random.choice(df['port']) # any port
                qclass = np.random.choice(df['qclass']) # any class
                qclass_name = np.random.choice(df['qclass_name']) # any qclassname
                qtype = np.random.choice(df['qtype']) # any qtype
                qtype_name = np.random.choice(df['qtype_name']) # any qtype_name
                rcode = np.random.choice(df['rcode']) # any rcode
                rcode_name = np.random.choice(df['rcode_name']) # any rcode_name
                QR = np.random.choice(df['QR']) # any QR
                AA = np.random.choice(df['AA']) # any AA
                TC = np.random.choice(df['TC']) # any TC
                RD = np.random.choice(df['RD']) # any RD
                Z = np.random.choice(df['Z']) # any Z
                answers = np.random.choice(df['answers']) # any answer - can be simulated. Likely also be encrypted message but query is focus
                TTLs = np.random.choice(reasonably_low_TTLs) # at least one TTL should be low, just assume 1 here but doesn't matter since we look at min
                rejected = np.random.choice(df['rejected'])
                ID = 'HEARTBEAT'
                heartbeat_values.append([datetime, ts, uid, id_orig_h, id_orig_p, id_resp_h, id_resp_p, proto, port, query, qclass, qclass_name, qtype, qtype_name, rcode, rcode_name, QR, AA, TC, RD, Z, answers, TTLs, rejected]) 
    
        malicious_df = pd.DataFrame(data=heartbeat_values, columns=df_no_new_features.columns)
    
        return(malicious_df)
    

    def simulate_DNS_tunnelling_fast_flux(df_no_new_features, df, num_simulations):
        '''simulates DNS tunnelling fast flux values, should be used to make a small dataframe of simulated values, with the same columns as the original dns logs, before applying feature engineering and concatenating onto the main dataframe'''
        
        
        from datetime import datetime
        
        letters = string.ascii_lowercase
    
        latest_date_in_df = max(df.index).to_pydatetime().replace(microsecond=0)
        first_date_in_df = min(df.index).to_pydatetime().replace(microsecond=0)
        latest_possible_start = latest_date_in_df - timedelta(hours=12)
    
        latest_date_in_df = datetime.strptime(str(latest_date_in_df), '%Y-%m-%d %H:%M:%S') 
        latest_date_in_df = latest_date_in_df.strftime('%Y-%m-%d %H:%M:%S')
    
        first_date_in_df = datetime.strptime(str(first_date_in_df), '%Y-%m-%d %H:%M:%S') 
        first_date_in_df = first_date_in_df.strftime('%Y-%m-%d %H:%M:%S')
    
        latest_possible_start = datetime.strptime(str(latest_possible_start), '%Y-%m-%d %H:%M:%S') 
        latest_possible_start = latest_possible_start.strftime('%Y-%m-%d %H:%M:%S')    
        
        start_times = df[str(first_date_in_df): str(latest_possible_start)].reset_index()['dt']
        
        dns_tunnelling_values = []
        
        fast_flux_domains = []
        for simulation_number in tqdm_notebook(range(num_simulations)):
            for i in range(5):
                random_length_from_original_distribution = len(str(np.random.choice(df['parent_domain'])))
                random_letters = ''.join(random.choice(letters) for i in range(20))
                parent_domain_encoded = random_letters.encode()
                key = Fernet.generate_key()
                f = Fernet(key)    
                encrypted_parent_domain = f.encrypt(parent_domain_encoded)
                encrypted_parent_domain = str.lower(str(encrypted_parent_domain)[2:])
                encrypted_parent_domain = encrypted_parent_domain.translate(str.maketrans('', '', '=-_'))
    
                start_index = np.random.choice([3, 5, 10, 15, 20, 25, 30])
                end_index = start_index + random_length_from_original_distribution
                encrypted_parent_domain = encrypted_parent_domain[start_index:end_index]
                fast_flux_domains.append(encrypted_parent_domain)
    
            relevant_top_level_domain_tunnelling = 'com'
    
            datetime = start_times[np.random.choice(len(start_times))]
            first_datetime = datetime
    
            num_instances = np.random.choice([2, 3, 4, 5])
            for fast_flux_name in fast_flux_domains:
                parent_domain = fast_flux_name
    
                for i in range(num_instances):
                    random_interal_spacing = np.random.choice([1, 2, 5, 10]) # in seconds
                    random_length_from_original_distribution = len(str(np.random.choice(df['query'])))
                    random_letters = ''.join(random.choice(letters) for i in range(20))
                    outgoing_message = random_letters.encode()
                    key = Fernet.generate_key()
                    f = Fernet(key)
                    encrypted_outgoing_message = f.encrypt(outgoing_message)
                    encrypted_outgoing_message = str.lower(str(encrypted_outgoing_message)[2:])
                    encrypted_outgoing_message = encrypted_outgoing_message.translate(str.maketrans('', '', '=-_'))
                    start_index = np.random.choice([3, 5, 10, 15, 20, 25, 30])
                    end_index = start_index + random_length_from_original_distribution
                    encrypted_outgoing_message = encrypted_outgoing_message[start_index:end_index]
    
                    datetime = datetime + timedelta(seconds=int(random_interal_spacing)) + timedelta(seconds=int(np.random.choice([-5,-3,-1,1,3,5]))) # some randomness but mostly evenly spaced
                    ts = np.random.choice(df['ts'])
                    uid = np.random.choice(df['uid'])
                    id_orig_h = np.random.choice(df['id.orig_h'])
                    id_orig_p = np.random.choice(df['id.orig_p'])
                    id_resp_h = np.random.choice(df['id.resp_h'])
                    id_resp_p = np.random.choice(df['id.resp_p'])
                    proto = np.random.choice(df['proto'])
                    port = np.random.choice(df['port'])
                    query = encrypted_outgoing_message + "." + parent_domain + "." + relevant_top_level_domain_tunnelling # a specifc parent domain
                    qclass = np.random.choice(df['qclass'])
                    qclass_name = np.random.choice(df['qclass_name'])
                    qtype = np.random.choice(df['qtype'])
                    qtype_name = np.random.choice(df['qtype_name'])
                    rcode = np.random.choice(df['rcode'])
                    rcode_name = np.random.choice(df['rcode_name'])
                    QR = np.random.choice(df['QR'])
                    AA = np.random.choice(df['AA'])
                    TC = np.random.choice(df['TC'])
                    RD = np.random.choice(df['RD'])
                    Z = np.random.choice(df['Z'])
                    answers = np.random.choice(df['answers']) # could be anything, attacker may have simulated
                    TTLs = np.random.choice(df['TTLs'])
                    rejected = np.random.choice(df['rejected'])
                    ID = 'DNS_TUNNELLING'
    
                    dns_tunnelling_values.append([datetime, ts, uid, id_orig_h, id_orig_p, id_resp_h, id_resp_p, proto, port, query, qclass, qclass_name, qtype, qtype_name, rcode, rcode_name, QR, AA, TC, RD, Z, answers, TTLs, rejected]) 
    
        malicious_df = pd.DataFrame(data=dns_tunnelling_values, columns=df_no_new_features.columns)
        df_no_new_features = pd.concat([df_no_new_features.reset_index(), malicious_df])
        return(df_no_new_features)




    def train_model_with_simulated_values(df, exploit):
        exploit_types = ['heartbeat', 'tunnelling', 'dns_ddos', 'amp_ref_dns_ddos', 'heartbeat_fast_flux', 'tunnelling_fast_flux']
        if exploit not in exploit_types:
            raise ValueError("Invalid exploit type. Expected one of: %s" % exploit_types)
        
        # add the simulations to the dataframe
        if exploit == 'heartbeat':
            
            # add simulations
            num_simulations = 10
            malicious_df = simulate_heartbeat(df_no_new_features=df_no_new_features, df=df, num_simulations=4*num_simulations)
            malicious_df = apply_feature_engineering_targeted(malicious_df)
    
            # apply labels
            df['malicious_label'] = 0
            malicious_df['malicious_label'] = 1
    
            # add the malicious values to the already engineered dataframe, and call a new name
            df_inc_simulated = pd.concat([df, malicious_df])

            # get the feature dataframe
            heartbeat_simulated_df = get_heartbeat_df(df_inc_simulated, add_simulated_labels=True, classify=False)
            # print what proportion of data is simulated
            print('proportion of data simulated:', len(heartbeat_simulated_df[heartbeat_simulated_df['malicious_label'] == 1])/len(heartbeat_simulated_df))
            
            # train test split
            X_train, X_test, y_train, y_test = train_test_split(heartbeat_simulated_df.drop(['malicious_label', 'parent_domain'],axis=1),heartbeat_simulated_df['malicious_label'],test_size=0.2)
            
            # visualise single tree
            heartbeat_tree_model = tree.DecisionTreeClassifier(criterion = "entropy", max_depth = 10)
            heartbeat_tree_model.fit(X_train, y_train)
            
            # see  score
            print('tree model score:', heartbeat_tree_model.score(X_test, y_test))
            
            # save model
            pickle.dump(heartbeat_tree_model, open(folder_path + 'heartbeat_classifier', 'wb'))
            
            # load model
            heartbeat_tree_model = pickle.load(open(folder_path + 'heartbeat_classifier', 'rb'))
    
            # use on dataframe
            heartbeat_simulated_df['classification'] = heartbeat_tree_model.predict(heartbeat_simulated_df.drop(['parent_domain', 'malicious_label'], axis=1))
            heartbeat_simulated_df['incorrect_classification'] = heartbeat_simulated_df['classification'] != heartbeat_simulated_df['malicious_label'] 
            heartbeat_simulated_df['classification'] = heartbeat_simulated_df['classification'].apply(int)
            conf_mat = confusion_matrix(heartbeat_simulated_df['malicious_label'], heartbeat_simulated_df['classification'])
            
            # show graph
            dot_data = StringIO()
            tree.export_graphviz(heartbeat_tree_model, 
             out_file=dot_data, 
             class_names=['0', '1'], # the target names.
             feature_names=X_train.columns, # the feature names.
             filled=True, # Whether to fill in the boxes with colours.
             rounded=True, # Whether to round the corners of the boxes.
             special_characters=True)
            (graph,) = pydot.graph_from_dot_data(dot_data.getvalue())
            graph.write_png(folder_path + "heartbeat_tree.png")
            return(heartbeat_simulated_df, conf_mat)
        
        if exploit == 'tunnelling':
            
            # add simulations
            num_simulations = 50
            malicious_df = simulate_DNS_tunnelling(df_no_new_features=df_no_new_features, df=df, num_simulations=4*num_simulations)
            malicious_df = apply_feature_engineering_targeted(malicious_df)
    
            # apply labels
            df['malicious_label'] = 0
            malicious_df['malicious_label'] = 1
    
            # add the malicious values to the already engineered dataframe, and call a new name
            df_inc_simulated = pd.concat([df, malicious_df])
            
            # get the feature dataframe
            tunnelling_simulated_df = get_tunnelling_df(df_inc_simulated, add_simulated_labels=True, classify=False)
            # print what proportion of data is simulated
            print('proportion of data simulated:', len(tunnelling_simulated_df[tunnelling_simulated_df['malicious_label'] == 1])/len(tunnelling_simulated_df))
            
            # train test split
            X_train, X_test, y_train, y_test = train_test_split(tunnelling_simulated_df.drop(['malicious_label'],axis=1),tunnelling_simulated_df['malicious_label'],test_size=0.2)
            
            # visualise single tree
            tunnelling_tree_model = tree.DecisionTreeClassifier(criterion = "entropy", max_depth = 10)
            tunnelling_tree_model.fit(X_train, y_train)
            
            # see  score
            print('tree model score:', tunnelling_tree_model.score(X_test, y_test))
            
            # save model
            pickle.dump(tunnelling_tree_model, open(folder_path + 'tunnelling_classifier', 'wb'))
            
            # load model
            tunnelling_tree_model = pickle.load(open(folder_path + 'tunnelling_classifier', 'rb'))
    
            # use on dataframe
            tunnelling_simulated_df['classification'] = tunnelling_tree_model.predict(tunnelling_simulated_df.drop(['malicious_label'], axis=1))
            tunnelling_simulated_df['incorrect_classification'] = tunnelling_simulated_df['classification'] != tunnelling_simulated_df['malicious_label'] 
            tunnelling_simulated_df['classification'] = tunnelling_simulated_df['classification'].apply(int)
            conf_mat = confusion_matrix(tunnelling_simulated_df['malicious_label'], tunnelling_simulated_df['classification'])
            
            # show graph
            dot_data = StringIO()
            tree.export_graphviz(tunnelling_tree_model, 
             out_file=dot_data, 
             class_names=['0', '1'], # the target names.
             feature_names=X_train.columns, # the feature names.
             filled=True, # Whether to fill in the boxes with colours.
             rounded=True, # Whether to round the corners of the boxes.
             special_characters=True)
            (graph,) = pydot.graph_from_dot_data(dot_data.getvalue())
            graph.write_png(folder_path + "tunnelling_tree.png")
            return(tunnelling_simulated_df, conf_mat)
        
        if exploit == 'dns_ddos':
            
            # add simulations
            num_simulations = 50
            malicious_df = simulate_DNS_DDoS_Attack(df_no_new_features=df_no_new_features, df=df, num_simulations=num_simulations)
            malicious_df = apply_feature_engineering_targeted(malicious_df)
    
            # apply labels
            df['malicious_label'] = 0
            malicious_df['malicious_label'] = 1
    
            # add the malicious values to the already engineered dataframe, and call a new name
            df_inc_simulated = pd.concat([df, malicious_df])
            
            # get the feature dataframe
            dns_ddos_simulated_df = get_DNS_DDoS_Attack_df(df_inc_simulated, add_simulated_labels=True, classify=False)
            # print what proportion of data is simulated
            print('proportion of data simulated:', len(dns_ddos_simulated_df[dns_ddos_simulated_df['malicious_label'] == 1])/len(dns_ddos_simulated_df))
            
            # train test split
            X_train, X_test, y_train, y_test = train_test_split(dns_ddos_simulated_df.drop(['malicious_label'],axis=1),dns_ddos_simulated_df['malicious_label'],test_size=0.2)
            
            # visualise single tree
            dns_ddos_tree_model = tree.DecisionTreeClassifier(criterion = "entropy", max_depth = 10)
            dns_ddos_tree_model.fit(X_train, y_train)
            
            # see  score
            print('tree model score:', dns_ddos_tree_model.score(X_test, y_test))
            
            # save model
            pickle.dump(dns_ddos_tree_model, open(folder_path + 'dns_ddos_classifier', 'wb'))
            
            # load model
            dns_ddos_tree_model = pickle.load(open(folder_path + 'dns_ddos_classifier', 'rb'))
    
            # use on dataframe
            dns_ddos_simulated_df['classification'] = dns_ddos_tree_model.predict(dns_ddos_simulated_df.drop(['malicious_label'], axis=1))
            dns_ddos_simulated_df['incorrect_classification'] = dns_ddos_simulated_df['classification'] != dns_ddos_simulated_df['malicious_label'] 
            dns_ddos_simulated_df['classification'] = dns_ddos_simulated_df['classification'].apply(int)
            conf_mat = confusion_matrix(dns_ddos_simulated_df['malicious_label'], dns_ddos_simulated_df['classification'])
            
            # show graph
            dot_data = StringIO()
            tree.export_graphviz(dns_ddos_tree_model, 
             out_file=dot_data, 
             class_names=['0', '1'], # the target names.
             feature_names=X_train.columns, # the feature names.
             filled=True, # Whether to fill in the boxes with colours.
             rounded=True, # Whether to round the corners of the boxes.
             special_characters=True)
            (graph,) = pydot.graph_from_dot_data(dot_data.getvalue())
            graph.write_png(folder_path + "dns_ddos_tree.png")
            return(dns_ddos_simulated_df, conf_mat)
        
        if exploit == 'amp_ref_dns_ddos':
            
            # add simulations
            num_simulations = 50
            malicious_df = simulate_amp_ref_DNS_DDoS_Attack(df_no_new_features=df_no_new_features, df=df, num_simulations=num_simulations)
            malicious_df = apply_feature_engineering_targeted(malicious_df)
    
            # apply labels
            df['malicious_label'] = 0
            malicious_df['malicious_label'] = 1
    
            # add the malicious values to the already engineered dataframe, and call a new name
            df_inc_simulated = pd.concat([df, malicious_df])
            
            # get the feature dataframe
            amp_ref_dns_ddos_simulated_df = get_amp_ref_DNS_DDoS_Attack_df(df_inc_simulated, add_simulated_labels=True, classify=False)
            # print what proportion of data is simulated
            print('proportion of data simulated:', len(amp_ref_dns_ddos_simulated_df[amp_ref_dns_ddos_simulated_df['malicious_label'] == 1])/len(amp_ref_dns_ddos_simulated_df))
            
            # train test split
            X_train, X_test, y_train, y_test = train_test_split(amp_ref_dns_ddos_simulated_df.drop(['malicious_label'],axis=1),amp_ref_dns_ddos_simulated_df['malicious_label'],test_size=0.2)
            
            # visualise single tree
            amp_ref_dns_ddos_tree_model = tree.DecisionTreeClassifier(criterion = "entropy", max_depth = 10)
            amp_ref_dns_ddos_tree_model.fit(X_train, y_train)
            
            # see  score
            print('tree model score:', amp_ref_dns_ddos_tree_model.score(X_test, y_test))
            
            # save model
            pickle.dump(amp_ref_dns_ddos_tree_model, open(folder_path + 'amp_ref_dns_ddos_classifier', 'wb'))
            
            # load model
            amp_ref_dns_ddos_tree_model = pickle.load(open(folder_path + 'amp_ref_dns_ddos_classifier', 'rb'))
    
            # use on dataframe
            amp_ref_dns_ddos_simulated_df['classification'] = amp_ref_dns_ddos_tree_model.predict(amp_ref_dns_ddos_simulated_df.drop(['malicious_label'], axis=1))
            amp_ref_dns_ddos_simulated_df['incorrect_classification'] = amp_ref_dns_ddos_simulated_df['classification'] != amp_ref_dns_ddos_simulated_df['malicious_label'] 
            amp_ref_dns_ddos_simulated_df['classification'] = amp_ref_dns_ddos_simulated_df['classification'].apply(int)
            conf_mat = confusion_matrix(amp_ref_dns_ddos_simulated_df['malicious_label'], amp_ref_dns_ddos_simulated_df['classification'])
            
            # show graph
            dot_data = StringIO()
            tree.export_graphviz(amp_ref_dns_ddos_tree_model, 
             out_file=dot_data, 
             class_names=['0', '1'], # the target names.
             feature_names=X_train.columns, # the feature names.
             filled=True, # Whether to fill in the boxes with colours.
             rounded=True, # Whether to round the corners of the boxes.
             special_characters=True)
            (graph,) = pydot.graph_from_dot_data(dot_data.getvalue())
            graph.write_png(folder_path + "amp_ref_dns_ddos_tree.png")
            return(amp_ref_dns_ddos_simulated_df, conf_mat)
        
        if exploit == 'heartbeat_fast_flux':
            
            # add simulations
            num_simulations = 500
            malicious_df = simulate_heartbeat_fast_flux(df_no_new_features=df_no_new_features, df=df, num_simulations=num_simulations)
            malicious_df = apply_feature_engineering_targeted(malicious_df)
    
            # need to add parent information level feature
            malicious_df = add_parent_randomness_level_LSTM_feature(malicious_df)
            
            # apply labels
            df['malicious_label'] = 0
            malicious_df['malicious_label'] = 1
            # add the malicious values to the already engineered dataframe, and call a new name
            df_inc_simulated = pd.concat([df, malicious_df])
            
    
            # get the feature dataframe
            heartbeat_fast_flux_simulated_df = get_heartbeat_fast_flux_df(df_inc_simulated, add_simulated_labels=True, classify=False)
            # print what proportion of data is simulated
            print('proportion of data simulated:', len(heartbeat_fast_flux_simulated_df[heartbeat_fast_flux_simulated_df['malicious_label'] == 1])/len(heartbeat_fast_flux_simulated_df))
            
            # train test split
            X_train, X_test, y_train, y_test = train_test_split(heartbeat_fast_flux_simulated_df.drop(['malicious_label'],axis=1),heartbeat_fast_flux_simulated_df['malicious_label'],test_size=0.2)
            
            # visualise single tree
            heartbeat_fast_flux_tree_model = tree.DecisionTreeClassifier(criterion = "entropy", max_depth = 10)
            heartbeat_fast_flux_tree_model.fit(X_train, y_train)
            
            # see  score
            print('tree model score:', heartbeat_fast_flux_tree_model.score(X_test, y_test))
            
            # save model
            pickle.dump(heartbeat_fast_flux_tree_model, open(folder_path + 'heartbeat_fast_flux_classifier', 'wb'))
            
            # load model
            heartbeat_fast_flux_tree_model = pickle.load(open(folder_path + 'heartbeat_fast_flux_classifier', 'rb'))
    
            # use on dataframe
            heartbeat_fast_flux_simulated_df['classification'] = heartbeat_fast_flux_tree_model.predict(heartbeat_fast_flux_simulated_df.drop(['malicious_label'], axis=1))
            heartbeat_fast_flux_simulated_df['incorrect_classification'] = heartbeat_fast_flux_simulated_df['classification'] != heartbeat_fast_flux_simulated_df['malicious_label'] 
            heartbeat_fast_flux_simulated_df['classification'] = heartbeat_fast_flux_simulated_df['classification'].apply(int)
            conf_mat = confusion_matrix(heartbeat_fast_flux_simulated_df['malicious_label'], heartbeat_fast_flux_simulated_df['classification'])
            
            # show graph
            dot_data = StringIO()
            tree.export_graphviz(heartbeat_fast_flux_tree_model, 
             out_file=dot_data, 
             class_names=['0', '1'], # the target names.
             feature_names=X_train.columns, # the feature names.
             filled=True, # Whether to fill in the boxes with colours.
             rounded=True, # Whether to round the corners of the boxes.
             special_characters=True)
            (graph,) = pydot.graph_from_dot_data(dot_data.getvalue())
            graph.write_png(folder_path + "heartbeat_fast_flux_tree.png")
            return(heartbeat_fast_flux_simulated_df, conf_mat)
    
        if exploit == 'tunnelling_fast_flux':
            # add simulations
            num_simulations = 100
            malicious_df = simulate_DNS_tunnelling_fast_flux(df_no_new_features=df_no_new_features, df=df, num_simulations=num_simulations)
            malicious_df = apply_feature_engineering_targeted(malicious_df)
    
            # need to add parent information level feature
            malicious_df = add_parent_randomness_level_LSTM_feature(malicious_df)
            
            # apply labels
            df['malicious_label'] = 0
            malicious_df['malicious_label'] = 1
            # add the malicious values to the already engineered dataframe, and call a new name
            df_inc_simulated = pd.concat([df, malicious_df])
            
            # get the feature dataframe
            tunnelling_fast_flux_simulated_df = get_tunnelling_fast_flux_df(df_inc_simulated, add_simulated_labels=True, classify=False)
            # print what proportion of data is simulated
            print('proportion of data simulated:', len(tunnelling_fast_flux_simulated_df[tunnelling_fast_flux_simulated_df['malicious_label'] == 1])/len(tunnelling_fast_flux_simulated_df))
            
            # train test split
            X_train, X_test, y_train, y_test = train_test_split(tunnelling_fast_flux_simulated_df.drop(['malicious_label'],axis=1),tunnelling_fast_flux_simulated_df['malicious_label'],test_size=0.2)
            
            # visualise single tree
            tunnelling_fast_flux_tree_model = tree.DecisionTreeClassifier(criterion = "entropy", max_depth = 10)
            tunnelling_fast_flux_tree_model.fit(X_train, y_train)
            
            # see  score
            print('tree model score:', tunnelling_fast_flux_tree_model.score(X_test, y_test))
            
            # save model
            pickle.dump(tunnelling_fast_flux_tree_model, open(folder_path + 'tunnelling_fast_flux_classifier', 'wb'))
            
            # load model
            tunnelling_fast_flux_tree_model = pickle.load(open(folder_path + 'tunnelling_fast_flux_classifier', 'rb'))
    
            # use on dataframe
            tunnelling_fast_flux_simulated_df['classification'] = tunnelling_fast_flux_tree_model.predict(tunnelling_fast_flux_simulated_df.drop(['malicious_label'], axis=1))
            tunnelling_fast_flux_simulated_df['incorrect_classification'] = tunnelling_fast_flux_simulated_df['classification'] != tunnelling_fast_flux_simulated_df['malicious_label'] 
            tunnelling_fast_flux_simulated_df['classification'] = tunnelling_fast_flux_simulated_df['classification'].apply(int)
            conf_mat = confusion_matrix(tunnelling_fast_flux_simulated_df['malicious_label'], tunnelling_fast_flux_simulated_df['classification'])
            
            # show graph
            dot_data = StringIO()
            tree.export_graphviz(tunnelling_fast_flux_tree_model, 
             out_file=dot_data, 
             class_names=['0', '1'], # the target names.
             feature_names=X_train.columns, # the feature names.
             filled=True, # Whether to fill in the boxes with colours.
             rounded=True, # Whether to round the corners of the boxes.
             special_characters=True)
            (graph,) = pydot.graph_from_dot_data(dot_data.getvalue())
            graph.write_png(folder_path + "tunnelling_fast_flux_tree.png")
            return(tunnelling_fast_flux_simulated_df, conf_mat)

    
    def get_heartbeat_score(df):
        
        score_1 = 0
        score_2 = 0
        score_3 = 0
        score_4 = 0
        score_5 = 0
        score_6 = 0
        score_7 = 0
        
        if (df['mean_seconds_from_last_exact_query'] <= mean_second_upper) & (df['mean_seconds_from_last_exact_query'] >= mean_second_lower):
            score_1 = mean_second_weight
        if (df['std_of_seconds_from_last_exact_query'] <= std_second_upper):
            score_2 = std_second_weight
        if (df['query_count'] >= query_count_lower):
            score_3 = query_count_weight
        if (df['mean_subquery_information_level'] >= mean_subquery_information_level_lower):
            score_4 = mean_subquery_information_level_weight
        if (df['parent_domain_unique_value_counts'] <= parent_domain_unique_value_counts_upper):
            score_5 = parent_domain_unique_value_counts_weight
        if (df['n_unique_IP_addresses_making_exact_query'] <= unique_ips_making_query_upper):
            score_6 = unique_ips_making_query_weight
        if (df['n_unique_IP_addresses_querying_parent_domain'] <= unique_ips_querying_domain_upper):
            score_7 = unique_ips_querying_domain_weight
            
        return(score_1 + score_2 + score_3 + score_4 + score_5 + score_6 + score_7)    
            
        
        
        
    def get_tunnelling_score(df):
        
        score_1 = 0
        score_2 = 0
        score_3 = 0
    
        if (df['query_count_in_parent_domain'] >= query_count_in_parent_domain_lower):
            score_1 = query_count_in_parent_domain_weight
        if (df['proportion_queries_unique_in_parent_domain'] >= proportion_queries_unique_in_parent_domain_lower):
            score_2 = proportion_queries_unique_in_parent_domain_weight
        if (df['mean_subquery_information_level_in_parent_domain'] >= mean_subquery_information_level_in_parent_domain_lower):
            score_3 = mean_subquery_information_level_in_parent_domain_weight
            
        return(score_1 + score_2 + score_3)
        
        
    def get_DDoS_Attack_score(df):
    
        score_1 = 0
        score_2 = 0
        score_3 = 0
    
        if (df['minute_counts_to_parent_domain'] >= minute_counts_to_parent_domain_lower):
            score_1 = minute_counts_to_parent_domain_weight
        if (df['minute_proportion_of_unique_query_counts'] >= minute_proportion_of_unique_query_counts_lower):
            score_2 = minute_proportion_of_unique_query_counts_weight
        if (df['minute_proportion_nxdomain_or_blank_responses_from_parent_domain'] >= minute_proportion_nxdomain_or_blank_responses_from_parent_domain_lower):
            score_3 = minute_proportion_nxdomain_or_blank_responses_from_parent_domain_weight
    
        return(score_1 + score_2 + score_3)
        
    def get_amp_ref_score(df):
    
        score_1 = 0
        score_2 = 0
        score_3 = 0
        score_4 = 0
    
        if (df['minute_counts_from_unique_ip'] >= minute_counts_from_unique_ip_lower):
            score_1 = minute_counts_from_unique_ip_weight
        if (df['minute_total_proportion_of_unique_query_counts'] >= minute_total_proportion_of_unique_query_counts_lower):
            score_2 = minute_total_proportion_of_unique_query_counts_weight
        if (df['minute_proportion_ANY_or_SOA_queries_from_unique_ip'] >= minute_proportion_ANY_or_SOA_queries_from_unique_ip_lower):
            score_3 = minute_proportion_ANY_or_SOA_queries_from_unique_ip_weight
        if (df['minute_proportion_no_responses_from_unique_ip'] >= minute_proportion_no_responses_from_unique_ip_lower):
            score_4 = minute_proportion_no_responses_from_unique_ip_weight
    
        return(score_1 + score_2 + score_3 + score_4)
        
    def get_heartbeat_fast_flux_score(df):
    
        score_1 = 0
        score_2 = 0
        score_3 = 0
        score_4 = 0
        score_5 = 0
        score_6 = 0
    
        if (df['mean_seconds_from_last_exact_subquery'] <= mean_seconds_from_last_exact_subquery_upper) & (df['mean_seconds_from_last_exact_subquery'] >= mean_seconds_from_last_exact_subquery_lower):
            score_1 = mean_seconds_from_last_exact_subquery_weight
        if (df['std_seconds_from_last_exact_subquery'] <= std_seconds_from_last_exact_subquery_upper):
            score_2 = std_seconds_from_last_exact_subquery_weight
        if (df['subquery_count'] >= subquery_count_lower):
            score_3 = subquery_count_weight
        if (df['subquery_information_level'] >= subquery_information_level_lower):
            score_4 = subquery_information_level_weight
        if (df['mean_parent_randomness_level_for_subquery'] >= mean_parent_randomness_level_for_subquery_lower):
            score_5 = mean_parent_randomness_level_for_subquery_weight
        if (df['n_unique_IP_addresses_making_exact_subquery'] <= n_unique_IP_addresses_making_exact_subquery_upper):
            score_6 = n_unique_IP_addresses_making_exact_subquery_weight
        return(score_1 + score_2 + score_3 + score_4 + score_5 + score_6)


    def get_tunnelling_fast_flux_score(df):
    
        score_1 = 0
        score_2 = 0
        score_3 = 0
        score_4 = 0
    
    
        if (df['query_count_in_parent_domain'] >= query_count_in_parent_domain_lower):
            score_1 = query_count_in_parent_domain_weight
        if (df['proportion_queries_unique_in_parent_domain'] >= proportion_queries_unique_in_parent_domain_lower):
            score_2 = proportion_queries_unique_in_parent_domain_weight
        if (df['mean_subquery_information_level_in_parent_domain'] >= mean_subquery_information_level_in_parent_domain_lower):
            score_3 = mean_subquery_information_level_in_parent_domain_weight
        if (df['parent_randomness_level'] >= parent_randomness_level_lower):
            score_4 = parent_randomness_level_weight
        return(score_1 + score_2 + score_3 + score_4)
# Heartbeat
# -----------------------------------------------------------------------------
    if attack_type=="Heartbeat":
        st.write("A heartbeat is a unique code sent by a compromised client to an attacker's command and control, to identify itself and request instructions.")
        st.write("Exploit detection will be successful if the heartbeat query is identified.")
        st.write("""A heartbeat is identified by:  
                  - A short time between queries  
                  - Low standard deviation of time between identical queries  
                  - At least a certain number of queries having been made  
                  - High subquery information level  
                  - Low number of unique queries in parent domain  
                  - A low number of IP addresses making the query  
                  - A low number of IP addresses querying the parent domain""")
    

        mean_second_upper =  st.text_input('Mean Seconds Between Identical Queries: Upper Threshold', 7200)
        mean_second_lower = st.text_input('Mean Seconds Between Identical Queries: Lower Threshold', 30)
        
        mean_second_weight = st.text_input('Mean Seconds Between Identical Queries: Weight', 0.125)
        
        std_second_upper = st.text_input('Standard Deviation of Seconds Between Identical Queries: Upper Threshold', 30)
        std_second_weight = st.text_input('Standard Deviation of Seconds Between Identical Queries: Weight', 0.25)
        
        query_count_lower = st.text_input('Counts of Exact Query: Lower Threshold', 3)
        query_count_weight = st.text_input('Counts of Exact Query: Weight', 0.125)
        
        mean_subquery_information_level_lower = st.text_input('Encrpytion Level of Subquery (Shannon Entrpy) : Lower Threshold', 3)
        mean_subquery_information_level_weight = st.text_input('Encrpytion Level of Subquery (Shannon Entrpy) : Weight', 0.125)
        
        parent_domain_unique_value_counts_upper = st.text_input('Unique Queries in Parent Domain: Upper Threshold', 10)
        parent_domain_unique_value_counts_weight = st.text_input('Unique Queries in Parent Domain: Weight', 0.125)
        
        unique_ips_making_query_upper = st.text_input('Counts of Unique IPs Making Exact Query: Upper Threshold', 1)
        unique_ips_making_query_weight = st.text_input('Counts of Unique IPs Making Exact Query: Weight', 0.125)
        
        unique_ips_querying_domain_upper = st.text_input('Counts of Unique IPs Querying Parent Domain: Upper Threshold', 3)
        unique_ips_querying_domain_weight = st.text_input('Counts of Unique IPs Querying Parent Domain: Weight', 0.125)
        
        
        mean_second_upper=float(mean_second_upper)
        mean_second_lower=float(mean_second_lower)
        std_second_upper=float(std_second_upper)
        query_count_lower=float(query_count_lower)
        mean_subquery_information_level_lower=float(mean_subquery_information_level_lower)
        parent_domain_unique_value_counts_upper=float(parent_domain_unique_value_counts_upper)
        unique_ips_making_query_upper=float(unique_ips_making_query_upper)
        unique_ips_querying_domain_upper=float(unique_ips_querying_domain_upper)
            
        mean_second_weight=float(mean_second_weight)
        std_second_weight=float(std_second_weight)
        query_count_weight=float(query_count_weight)
        mean_subquery_information_level_weight=float(mean_subquery_information_level_weight)
        parent_domain_unique_value_counts_weight=float(parent_domain_unique_value_counts_weight)
        unique_ips_making_query_weight=float(unique_ips_making_query_weight)
        unique_ips_querying_domain_weight=float(unique_ips_querying_domain_weight)
        
        if round(np.sum([mean_second_weight, std_second_weight, query_count_weight, mean_subquery_information_level_weight, parent_domain_unique_value_counts_weight, unique_ips_making_query_weight, unique_ips_querying_domain_weight]), 3) == 1.000:
            st.success("Feature weights sum to 1")
        else:
            st.warning("Feature weights do not sum to 1")
            
        get_heartbeat_with_scores = st.button("Get Heartbeat Feature Values and Scores", key='get_heartbeat_with_scores')            
        if get_heartbeat_with_scores:
            scored_df = get_heartbeat_df(df, classify=False)
            scored_df['score'] = scored_df.apply(get_heartbeat_score, axis=1)
            scored_df = scored_df.sort_values('score', ascending=False)
            plt.hist(scored_df['score'])
            plt.title("Risk Score Distribution")
            st.pyplot()
            scored_df.to_csv('C:\\Users\\roder\\Projects\\CyberSecurity\\heartbeat_df.csv')
            st.success("anomalies saved as 'heartbeat_df.csv'")
            st.table(scored_df)
        
# Infected Cache
# -----------------------------------------------------------------------------            
    if attack_type=="Infected Cache":
        st.write("An Infected Cache occurs when a DNS resolver has been compromised such that it will provide a malicous IP when a certain parent domain is requested.")
        st.write("Exploit detection will be successful if an IP that points to a malicious domain is identified.")
        st.write("An Infected Cache is identified by a new IP address appearing for a domain name, with what will likely be a large Time to Live, in order to redirect as much traffic as possible to the malicious site.")

        check_for_infected_cache = st.button("Check for Infected Cache", key='check_for_infected_cache')
        if check_for_infected_cache:
            infected_cache_df = get_infected_cache_df(df, date_to)
            st.table(infected_cache_df)
            infected_cache_df.to_csv('C:\\Users\\roder\\Projects\\CyberSecurity\\infected_cache_df.csv')
            st.success("anomalies saved as 'infected_cache_df.csv'")
                
# DNS Tunnelling
# ----------------------------------------------------------------------------
    if attack_type=="DNS Tunnelling":
        
        st.write("DNS Tunnelling occurs when a compromised client sends informaiton to a Command and Control though DNS queries to an attacker-controlled domain.")
        st.write("Exploit detection will be successful if the attacker's domain is identified.")
        st.write("""DNS Tunnelling is identified by:  
                  - At least a certain number of queries to a parent domain  
                  - high proportion of unique queries to parent domain  
                  - high mean subquery information level for the parent domain""")
            
        query_count_in_parent_domain_lower =  st.text_input('Query Count in Parent Domain: Lower Threshold', 7200)
        query_count_in_parent_domain_weight =  st.text_input('Query Count in Parent Domain: Weight', 0.33)
        
        proportion_queries_unique_in_parent_domain_lower = st.text_input('Proportion of Queries Unique in Parent Domain: Lower Threshold', 0.6)
        proportion_queries_unique_in_parent_domain_weight = st.text_input('Proportion of Queries Unique in Parent Domain: Weight', 0.33)
        
        mean_subquery_information_level_in_parent_domain_lower = st.text_input('Mean Subquery Information Level in Parent Domain (Shannon Entropy): Lower Threshold', 3)
        mean_subquery_information_level_in_parent_domain_weight = st.text_input('Mean Subquery Information Level in Parent Domain (Shannon Entropy): Weight', 0.34)
         

        query_count_in_parent_domain_lower=float(query_count_in_parent_domain_lower)
        proportion_queries_unique_in_parent_domain_lower=float(proportion_queries_unique_in_parent_domain_lower)
        mean_subquery_information_level_in_parent_domain_lower=float(mean_subquery_information_level_in_parent_domain_lower)
        
        query_count_in_parent_domain_weight=float(query_count_in_parent_domain_weight)
        proportion_queries_unique_in_parent_domain_weight=float(proportion_queries_unique_in_parent_domain_weight)
        mean_subquery_information_level_in_parent_domain_weight=float(mean_subquery_information_level_in_parent_domain_weight)
        
        if round(np.sum([query_count_in_parent_domain_weight, proportion_queries_unique_in_parent_domain_weight, mean_subquery_information_level_in_parent_domain_weight]), 3) == 1.000:
            st.success("Feature weights sum to 1")
        else:
            st.warning("Feature weights do not sum to 1")
            
        get_tunnelling_with_score = st.button("Get Tunnelling Feature Values and Scores", key='get_tunnelling_with_scores')            
        if get_tunnelling_with_score:
            scored_df = get_tunnelling_df(df, classify=False)
            scored_df['score'] = scored_df.apply(get_tunnelling_score, axis=1)
            scored_df = scored_df.sort_values('score', ascending=False)
            plt.hist(scored_df['score'])
            plt.title("Risk Score Distribution")
            st.pyplot()            
            scored_df.to_csv('C:\\Users\\roder\\Projects\\CyberSecurity\\tunnelling_df.csv')
            st.success("anomalies saved as 'tunnelling_df.csv'")
            st.table(scored_df)
        
# DNS DDoS Attack    
# ----------------------------------------------------------------------------
             
    if attack_type=="DNS DDoS Attack":
        
        st.write("A DNS DDoS Attack occurs when a compromised clients in a botnet recieve instructions to flood a target domain with requests, such that the bandwidth to a target domain's Authoritative Name Server is overwhelmed, and legitimate users will not be able to access the server.")
        st.write("A DNS DDoS Attack would typically be identifed by a increase in a specific type of traffic volume, though in a large botnet, the number of requests that each client would need to make is significantly smaller, and it my evade detection as a result.")
        st.write("An attack will have been successfully identified if the time-interval of the attack and the victim's domain are identified.")
        st.write("""DNS DDoS Attacks are identified by:  
                  - high number of queries to a domain in a time interval  
                  - a high proportion of unique queries to a domain in a time interval  
                  - high number of queries with NXDOMAIN responses or no response to domain in a time interval""")

  
        minute_counts_to_parent_domain_lower =  st.text_input('Minute Query Counts to a Parent Domain: Lower Threshold', 100)
        minute_counts_to_parent_domain_weight =  st.text_input('Minute Query Counts to a Parent Domain: Weight', 0.6)

        minute_proportion_of_unique_query_counts_lower = st.text_input('Proportion of Minute Query Counts to a Parent Domain that are Unique: Lower Threshold', 0.5)
        minute_proportion_of_unique_query_counts_weight = st.text_input('Proportion of Minute Query Counts to a Parent Domain that are Unique: Weight', 0.2)
        
        minute_proportion_nxdomain_or_blank_responses_from_parent_domain_lower = st.text_input('Proportion of Minute Query Counts to a Parent Domain that are NXDOMAIN or Blank Responses: Lower Threshold', 0.5)
        minute_proportion_nxdomain_or_blank_responses_from_parent_domain_weight = st.text_input('Proportion of Minute Query Counts to a Parent Domain that are NXDOMAIN or Blank Responses: Weight', 0.2)
        
        
        minute_counts_to_parent_domain_lower=float(minute_counts_to_parent_domain_lower)
        minute_proportion_of_unique_query_counts_lower =float(minute_proportion_of_unique_query_counts_lower)
        minute_proportion_nxdomain_or_blank_responses_from_parent_domain_lower=float(minute_proportion_nxdomain_or_blank_responses_from_parent_domain_lower)
        
        minute_counts_to_parent_domain_weight = float(minute_counts_to_parent_domain_weight)
        minute_proportion_of_unique_query_counts_weight = float(minute_proportion_of_unique_query_counts_weight)
        minute_proportion_nxdomain_or_blank_responses_from_parent_domain_weight = float(minute_proportion_nxdomain_or_blank_responses_from_parent_domain_weight)
        
        
        if round(np.sum([minute_counts_to_parent_domain_weight, minute_proportion_of_unique_query_counts_weight, minute_proportion_nxdomain_or_blank_responses_from_parent_domain_weight]), 3) == 1.000:
            st.success("Feature weights sum to 1")
        else:
            st.warning("Feature weights do not sum to 1")
        
        
        get_DNS_DDoS_Attack_with_score = st.button("Get DNS DDoS Attack Feature Values and Scores", key='get_DNS_DDoS_Attack_with_score ')            
        if get_DNS_DDoS_Attack_with_score:
            scored_df = get_DNS_DDoS_Attack_df(df, classify=False)
            scored_df['score'] = scored_df.apply(get_DDoS_Attack_score, axis=1)
            scored_df = scored_df.sort_values('score', ascending=False)
            plt.hist(scored_df['score'])
            plt.title("Risk Score Distribution")
            st.pyplot()            
            scored_df.to_csv('C:\\Users\\roder\\Projects\\CyberSecurity\\DNS_DDoS_Attack_df.csv')
            st.success("anomalies saved as 'DNS_DDoS_Attack_df.csv'")
            st.table(scored_df)

# ----------------------------------------------------------------------------
                
    if attack_type=="Amplified Reflection DNS DDoS Attack":
        
        st.write("An Amplified Reflective DDoS Attack occurs when compromised clients in a botnet recieve instructions to request large information (typically an 'ANY' request) from a large domain, and use a spoofed IP address of a client that they wish to target. The large domain's server, or DNS resolver if the information has been cached, will send a large amount of unrequested information to the victim's client or server which will exhaust their bandwidth and restrict legitimate usage.")
        st.write("An attack will have been successfully identified if the time of the attack and the victim's IP address are identified.")
        st.write("""Amplified Reflectiion DNS DDoS Attacks are identified by:  
                  - high number of queries from a certain IP address in a time interval  
                  - a high proportion of unque queries from a certain IP address in a time interval  
                  - A high number of "ANY" or "SOA" queries from a certain IP address in a time interval  
                  - a large number of queries with no response from a certain IP address in a time interval""")

            
        minute_counts_from_unique_ip_lower =  st.text_input('Minute Query Counts from an IP address: Lower Threshold', 100)
        minute_counts_from_unique_ip_weight =  st.text_input('Minute Query Counts from an IP address: Weight', 0.4)
        
        minute_total_proportion_of_unique_query_counts_lower = st.text_input('Proportion of Minute Query Counts from an IP address that are Unique: Lower Threshold', 0.5)
        minute_total_proportion_of_unique_query_counts_weight = st.text_input('Proportion of Minute Query Counts from an IP address that are Unique: Weight', 0.2)        
        
        minute_proportion_ANY_or_SOA_queries_from_unique_ip_lower = st.text_input('Proportion of ANY or SOA Minute Query Counts from an IP address: Lower Threshold', 0.5)
        minute_proportion_ANY_or_SOA_queries_from_unique_ip_weight = st.text_input('Proportion of ANY or SOA Minute Query Counts from an IP address: Weight', 0.2)        
        
        minute_proportion_no_responses_from_unique_ip_lower = st.text_input('Proportion of Minute Query Counts from an IP address that contain no response: Lower Threshold', 0.5)
        minute_proportion_no_responses_from_unique_ip_weight = st.text_input('Proportion of Minute Query Counts from an IP address that contain no response: Weight', 0.2)        
        

        minute_counts_from_unique_ip_lower=float(minute_counts_from_unique_ip_lower)
        minute_total_proportion_of_unique_query_counts_lower =float(minute_total_proportion_of_unique_query_counts_lower)
        minute_proportion_ANY_or_SOA_queries_from_unique_ip_lower=float(minute_proportion_ANY_or_SOA_queries_from_unique_ip_lower)
        minute_proportion_no_responses_from_unique_ip_lower=float(minute_proportion_no_responses_from_unique_ip_lower)
        
        minute_counts_from_unique_ip_weight = float(minute_counts_from_unique_ip_weight)
        minute_total_proportion_of_unique_query_counts_weight = float(minute_total_proportion_of_unique_query_counts_weight)
        minute_proportion_ANY_or_SOA_queries_from_unique_ip_weight = float(minute_proportion_ANY_or_SOA_queries_from_unique_ip_weight)
        minute_proportion_no_responses_from_unique_ip_weight = float(minute_proportion_no_responses_from_unique_ip_weight)
        
        if round(np.sum([minute_counts_from_unique_ip_weight, minute_total_proportion_of_unique_query_counts_weight, minute_proportion_ANY_or_SOA_queries_from_unique_ip_weight, minute_proportion_no_responses_from_unique_ip_weight]), 3) == 1.000:
            st.success("Feature weights sum to 1")
        else:
            st.warning("Feature weights do not sum to 1")
        
        
        get_amp_ref_DNS_DDoS_Attack_with_score = st.button("Get Amplified Reflective DNS DDoS Attack Feature Values and Scores", key='get_amp_ref_DNS_DDoS_Attack_with_score')            
        if get_amp_ref_DNS_DDoS_Attack_with_score:
            scored_df = get_amp_ref_DNS_DDoS_Attack_df(df, classify=False)
            scored_df['score'] = scored_df.apply(get_amp_ref_score, axis=1)
            scored_df = scored_df.sort_values('score', ascending=False)
            plt.hist(scored_df['score'])
            plt.title("Risk Score Distribution")
            st.pyplot()            
            scored_df.to_csv('C:\\Users\\roder\\Projects\\CyberSecurity\\Amp_Ref_DNS_DDoS_Attack_df.csv')
            st.success("anomalies saved as 'Amp_Ref_DNS_DDoS_Attack_df.csv'")
            st.table(scored_df)            
            
# Heartbeat Fast Flux
# -----------------------------------------------------------------------------
    if attack_type=="Heartbeat Fast Flux":
        st.write("A heartbeat is a unique code sent by a compromised client to an attacker's command and control, to identify itself and request instructions. Since malicious parent domains are easily identified, the parent domain can be regularly changed according to an algorithm whose inputs are known by the client and attacker.")
        st.write("The algorithm creates a randomly generated parent domain, and the inputs could be any combination of time related features, like a time of day with a certain company's stock price.")
        st.write("Exploit detection will be successful if the domain-flux heartbeat subquery is identified.")
        st.write("""A domain-flux heartbeat is typically identified by:  
                  - A short time between subqueries  
                  - Low standard deviation of time between subqueries  
                  - Number of subqueries greater than a certain value  
                  - High subquery information level  
                  - High mean randomness level of parent domains that the subquery is made to  
                  - A low number of IP addresses making that exact subquery""")
    
        
            
        mean_seconds_from_last_exact_subquery_upper =  st.text_input('Mean seconds between exact subqueries: Upper Threshold', 7200)
        mean_seconds_from_last_exact_subquery_lower = st.text_input('Mean seconds between exact subqueries: Lower Threshold', 600)
        mean_seconds_from_last_exact_subquery_weight = st.text_input('Mean seconds between exact subqueries: Weight', 0.1)
        
        
        std_seconds_from_last_exact_subquery_upper = st.text_input('Standard deviation of seconds between exact subqueries: Upper Threshold', 0.5)
        std_seconds_from_last_exact_subquery_weight = st.text_input('Standard deviation of seconds between exact subqueries: Weight', 0.1)
        
        subquery_count_lower = st.text_input('Count of exact subquery: Lower Threshold', 0.5)
        subquery_count_weight = st.text_input('Count of exact subquery: Lower Threshold', 0.1)
        
        subquery_information_level_lower = st.text_input('Information level of subquery (Shannon Entropy): Lower Threshold', 3)
        subquery_information_level_weight = st.text_input('Information level of subquery (Shannon Entropy): Weight', 0.2)
        
        mean_parent_randomness_level_for_subquery_lower = st.text_input('Mean Parent Domain Randomness Level for Parent Domains using the exact subquery (difference from verified domain names, scale 0-1): Lower Threshold', 0.5)
        mean_parent_randomness_level_for_subquery_weight = st.text_input('Mean Parent Domain Randomness Level for Parent Domains using the exact subquery (difference from verified domain names, scale 0-1): Weight', 0.4)
        
        n_unique_IP_addresses_making_exact_subquery_upper = st.text_input('Number of unique IP addresses using this exact subquery: Lower Threshold', 0.5)
        n_unique_IP_addresses_making_exact_subquery_weight = st.text_input('Number of unique IP addresses using this exact subquery: Weight', 0.1)


        mean_seconds_from_last_exact_subquery_upper=float(mean_seconds_from_last_exact_subquery_upper)
        mean_seconds_from_last_exact_subquery_lower=float(mean_seconds_from_last_exact_subquery_lower)
        std_seconds_from_last_exact_subquery_upper =float(std_seconds_from_last_exact_subquery_upper)
        subquery_count_lower=float(subquery_count_lower)
        subquery_information_level_lower=float(subquery_information_level_lower)
        mean_parent_randomness_level_for_subquery_lower=float(mean_parent_randomness_level_for_subquery_lower)
        n_unique_IP_addresses_making_exact_subquery_upper=float(n_unique_IP_addresses_making_exact_subquery_upper)
        
        mean_seconds_from_last_exact_subquery_weight = float(mean_seconds_from_last_exact_subquery_weight)
        std_seconds_from_last_exact_subquery_weight = float(std_seconds_from_last_exact_subquery_weight)
        subquery_count_weight = float(subquery_count_weight)
        subquery_information_level_weight = float(subquery_information_level_weight)
        mean_parent_randomness_level_for_subquery_weight = float(mean_parent_randomness_level_for_subquery_weight)
        n_unique_IP_addresses_making_exact_subquery_weight = float(n_unique_IP_addresses_making_exact_subquery_weight)
        
        if round(np.sum([mean_seconds_from_last_exact_subquery_weight, std_seconds_from_last_exact_subquery_weight, subquery_count_weight, subquery_information_level_weight, mean_parent_randomness_level_for_subquery_weight, n_unique_IP_addresses_making_exact_subquery_weight]), 3) == 1.000:
            st.success("Feature weights sum to 1")
        else:
            st.warning("Feature weights do not sum to 1")
            
        get_heartbeat_fast_flux_with_score = st.button("Get Heartbeat Domain Flux Feature Values and Scores", key='get_heartbeat_fast_flux_with_score')            
        if get_heartbeat_fast_flux_with_score:
            scored_df = get_heartbeat_fast_flux_df(df, classify=False)
            scored_df['score'] = scored_df.apply(get_heartbeat_fast_flux_score, axis=1)
            scored_df = scored_df.sort_values('score', ascending=False)
            plt.hist(scored_df['score'])
            plt.title("Risk Score Distribution")
            st.pyplot()            
            scored_df.to_csv('C:\\Users\\roder\\Projects\\CyberSecurity\\Heartbeat_Domain_Flux_df.csv')
            st.success("anomalies saved as 'Heartbeat_Domain_Flux_df.csv'")
            st.table(scored_df)          

# DNS Tunnelling Fast Flux
# -----------------------------------------------------------------------------
    if attack_type=="DNS Tunnelling Fast Flux":
        
        st.write("DNS Tunnelling occurs when a compromised client sends informaiton to a Command and Control though DNS queries to an attacker-controlled domain. Since malicious parent domains are easily identified, the parent domain can be regularly changed according to an algorithm whose inputs are known by the client and attacker.")
        st.write("The algorithm creates a randomly generated parent domain, and the inputs could be any combination of time related features, like a time of day with a certain company's stock price.")
        st.write("Exploit detection will be successful if the attacker domain(s) are identified.")
        st.write("""DNS Tunnelling using domain-flux is typically identified by:  
                  - Count of queries to the parent domain above a certain value  
                  - High proportion of unique queries for parent domain  
                  - High mean subquery information level for queries to the parent domain  
                  - High randomness level of parent domain""")

            
            
            
        query_count_in_parent_domain_lower =  st.text_input('Count of queries in parent domain: Lower Threshold', 20)
        query_count_in_parent_domain_weight =  st.text_input('Count of queries in parent domain: Weight', 0.2)
        
        proportion_queries_unique_in_parent_domain_lower = st.text_input('Proportion of queries in parent domain that are unique: Lower Threshold', 0.5)
        proportion_queries_unique_in_parent_domain_weight = st.text_input('Proportion of queries in parent domain that are unique: Weight', 0.2)
        
        mean_subquery_information_level_in_parent_domain_lower = st.text_input('Mean subquery information level for queries in parent domain: Lower Threshold', 3)
        mean_subquery_information_level_in_parent_domain_weight = st.text_input('Mean subquery information level for queries in parent domain: Weight', 0.2)        
        
        parent_randomness_level_lower = st.text_input('Parent Domain Randomness Level (difference from verified domain names, scale 0-1): Lower Threshold', 0.5)
        parent_randomness_level_weight = st.text_input('Parent Domain Randomness Level (difference from verified domain names, scale 0-1): Weight', 0.4)

        query_count_in_parent_domain_lower=float(query_count_in_parent_domain_lower)
        proportion_queries_unique_in_parent_domain_lower=float(proportion_queries_unique_in_parent_domain_lower)
        mean_subquery_information_level_in_parent_domain_lower =float(mean_subquery_information_level_in_parent_domain_lower)
        parent_randomness_level_lower =float(parent_randomness_level_lower)

        query_count_in_parent_domain_weight = float(query_count_in_parent_domain_weight)
        proportion_queries_unique_in_parent_domain_weight = float(proportion_queries_unique_in_parent_domain_weight)
        mean_subquery_information_level_in_parent_domain_weight = float(mean_subquery_information_level_in_parent_domain_weight)
        parent_randomness_level_weight = float(parent_randomness_level_weight)

        get_tunnelling_fast_flux_with_score = st.button("Get Tunnelling Domain Flux Feature Values and Scores", key='get_tunnelling_fast_flux_with_score')            
        if get_tunnelling_fast_flux_with_score:
            scored_df = get_tunnelling_fast_flux_df(df, classify=False)
            scored_df['score'] = scored_df.apply(get_tunnelling_fast_flux_score, axis=1)
            scored_df = scored_df.sort_values('score', ascending=False)
            plt.hist(scored_df['score'])
            plt.title("Risk Score Distribution")
            st.pyplot()            
            scored_df.to_csv('C:\\Users\\roder\\Projects\\CyberSecurity\\Tunnelling_Domain_Flux_df.csv')
            st.success("anomalies saved as 'Tunnelling_Domain_Flux_df.csv'")
            st.table(scored_df)      
# Targeted
# ----------------------------------------------------------------------------
# ****************************************************************************
# ----------------------------------------------------------------------------
if tabs == "Exploit Detection - Manage Infected Cache IPs":
    # not yet  complete
    # tabs to amend IPs in the bad IP list
    st.write('## Exploit Detection - Manage Infected Cache IPs')
    amend_bad_IPs = st.sidebar.radio(
            "Amend IPs in bad IP list?:",
            ("No", "Add IPs to list (import dataframe of IP addresses 'IPs_to_add_to_bad_IP_list.xlsx')", "remove IPs from bad IP list (import dataframe of IP addresses 'IPs_to_remove_from_bad_IP_list.xlsx'", "See list of bad IPs", "Clear list of bad IPs"),
            key='amend_bad_IPs'
            )
            
    if amend_bad_IPs == "Add IPs to list (import dataframe of IP addresses 'IPs_to_add_to_bad_IP_list.xlsx')":
        new_bad_IPs_df = pd.read_excel("C:\\Users\\roder\\Projects\\CyberSecurity\\IPs_to_add_to_bad_IP_list.xlsx")
        new_bad_IPs_df.set_index('IP', inplace=True)
        add_ips_to_bad_IP_list_from_dataframe(new_bad_IPs_df)
        st.success('IPs added to disregard list')
        st.code("Number of unique bad IPs:" + str(len(get_bad_ips())))
    if amend_bad_IPs == "remove IPs from bad IP list (import dataframe of IP addresses 'IPs_to_remove_from_bad_IP_list.xlsx'":
        ips_to_remove_df = pd.read_excel("C:\\Users\\roder\\Projects\\CyberSecurity\\IPs_to_remove_from_bad_IP_list.xlsx")
        ips_to_remove_df.set_index('IP', inplace=True)    
        remove_ips_from_bad_IPs_list_from_dataframe(ips_to_remove_df)
        st.success('IPs removed from bad IP list')
        st.code("Number of unique IPs in bad IP list:" + str(len(get_bad_ips())))
    if amend_bad_IPs == "See list of bad IPs":
        st.code("Number of unique IPs in bad IP list::" + str(len(get_bad_ips())))
        st.table(get_bad_ips())
    if amend_bad_IPs == "Clear list of bad IPs":
        remove_ip_from_bad_ips(remove_all=True)
        st.success('bad IP list has been cleared')
        st.code("Number of unique IPs in bad IP list:" + str(len(get_bad_ips()))) 
        
        

        
