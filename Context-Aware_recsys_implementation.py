import os, warnings, folium
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from geopy.distance import great_circle
from shapely.geometry import MultiPoint
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances, mean_squared_error
from sklearn.preprocessing import minmax_scale, MultiLabelBinarizer
from sklearn.decomposition import NMF
from random import randint
from download_csv_file import create_download_link
from ml_metrics import mapk, apk


# Find ratings
train_rating = X_train.groupby(['location_id','user_id'])['visit_time'].count().reset_index(name='rating')
train_rating.head(10)

def normalize(df):
    # Normalize number of visit into a range of 1 to 5
    df['rating'] = minmax_scale(df.rating, feature_range=[1,5])
    return df

r_df = normalize(train_rating)

# Create a rating matrix
r_df = train_rating.pivot_table(
    index='user_id', 
    columns='location_id', 
    values='rating', 
    fill_value=0
)
    
# Calculate the sparcity percentage of matrix
def calSparcity(m):
    m = m.fillna(0)
    non_zeros = np.count_nonzero(m)/np.prod(m.shape) * 100
    sparcity = 100 - non_zeros
    print(f'The sparcity percentage of matrix is %{round(sparcity,2)}')

display(r_df.head())
calSparcity(r_df)

# Create user-user similarity matrix
def improved_asym_cosine(m, mf=False,**kwarg):
    # Cosine similarity matrix distance
    cosine = cosine_similarity(m)

    # Asymmetric coefficient
    def asymCo(X,Y):
        co_rated_item = np.intersect1d(np.nonzero(X),np.nonzero(Y)).size
        coeff = co_rated_item / np.count_nonzero(X)
        return coeff
    asym_ind = pairwise_distances(m, metric=asymCo)

    # Sorensen similarity matrix distance
    sorensen = 1 - pairwise_distances(np.array(m, dtype=bool), metric='dice')

    # User influence coefficient
    def usrInfCo(m):
        binary = m.transform(lambda x: x >= x[x!=0].mean(), axis=1)*1
        res = pairwise_distances(binary, metric=lambda x,y: (x*y).sum()/y.sum() if y.sum()!=0 else 0)
        return res       
    usr_inf_ind = usrInfCo(m)

    similarity_matrix = np.multiply(np.multiply(cosine,asym_ind),np.multiply(sorensen,usr_inf_ind))

    usim = pd.DataFrame(similarity_matrix, m.index, m.index)
    
    # Check if matrix factorization was True
    if mf:
        # Binary similarity matrix
        binary = np.invert(usim.values.astype(bool))*1
        model = NMF(**kwarg)
        W = model.fit_transform(usim)
        H = model.components_
        factorized_usim = np.dot(W,H)*binary + usim
        usim = pd.DataFrame(factorized_usim, m.index, m.index)
                
    return usim

s_df = improved_asym_cosine(r_df)
display(s_df.head())
calSparcity(s_df)


# Find probability of contexts
contexts = X_train.filter(['season','daytime','weather']).apply(lambda x: (x.season,x.daytime,x.weather), axis=1).reset_index(name='context')
IF = contexts.groupby(['location_id','context'])['context'].count()/contexts.groupby(['context'])['context'].count()
IDF = np.log10(contexts.groupby(['location_id','user_id'])['user_id'].count().sum()/contexts.groupby(['location_id'])['user_id'].count())
contexts_weight = (IF * IDF).to_frame().rename(columns={0: 'weight'})

# Create a context-location matrix
lc_df = contexts_weight.pivot_table(
    index='context', 
    columns='location_id', 
    values='weight',
    fill_value=0
)


display(lc_df.head())
calSparcity(lc_df)

cs_df = pd.DataFrame(cosine_similarity(lc_df), index=lc_df.index, columns=lc_df.index)
display(cs_df.head())
calSparcity(cs_df)


"Final Recommendation"

def CF(user_id, location_id, s_matrix):
    r = np.array(r_df)
    s = np.array(s_matrix)
    users = r_df.index
    locations = r_df.columns
    l = np.where(locations==location_id)[0]
    u_idx = np.where(users==user_id)[0]
        
    # Means of all users
    means = np.array([np.mean(row[row!=0]) for row in r])
    
    # Check if l is in r_rating
    if location_id in r_df:
        # Find similar users rated the location that target user hasn't visited
        idx = np.nonzero(r[:,l])[0]
        sim_scores = s[u_idx,idx].flatten()
        sim_users = zip(idx,sim_scores)
    
        # Check if there is any similar user to target user
        if idx.any():
            sim_ratings = r[idx,l]
            sim_means = means[idx]
            numerator = (sim_scores * (sim_ratings - sim_means)).sum()
            denominator = np.absolute(sim_scores).sum()
            weight = (numerator/denominator) if denominator!=0 else 0
            wmean = means[u_idx] + weight
            wmean_rating = wmean[0]
            
    else:
        wmean_rating = 0

    return wmean_rating


# Collaborative filtering with post-filtered contexts
def CaCF_Post(user_id, location_id, s_matrix, c_current, delta):
    
    # Calculate cf
    initial_pred = CF(user_id, location_id, s_matrix)
    
    if location_id in r_df:
        r = np.array(r_df)
        users = r_df.index
        locations = r_df.columns
        l = np.where(locations==location_id)[0]
        c_profile = contexts
        all_cnx = contexts.context.unique().tolist()
        c = np.array(c_profile)
        u_idx = np.where(users==user_id)[0]
        c_current = tuple(c_current)
        
        # Get contexts of similar users visited the location
        l_cnx = np.array(c_profile.loc[c_profile.location_id==location_id,['user_id','context']])
                
        if c_current in all_cnx:
            # Find similarity of the current context to location contexts
            cnx_scores = np.array([[uid, cs_df[c_current][cx]] for uid,cx in l_cnx])

            # Filter users whose similarity bigger than delta
            filtered_scores = cnx_scores[cnx_scores[:,1].astype(float)>delta]

            # Location popularity based on current context
            visit_prob = len(filtered_scores) / len(cnx_scores)
            
        else:
            visit_prob = 1

        return initial_pred * visit_prob

    else:
        return initial_pred
    

# Find ratings
test_rating = X_test.groupby(['location_id','user_id'])['visit_time'].count().reset_index(name='rating')
test_rating = normalize(test_rating)
r_df_test = test_rating.pivot_table(index='user_id', columns='location_id', values='rating', fill_value=0)

# Proposed approach
def EACOS_CaCF_Post(user_id, location_id, c_current, delta):
    res = CaCF_Post(user_id, location_id, s_df, c_current, delta)
    return res

# Recommendation
def predict(target_user, model, option=None):
    true = r_df_test.loc[target_user]
    
    # Check if model is context-aware 
    if option:
        pred_val = []
        for l in true.index:
            delta = option.get('delta')
            c_current = tuple(X_test.xs(target_user)[['season','daytime','weather']].head(1).values[0])
            r = model(user_id=target_user, location_id=l, c_current=c_current, delta=delta)
            pred_val.append(r)
    else:
        pred_val = [model(user_id=target_user, location_id=l) for l in true.index]

    pred = pd.Series(pred_val, index=true.index)

    return pred



