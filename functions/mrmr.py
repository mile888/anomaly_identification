#!/usr/bin/env python

### Minimum Redundancy Maximum Relevance for classification ###


import tqdm
from joblib import Parallel, delayed
from multiprocessing import cpu_count

import pandas as pd
import numpy as np

from sklearn.feature_selection import f_classif, mutual_info_classif

import warnings 
warnings.filterwarnings("ignore")

FLOOR = 1e-5



### Functions for parallelization ###
def parallel_df(func, df, series):
    n_jobs = min(cpu_count(), len(df.columns))
    col_chunks = np.array_split(range(len(df.columns)), n_jobs)
    lst = Parallel(n_jobs = n_jobs)(
        delayed (func) (df.iloc[:, col_chunk], series) 
        for col_chunk in col_chunks
    )
    return pd.concat(lst)



### Functions for computing relevance and redundancy ###
# relevance
def _anova_classif(X, y):
    def _anova_classif_series(x, y):
        x_not_na = ~ x.isna()
        if x_not_na.sum() == 0:
            return 0
        return f_classif(x[x_not_na].to_frame(), y[x_not_na])[0][0]
    return X.apply(lambda col: _anova_classif_series(col, y)).fillna(0.0)

def _mi_classif(X, y):
    def _mi_classif_series(x, y):
        x_not_na = ~ x.isna()
        if x_not_na.sum() == 0:
            return 0
        return mutual_info_classif(x[x_not_na].to_frame(), y[x_not_na])[0]
    return X.apply(lambda col: _mi_classif_series(col, y)).fillna(0.0)

# redundancy
def _corr_pearson(A, b):
    return A.corrwith(b, method='pearson').fillna(0.0).abs().clip(FLOOR)

def _corr_spearman(A, b):
    return A.corrwith(b, method='spearman').fillna(0.0).abs().clip(FLOOR)



### Functions for computing relevance and redundancy - Parallelized versions ### 
# relevance
def anova(X, y):
    """ ANOVA F-statistic - parallelized version """
    return parallel_df(_anova_classif, X, y)

def mi(X, y):
    """ Mutual Information - parallelized versions """
    return parallel_df(_mi_classif, X, y)

# redundancy
def corr_pearson(A, b):
    """ Pearson correlation - parallelized versions """
    return parallel_df(_corr_pearson, A, b)

def corr_spearman(A, b):
    """ Spearman correlation - parallelized versions """
    return parallel_df(_corr_spearman, A, b)




# MRMR selection
def mrmr(X, y, K, func_relevance, func_redundancy, func_denominator):
    '''
    Do MRMR selection.
    
    Args:
        X:                 (pandas.DataFrame) A Dataframe consisting of numeric features only.
        y:                 (pandas.Series) A Series containing the target variable.
        K:                 (int) Number of features to select.
        func_relevance:    (func) Relevance function.
        func_redundancy:   (func) Redundancy function.
        func_denominator:  (func) Synthesis function to apply to the denominator.
    Returns:
        (list) List of K names of selected features (sorted by importance).
    '''
        
    # compute relevance
    rel = func_relevance(X, y)
            
    # keep only columns that have positive relevance
    columns = rel[rel.fillna(0) > 0].index.to_list()
    K = min(K, len(columns))
    rel = rel.loc[columns]
    
    # init
    red = pd.DataFrame(FLOOR, index = columns, columns = columns)
    selected = []
    not_selected = columns.copy()
    
    for i in tqdm.tqdm(range(K)):
        
        # compute score numerator
        score_numerator = rel.loc[not_selected]

        # compute score denominator
        if i > 0:

            last_selected = selected[-1]
            not_selected_subset = [c for c in not_selected if c.split('_')[0] == last_selected.split('_')[0]]
                                
            if not_selected_subset:
                red.loc[not_selected_subset, last_selected] = func_redundancy(X[not_selected_subset], X[last_selected]).abs().clip(FLOOR).fillna(FLOOR)
                score_denominator = red.loc[not_selected, selected].apply(func_denominator, axis = 1).round(5).replace(1.0, float('Inf'))
                
        else:
            score_denominator = pd.Series(1, index = columns)
            
        # compute score and select best
        score = score_numerator / score_denominator
        best = score.index[score.argmax()]
        selected.append(best)
        not_selected.remove(best)
        
    return selected
    
    
    
def mrmr_classif(X, y, K, relevance = 'mi', redundancy = 'spearman', denominator = 'mean'):
    '''
    Do MRMR feature selection on classification task.
    
    Args:
        X:                 (pandas.DataFrame) A Dataframe consisting of numeric features only.
        y:                 (pandas.Series) A Series containing the (categorical) target variable.
        K:                 (int) Number of features to select.
        relevance:         (str) Relevance method. 
                           string - name of supported methods: 'anova' (f-statistic), 'mi' (mutual information).
        redundancy:        (str) Redundancy method. 
                           string - name of supported methods: 'pearson' (Pearson correlation), 'spearman' (Spearman correlation)
        denominator:       (str) Synthesis function to apply to the denominator of MRMR score.
                           string - name of supported methods: 'max', 'mean'
    Returns:
        (list) List of K names of selected features (sorted by importance).
    '''
    
    FUNCS = {
        'anova': anova,
        'mi': mi,
        'pearson': corr_pearson,
        'spearman': corr_spearman,
        'mean': np.mean,
        'max': np.max
    }
    
    func_relevance = FUNCS[relevance] if relevance in FUNCS.keys() else relevance
    func_redundancy = FUNCS[redundancy] if redundancy in FUNCS.keys() else redundancy
    func_denominator = FUNCS[denominator] if denominator in FUNCS.keys() else denominator
    
    return mrmr(X, y, K, func_relevance, func_redundancy, func_denominator) 
        





