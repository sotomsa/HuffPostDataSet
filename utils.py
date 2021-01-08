import os
import numpy as np
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
from joblib import load, dump

import nltk
#nltk.download('punkt') # download punkt in order to use word tokenizer
#nltk.download('stopwords') # download stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

from sklearn.metrics import classification_report

def tokenization_stopwords_stemming(df, col_name='headline', tok_col_name='tok'):
    """
    Function that, given a pandas dataframe, a text column name and token column name,
    applies tokenization, stopwords(english) and stemming(english).
    Returns dataframe with column tok_col_name added
    """
    # Tokenization
    df[tok_col_name] = df[col_name].apply(word_tokenize)
    df_tok = df.explode(tok_col_name)
    
    # Convert tokens to lower case
    df_tok.loc[:,[tok_col_name]] = df_tok.apply(lambda x: str(x[tok_col_name]).lower(), axis = 1)
    
    # Get stopwords
    stop_words = [x for x in nltk.corpus.stopwords.words('english')]
    # Add some custom punctation (some punctation could be usefull)
    stop_words.extend(['-','(',')','.',':',',',"'","'s",'?',"n't","’","_"])
    
    # Clean stopwords and some punctuation
    df_tok.loc[:,['in_stopwords']] = df_tok.apply(lambda x: x[tok_col_name] in stop_words, axis = 1)
    df_tok_clean = df_tok[~df_tok['in_stopwords']]
    
    # Apply Snowball stemmer
    stemmer = nltk.stem.SnowballStemmer('english')
    df_tok_clean.loc[:,[tok_col_name]] = df_tok_clean[tok_col_name].astype(str)
    df_tok_clean.loc[:,[tok_col_name]] = df_tok_clean.apply(lambda x: stemmer.stem(x[tok_col_name]), axis = 1)
    
    return(df_tok_clean)

def to_liquid_text(df_tok_clean, class_col_name='category', tok_col_name='tok'):
    """
    Function that, given a pandas dataframe, a class_col_name and tok_col_name,
    joins the column tok_col_name with a space for each "id" in order to form a liquid text column.
    Returns dataframe with column 'text' added
    """
    return(df_tok_clean
               .reset_index()
               .filter(['id',class_col_name,tok_col_name], axis=1)
               .groupby(['id',class_col_name])[tok_col_name]
               .apply(' '.join)
               .reset_index()
               .rename(columns = {tok_col_name:'text'}))

def ETL(df, col_name= 'headline', class_col_name='category', tok_col_name='tok'):
    """
    Function that combines all steps for ETL
    """
    # Primer paso: Tokenizacion, stopwords y stemming
    df_tok_clean = tokenization_stopwords_stemming(df, col_name=col_name, tok_col_name=tok_col_name)
    
    # Segundo paso: Crear texto liquido del dataframe limpio
    df_liquid_text = to_liquid_text(df_tok_clean, class_col_name=class_col_name, tok_col_name=tok_col_name)
    
    return df_tok_clean,df_liquid_text

def plot_freq_x_context(df_tok_clean, class_col_name='category', tok_col_name='tok', n=5):
    """
    Function that, given a pandas dataframe, a text column name and token column name,
    plots the n largest frequency tokens by class_col_name.
    Returns altair chart
    """
    # Group tokens by class_col_name and tokname
    grouped_toks = (df_tok_clean
                        .reset_index()
                        .filter(['id',class_col_name,tok_col_name], axis=1)
                        .groupby([class_col_name,tok_col_name])
                        .agg({'id':['count']})
                        .sort_values([class_col_name,('id', 'count')], ascending = False)
    )
    grouped_toks.columns = grouped_toks.columns.map('_'.join)
    grouped_toks = grouped_toks.rename(columns = {'id_count':'count'}).reset_index()
    
    # Return altair facet graph
    return(alt.Chart(grouped_toks
                  .groupby(class_col_name)
                  .head(n)
    ).mark_bar().encode(
        x='count:Q',
        y=alt.Y(tok_col_name + ':N', sort='-x'),
        tooltip=[tok_col_name,'count']
    ).properties(
        width=120,
        height=50
    ).facet(
        facet= class_col_name + ':N',
        columns=4
    ).resolve_scale(
      x='independent',
      y='independent'
    ))

def GridSearchResultToDF(search):
    """
    Function that extracts the results from GridSearchCV and
    converts them to a pandas dataframe
    """
    return(pd.concat([pd.DataFrame(data=search.cv_results_['params']),
                      pd.DataFrame(data={'mean': search.cv_results_['mean_test_score'],
                                           'std': search.cv_results_['std_test_score']}),
                      pd.DataFrame(data={'mean_fit_time': search.cv_results_['mean_fit_time']})],
                     axis = 1))

def PlotComparison(result_values, descrete, continuous, jitter=100):
    """
    Function that takes the result from GridSearchResultToDF function and
    plots some of their values
    """
    df = result_values.copy()
    np.random.seed(0)
    df[continuous] = df[continuous] + np.random.randint(low=-jitter, high=jitter, size=len(df))
    base = alt.Chart(df).transform_calculate(
        ymin="datum.mean-2*datum.std",
        ymax="datum.mean+2*datum.std",
    ).properties(
        title = 'Accuracy by Params'
    )
    
    points = base.mark_point(
        filled=True,
        size=10
    ).encode(
        x=continuous,
        y=alt.Y('mean:Q'),#, scale=alt.Scale(domain=(0.55, 0.7))),
        color=descrete,
        tooltip=['mean','std']
    )

    errorbars = base.mark_errorbar().encode(
        x=continuous,
        y=alt.Y("ymin:Q",title='Accuracy'),
        y2="ymax:Q",
        color=descrete,
    )

    return(points + errorbars)

def get_model_filename(model_name, folder_path='Data', model_name_prefix='model_'):
    return(os.path.join(folder_path,f"{model_name_prefix}{model_name}.joblib"))

def save_model(model_dict):
    filename = get_model_filename(model_dict['model_name'])
    dump(model_dict,filename)

def create_model_dict(model_name, model_description, search, test, predicted_test, rewrite=False):
    """
    Function that creates and saves all information related to a model run
    """
    filename = get_model_filename(model_name)
    if not os.path.exists(filename) or rewrite:
        test_copy = test.copy()
        test_copy['pred'] = predicted_test
        model_dict = {
            'model_name': model_name,
            'model_description': model_description,
            'model_CV': search,
            'model_results': test_copy
        }
        save_model(model_dict)
        
def consolidate_results(path='./Data'):
    """
    Function that loads model result files from given path and
    creates a Pandas dataframe with relevant information
    """
    model_files = [load(os.path.join(path, f)) 
                   for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.startswith('model_')]
    df_final = pd.DataFrame(columns=['model_name','train_accuracy','test_accuracy',
                                     'macro_avg_precision','macro_avg_recall',
                                     'macro_avg_f1-score','weighted_avg_precision',
                                     'weighted_avg_recall','weighted_avg_f1-score'])
    for model_file in model_files:
        results = model_file['model_results']
        class_report = classification_report(results.category, results.pred, output_dict=True)
        df_final = df_final.append({'model_name':model_file['model_name'],
                                    'train_accuracy':'{0:.2f}'.format(model_file['model_CV'].best_score_),
                                    'test_accuracy':'{0:.2f}'.format(class_report['accuracy']),
                                    'macro_avg_precision':class_report['macro avg']['precision'],
                                    'macro_avg_recall':class_report['macro avg']['recall'],
                                    'macro_avg_f1-score':class_report['macro avg']['f1-score'],
                                    'weighted_avg_precision':class_report['weighted avg']['precision'],
                                    'weighted_avg_recall':class_report['weighted avg']['recall'],
                                    'weighted_avg_f1-score':class_report['weighted avg']['f1-score']
                                   },ignore_index=True)
    return(df_final)

class GridSearchSimulation(object):
    # Class to include parameter best_score_
    # so that it does not break the code.
    def __init__(self,train_score):
        self.best_score_ = train_score