#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 10:35:09 2020

@author: alfredogarcia
"""
import os
import re
import pickle
#from pathlib import Path
import datetime
import numpy as np
import pandas as pd
from pandas import ExcelWriter
import calendar
import gensim
from gensim import corpora, models
from gensim.corpora import Dictionary
from gensim.parsing.preprocessing import preprocess_string
from gensim.parsing.preprocessing import STOPWORDS
import matplotlib.pyplot as plt
#from matplotlib.backends.backend_pdf import PdfPages
#from matplotlib.dates import MonthLocator, WeekdayLocator, DateFormatter
import scipy.stats as stats
from scipy.stats import normaltest
from scipy.stats import shapiro
from scipy.stats import anderson
import general_config as gconfig
import streamlit as st

def uncategorize(df):
    '''uncategorize columns'''
    for y in df.columns:
        #categorize columns
        if df[y].dtype.name == 'category':
            df[y] = df[y].astype(str)
    return df

def clean_normalize_fields(df, timedelta=0, sla_map=gconfig.SLAS_DEFAULT):
    '''clean name columns, categorize columns, downcast columns
    generate x_ columns for standarize columns
    drop rows with duplicate number columns, 
    caclulate cycletime in days closed - opened
    split day in bins,8-17, 18-2 2-7
    timedelta adjust time , 0= est timeframe'''

    df= categorize_columns(df)
    #df, x_desc, x_shortdesc, records= setup_descriptioncolumns(df)
    #For tickets file
    print(df.columns)
    df.drop_duplicates(['x_number'], inplace= True)
    #add fields with correct hrs in EST
    df['x_opened_at'] = df['x_opened_at'] -  pd.to_timedelta(timedelta, unit='h')
    df['x_closed_at'] = df['x_closed_at'] -  pd.to_timedelta(timedelta, unit='h')
    #add weekdate base on x_opened_at
    df['x_weekdate'] = df['x_opened_at'] - pd.to_timedelta(df['x_opened_at'].dt.dayofweek, unit='d')
    df['x_weekdate'] = df['x_weekdate'].dt.date
    df['x_yearmonth']= df['x_opened_at'].dt.strftime('%Y%m')
    df['x_dayweek']= df['x_opened_at'].dt.strftime('%w')
    date = pd.to_datetime(df['x_opened_at'])
    df['x_hourday']= date.dt.hour
    df['x_year']= df['x_opened_at'].dt.strftime('%Y')
    #add fields based on x_closed_at
    df['x_weekdate_closed'] = df['x_closed_at'] - pd.to_timedelta(df['x_closed_at'].dt.dayofweek, unit='d')
    df['x_weekdate_closed'] = df['x_weekdate_closed'].dt.date
    df['x_yearmonth_closed']= df['x_closed_at'].dt.strftime('%Y%m')
    df['x_dayweek_closed']= df['x_closed_at'].dt.strftime('%w')
    date = pd.to_datetime(df['x_closed_at'])
    df['x_hourday_closed']= date.dt.hour
    df['x_year_closed']= df['x_closed_at'].dt.strftime('%Y')
    #set bins for the day
    '''adjust hourday to EST timezone and separate in 3 bins
    0,1  ,2,3,4,5,6,7,  8,9,10,11,12,13,14,15,16,17, 18,19,20,21,22,23,0, 1,   
         '2-8'          '8-17'                       '18-2'           '''
    df['x_bins_day']= pd.cut(df['x_hourday'], bins=[-float("inf"),1,7,18], 
                                   labels=['18-2','2-8','8-17'])
    df['x_bins_day']= df['x_bins_day'].fillna('18-2')

    #cycletime in days
    df['x_cycletime'] = df['x_closed_at'] - df['x_opened_at']
    df['x_cycletime'] = df['x_cycletime']/np.timedelta64(1,'D')
    df['x_cycletime'] = df['x_cycletime'].fillna(value=0, downcast='infer')

    #add calculate SLA , due_date and In_SLA
    days_sla= sla_map
    #print(days_sla)
    df['x_priority'] = np.where(df['x_priority'] == '3 - Low', '4 - Low', df['x_priority'])
    df['x_days_sla']= df['x_priority'].map(days_sla).fillna(5)
    df['x_days_sla'].dtype == np.float64
    df['x_aging'] = np.where(df['x_state'].isin(gconfig.OPEN_CASES),
                                      1,0) 
    #consider weekends in the duedate and verify if it is in SLA or OUTSLA
    df['x_due_date2'] = df.apply(lambda row: add_business_days(row['x_opened_at'],
                                                                      row['x_days_sla']), axis=1)
    df['x_inSLA2'] = np.where(df['x_closed_at'] > df['x_due_date2'],
                              0, np.where(df['x_closed_at'].isnull(), 0, 1))
    
    np_concat = np.vectorize(concat)
    df['x_description'] = np_concat(df['x_description'], df['x_shortdesc'])
    print('Done clean_normalize_fields')
    return df

def outliers_iqr(df, column, output= 'x_outlier', remove=True):
    '''Interquartile range method to detect outliers
    return a df with column for outlier default name x_outlier
    if remove is false only mark outliers but keep them
    '''
    quartile_1, quartile_3 = np.percentile(df[column], [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    old_column = column + '_old'
    df[output] = np.where((df[column] > upper_bound) | (df[column] < lower_bound),1,0)
    if remove:
        df_filtered = df[df[output]==0]
        df_outliers = df[df[output]==1]
        df[old_column] = df[column]
        df[column] = np.where((df[column] > upper_bound) | (df[column] < lower_bound),np.nan,df[column])
    else:
        df[old_column] = df[column]
        df_filtered = df.copy()
        df_outliers = pd.DataFrame()
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    red_square = dict(markerfacecolor='r', marker='s')
    ax1.boxplot(df[old_column], vert=False, flierprops=red_square)
    ax1.set_title('{} Before'.format(column))
    ax2.boxplot(df_filtered[column], vert=False, flierprops=red_square)
    ax2.set_title('{} After {} outliers removed'.format(column, len(df_outliers)))
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.6, wspace=0.35)
    #plt.show()
    return df, fig, df_outliers

def cleanlines2(text):
    '''Clean text removing urls, punctation, numbers, whitespace and convert
    to lowecase'''
    text1 = str(text).lower()

    lines = []   #split in lines
    for line in text1.split('\n'):
        line = str(line)
        line = line.strip('\n')
        if line:
            lines.append(line)
    cleantext = ''
    for line in lines:
        filterreg = gconfig.LABELREGEX.search(line)
        if filterreg is None:
            cleantext = cleantext + line #+ '\n'
        else:
            if filterreg.group():
                pass
            else:
                cleantext = cleantext + line #+ '\n'
    cleantext = str(cleantext)
    text1 = re.sub('\\S*@\\S*\\s?', '', cleantext)  # Remove Emails
    text1 = re.sub("\'", "", text1)                 #remove single quotes
    text1 = re.sub('\\s+', ' ', text1)              #remove new line character
    text1 = re.sub(r'http\S+', '', text1)           #remove URLs
    text1 = tokenize(str(text1))
    text1 = str(text1)
    #using gensim to remove numbers, punctuation, whitespace, stopwords,
    #non-alfa, convert lowercase and stem
    text1 = ' '.join(preprocess_string(str(text1)))
    return text1

def tokenize(text, biagram = True):
    '''Lematization and tokenization of text'''
    output = []
    text1 = gconfig.NLP(text)

    doc = [token.lemma_ for token in text1 \
           if token.is_alpha and not token.is_stop and not token.is_digit]
    doc = [token for token in doc if token not in STOPWORDS]
    output.append(' '.join(doc))
    return output

def train_bigrams(str3, genbig=False, prefix = ''):
    '''generate bigrams and trigrams with text use to train model'''
    #train bigrams and trigrams
    processed_docs = []
    batchsize = 1
    if genbig:
        batchsize = 100
    for doc in gconfig.NLP.pipe(str3, n_threads=4, batch_size=batchsize):
        # Process document using Spacy NLP pipeline.
        ents = doc.ents  # Named entities.
        # Lemmatize tokens
        doc = [token.lemma_ for token in doc
               if token.is_alpha and not token.is_stop and not token.is_digit]
        doc = [token for token in doc if token not in STOPWORDS]
        # Remove common words from a stopword list.
        # Add named entities, but only if they are a compound of more than word
        doc.extend([str(entity) for entity in ents if len(entity) > 1])
        processed_docs.append(doc)

    #use bigram and trigrams
    if genbig:   #if true generate biagram model
        bigram = gensim.models.phrases.Phrases(processed_docs, min_count=50)
        trigram = gensim.models.phrases.Phrases(bigram[processed_docs],
                                                min_count=10)
        bigram = gensim.models.phrases.Phraser(trigram)
        bigram.save(gconfig.BIGRAMFILE + prefix)
    return processed_docs

def ldamodel(processed_docs, bigram, trainlda = False, prefix = '',
             min_wordcount = 5, max_freq= 0.5, topics = 20 ):
    '''Generate LDA model and dictionary
    use biagram and returns lda model and dictionary
    save lda and dictionary according to config file'''
    print('Generating dictionary and ldamodel')
    
    processed_docs1 = []
    for doc in processed_docs:
        if doc in [' nan', 'nan nan']:
            pass #description or short description was NaN from source
        else:
            bigram_sentence = ' '.join(bigram[doc])
            processed_docs1.append(bigram_sentence)

    processed_docs = []    
    for doc in gconfig.NLP.pipe(processed_docs1, n_threads=4, batch_size=100):
        ents = doc.ents  # Named entities.
        doc = [token.lemma_ for token in doc]  # Lemmatize tokens,
        doc = [token for token in doc if token not in STOPWORDS]  # Remove common words from a stopword list.
        # Add named entities, but only if they are a compound of more than word.
        doc.extend([str(entity) for entity in ents if len(entity) > 1])    
        processed_docs.append(doc)
    del processed_docs1

    for num, entity in enumerate(ents):
        print ('Entity {%d}-{%s} : {%s}' % (num+1, entity, entity.label_)) 
    if trainlda:
        print('Generating LDAModel, LSImodel, Dictionary & corpus')
        # Create a dictionary representation of the documents, and filter out frequent and rare words.
        dictionary = Dictionary(processed_docs)
        # Remove rare and common tokens.  Filter out words that occur too frequently or too rarely.
        dictionary.filter_extremes(no_below=min_wordcount, no_above = max_freq)
        _ = dictionary[0]  # This sort of "initializes" dictionary.id2token.
    
        dictionary = corpora.Dictionary(processed_docs) # turn our tokenized documents into a id <-> term dictionary
    
        dictionary.save(gconfig.DICTIONARYFILE + prefix)  #save dictionary
        corpus = [dictionary.doc2bow(text) for text in processed_docs] # convert tokenized documents into a document-term matrix
        pickle.dump(corpus, open(gconfig.CORPUS + prefix, 'wb'))
        # generate LDA model
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics= topics, 
                                                   id2word = dictionary, update_every=2, 
                                                   chunksize=5000, passes=5)
        # generate LSI model
        lsimodel = gensim.models.LsiModel(corpus, num_topics= topics,
                                          id2word = dictionary,
                                          chunksize=5000, power_iters=5)
        lsimodel.save(gconfig.LSIMODEL + prefix) #save LSI model
        ldamodel.save(gconfig.LDAMODEL + prefix) #save LDA model
        #return ldamodel, dictionary, processed_docs, lsimodel
    return processed_docs

def listmonths(start_date, end_date):
    return pd.date_range(start_date , end_date, 
              freq='MS').strftime("%Y-%m").tolist()

def get_sheets(filename):
    rows_file = pd.ExcelFile(filename, engine='openpyxl')
    return rows_file.sheet_names

def read_file(filename, columns= None, skip= 0, sheet= None, ext = None,
              fileobj= None, encoding = None):
    '''read file and return dataframe, can read PKL, CVS or excel'''
    if ext is None:
        filepm, file_extension = os.path.splitext(filename)
    else: file_extension = '.' + ext
    if fileobj is not None:
        filename = fileobj
    if file_extension == '.pkl':
        df2 = pd.read_pickle(filename, 'gzip')
        dfreadfile = df2.copy()
    elif file_extension == '.csv':
        dfreadfile = pd.read_csv(filename, encoding)
    elif file_extension in ['.xlsx', '.xls']:
        rows_file = pd.ExcelFile(filename, engine='openpyxl')
        print(f'Sheets in the file: {rows_file.sheet_names}')
        if sheet == None:
            if  fileobj is not None:
                sheetname = rows_file.sheet_names[0]
                print(sheetname)
            else:
                sheetname = input(f'Sheet to load: [{rows_file.sheet_names[0]}] : ') \
                            or rows_file.sheet_names[0]
        else:
            print(f'Sheet to load: {sheet}')
            sheetname = sheet
        #skiprows = input("Skip rows: [0]: ") or str(0)
        dfreadfile = rows_file.parse(sheetname, skiprows=skip)    
    if not dfreadfile.empty:
        if columns != None:
            dfreadfile.columns = columns
    print(f'{dfreadfile.shape[0]} Rows with {dfreadfile.shape[1]} Columns: {dfreadfile.columns}')
    return dfreadfile

# def setup_descriptioncolumns(df):
#     '''get names for description and short_description and setup xshort_title if 
#     there is no short_desciotion in the file'''
#     column_desc = 'description'
#     column_short_desc = 'xshort_title'
#     found_records = False
#     if not df.empty:
#         for col in df.columns:
#             if col in ['description', 'Description', 'desc', 'Desc']:
#                 column_desc = col
#             if col in ['short_description', 'Short_Description', 'short_desc', 'Short_Desc']:
#                 column_short_desc = col
#         found_records = True
#         column_desc = input(f'Column for [{column_desc}]: ') or {column_desc}
#         column_short_desc = input(f'Column for Short Description [{column_short_desc}]: ') or {column_short_desc}
#         if column_short_desc == 'xshort_title':
#             df['xshort_title'] = ''
#     return df, column_desc, column_short_desc, found_records

def concat(*args):
    '''concatenate 2 columns if 1 in null keep value of other col
    use:
        np_concat = np.vectorize(concat)
        df['z_classtxt'] = np_concat(df[column_short_desc],
                                     df[column_desc])
    '''
    strs = [str(arg) for arg in args if not pd.isnull(arg)]
    return ' '.join(strs) if strs else np.nan

def categorize_columns(df, numunique= 205):
    '''clean names of columns and caegorize object columns if unique is less than , numunique
       downcast for int and floats'''
    # https://stackoverflow.com/questions/30763351/removing-space-in-dataframe-python
    df.rename(columns = str.lower, inplace= True)
    df.columns = [x.strip() for x in df.columns]
    df.rename(columns = lambda x: x.replace(" ","_"), inplace= True)
    df.rename(columns = lambda x: x.replace(":",""), inplace= True)
    df.rename(columns = lambda x: x.replace(",",""), inplace= True)
    df.rename(columns = lambda x: x.replace("'",""), inplace= True)
    df.rename(columns = lambda x: x.replace(".","_"), inplace= True)
    df.rename(columns = lambda x: x.replace("&","and"), inplace= True)
    df.rename(columns = lambda x: x.replace("(",""), inplace= True)
    df.rename(columns = lambda x: x.replace(")",""), inplace= True)
    df.rename(columns = lambda x: x.replace("/",""), inplace= True)
    df.rename(columns = lambda x: x.replace("-","_"), inplace= True)
    df.rename(columns = lambda x: x.replace("#","num"), inplace= True)
    df.rename(columns = lambda x: x.replace("%","percent"), inplace= True)
    df.rename(columns = lambda x: x.replace("+","_plus_"), inplace= True)
    for y in df.columns:
        #categorize columns
        if df[y].dtype == np.object:
            if len(df[y].unique()) <= numunique:
                #print('converted ' + y + ' ' +str(df[y].dtype) + ' records=' + str(len(df[y].unique())))
                df[y] = df[y].astype('category')
        elif(df[y].dtype == np.float64 or df[y].dtype == np.int64):
            df[y] = pd.to_numeric(df[y], downcast='unsigned')
            #print('DOWNCAST ' +  y + ' ' +str(df[y].dtype))
    return df

def save_file(dfdic, filename, path = gconfig.PATHSAVE, index = False, savepkl= True, dicimage=None):
    '''save file in directory according to config file'''
    filetosavepkl=''
    filetosavexls= os.path.join(path, filename + '{:%m%d%y}.xlsx').format(gconfig.TODAY)
    #df = dflist[0]
    listkeys= list(dfdic.keys())
    one = listkeys[0]
    df = dfdic.get(one)
    if savepkl:
        filetosavepkl= os.path.join(path, filename + '{:%m%d%y}.pkl').format(gconfig.TODAY)
        df.to_pickle(filetosavepkl, 'gzip')
        print(f'File saved as {filetosavepkl}')
    writer = ExcelWriter(filetosavexls,
                         engine='xlsxwriter',
                         options={'strings_to_formulas': False, 'strings_to_urls': False})
    #for sheet_name in dflist.keys():
    workbook= writer.book
    if dicimage is not None:
        for sheetimage, image in dicimage.items():
            wks1=workbook.add_worksheet(sheetimage)
            wks1.write(0,0,sheetimage)
            wks1.insert_image(2,2, image, {'x_scale': 1.5, 'y_scale': 1.5})
    for sheet_name, df in dfdic.items():
        df.to_excel(writer, sheet_name=sheet_name, index=index)

    writer.save()
    print(f'File saved as {filetosavexls}')
    return filetosavexls, filetosavepkl

def gen_periods(start, end):
    '''generate list of periods (months) with start end date'''
    months= listmonths(start, end)
    list2= []
    for n in months:
        s= n.split('-')
        #print(s)
        list2.append(s)
    datelist = []
    periods= []
    for l in range(len(list2)):
        fd, ld = calendar.monthrange(int(list2[l][0]),int(list2[l][1]))
        date1 = list2[l][0] + '-' + list2[l][1] + '-01' 
        date2 = list2[l][0] + '-' + list2[l][1] + '-' + str(ld) 
        datelist.append(date1)
        datelist.append(date2)
        periods.append(datelist)
        datelist= []
    return periods

def add_business_days(from_date, ndays):
    '''Consider weekends when add days to a date'''
    business_days_to_add = abs(ndays)
    current_date = from_date
    sign = ndays/abs(ndays)
    while business_days_to_add > 0:
        current_date += datetime.timedelta(sign * 1)
        weekday = current_date.weekday()
        if weekday >= 5: # sunday = 6
            continue
        business_days_to_add -= 1
    return current_date

def cleaning_preprocessing(dfreadfile, column_desc, column_short_desc,
                           trainbiagrams=False, min_wordcount = 20,
                           max_freq= 0.5, topics = 20):
    '''Clean and preprocessing file, generating bigrams for the text and
    hasigh vector returns df with clean text'''
    print('Cleaning and preprocessing...')
    #concatenate short and long descriptions
    dfreadfile['z_classtxt'] = dfreadfile[column_short_desc].apply(str) +\
                               ' ' + dfreadfile[column_desc].apply(str)

    #remove description null
    dfreadfile = dfreadfile[dfreadfile.z_classtxt.notnull()]

    dfreadfile['z_cleanclasstxt2'] = dfreadfile.apply(lambda \
                                                      row: \
                                                      cleanlines2(row['z_classtxt']),
                                                      axis=1)

    ##generate bigrams and trigrams from file and save
    print('Generating bigrams')
    dfreadfile.reset_index(drop=True)
    str3 = dfreadfile['z_cleanclasstxt2']
    processed_docs = train_bigrams(str3, trainbiagrams)
    bigram = gensim.models.phrases.Phraser.load(gconfig.BIGRAMFILE)
    #put back bigrams in dfreadfile
    #ldamodelresult, dictionary, docs, lsimodelresults = ldamodel(processed_docs, bigram,
    docs = ldamodel(processed_docs, bigram,
                                                min_wordcount, max_freq,
                                                topics)
    docs1 = []
    for doc in docs:
        docs1.append(' '.join(doc))
    dfcleantxtnew = pd.DataFrame(docs1, columns=['z_newclean'])
    dfreadfile.reset_index(drop=True)
    dfcleantxtnew.reset_index(drop=True)
    dfreadfile = pd.concat([dfreadfile.reset_index(drop=True),
                            dfcleantxtnew.reset_index(drop=True)], axis=1)
    dfreadfile['z_cleanclasstxt2'] = dfreadfile['z_newclean']
    dfreadfile.drop(['z_newclean'], axis=1, inplace=True)
    print(f'dfreadfile {dfreadfile.shape}')
    print('Generating vectors...')
    # create hasing vectors
    dfreadfile['z_hashvector'] = list(gconfig.\
              VECTORIZER.transform(dfreadfile['z_cleanclasstxt2']).toarray())
    print('Done cleaning and preprocessing')
    return dfreadfile

def group_file(df, col= 'x_opened_at', ci= 'x_number', 
               freq= 'D', floor = None, ncap = None, oper='count'):
    '''convert Df to a timeseries and resample by month or daily''' 
    #print(df.shape)
    #df= df1.copy()
    df.index = df[col]
    del df[col]
    if freq != 'N':
        if oper == 'count':
            df= df.resample(freq).count()
        elif oper == 'sum':
            df= df.resample(freq).sum()
    for colum in df.columns:
        if colum != ci:
            del df[colum]
    df= df.reset_index()
    df.rename(columns={col: 'ds', ci: 'y'}, inplace=True)
    if ncap is not None:
        if ncap == 'max':
            df['cap'] = np.log(df['y'].max()+1)
        else: df['cap'] = np.log(int(ncap)+1)
    if floor is not None:
        if floor == 'min':
            df['floor'] = np.log(df['y'].min()+1)
        else: df['floor']= np.log(int(floor)+1)
    df['y_orig'] = df['y'] # to save a copy of the original data. 
    df['y'] = np.log(df['y']+1) # log-transform y
    df = df.sort_values(by=['ds'], ascending=[True])
    return df

def gen_control_chart(df, ds, y, url= None, lrl= None,
                      title= 'Control Chart', xtitle= 'Days', ytitle='Tickets'):
    '''generate fig control chart
    calculate upper and lower limits and draw chart'''
    stdev = df[y].std()
    average = df[y].mean()
    ucl = average + (3 * stdev)
    lcl = average - (3 * stdev)
    df['mean']= average
    df['ucl'] = ucl
    df['lcl'] = lcl
    if url is not None:
        df['url']= url
    if lrl is not None:
        df['lrl']= lrl
    fig, ax1 = plt.subplots()
    x1 = df[ds]
    ax1.set_title(title) 
    labelucl = f'ucl= {round(ucl,2)}'
    labelX = f'x\u0304= {round(df[y].mean(),2)}'
    labellcl = f'lcl= {round(lcl,2)}'

    ax1.plot(x1, df['ucl'], label = labelucl, color= 'red')
    if url is not None:
        labelurl = f'url= {round(url,2)}'
        ax1.plot(x1, df['url'], label = labelurl, color= 'green')
    ax1.plot(x1, df['mean'], label = labelX, color= 'purple') 
    ax1.plot(x1, df[y], color= 'blue',  marker='.') 

    if lrl is not None:
        labellrl = f'lrl= {round(lrl,2)}'
        ax1.plot(x1, df['lrl'], label = labellrl, color= 'green')
    ax1.plot(x1, df['lcl'], label = labellcl, color= 'red')
    ax1.set_ylabel(ytitle)
    ax1.set_xlabel(xtitle)
    ax1.legend(loc='upper center', fontsize='xx-small', ncol=5)#, bbox_to_anchor=(.5, -0.15),shadow=True, ncol=5) 
    return fig, df

#@st.cache(suppress_st_warning=True)
def matrix(df, col1= 'x_type', col2= 'x_state', normalize=False):
    '''Generate crosstab of 2 columns '''
    dfmatrix = pd.crosstab([df[col2]],[df[col1]],
                           margins=True, normalize = normalize,
                           dropna=False).sort_values('All', ascending=False)
    return dfmatrix  

def getcolumnindex(df, column = 'cmdb_ci'):
    '''Get inde of a column to select a column
    return index of the column'''
    i=0
    colidx = None
    for col in df.columns:
        if col == column:
            colidx = i
            break
        i +=1
    return colidx

def view_dataframe(df,prefilter_cols = None, filter_by_col= None, crosstab = False, filter_by_value = False):
    '''view dataframe with streamlite
    Parameters:
    df (dataframe): dataframe to view
    prefilter_cols : list of columns to display, if 'default'is passed a set
                     of predefined columns will be used
    filter_by_col: Name of the column to filter by, if it is None will use first
                   Column as a filter.
    Returns:
    None
    '''
    filtered_data_cols = df.columns
    filtered_data_cols = list(filtered_data_cols)
    print(f'dff {df.columns}')
    print(f'filtered_data_cols {filtered_data_cols}')
    if prefilter_cols == 'default':
        prefilter_cols = ['x_number','x_opened_at','x_opened_by', 'x_description','x_type',
                          'x_closed_at','x_closed_by','x_cmdb_ci','x_priority','x_state']
    cols_selected = st.multiselect('Columns', filtered_data_cols,
                                   default= prefilter_cols)
    cols_selected = list(cols_selected)
    print(f'filter_by_col   XXXXXX {filter_by_col}')
    if filter_by_col == 'ALL':
        print('AAALLLLL')
        filter_by_df = df.copy()
    else: 
        if filter_by_col is None:
            colindex = 0
        else: colindex = getcolumnindex(df, filter_by_col )
        filter_by_col = st.selectbox('Filter by:', filtered_data_cols, colindex,
                                     key='opt6')
        if filter_by_value:
            filter_value = st.text_input('value')
            filter_unique_vals=[]
            filter_unique_vals.append(filter_value)
        else:
            unique_vals = df[filter_by_col].unique()
        
            unique_vals = list(unique_vals[0:5])
            filter_unique_vals = st.multiselect('select values', unique_vals,
                                            default=unique_vals)
        filter_by_df = df[df[filter_by_col].isin(filter_unique_vals)]
    st.markdown(f'Records: {len(filter_by_df)}')
    st.dataframe(filter_by_df[cols_selected])
    if crosstab:
        columns = st.selectbox('Column:', filtered_data_cols, colindex,
                                 key='opt7')
        rows = st.selectbox('Row:', filtered_data_cols, colindex,
                                 key='opt8')
        normalize = st.checkbox('Normalize')
        ####
        ####
        filter_by_df[rows] = filter_by_df[rows].astype('string')
        #print(filter_by_df.type())
        crosstabdf = matrix(filter_by_df, col1=columns, col2=rows, normalize=normalize)
        print(f'crosstabdf {crosstabdf}')
        st.dataframe(crosstabdf)

def verify_normality(df, column, hist= True):
    ''' graph distribution for a column, with values > 0
    Parameters:
    df: datframe to verify
    
    Returns:
    fig: figure 1 normal distribution
    fig2: figure 2 prob plot
    '''
    print(df[column].describe())
    df2= df[df[column] > 0]
    arr = df2[column]
    mean=arr.mean()
    median=arr.median()
    mode=arr.mode()
    print('Mean: ',mean,'\nMedian: ',median,'\nMode: ',mode[0])

    arr = sorted(arr)
    fit = stats.norm.pdf(arr, np.mean(arr), np.std(arr)) 
 
    #plot both series on the histogram
    fig, ax = plt.subplots() 
    plt.axvline(mean,color='red',label='Mean')
    plt.axvline(median,color='yellow',label='Median')
    plt.axvline(mode[0],color='green',label='Mode')
    plt.plot(arr,fit,'-',linewidth = 2,label="Normal distribution with same mean and var")
    plt.hist(arr,density=True,bins = 10,label="Actual distribution")   
    ax.patch.set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    #plt.title('Histogram {}'.format(column))
    plt.legend()
    #plt.show()
    
    fig2 = plt.figure()
    if hist == True:
        ax1 = fig2.add_subplot(211)
        prob = stats.probplot(df2[column], dist=stats.norm, plot=ax1)
        ax1.set_xlabel('')
        # x= 'Theoreticl Quntities'
        #y= 'Sample Quantities'
        ax1.set_title('Prob plot against normal distribution')
    return fig, fig2


def nomality_tests(df, column, alpha= 0.05):
    '''Test normality using D'Angostino & Pearson, Shapiro, Anderson-Darling
    '''
    dfnt = pd.DataFrame(columns=['test', 'txtresult','result','pvalue'])
    x= df[column]
    stat, p = normaltest(x)  #D'Angostino & Pearson test
    print(' D Angostino = {:.3f} pvalue = {:.4f}'.format(stat, p))
    if p > alpha:
        result = 'data looks normal (fail to reject H0)'
    else:
        result = 'data does not look normal (reject H0)'

    dfnt = dfnt.append({'test' : "D'Angostino & Pearson", 'txtresult' : result,
                       'result' : stat, 'pvalue' : p}, ignore_index=True)
    if len(x) < 5000:  #Shapiro test is reliable with less than 5K records
        stat, p = shapiro(x)
        print(' Shapiro = {:.3f} pvalue = {:.4f}'.format(stat, p))
        if p > alpha:
            result = 'data looks normal (fail to reject H0)'
        else:
            result = 'data does not look normal (reject H0)'
        dfnt = dfnt.append({'test' : "Shapiro", 'txtresult' : result,
                            'result' : stat, 'pvalue' : p}, ignore_index=True)
    stat = anderson(x, dist='norm')
    print(' Anderson = {:.3f}  '.format(stat.statistic))
    dfnt = dfnt.append({'test' : "Anderson-Darling", 'txtresult' : 'NA',
                        'result' : stat.statistic, 'pvalue' : 'NA'},
                       ignore_index=True)
    for i in range(len(stat.critical_values)):
        sl, cv = stat.significance_level[i], stat.critical_values[i]
        if stat.statistic < stat.critical_values[i]:
            result = 'data looks normal (fail to reject H0)'
        else:
            result = 'data does not look normal (reject H0)'
        dfnt = dfnt.append({'test' : "Anderson-Darling", 'txtresult' : result,
                            'result' : sl/100, 'pvalue' : cv}, ignore_index=True)                                                                                       
    return dfnt
