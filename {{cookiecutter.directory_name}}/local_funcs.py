#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 10:35:09 2023

@author: alfredogarcia
Local funcitions for this component
"""
import pandas as pd
import streamlit as st

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
    def matrix(df, col1= 'x_type', col2= 'x_state', normalize = False):
        '''Generate crosstab of 2 columns '''
        dfmatrix = pd.crosstab([df[col2]],[df[col1]], margins=True, normalize = normalize,
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
    
    filtered_data_cols = df.columns
    filtered_data_cols = list(filtered_data_cols)
    if prefilter_cols == 'default':
        prefilter_cols = ['x_number','x_opened_at','x_opened_by', 'x_description','x_type',
                          'x_closed_at','x_closed_by','x_cmdb_ci','x_priority','x_state']
    cols_selected = st.multiselect('Columns', filtered_data_cols, default= prefilter_cols)
    cols_selected = list(cols_selected)
    if filter_by_col == 'ALL':
        filter_by_df = df.copy()
    else: 
        if filter_by_col is None:
            colindex = 0
        else: colindex = getcolumnindex(df, filter_by_col )
        filter_by_col = st.selectbox('Filter by:', filtered_data_cols, colindex, key='opt6')
        if filter_by_value:
            filter_value = st.text_input('value')
            filter_unique_vals=[]
            filter_unique_vals.append(filter_value)
        else:
            unique_vals = df[filter_by_col].unique()
        
            unique_vals = list(unique_vals[0:5])
            filter_unique_vals = st.multiselect('select values', unique_vals, default=unique_vals)
        filter_by_df = df[df[filter_by_col].isin(filter_unique_vals)]
    st.markdown(f'Records: {len(filter_by_df)}')
    st.dataframe(filter_by_df[cols_selected])
    if crosstab:
        columns = st.selectbox('Column:', filtered_data_cols, colindex, key='opt7')
        rows = st.selectbox('Row:', filtered_data_cols, colindex, key='opt8')
        normalize = st.checkbox('Normalize')
        filter_by_df[rows] = filter_by_df[rows].astype('string')
        crosstabdf = matrix(filter_by_df, col1=columns, col2=rows, normalize=normalize)
        print(f'crosstabdf {crosstabdf}')
        st.dataframe(crosstabdf)