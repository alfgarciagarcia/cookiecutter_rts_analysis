#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  18 22:55:00 2020

@author: alfredogarcia

Config for Tickets Classification model
"""
import re
import os
from pathlib import Path
import datetime
import spacy
from sklearn.feature_extraction.text import HashingVectorizer
#from sklearn.externals import joblib
from sklearn_pandas import DataFrameMapper
import joblib

NLP = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

STOPWORDS = ['FW', 'ssw', 'SSW', 'SSW:', 'thank', '\n', '\n\n', '\n\n\n\n',\
             'thank', '\t', '\t ', 'â', ':', '/', '-', '_', 'pm', 'am', ',', 'PM']

LABELREGEX = re.compile(r'SSW:|Department:|Work Extension:|From:|Sent:|To:|\
                        Cc:|Please provide your application logon name:|\
                        The information transmitted in this email|\
                        .@staples.com|Tel:|Mob:|Preferred.ca')


CATGORIES = ['Application', 'Collaboration', 'Databases',
             'DataCenter', 'End User Support', 'ITSM',
             'Networking', 'Servers', 'Storage & Backup',
             'Virtualization']

OPEN_CASES = ['Pending', 'Work in Progress','Open', 'In Progress',
              'On Hold', 'Submitted', 'Scheduled', 'Accepted']

#templates for files to upload
#customize the template online use the following definition
FIELD_DEFAULTS={
    'stk_project':('stk_project','Column for stk project, defailt [stk_project]'),
    'service_group':('service_group','Column for service_group, defailt [service_group]'),
    'x_description':('x_description', 'Column for x_desc, default [x_description]'),
    'x_shortdesc':('x_short_description', 'Column for x_shortdescription, default [x_short_description'),
    'x_number':('x_number', 'Column for "x_number", default [x_number]'),
    'x_opened_at': ('x_opened_at', 'Column for "x_opened_at", default [x_opened_at]'),
    'x_opened_by': ('x_opened_by', 'Column for "x_created_by", default [x_opened_by]'),
    'x_closed_at' : ('x_closed_at', 'Column for "x_closed_at", default [x_closed_at]'),
    'x_closed_by' : ('x_closed_by', 'Column for "x_closed_by", default [x_closed_by]'),
    'x_cmdb_ci' : ('x_cmdb_ci', 'Column for "x_cmdb_ci", default [cmdb_ci]'),
    'x_priority' : ('x_priority', 'Column for "x_priority", default [x_priority]'),
    'x_state' : ('x_state', 'Column for "x_state", default [state]'),
    'x_type' : ('x_type', 'Column for "x_type", default [cmdb_ci]')}
#template for OSMfile
FIELD_OSM={
    'service_group':'assignment_group',
    'x_description':'short_description',
    'x_shortdesc':'number',
    'x_number':'number',
    'x_opened_at': 'created',
    'x_opened_by': 'opened_by',
    'x_closed_at' : 'resolved',
    'x_closed_by' : 'assigned_to',
    'x_cmdb_ci' : 'configuration_item',
    'x_priority' : 'priority',
    'x_state' : 'status',
    'x_type' : 'type'}
#fixed values to use with the template
FIXED_VALUESOSM = {
    'stk_project': 'OSM'}

#SLAs definition per option in days
#default SLAs
SLAS_DEFAULT= {
    '4 - Low' : 7,
    '1 - Critical': 1,
    '2 - High': 3,
    '3 - Moderate': 5 }
SLAS_OSM= {
    '1 - Critical': 1,
    '2 - High': 3,
    '3 - Moderate': 7 }


SUBCATEGORIES = ['' 'Backup' 'Break fix'
'Cluster Request - Add, Modify or Delete' 'Configuration'
'Databases' 'Decommission' 'E-mail'
'File System Changes' 'LDAP Configuration'
'OS Upgrade' ''
'Patching' 'Reboot' 'Server down'
'ServiceNow' 'Test'
'VM - Resources - Add / Modify / Delete']

SUBCAT_APPLICATION = ['Authentication','Package - add / modify / delete']

SUBCAT_SERVERS = ['Cluster Request - Add, Modify or Delete', 'Configuration',
'File System Changes', 'LDAP Configuration', 'OS Upgrade',
'Patching', 'Reboot', 'Server down', 'Test']

DICTRCA = dict(zip(range(1, len(CATGORIES)+1, 1), CATGORIES))
DICTRCAL2 = dict(zip(range(1, len(SUBCATEGORIES)+1, 1), SUBCATEGORIES))

ORIGINAL_PATH = os.getcwd()
PATH = str(Path(ORIGINAL_PATH))
PATHSAVE = os.path.join(PATH, 'data', 'processed')
PATHRAW = os.path.join(PATH, 'data', 'raw')
PATHREPORT = os.path.join(PATH, 'data', 'reports')
PATHFIGURES= os.path.join(PATHREPORT,'figures')

MODELPATH = os.path.join(PATH, 'models')

#load model
MODNAME = 'ITISPRedictormodelRF_v1CF'
MODELFILENAME = os.path.join(MODELPATH, MODNAME + '.mod')
#MODEL = joblib.load(MODELFILENAME)

MODNAME3 = 'ITISPRedictormodelRFLevel2_v1'
MODELFILENAME3 = os.path.join(MODELPATH, MODNAME3 + '.mod')
#MODEL3 = joblib.load(MODELFILENAME3)

TODAY = datetime.datetime.today()
DIRECTORY_NAME = 'general'

INPUTFILE = 'ticketmaster.xlsx'

FILENAME, file_extension = os.path.splitext(INPUTFILE)

BIGRAMFILE = os.path.join(MODELPATH, DIRECTORY_NAME + 'bigramfile_lda')
DICTIONARYFILE = os.path.join(MODELPATH, DIRECTORY_NAME + 'dictionary_lda')
LDAMODEL = os.path.join(MODELPATH, DIRECTORY_NAME + 'ldamodel')
LSIMODEL = os.path.join(MODELPATH, DIRECTORY_NAME + 'lsimodel')
CORPUS = os.path.join(MODELPATH, DIRECTORY_NAME + 'corpus')
#outputfiles
FILE_W_PREDICTIONS_PKL = os.path.join(PATHSAVE,
                                      FILENAME + '{:%m%d%y}.pkl').format(TODAY)
FILE_W_PREDICTIONS_XLSX = os.path.join(PATHSAVE,
                                       FILENAME + '{:%m%d%y}.xlsx').format(TODAY)

WORDCLOUDFILE = os.path.join(PATH, 'data', 'processed',
                             'wordcloud_' + '{:%m%d%y}.png').format(TODAY)
# create hasing vectors
VECTORIZER = HashingVectorizer(n_features=60)

MAPPER2 = DataFrameMapper([
    ('z_RCA_predicted_desc', HashingVectorizer(n_features=60)),
    ('z_cleanclasstxt2', HashingVectorizer(n_features=60)),
])

VERBOSE = False