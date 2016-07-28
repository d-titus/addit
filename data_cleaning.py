from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string

acc = pd.read_csv('EXTR_Accessory_V.csv', dtype={'Major': object, 'Minor': object})
lookup = pd.read_csv('EXTR_LookUp.csv')
sales = pd.read_csv('king_county_sales.csv', dtype={'parcel_parcel_number': object})

# Input: columns = list, df = dataframe
# Output: dataframe with columns removed


def drop_cols(columns, df):
    for cols in columns:
        df.drop(cols, axis='columns', inplace=True)
    # Input: list of columns, dataframe
    # Output: none
    # drops multiple columns at once

def fill_nas(col_list, df, fill_with):
    if fill_with == 0:
        for col in col_list:
            df[col].fillna(value=fill_with, inplace=True)
    elif fill_with == 'mean':
        for col in col_list:
            df[col].fillna(value=df[col].mean(), inplace=True)
    elif fill_with == 'median':
        for col in col_list:
            df[col].fillna(value=df[col].median(), inplace=True)
    # Input: list of colums, dataframe, string
    # Output: none
    # Fill in nans with 0, mean, or median


def cleaning_info_generator(col_name, df):
    print col_name
    print ('Number of unique: {}'.format(len(df[col_name].unique())))
    print ('Type: {}'.format(df[col_name].dtype))
    if df[col_name].dtype == float or df[col_name].dtype == int:
        print 'Median: {}'.format(df[col_name].median())
        print 'Mean: {}'.format(df[col_name].mean())
        print 'Range: {}'.format(df[col_name].max() - df[col_name].min())
        print 'Max: {}'.format(df[col_name].max())
        print 'Min: {}'.format(df[col_name].min())
    print ('Number of records: {}'.format(len(df)))
    print ('Number of records filled: {}'.format(len(df)-df[col_name].isnull().sum()))
    print ('Number of records missing: {}'.format(df[col_name].isnull().sum()))
    if len(df[col_name].unique()) < 20:
        print ('Unique values: {}'.format(df[col_name].unique()))
    # Input: list of colums, dataframe
    # Output: useful features from the column
    # Use for EDA when deciding which columns to

def drop_rows_with_letters(column, df):
    df = df[df[column].str.isdigit]


letters = list(string.ascii_letters)
for letter in letters:
    sales = sales[sales['parcel_parcel_number'].str.contains(letter) == False]

# sales = sales[sales['parcel_parcel_number'].str.isdigit == False]

# acc = acc[acc['parcel_parcel_number'].str.isdigit]

len(sales)

# (--------------- Accessory and Lookup merge and preclean -------------------)
# Concating the unique identifers so that they can be merged into main file
acc['parcel_parcel_number'] = acc['Major']+acc['Minor']
'''
len(acc)

drop_rows_with_letters('parcel_parcel_number', acc)

len(acc)
len(sales)

drop_rows_with_letters('parcel_parcel_number', sales)

len(sales)
'''
# Remove strings from parcel_parcel_number
acc = acc[acc['parcel_parcel_number'] != acc['parcel_parcel_number'].str.contains(string.ascii_letters)]
sales = sales[sales['parcel_parcel_number'] != sales['parcel_parcel_number'].str.contains(string.ascii_letters)]

len(acc)
acc['parcel_parcel_number'].str.contains(string.ascii_letters).sum()
sales['parcel_parcel_number'].str.contains(string.ascii_letters).sum()

len(acc)

# Type 0 does not appear in the list, and is a small number of homes.
acc = acc[acc['AccyType'] != 0]

# removing objects, and columns that won't be useful to the model
acc_drop_list = ['Major',
                 'Minor',
                 'DateValued',
                 'UpdatedBy',
                 'UpdateDate',
                 'EffYr',
                 'Quantity',
                 'Size',
                 'Unit',
                 'Grade',
                 'AccyValue',
                 'PcntNetCondition']

drop_cols(acc_drop_list, acc)

# Getting text values for AccyType
lookup = lookup[lookup['LUType'] == 26]
lookup = lookup.drop('LUType', axis=1)

acc = pd.merge(lookup, acc, left_on='LUItem', right_on='AccyType')
acc = acc.drop('LUItem', axis=1)

# Need to fix these values
len(acc['parcel_parcel_number'])
len(acc['parcel_parcel_number'].unique())
len(acc['parcel_parcel_number'])-len(acc['parcel_parcel_number'].unique())

# (--------------------------Sales cleaning-----------------------------------)

# sales.groupby('structure_structure_type').count().T

# Merge all files into one

sales = pd.merge(sales, acc, on='parcel_parcel_number', how='left')

sales['parcel_parcel_number'].dropna(inplace=True)

len(sales)
# Remove all plexes and 'other' structure types
structure_type_removal = ['Other', 'Duplex', 'Triplex', 'Fourplex', 'Fiveplex']

for structure in structure_type_removal:
    sales = sales[sales['structure_structure_type'] != structure]

len(sales)

# Unneeded column removal
remove_cols = ['is_flip',
               'pre_flip_listing_source',
               'pre_flip_sale_deed',
               'pre_flip_sale_type',
               'pre_flip_sale_date',
               'pre_flip_sale_price',
               'pre_flip_sale_id',
               'parcel_latitude',
               'parcel_longitude',
               'parcel_street_address',
               'parcel_thumbnail_url',
               'sale_price_per_square_foot',
               'parcel_county_id',
               'parcel_neighborhood_id',
               'parcel_city',
               'parcel_county',
               'sale_listing_number',
               'street_scores',
               'parcel_id',
               'structure_id',
               'sale_id',
               'value_assessed_tax_value',
               'sale_listing_company',
               'sale_listing_status',
               'street_distances',
               'sale_pending_dates',
               'AccyType']

# Columns that may warrant further investigation
cols_to_investigate = ['sale_listing_remarks',
                       'parcel_parking_count_uncovered',
                       'parcel_school_district_name',
                       'photo_count',
                       'sale_listing_prices',
                       'sale_listing_prices_dates',
                       'AccyDescr']

drop_cols(remove_cols, sales)
drop_cols(cols_to_investigate, sales)

#  (------------------Removing NaNs______________________________)

cols_with_nan_to_zero = ['parcel_waterfront_footage',
                         'structure_square_feet_basement_finished',
                         'structure_square_feet_basement_unfinished',
                         'structure_square_feet_garage',
                         'structure_square_feet_decking']

cols_with_nan_to_med = ['structure_floors',
                        'parcel_mls_area',
                        'parcel_parking_count_covered',
                        'structure_quality',
                        'parcel_lot_square_feet',
                        'structure_square_feet_finished',
                        'street_score']

cols_with_nan_to_mean = ['parcel_block_group_pp_sqft',
                         'parcel_assessment_area',
                         'parcel_assessment_sub_area',
                         'parcel_school_district_rating',
                         'structure_condition',
                         'structure_square_feet_finished']

fill_nas(cols_with_nan_to_zero, sales, 0)
fill_nas(cols_with_nan_to_med, sales, 'median')
fill_nas(cols_with_nan_to_mean, sales, 'mean')

sales.info()

cleaning_info_generator('structure_style', sales)
sales.info()


# Categorical columns that need to be turned into features

cols_to_features = ['parcel_access_type',
                    'parcel_mls_neighborhood',
                    'parcel_view_type',
                    'parcel_waterfront_type',
                    'parcel_zip_code',
                    'structure_structure_type',
                    'structure_style',
                    'structure_hvac',
                    'structure_siding_cover',
                    'structure_roof_cover',
                    'sale_sale_deed',
                    'sale_sale_type',
                    'LUDescription']

sales_dummies = pd.get_dummies(sales, columns=cols_to_features, dummy_na=True, drop_first=True)

sales_dummies.info()

sales = pd.concat([sales, sales_dummies])

# Drop cols that have been turned into features
drop_cols(cols_to_features, sales)

# (------Pivot talbe set up-----)
sales = sales.pivot_table(sales, index='parcel_parcel_number', aggfunc='mean')

sales.to_csv('clean_data.csv')
