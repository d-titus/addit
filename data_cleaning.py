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

# (--------------- Accessory and Lookup merge and preclean -------------------)
# Concating the unique identifers so that they can be merged into main file
acc['parcel_parcel_number'] = acc['Major']+acc['Minor']
len(acc)


def strip_letters(column, df):
    letters = list(string.ascii_letters)
    for letter in letters:
        df = df[df[column].str.contains(letter) == False]


strip_letters('parcel_parcel_number', acc)

len(acc)
len(sales)

strip_letters('parcel_parcel_number', sales)

len(sales)
acc['parcel_parcel_number'].str.contains(string.ascii_letters).sum()
sales['parcel_parcel_number'].str.contains(string.ascii_letters).sum()

# Remove strings from parcel_parcel_number
acc = acc[acc['parcel_parcel_number'].str.contains(string.ascii_letters) != True]
sales = sales[sales['parcel_parcel_number'].str.contains(string.ascii_letters) != True]

acc['parcel_parcel_number'].str.contains(string.ascii_letters).sum()
sales['parcel_parcel_number'].str.contains(string.ascii_letters).sum()

len(acc)

# Type 0 does not appear in the list, and is a small number of homes.
acc = acc[acc['AccyType'] != 0]

# removing objects, and columns that won't be useful to the model
acc_drop_list = ['Major', 'Minor', 'DateValued', 'UpdatedBy',
                 'UpdateDate', 'EffYr', 'Quantity', 'Size', 'Unit',
                 'Grade', 'AccyValue', 'PcntNetCondition']

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


def cleaning_info_generator(col_name, df):
    print ('Number of unique: {}'.format(len(df[col_name].unique())))
    print ('Type: {}'.format(df[col_name].dtype))
    if df[col_name].dtype == float or df[col_name].dtype == int:
        print 'Median: {}'.format(df[col_name].median())
        print 'Mean: {}'.format(df[col_name].mean())
        print 'Range: {}'.format(df[col_name].max() - df[col_name].min())
        print 'Max: {}'.format(df[col_name].max())
        print 'Min: {}'.format(df[col_name].min())
    print ('Number of holes: {}'.format(len(df[col_name])-len(df)))
    if len(df[col_name].unique()) < 20:
        print ('Unique values: {}'.format(df[col_name].unique()))

len(sales)

# Unneeded column removal
remove_cols = ['is_flip', 'pre_flip_listing_source', 'pre_flip_sale_deed',
               'pre_flip_sale_type', 'pre_flip_sale_date', 'pre_flip_sale_price',
               'pre_flip_sale_id', 'parcel_latitude', 'parcel_longitude',
               'parcel_street_address', 'parcel_thumbnail_url',
               'sale_price_per_square_foot', 'parcel_county_id',
               'parcel_neighborhood_id', 'parcel_city', 'parcel_county',
               'sale_listing_number', 'street_scores', 'parcel_id', 'structure_id',
               'sale_id', 'value_assessed_tax_value', 'sale_listing_company',
               'sale_listing_status', 'street_distances', 'sale_pending_dates',
               'AccyType']

# Columns that may warrant further investigation
cols_to_investigate = ['sale_listing_remarks', 'parcel_parking_count_uncovered',
                        'parcel_school_district_name','photo_count', 'sale_listing_prices',
                        'sale_listing_prices_dates', 'AccyDescr']

drop_cols(remove_cols, sales)
drop_cols(cols_to_investigate, sales)

cols_to_features = ['parcel_access_type', 'parcel_mls_neighborhood', 'parcel_view_type',
                    'parcel_waterfront_type', 'parcel_zip_code', 'structure_structure_type',
                    'structure_style', 'structure_hvac', 'structure_siding_cover',
                    'structure_roof_cover', 'sale_sale_deed', 'sale_sale_type',
                    'LUDescription']

cols_with_nan_to_zero = ['parcel_waterfront_footage', 'structure_square_feet_basement_finished',
                         'structure_square_feet_basement_unfinished', 'structure_square_feet_garage',
                         'structure_square_feet_decking', ]

cols_with_nan_to_med = ['structure_floors','parcel_mls_area', 'parcel_parking_count_covered',
                         'structure_quality', 'parcel_lot_square_feet',
                        'structure_square_feet_finished']

cols_with_nan_to_mean = ['parcel_block_group_pp_sqft', 'parcel_assessment_area',
                        'parcel_assessment_subarea', 'parcel_school_district_rating',
                        'structure_condition', 'structure_square_feet_finished']

cleaning_info_generator('parcel_lot_square_feet', sales)

sales.info()


parcel = len(sales.parcel_parcel_number)
unique_parcel = len(sales.parcel_parcel_number.unique())

parcel - unique_parcel

sales.info()

exists = lambda x: 0 if x in [[], np.nan, 'None'] else 1

exist_cols = ['']


# (------Pivot talbe set up-----)
sales = sales.pivot_table(sales, index='parcel_parcel_number', aggfunc='mean')
