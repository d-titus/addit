from __future__ import division
import pandas as pd
import string

acc = pd.read_csv('EXTR_Accessory_V.csv', dtype={'Major': object, 'Minor': object})
lookup = pd.read_csv('EXTR_LookUp.csv')
sales = pd.read_csv('king_county_sales.csv', dtype={'parcel_parcel_number': object})


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


def get_first(i):
    if type(i) == list:
        return i[::-1]
    elif type(i) == str:
        return list(i)[::-1]
    else:
        return i
# Input: column with list inside
# Output: output column with last value of list



# (--------------------------Sales cleaning-----------------------------------)

# (---------Remove non existant parcel_parcel_numbers -----------------------)

letters = list(string.ascii_letters)
for letter in letters:
    sales = sales[sales['parcel_parcel_number'].str.contains(letter) == False]

sales['parcel_parcel_number'].dropna(inplace=True)

# Remove all plexes and 'other' structure types
structure_type_removal = ['Other', 'Duplex', 'Triplex', 'Fourplex', 'Fiveplex']

for structure in structure_type_removal:
    sales = sales[sales['structure_structure_type'] != structure]

# Remove houses that are extremely high or extremely low in value
sales['sale_sale_price'] = get_first(sales['sale_sale_price'])

sales = sales[sales['sale_sale_price'] < 2000000]
sales = sales[sales['sale_sale_price'] > 70000]

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
               'structure_square_feet_finished',
               # 'structure_quality',
               # 'structure_bathrooms'
               ]

# Columns that may warrant further investigation
cols_to_investigate = ['sale_listing_remarks',
                       'parcel_parking_count_uncovered',
                       'parcel_school_district_name',
                       'photo_count',
                       'sale_listing_prices',
                       'sale_listing_prices_dates',
                       'parcel_zip_code',
                       'sale_sale_date']

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
                        'street_score',
                        #'structure_square_feet_finished'
                        ]

cols_with_nan_to_mean = ['parcel_block_group_pp_sqft',
                         'parcel_assessment_area',
                         'parcel_assessment_sub_area',
                         'parcel_school_district_rating',
                         'structure_condition',
                         ]

fill_nas(cols_with_nan_to_zero, sales, 0)
fill_nas(cols_with_nan_to_med, sales, 'median')
fill_nas(cols_with_nan_to_mean, sales, 'mean')

# Categorical columns that need to be turned into features
cols_to_features = ['parcel_access_type',
                    'parcel_mls_neighborhood',
                    'parcel_view_type',
                    'parcel_waterfront_type',
                    'structure_structure_type',
                    'structure_style',
                    'structure_hvac',
                    'structure_siding_cover',
                    'structure_roof_cover',
                    'sale_sale_deed',
                    'sale_sale_type'
                    ]

sales_dummies = pd.get_dummies(sales, columns=cols_to_features, dummy_na=True,
                               drop_first=True)

drop_cols(cols_to_features, sales)

sales.to_csv('control_regression.csv')

regression_cols = sales.columns

drop_cols(regression_cols, sales_dummies)

# (Make control data set to see whether extra features improve the model)

sales_dummies.to_csv('categorical_control.csv')

sales_dummies['parcel_parcel_number'] = sales['parcel_parcel_number']

# (--------------- Accessory and Lookup merge and preclean -------------------)
# Concating the unique identifers so that they can be merged into main file
acc['parcel_parcel_number'] = acc['Major']+acc['Minor']

# Type 0 does not appear in the list of values, and is a small number of homes.
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

sales_dummies = pd.merge(sales_dummies, acc, on='parcel_parcel_number', how='left')

cols_in_sales_only = ['AccyType', 'AccyDescr']
drop_cols(cols_in_sales_only, sales_dummies)

acc_features = ['LUDescription']
sales_dummies = pd.get_dummies(sales_dummies, columns=acc_features, dummy_na=True, drop_first=True)

# (------Pivot talbe set up to deal with duplicate parcel numbers-----)
sales_dummies = sales_dummies.pivot_table(sales_dummies, index=['parcel_parcel_number'], aggfunc='sum')

sales_dummies.to_csv('test_categorical.csv')
