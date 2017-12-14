# Addit

I was tasked with determining if adding accessories to a property can improve overall price prediction for properties in King County, as an intern for the startup Pellego. Accessories can be things like pools, jacuzzis, additional structures like a mother in law house, or parking structures. Pellego suspected that these things might have some predictive value, and I was tasked with finding out whether they could improve the model or not. When faced with a data question I use the CRISP-DM methodology. This is a six-step method for data processing that helps to ensure that the business gets what it needs out of a data project.

### Business Understanding

Pellego's central question was whether to incorporate an additional data set that includes property additions to their model. These include any addition to a property that is not inside the house, such as carports, pools, mother in laws, garages, and a myriad of other miscellaneous property additions.

### Data Understanding

I started with several separate files; there was the Pellego housing sales data set, the King County property additions data set, and a file translation file that showed what each addition code meant. I started by translating each of the coded fields into their plain text fields so that I knew what each addition actually corresponded to. I also ensured that the parcel numbers actually matched up so that there weren't properties in the accessory table that did not exist.

### Data Preparation

I started cleaning my data by removing outlier houses that were under $70 thousand and over $2 million because my contact at Pellego said that was the range that they most frequently worked in. They were primarily interested in single-family homes, so I removed anything with a structure that as multiplex or 'Other.'

Then I selected features that would be likely to help predict property values such as year built, square footage, number of covered parking spots, number of photos in the listing and made dummies of categorical fields such as neighborhood, structure style, and siding type. I made dummy variables of the different accessory types and grouped them into three categories, Pools, Parking and Other due to the sparsity of each specific accessory type.

### Modeling

I kept two data sets, one with accessories added in and one without, so that I could compare my models with and without and see if adding them could really improve the predictive value. I tried several different approaches, I stated with linear regression but was unhappy with the R-squared values, so I tried both gradient boosting regression and random forest regression.

### Evaluation

After cross-validating each model I found that gradient boosting regressor showed the best R-squared values, both with and without accessories. The addition of accessories improved the R-squared value of the model from 0.85 to 0.87, while random forest regressor improved the model from 0.82 to 0.85.

### Deployment

After talking it over at Pellego, they decided that although the addition of accessories could improve their model prediction, it wasn't a large enough value that they could not spare the engineering staff to put this into their production code.
