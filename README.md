# Addit

In partnership with Pellego, I have been seeing if adding accessories to a property can improve overall price prediction for houses.  Accessories can be thinks like pools, jacuzzis, additional structures like a mother in law house.  Pellego suspected that these things might have some predictive value, and I was tasked with finding out.

I started with a several separate files; there was the Pellego housing sales data set, the King County additions data set, and a file that translated what each of the items in the look up table were. I was able to extract the relevant features from the two accessory databases, and match them with their corresponding property, and then merge all three into a single usable dataframe.

Then I selected features that would be likely to help predict house price.  And made dummies of categorical fields such as neighborhood, structure style.  I also made dummies of the different accessory types, like covered parking.
