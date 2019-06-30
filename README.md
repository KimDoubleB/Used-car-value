# Used-car-value

## Objective
The number of vehicles per person is increasing. Therefore, used car trade will be brisk and our project will be helpful here.

## Dataset to use
There are 20 columns and 371539 rows. And we use 9 columns out of 20. The columns used are as follows. Price, yearOfRegistraion, powerPS, kilometer, monthOfRegistration, nrOfPictures, postalCode.

## Data Inspection
Through ‘data inspection’, It was found that the data were relevant for the purpose because price, powerPS are essential to predicting used car prices. and we are also can find a feature that should be deleted in the process of preprocessing like ‘nrOfPicture’ which has only 0 as value. Through statistical description, we found that scale was different between features. Therefore, it was recognized that standardization should be performed in the preprocessing process.

### Correlation
Through head map, correlation of each features is displayed. ‘Price’ had direct relationship with ‘powerPS’ and indirect relationship with ‘kilometer’.

## Data Preprocessing

1. Data Restructuring
- Vertical Decomposition
We deleted Columns on two criteria. The first criterion is "no data diversity". We thought the lack of diversity in the data makes data unnecessary and it would not help when analyzing. The second criterion is "Information of Data Crawling." These data sets also had information about data crawling, including when it was. This information has been deleted because it is not required for price prediction.

2. Data Value changes
    1. Cleaning dirty data
        * Outlier<br>
        We could check out the outliers in the box plot. The outlier has been removed.
        * Wrong data<br>
        Whether the price of the car is 0 won or the powerPS of the car is 0, we've done this to eliminate misinformation that doesn't fit the information of the car.
        * Unusable data<br>
        Duplicate data was removed through a method called ‘drop_duplicates’.
        * Missing data<br>
        Our data had null values in four categories: ‘vehicleType’, ‘gearbox’, ‘fueltype’, ‘notRepairedDamage’. The three preceding types of ‘verticleType’, ‘gearbox’ and ‘fuelType’ were judged to be dependent on the vehicle model, and if the feature is a null value, the model was filled with the mode of the model. ‘notRepairedDamage’ was very important information in predicting vehicle prices. And because of the large amount of null values, it was too much to delete. So we created a new category called "not declared."
    1. Data standardization<br>
      We had a lot of quantity data. From the previous statistical figures, we can see that there is a large difference in values between features. So, we performed standardization using ‘StandardScaler’.
    1. Feature engineering
        * Feature creation<br>
        We encoded the categorical data. One hot encoding that we learned in class creates too much features. So, we used ‘LabelEncoder’ for encoding categorical data.
        * Feature reduction<br>
        PCA that we learned in class is used for feature reduction. 11 columns were reduced to 9 by PCA.

## Data analysis
We used random forest that we learned in class. And we used RandomForestRegressor for regression because ‘Price’ is continuous value.
First we used Decision Tree. However, while the decision tree has the advantage of clearly showing each step, overfitting problems occurred because the model was so optimized to training dataset. So, very low accuracy was calculated and we decided to use random forest to solve this problem.
In the RandomForestRegressor, there is a data called ‘out-of-bag score’. This is the data that shows accuracy by testing with data that was not used during a random test. We also performed the evaluation process through a MAPE method. 
However, two evaluations have shown that the current analysis has a very low accuracy rate. So we went back to Preprocessing and re-processed the data. As a result, we found something wrong with the PCA. We went ahead with PCA, and we thought, "Let's just use it." And when we came back, we realized that there was a problem with underfitting, because the number of features had decreased.
By erasing the PCA process and performing the data analysis again, we have higher accuracy rates for data analysis than ever before. We found 88% accuracy for Score and 80% accuracy for MAPE. And by comparing the results directly predicted, we can see that it's somewhat predictable.

## Conclusion
We used the Decision tree and PCA to analyze our data and achieve low accuracy. So we went back and we repeated the preprocessing process so many times that we were able to achieve this high level of accuracy. And what we've learned from this is that we can see the importance of preprocessing, as the professor said in class. I realized more realistically as I watched the accuracy change slightly depending on how the data was processed.
In conclusion, we were able to predict the price of used cars through this process, and we learned the importance of preprocessing.



For database information, visit below website.  
https://www.kaggle.com/orgesleka/used-cars-database
