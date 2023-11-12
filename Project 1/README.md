# AML Task1

## General Info

In this project, we are exploring the machine learning pipelines to solve the problem of transforming the MRI data into ages. In general, we tried imputer, scaler, feature selection models, outlier detection models and various estimators.

## What I am responsible for

I am responsible for the whole process of the project. Specifically, I tried various combinations of different imputers, scalers, feature selection models, outlier detection models and estimators, and I found the most powerful model for public datasets in our team. In addition, I constructed an ensemble model which is more friendly to the imbalanced data in this task.

## What we do in the project

In this project, we first fill in the missing data with the median and then use RobustScaler to normalise the data. Then we use SelectKBest to filter out the 200 most effective features. We tried different outlier detection methods and found that there is some imbalanced data mistakenly deleted by them. So we constructed an outlier detection model based on average and standard deviation to avoid deleting imbalanced data.

We tried different estimators to compare which one gives the best results. And we found the CatBoost is the best one. Finally, we build an ensemble model that considers the outputs of multiple estimators, like GP and CatBoost, and use some tricks to improve the model's performance on imbalanced data. During the validation, we found that our model has a poor performance on the data whose y >= 80 or y<=50. So we train the classifiers to detect such potential samples and predict them with specific models that have a better performance on such data. Finally, we average the new prediction result with the original predict result to prevent overfitting.

## Conclusion

In this project, we explored different combinations of data preprocessing and got the final pipeline which had a relatively good score. In the future, we will explore some more effective ways to discover the best hyperparameters to get a higher score.



## Version 1

In this project, we are exploring the machine learning pipelines to solve the problem of transforming the MRI data into ages.
I am responsible for the whole process of the project. Specifically, I tried various combinations of different imputers, scalers, feature selection models, outlier detection models and estimators, and I found the most powerful model for public datasets in our team. In addition, I constructed an ensemble model which is more friendly to the imbalanced data in this task.
In this project, we first fill in the missing data with the median and then use RobustScaler to normalise the data. Then we use SelectKBest to filter out the 200 most effective features. We tried different outlier detection methods and found that there is some imbalanced data mistakenly deleted by them. So we constructed an outlier detection model based on average and standard deviation to avoid deleting imbalanced data.
We tried different estimators to compare which one gives the best results. And we found the CatBoost is the best one. During the validation, we found that our model has a poor performance on the data whose y >= 80 or y<=50. Finally, we build an ensemble model that considers the outputs of multiple estimators, like GP and CatBoost, and use some tricks to improve the model's performance on imbalanced data.

In the future, we will explore some more effective ways to discover the best hyperparameters to get a higher score.