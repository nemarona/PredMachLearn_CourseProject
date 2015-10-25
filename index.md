---
title: "Practical Machine Learning Course Project Report"
author: "Eduardo Rodríguez"
date: "October 25, 2015"
output: html_document
---




## Introduction

This is the Course Project Report for the Practical Machine Learning class
at Coursera.

The goal of this project is to predict, using data collected from
wearable devices, whether a person is correctly performing an exercise or not.
The training data have information from several subjects,
who are each wearing four data-collecting devices.
Each record is classified as either correct ("A") or incorrect,
with four possible common mistakes identified as "B," "C," "D," and "E."


## Pre-processing

The training data comprise 19,622 observations of 160 variables.
The first seven variables identify the subject, the time, and provide some
additional information, none of which is relevant for our prediction algorithm.
The last variable, `classe`, is the one we want to predict.
This leaves us with 152 predictors, all of them numerical.
Closer inspection reveals that exactly 100 of these (about 2/3 of the total)
have a very high proportion of missing values (greater than 19,000/19,622),
and hence are useless for prediction purposes.

We will build our model using the remaining 52 predictors.


## Model building

We first use the `createDataPartition()` function from the R `caret` package
to split our data into "training" and "testing" sets, with about 70%
of all data allocated to the "training" set.

Our model is a random forest using all 52 predictors:

```
## train.default(x = trainX, y = trainY, method = "rf")
```
The most important variables, arranged according to mean decrease in Gini impurity
(higher decrease means lower Gini impurity, which is better),
turn out to be

```
##             feature MeanDecreaseGini
## 1         roll_belt        1427.7085
## 2     pitch_forearm         844.0624
## 3          yaw_belt         779.4472
## 4 magnet_dumbbell_z         642.6414
## 5        pitch_belt         642.5693
## 6 magnet_dumbbell_y         618.9603
```


## Cross-validation

Random forest algorithms perform cross validation as an integral part
of the training process, so it is not necessary to add an extra
cross-validation step.


## Out-of-sample error



The random forest algorithm provides predictions for all rows in the
training data set, computed using only out-of-bag samples.
The out-of-bag error rate is computed for each observation.
Its minimum, mean, and maximum values are

```
## [1] 0.001280082
```

```
## [1] 0.01006085
```

```
## [1] 0.1281139
```
The confusion matrix shows how accurately can we predict a particular result,
how many times (and how) we get it wrong, and the classification error.
It is also based on out-of-bag samples:

```
##      A    B    C    D    E class.error
## A 3900    4    1    0    1 0.001536098
## B   22 2629    6    1    0 0.010910459
## C    0   16 2373    7    0 0.009599332
## D    0    0   30 2219    3 0.014653641
## E    0    1    6    9 2509 0.006336634
```

```
##      A    B    C    D    E class.error
## A 3900    4    1    0    1 0.001536098
## B   22 2629    6    1    0 0.010910459
## C    0   16 2373    7    0 0.009599332
## D    0    0   30 2219    3 0.014653641
## E    0    1    6    9 2509 0.006336634
```
To check the accuracy of these estimates, we compute a confusion matrix
"by hand" by applying our prediction algorithm to the test data
(that 30% of all data that we reserved at the beginning).

```r
cm <- table(p1, testY)
cerr <- numeric(5)
for (r in 1:5) {
    cerr[r] <- 1 - cm[r, r]/sum(cm[r,])
}
cm <- cbind(cm ,cerr)
colnames(cm)[6] <- "class.error"
print(cm)
```

```
##      A    B    C   D    E class.error
## A 1674    6    0   0    0 0.003571429
## B    0 1128    2   1    0 0.002652520
## C    0    5 1017   9    1 0.014534884
## D    0    0    7 952    5 0.012448133
## E    0    0    0   2 1076 0.001855288
```
Classification errors can be up to a factor of two greater in this
confusion matrix than in the one computed automatically by the algorithm,
but nevertheless never become higher than about 1%.


## Predictions

The following plot shows predicted values for all the testing data.
Different classes are given different colors;
right and wrong predictions are distinguished by different shapes.
<img src="figure/unnamed-chunk-8-1.png" title="plot of chunk unnamed-chunk-8" alt="plot of chunk unnamed-chunk-8" style="display: block; margin: auto;" />
The fact that one class can be found in several different places in this plot
shows that these two variables are insufficient by themselves to produce
a prediction.
Choosing a different pair of variables gives a fresh perspective:
<img src="figure/unnamed-chunk-9-1.png" title="plot of chunk unnamed-chunk-9" alt="plot of chunk unnamed-chunk-9" style="display: block; margin: auto;" />


## Final comments

To improve interpretability and reduce overfitting,
it would be desirable to restrict the number of features included in the model
to the bare minimum necessary to achieve an acceptable prediction accuracy.

The algorithms in the `RRF` (Regularized Random Forest) package offer
automated feature selection, which may help reduce the total number of features.
Unfortunately, use of `method = "RRF"` in our call to the `train` function
always resulted in R crashing.

