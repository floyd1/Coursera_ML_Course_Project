# Human Activity Recognition ML Project
Vladimir Goldin  
August 22, 2015  

### Background

Six young healthy participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E).  
Read more: <http://groupware.les.inf.puc-rio.br/har#ixzz3jcYf6pTX>

In this project, we will attempt to use the HAR dataset to correctly classify the fashion in which the Unilateral Dumbbell Biceps Curl exercise was performed.

### Load the data


```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
pml_full <- read.csv("pml-training.csv")
```

### Feature Extraction

`str(pml_full)` reveals that many features have `NA` or `#DIV/0!` values. Furthermore, some features are irrelevant for the prediction model (such as features `X`, `user_name`, etc.). And upon closer inspection, many others are derived from the collected instrument data, such as `avg`, `stddev`, etc. 

So, to complete the feature extraction, we will keep only the primary instrument data, with the following derivative data removed: `avg`, `min`, `max`, `var`, `stddev`, `amplitude`, `kurtosis`, `skewness`


```r
pml <- pml_full[,-c(1:7, grep("amplitude|min|max|avg|var|stddev|skewness|kurtosis", names(pml_full), ignore.case = TRUE))]

cols <- ncol(pml)
```

As a result, of the 160 columns in the original dataset, we are left with 53 columns that we'll be using to build our prediction model.

### Partitioning the Data

We will partition the original dataset into training and testing sets, 75% and 25%, respectively.


```r
set.seed(10505)
inTrain <- createDataPartition(y = pml$classe, p = .75, list = FALSE)
training <- pml[inTrain,]
testing <- pml[-inTrain,]
```

### Building a k-Nearest Neighbor Model with Cross-Validation

We've attempted to fit the data using the Partial Least Squares model, which yielded an accuracy rate of only 38%. With such a dismal result, we've decided to abandon the linear regression route altogether.

Instead, since we are dealing with a classification problem, the kNN Model seemed like a sensible choice.


```r
knnFit <- train(training[,1:(cols-1)], training[,cols],
                 method = "knn",
                 preProcess = c("center", "scale"),
                 tuneLength = 10,
                 trControl = trainControl(method = "cv"))

knnFit
```

```
## k-Nearest Neighbors 
## 
## 14718 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## Pre-processing: centered, scaled 
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 13246, 13246, 13247, 13245, 13245, 13248, ... 
## Resampling results across tuning parameters:
## 
##   k   Accuracy   Kappa      Accuracy SD  Kappa SD   
##    5  0.9629719  0.9531520  0.003884553  0.004923691
##    7  0.9506748  0.9375803  0.005222228  0.006619933
##    9  0.9402109  0.9243324  0.007110269  0.009007607
##   11  0.9311740  0.9128889  0.007601139  0.009646022
##   13  0.9207772  0.8997230  0.007201140  0.009152958
##   15  0.9126921  0.8894854  0.008346299  0.010601282
##   17  0.9025027  0.8765767  0.008645326  0.010977378
##   19  0.8948922  0.8669469  0.008548272  0.010849077
##   21  0.8878939  0.8580893  0.007397837  0.009397475
##   23  0.8800806  0.8481948  0.007578669  0.009620575
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was k = 5.
```

For k = 5, the accuracy was 96.3%.

### Cross-Validation Plot


```r
plot(knnFit)
```

![](pml-project_files/figure-html/unnamed-chunk-5-1.png) 

The plot clearly shows that for k > 5, the model results in progressively worsening accuracy. This makes sense since our dataset consists of only 5 classes: {A, B, C, D, E}

### Predicting on the `testing` set


```r
test_knnFit <- predict(knnFit, newdata = testing[,1:(cols-1)])
(conMat <- confusionMatrix(test_knnFit, testing[,cols]))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1378   33    2    2    1
##          B    7  885    4    0    7
##          C    2   27  833   26    6
##          D    4    0   14  774    6
##          E    4    4    2    2  881
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9688          
##                  95% CI : (0.9635, 0.9735)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9605          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9878   0.9326   0.9743   0.9627   0.9778
## Specificity            0.9892   0.9954   0.9849   0.9941   0.9970
## Pos Pred Value         0.9732   0.9801   0.9318   0.9699   0.9866
## Neg Pred Value         0.9951   0.9840   0.9945   0.9927   0.9950
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2810   0.1805   0.1699   0.1578   0.1796
## Detection Prevalence   0.2887   0.1841   0.1823   0.1627   0.1821
## Balanced Accuracy      0.9885   0.9640   0.9796   0.9784   0.9874
```

The Confusion Matrix shows an accuracy of 96.9% on the `testing` dataset. And the error rate is 3.1%.
