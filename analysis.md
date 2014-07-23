# DETERMINING TYPE OF WEIGHT LIFTING EXERCISE FROM ACTIVITY DATA
*In this document we report our analysis of activity data for determining the type of weight lifting exercise. The dataset is provided by Groupware@LES (http://groupware.les.inf.puc-rio.br/har), and it consists of nearly 20,000 observations and 160 variables obtained from 6 subjects of the experiment. Each variable represents a measurement of body activity, tracked using accelerometers attached in various parts of the body.*

*After simple data preprocessing, i.e. removing predictors that contain mostly empty values, we train an SVM classifier (*`svmLinear` *in* `caret` *package) with 5-fold cross validation. We obtain* **0.916 accuracy** *with standard deviation of* **0.0067** *as our estimate of out-of-sample error.*

## 1. Data Processing
This section describes how the data is loaded and processed.

### Load the data
From exploratory analysis, it was discovered that many numeric predictors contain NA and invalid value ('#DIV/0!'). Predictors that have at least 90% of these values are removed from the dataset.

There are also predictors that will likely not discriminate between the classes. They are:
- `X`, which is an ID / row number
- `raw_timestamp_part_2`, which is a timestamp of the activity
- `new_window`

After removing these predictors, there are 56 predictors remaining in the dataset. Boxplots of these predictors conditioned on the class variable were made (not shown), and from there we found few outlying data points. These are removed from the dataset.


**Load data and print its dimension. Also do a quick check of class distribution**

```r
x = read.csv('pml-training.csv')
dim(x)
```

```
## [1] 19622   160
```

```r
plot(x$classe)
```

![plot of chunk load](figure/load.png) 

The classes are roughly balanced, so accuracy is okay for measuring performance.


**Remove columns which are mostly empty or will likely not help in classification**

```r
c(length(unique(x$X)), length(unique(x$raw_timestamp_part_2)))
```

```
## [1] 19622 16783
```

```r
table(x$new_window, x$classe)
```

```
##      
##          A    B    C    D    E
##   no  5471 3718 3352 3147 3528
##   yes  109   79   70   69   79
```

```r
limit.rows = 0.9*nrow(x)
unused.cols = sapply(colnames(x), function(colname) {
    na = is.na(x[, colname])
    empty = x[, colname] %in% c('', '#DIV/0!')
    sum(na | empty) > limit.rows
})
unused.cols = colnames(x)[unused.cols]
unused.cols = c(unused.cols, 'new_window', 'X', 'raw_timestamp_part_2')

x = x[, !(colnames(x) %in% unused.cols)]
```


**Plot example column that has outlier, and remove the outliers**

```r
# Plot example column that has outlier
plot(x[,36] ~ x$classe)
```

![plot of chunk remove outliers](figure/remove outliers.png) 

```r
unused.rows = c(5373, 9274, 16025, 16026, 17956)
x = x[-unused.rows,]
dim(x)
```

```
## [1] 19617    57
```

### Train classification model
After data preprocessing, we train a classification model. Linear SVM is used with 5-fold cross validation.


```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
n.fold = 5
folds.train = createFolds(y=x$classe, k=n.fold, list=T, returnTrain=T)
folds.test = createFolds(y=x$classe, k=n.fold, list=T, returnTrain=F)

result = rep(NA, n.fold)
for (i in 1:n.fold) {
    x.train = x[folds.train[[i]], ]
    x.test = x[folds.test[[i]], ]
    
    mod = train(classe ~ ., data=x.train, method='svmLinear')
    preds = predict(mod, x.test)
    result[i] = confusionMatrix(preds, x.test$classe)$overall['Accuracy']
}
```

```
## Loading required package: kernlab
```

```
## Warning: package 'kernlab' was built under R version 3.1.1
```


## 2. Results

### Print the result from cross validation


```r
mean(result)
```

```
## [1] 0.916
```

```r
sd(result)
```

```
## [1] 0.006701
```

### Retrain with the whole training set and predict for test set

```r
mod.final = train(classe ~ ., data=x, method='svmLinear')

x.val = read.csv('pml-testing.csv')
x.val = x.val[, !(colnames(x.val) %in% unused.cols)]
x.val = x.val[, -dim(x)[2]]

preds.val = predict(mod.final, x.val)
```
These predictions are used for course submission.
