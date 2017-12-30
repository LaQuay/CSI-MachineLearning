####################################################################
#################### Project 1: the Bank Marketing Data Set
####################################################################

set.seed (6046)

## Direct marketing campaigns (phone calls) of a Portuguese banking institution. The classification goal is to predict if the client will subscribe a term deposit

## Getting the dataset

deposit <- read.table ("dataset/bank-full.csv", header=TRUE, stringsAsFactors=TRUE, sep=";")

dim(deposit)
summary(deposit)

## This dataset needs a lot of pre-processing ... also it displays a good mixture of categorical and numeric variables: LDA/QDA may not the best choice (better use LogReg)

# 'age' has 77 values, seems OK, age and number of people with each 'age'
table(deposit$age)

# 'job' has 12 values, seems OK
# 288 UNKNOWN
table(deposit$job)

# 'education' has 4 values, seems OK
# 4 UNKNOWN
table(deposit$education)

# 'month' has 12 values, looks very suspicious ... but is OK
table(deposit$month)

# 'duration' has more than 500 values, highly skewed ...
hist(deposit$duration)
hist(log(deposit$duration))

deposit$duration <- log(deposit$duration+0.001)

# 'pdays' and 'previous'? it is not clear... we leave as it is

## the 'unknown' and "-1" may need some traatment

## The rest seem OK (it would take a careful analysis, and a lot of domain knowledge)

# Rename the target value from 'y' to 'subscribed'

colnames(deposit)[ncol(deposit)] <- "subscribed"

dim(deposit)
summary(deposit)

## Let's have a visual inspection 

library(reshape2)
library(ggplot2)

## Visual inspection of the continuous variables

d.cont <- melt(deposit[,c("age","balance","duration","campaign","pdays","previous")])
ggplot(d.cont,aes(x = value)) + 
  facet_wrap(~variable,scales = "free") + geom_histogram()

## Visual inspection of the factor variables

d.categ <- melt(deposit, measure.vars=c("job","marital","education","housing","loan","contact","default", "poutcome"))
ggplot(d.categ,aes(x = value)) + facet_wrap(~variable,scales = "free") + geom_bar()

## Preparing dataset for modeling, we should consider:
# taking log10 of other variables,
# scaling the continuous ones (after the eventual log10s),
# treat the '999' and 'unknown', 
# balance the errors (if need be) ...

## Since we want to use different methods, we need CV and a separate test set:

N <- nrow(deposit)
all.indexes <- 1:N

learn.indexes <- sample(1:N, round(2*N/3))
test.indexes <- all.indexes[-learn.indexes]

learn.data <- deposit[learn.indexes,]

nlearn <- length(learn.indexes)
ntest <- N - nlearn

...

### Marc:

### Naive Bayes Classifier ###

library (e1071)
set.seed (6046)

deposit <- read.table ("dataset/bank-full.csv", header=TRUE, stringsAsFactors=TRUE, sep=";")

## Classify JOB - Marital
depJobMar <- deposit[c("job", "marital")]
colnames(depJobMar) <- c("job", "marital")

summary(depJobMar)

N <- nrow(depJobMar)

## We first split the available data into learning and test sets, 
## selecting randomly 2/3 and 1/3 of the data
## We do this for a honest estimation of prediction performance

learn <- sample(1:N, round(2*N/3))
nlearn <- length(learn)
test <- -learn
ntest <- N - nlearn

modelJobMar <- naiveBayes(job ~ marital, data = depJobMar[learn,])
modelJobMar

pred <- predict(modelJobMar, depJobMar[-learn,])

dfJobMar <- data.frame(prediction=pred, actual=depJobMar[-learn,]$job)
table(dfJobMar)

## Classify JOB - Education
depJobEdu <- deposit[c("job", "education")]
colnames(depJobEdu) <- c("job", "education")

summary(depJobEdu)

N <- nrow(depJobEdu)

## We first split the available data into learning and test sets, 
## selecting randomly 2/3 and 1/3 of the data
## We do this for a honest estimation of prediction performance

learn <- sample(1:N, round(2*N/3))
nlearn <- length(learn)
test <- -learn
ntest <- N - nlearn

modelJobEdu <- naiveBayes(job ~ education, data = depJobEdu[learn,])
modelJobEdu

pred <- predict(modelJobEdu, depJobEdu[-learn,])

dfJobEdu <- data.frame(prediction=pred, actual=depJobEdu[-learn,]$job)
table(dfJobEdu)
