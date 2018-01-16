####################################################################
#################### Project 1: the Bank Marketing Data Set
#################### Robert Carausu & Marc Vila, CSI - MEI 2017-2018
####################################################################
set.seed (6046)
library(reshape2)
library(ggplot2)

## Direct marketing campaigns (phone calls) of a Portuguese banking institution. 
## The classification goal is to predict if the client will subscribe a term deposit

## Getting the dataset
deposit <- read.table(file="./data/bank.csv", header=TRUE, stringsAsFactors=TRUE, sep=";")
# We rename the target variable
colnames(deposit)[ncol(deposit)] <- "subscribed"
original_data = deposit # We make a copy to compare it later with our pre-processed data

# 45211 observations and 17 different variables 
# (9 categorical: job, marital, education, default, housing, loan, contact, month, poutcome and y)
dim(deposit)
summary(deposit)
# 11.70% of subscribed, so our model sholdn't have a higher error than this
# Data is very unbalanced so some models will adjust worse than others
sum(deposit$subscribed=="yes")/sum(length(deposit$subscribed))*100

## Let's have a visual inspection of the continuous variables before pre-processing
# Age seems ok
# The other variables are highly skewed so we will try to scale and apply log where we can
# We can do it for duration, not for balance since it has negative values and we don't want to lose data
# pdays can be converted to categorical: "not_contacted" (in previous campaign) and "contacted"
d.cont <- melt(deposit[,c("age","balance","duration","campaign","pdays","previous")])
ggplot(d.cont,aes(x = value)) + facet_wrap(~variable,scales = "free") + geom_histogram() + theme(axis.text.x=element_text(angle=90,hjust=1,vjust=0.5))

## Let's have a visual inspection of the factor variables before pre-processing
# They seem ok so we won't be touching these variables
d.categ <- melt(deposit, measure.vars=c("job","marital","education","housing","loan","contact","default", "poutcome"))
ggplot(d.categ,aes(x = value)) + facet_wrap(~variable,scales = "free") + geom_bar() + theme(axis.text.x=element_text(angle=90,hjust=1,vjust=0.5))

# This dataset needs a lot of pre-processing ... also it displays a good mixture of categorical and numeric variables
# In conclusion LDA/QDA may not the best choice, a good choice may be Logistic Regression. We will test also Naive Bayes and Random Forest and 
# choose the best model that fits our problem

#### PRE-PROCESSING ####
#### Fixing skewness and scaling continuous variables
# The balance has negative values, so we can only scale it
hist(deposit$balance, col='lightskyblue', border='lightskyblue4', xlab='balance', main='balance histogram', density=50)
# There are 3766 for negative balance
# The only way to fix it is to delete this observations so we choose to leave it as it is since we don't want to lose data
sum(deposit$balance<0)
deposit$balance = scale(deposit$balance)
# Scaled balance
hist(scale(deposit$balance), col='lightskyblue', border='lightskyblue4', xlab='balance', main='balance histogram', density=50)

# duration, campaign and previous are all skewed, so we apply log and scale
hist(deposit$duration, col='lightskyblue', border='lightskyblue4', xlab='duration', main='duration histogram', density=50)
deposit$duration = log(deposit$duration+0.001) # +0.001 to avoid -Inf
hist(deposit$duration, col='lightskyblue', border='lightskyblue4', xlab='duration', main='duration histogram', density=50)
deposit$duration = scale(deposit$duration) 
hist(deposit$duration, col='lightskyblue', border='lightskyblue4', xlab='duration', main='duration histogram', density=50)

# Applying log and scale to campaign and previous has some undesired effects, so we will leave them as they are
hist(scale(log(deposit$campaign+0.001)), col='lightskyblue', border='lightskyblue4', xlab='campaign', main='scale(log(campaign)) histogram', density=50)
hist(scale(log(deposit$previous+0.001)), col='lightskyblue', border='lightskyblue4', xlab='previous', main='scale(log(previous)) histogram', density=50)

# pdays has most of values -1 (not contacted previously). 
# We make a categorical value with "contacted" for pdays!=-1 and "not contacted" previously for pdays=-1
hist(deposit$pdays, col='lightskyblue', border='lightskyblue4', xlab='pdays', main='pdays histogram', density=50)
deposit$pdays = cut(deposit$pdays, breaks=c(-Inf, 0.0, Inf), labels=c("not_contacted", "contacted"))
table(deposit$pdays)
plot(deposit$pdays)

#### Fixing "unknown" values

# There are 288 subscriptions for unknown job, we leave it as it is since we don't want to delete this data
summary(deposit[deposit$job=="unknown",]) 

# We could change the unknown values to NA (as well as the 0 previous contacts variables), this is useful if we use a Random Forest algorythm,
# but since it is not the case we leave it as it is

# We plot again after pre-processing
d.cont <- melt(deposit[,c("age","balance","duration","campaign","previous")])
ggplot(d.cont,aes(x = value)) + facet_wrap(~variable,scales = "free") + geom_histogram() + theme(axis.text.x=element_text(angle=90,hjust=1,vjust=0.5))

# Now pdays is categorical
d.categ <- melt(deposit, measure.vars=c("job","marital","education","housing","loan","contact","default", "poutcome", "pdays"))
ggplot(d.categ,aes(x = value)) + facet_wrap(~variable,scales = "free") + geom_bar() + theme(axis.text.x=element_text(angle=90,hjust=1,vjust=0.5))

library(caret)
library(MASS)
library(e1071)
library(randomForest)

# PREPARING THE TRAINING AND TEST DATA
## Since we want to use different methods, we need CV and a separate test set:

N <- nrow(deposit)
all.indexes <- 1:N

learn.indexes <- sample(1:N, round(2*N/3))
test.indexes <- all.indexes[-learn.indexes]

learn.data <- deposit[learn.indexes,]
original.learn.data <- original_data[learn.indexes,]
test.data <- deposit[test.indexes,]
original.test.data <- original_data[test.indexes,]

nlearn <- length(learn.indexes)
ntest <- N - nlearn

#### MODELLING ####
############ LOGISTIC REGRESSION ##########
# We use Logistic Regression as recommended since it doesn't need a lot of preprocessing of the data and we also have a lot of categorical variables

# ORIGINAL DATA
# First aproximation with the original unchanged data & all variables
glm.fit = glm(subscribed~., data=original.learn.data, family="binomial")

# Observing the p-values, we can have an idea of the variables that have more importance in predicting our model,
# a low p-value indicates that we can reject the null hipotesis, thus that variable has an importance on our model,
# a higher p-value means that we can discard that variable
# so we can fit the mode again with just the variable that actually have an influence on our model
# We can discard the following since they affect our model less: age, job, marital, default, balance, pdays and previous
summary(glm.fit)

# We calculate the prediction with and without the discarded variables and compare the errors
glm.probs=predict(glm.fit, original.test.data, type="response")
glm.pred=rep("no", length(glm.probs))
glm.pred[glm.probs>.5]="yes"

# We choose 3 values to represent our model performance: accuracy, error and precision, the last one is important
# because the bank wants to contact only those clients that are more probable to subscribe to the loan
# We can see that accuracy is high (90.07 %), but precission is low (34.15%), to solve this we lower the threshold
# for which a client may subscribe a loan (the probability) compare the values again, since for the bank clients
# with 30% probability of subscribing is probably worth to spend it's ressources contacting them
res.performance = table(glm.pred, original.test.data$subscribed)
res.accuracy = (res.performance[2,2]+res.performance[1,1])/sum(res.performance)*100
res.error = 100-res.accuracy
res.precision = (res.performance[2,2])/(res.performance[2,2]+res.performance[1,2])*100

# Accuracy is slightly lower (90.03%) but precission has almost doubled (53.78%)
glm.pred[glm.probs>.3]="yes"
res.performance = table(glm.pred, original.test.data$subscribed)
res.accuracy = (res.performance[2,2]+res.performance[1,1])/sum(res.performance)*100
res.error = 100-res.accuracy
res.precision = (res.performance[2,2])/(res.performance[2,2]+res.performance[1,2])*100

# Now we fit the model without the variables that had less of an impact
glm.fit = glm(subscribed~.-age-job-marital-default-balance-pdays-previous, data=original.learn.data, family="binomial")
glm.probs=predict(glm.fit, original.test.data, type="response")
glm.pred=rep("no", length(glm.probs))

# The total accuracy decreases to 89.93% and precission to 53.32%, so using less variables makes our model a bit less accurate
# but the difference is really small so it's not really important to discard those variables
# If we have too many variables and computation time is important,
# we can also see that removing the ones we selected won't affect so much our model prediction
glm.pred[glm.probs>.3]="yes"
res.performance = table(glm.pred, original.test.data$subscribed)
res.accuracy = (res.performance[2,2]+res.performance[1,1])/sum(res.performance)*100
res.error = 100-res.accuracy
res.precision = (res.performance[2,2])/(res.performance[2,2]+res.performance[1,2])*100

# PREPROCESSED DATA
# We will fit with all the variables and also removing the ones that we mentioned before
glm.fit = glm(subscribed~., data=learn.data, family="binomial")
glm.probs=predict(glm.fit, test.data, type="response")
glm.pred=rep("no", length(glm.probs))

# Accuracy is 89.40% and precission is 57.95%, so our model is much more precise detecting clients 
# that will probably buy the finantial product of the bank with the preprocessed data
glm.pred[glm.probs>.3]="yes"
res.performance = table(glm.pred, original.test.data$subscribed)
res.accuracy = (res.performance[2,2]+res.performance[1,1])/sum(res.performance)*100
res.error = 100-res.accuracy
res.precision = (res.performance[2,2])/(res.performance[2,2]+res.performance[1,2])*100

# Now we fit the model without the variables that had less of an impact
glm.fit = glm(subscribed~.-age-job-marital-default-balance-pdays-previous, data=learn.data, family="binomial")
glm.probs=predict(glm.fit, test.data, type="response")
glm.pred=rep("no", length(glm.probs))

# Accuracy: 89.47%, precision: 57.44%
# As before, there is a small reduction in accuracy and precision but the results with preprocessed data are better
glm.pred[glm.probs>.3]="yes"
res.performance = table(glm.pred, original.test.data$subscribed)
res.accuracy = (res.performance[2,2]+res.performance[1,1])/sum(res.performance)*100
res.error = 100-res.accuracy
res.precision = (res.performance[2,2])/(res.performance[2,2]+res.performance[1,2])*100

#### To get a better grasp at the performance of our model, we do k-fold cross validation
precision <- NULL
accuracy <- NULL
error <- NULL
k <- 100

# It may take a while to compute
for (i in 1:k) 
{
  N <- nrow(deposit)
  all.indexes <- 1:N
  
  # we choose 9/10s of the data as training data and the rest as test data
  learn.indexes <- sample(1:N, round(9*N/10))
  test.indexes <- all.indexes[-learn.indexes]
  
  learn.data <- deposit[learn.indexes,]
  test.data <- deposit[test.indexes,]
  
  nlearn <- length(learn.indexes)
  ntest <- N - nlearn
  glm.fit = glm(subscribed ~ ., data=learn.data, family="binomial")
  #glm.fit = glm(subscribed ~ .-age-job-marital-default-balance-pdays-previous, data=learn.data, family="binomial")
  glm.probs=predict(glm.fit, test.data, type="response")
  
  glm.pred=rep("no", length(glm.probs))
  glm.pred[glm.probs>.3]="yes"
  
  res.performance = table(glm.pred, test.data$subscribed)
  accuracy[i] <- (res.performance[2,2]+res.performance[1,1])/sum(res.performance)*100
  error[i] <- 100-accuracy[i]
  precision[i] <- (res.performance[2,2])/(res.performance[2,2]+res.performance[1,2])*100
}

# We can see that our model performs pretty well, even though the data is highly unbalanced
# Mean values with all the variables
# accuracy: 89.69%
# error: 10.31%
# precision: 58.80 %

# Mean values without the variables that influence less our model (swap the commented code in the previous bucle)
# accuracy: 89.74%
# error: 10.26%
# precision: 58.37 %
mean(accuracy)
mean(error)
mean(precision)

par(mfrow=c(1,3))
hist(accuracy, col='lightskyblue', border='lightskyblue4', xlab='Acuracy', main='Acuracy for CV', density=50)
hist(error, col='lightskyblue', border='lightskyblue4', xlab='Error', main='Error for CV', density=50)
hist(precision, col='lightskyblue', border='lightskyblue4', xlab='Precision', main='Precision for CV', density=50)

boxplot(accuracy, horizontal=T, col='lightskyblue', border='lightskyblue4', xlab='Acuracy', main='Acuracy for CV')
boxplot(error, horizontal=T, col='lightskyblue', border='lightskyblue4', xlab='Error', main='Error for CV')
boxplot(precision, horizontal=T, col='lightskyblue', border='lightskyblue4', xlab='Precision', main='Precision for CV')
dev.off()

# To compare the performance of our model we will also model with LDA and QDA and analyze their performances.
# Also we will test NaiveBayes and RandomForest
N <- nrow(deposit)
all.indexes <- 1:N

learn.indexes <- sample(1:N, round(2*N/3))
test.indexes <- all.indexes[-learn.indexes]

learn.data <- deposit[learn.indexes,]
test.data <- deposit[test.indexes,]

nlearn <- length(learn.indexes)
ntest <- N - nlearn

#################### LDA ##################
# With LDA the precision is much lower, so we won't be using this model
# 10.73% error, 89.27% accuracy, 28.51% precision
lda.fit = lda(subscribed ~ ., data=learn.data)
lda.pred = predict(lda.fit, test.data)
lda.class = lda.pred$class

res.performance = table(lda.class, test.data$subscribed)
res.accuracy = (res.performance[2,2]+res.performance[1,1])/sum(res.performance)*100
res.error = 100-res.accuracy
res.precision = (res.performance[2,2])/(res.performance[2,2]+res.performance[1,2])*100

#################### QDA ##################
# Performs worse than LDA, but the precision is a bit higher so it detects better the subscriptions
# We confirm that both LDA and QDA are not suitable models to fit our problem
# 13.43% error, 86.57% accuracy, 37.99% precision
qda.fit <- qda(subscribed ~ ., data=learn.data)
qda.pred = predict(qda.fit, test.data)
qda.class = qda.pred$class

res.performance = table(qda.class, test.data$subscribed)
res.accuracy = (res.performance[2,2]+res.performance[1,1])/sum(res.performance)*100
res.error = 100-res.accuracy
res.precision = (res.performance[2,2])/(res.performance[2,2]+res.performance[1,2])*100

################# NAIVE BAYES ##################
# It performs better than LDA and QDA, but worse than logistic regression
# 12.57% error, 87.43% accuracy, 43.89% precision
bayes.fit <- naiveBayes(subscribed ~ ., data = learn.data)
bayes.pred <- predict(bayes.fit, test.data)

res.performance = table(bayes.pred, test.data$subscribed)
res.accuracy = (res.performance[2,2]+res.performance[1,1])/sum(res.performance)*100
res.error = 100-res.accuracy
res.precision = (res.performance[2,2])/(res.performance[2,2]+res.performance[1,2])*100

#################### Random Forest ##################
# 9.30% error, 90.70% accuracy, 64.66% precision, so far the best method
rf <- randomForest(subscribed ~ ., data = original_data[learn.indexes,], ntree=300, proximity=FALSE)
rf.pred <- predict(rf, newdata=original_data[-learn.indexes,])

res.performance = table(Truth=original_data[-learn.indexes,]$subscribe,Pred=rf.pred)
res.accuracy = (res.performance[2,2]+res.performance[1,1])/sum(res.performance)*100
res.error = 100-res.accuracy
res.precision = (res.performance[2,2])/(res.performance[2,2]+res.performance[1,2])*100

#### As with logistic regresion we do k-fold CV to confirm our model accuracy and precision
# We choose k=10 since randomForest has a high computation time
precision <- NULL
accuracy <- NULL
error <- NULL
k <- 10

# It may take quite a while to compute
for (i in 1:k) 
{
  N <- nrow(deposit)
  all.indexes <- 1:N
  
  # we choose 9/10s of the data as training data and the rest as test data
  learn.indexes <- sample(1:N, round(9*N/10))
  test.indexes <- all.indexes[-learn.indexes]
  
  nlearn <- length(learn.indexes)
  ntest <- N - nlearn
  #glm.fit = glm(subscribed ~ ., data=learn.data, family="binomial")
  rf <- randomForest(subscribed ~ ., data = original_data[learn.indexes,], ntree=300, proximity=FALSE)
  rf.pred <- predict(rf, newdata=original_data[-learn.indexes,])
  
  res.performance = table(Truth=original_data[-learn.indexes,]$subscribe,Pred=rf.pred)
  accuracy[i] <- (res.performance[2,2]+res.performance[1,1])/sum(res.performance)*100
  error[i] <- 100-accuracy[i]
  precision[i] <- (res.performance[2,2])/(res.performance[2,2]+res.performance[1,2])*100
}
# Mean values
# accuracy: 90.83%
# error: 9.17 %
# precision: 64.66%
mean(accuracy)
mean(error)
mean(precision)

par(mfrow=c(1,3))
hist(accuracy, col='lightskyblue', border='lightskyblue4', xlab='Acuracy', main='Acuracy for CV', density=50)
hist(error, col='lightskyblue', border='lightskyblue4', xlab='Error', main='Error for CV', density=50)
hist(precision, col='lightskyblue', border='lightskyblue4', xlab='Precision', main='Precision for CV', density=50)

boxplot(accuracy, horizontal=T, col='lightskyblue', border='lightskyblue4', xlab='Acuracy', main='Acuracy for CV')
boxplot(error, horizontal=T, col='lightskyblue', border='lightskyblue4', xlab='Error', main='Error for CV')
boxplot(precision, horizontal=T, col='lightskyblue', border='lightskyblue4', xlab='Precision', main='Precision for CV')
dev.off()

# To conclude, random forest is the best method, followed by logistic regression, according to the results
