####################################################################
#################### Project 1: the Bank Marketing Data Set
#################### Robert Carausu & Marc Vila, 2018
####################################################################

## Direct marketing campaigns (phone calls) of a Portuguese banking institution. 
## The classification goal is to predict if the client will subscribe a term deposit
library(reshape2)
library(ggplot2)

## Getting the dataset
set.seed (6046)
deposit <- read.table(file="./data/bank.csv", header=TRUE, stringsAsFactors=TRUE, sep=";")
# We rename the target variable
colnames(deposit)[ncol(deposit)] <- "subscribed"
original_data = deposit # We make a copy to compare it later with our pre-processed data

# 45211 observations and 17 different variables 
# (9 categorical: job, marital, education, default, housing, loan, contanct, month, poutcome and y)
dim(deposit)
summary(deposit)
# 11.70% of subscribed
sum(deposit$subscribed=="yes")/sum(length(deposit$subscribed))*100

## Let's have a visual inspection of the continuous variables before pre-processing
# Age seems ok
# The other variables are highly skeewed so we will try to scale and apply log where we can
# We can do it for duration, not for balance since it has negative values and we don't want to lose data
# pdays can be converted to categorical: "not_contacted" (in previous campaign) and "contacted"
d.cont <- melt(deposit[,c("age","balance","duration","campaign","pdays","previous")])
ggplot(d.cont,aes(x = value)) + facet_wrap(~variable,scales = "free") + geom_histogram() + theme(axis.text.x=element_text(angle=90,hjust=1,vjust=0.5))

## Let's have a visual inspection of the factor variables before pre-processing
# They seem ok so we won't be touching these variables
d.categ <- melt(deposit, measure.vars=c("job","marital","education","housing","loan","contact","default", "poutcome"))
ggplot(d.categ,aes(x = value)) + facet_wrap(~variable,scales = "free") + geom_bar() + theme(axis.text.x=element_text(angle=90,hjust=1,vjust=0.5))

# This dataset needs a lot of pre-processing ... also it displays a good mixture of categorical and numeric variables
# In conclusion LDA/QDA may not the best choice so the best method to use is Logistic Regression

                                                      #### PRE-PROCESSING ####

# The duration of the contact duration is highy skewed so we fix it applying log
hist(deposit$duration)
hist(log(deposit$duration))
deposit$duration = log(deposit$duration+0.001) # +0.001 to avoid -Inf

# The balance is also highly skewed and it has negative values
hist(deposit$balance)
# There are 3766 for negative balance
# The only way to fix it is to delete this observations so we choose to leave it as it is since we don't want to lose data
# Is it better to delete them?
sum(deposit$balance<0)

# pdays has most of values -1 (not contacted previously). 
# We make a categorical value with "contacted" for pdays!=-1 and "not contacted" previously for pdays=-1
hist(deposit$pdays)
deposit$pdays = cut(deposit$pdays, breaks=c(-Inf, 0.0, Inf), labels=c("not_contacted", "contacted"))
table(deposit$pdays)
plot(deposit$pdays)

#### Fixing "unknown" values

# There are 288 subscriptions for unknown job, we leave it as it is since we don't want to delete this data
summary(deposit[deposit$job=="unknown",]) 

# We plot again after pre-processing
d.cont <- melt(deposit[,c("age","balance","duration","campaign","previous")])
ggplot(d.cont,aes(x = value)) + facet_wrap(~variable,scales = "free") + geom_histogram() + theme(axis.text.x=element_text(angle=90,hjust=1,vjust=0.5))

# Now pdays is categorical
d.categ <- melt(deposit, measure.vars=c("job","marital","education","housing","loan","contact","default", "poutcome", "pdays"))
ggplot(d.categ,aes(x = value)) + facet_wrap(~variable,scales = "free") + geom_bar() + theme(axis.text.x=element_text(angle=90,hjust=1,vjust=0.5))

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
# LOGISTIC REGRESSION
# We use Logistic Regression as recommended since it doesn't need a lot of preprocessing of the data and we also have a lot of categorical variables

# ORIGINAL DATA
# First aproximation with the original unchanged data & all variables
glm.fit = glm(subscribed~., data=original.learn.data, family="binomial")

# Observing the p-values, we can have an idea of the variables that have more importance in predicting our model,
# so we can fit the mode again with just the variable that actually have an influence on our model
# We can discard the following since they affect our model less: age, job, marital, default, balance and poutcome
summary(glm.fit)

# We calculate the prediction with and without the discarded variables and compare the errors

glm.probs=predict(glm.fit, original.test.data, type="response")
glm.pred=rep("no", length(glm.probs))

glm.pred[glm.probs>.5]="yes"

# We can see that our model is pretty accurate (90.07%) in general, but it has a low accuracy for predicting
# positive subscriptions (597/(597+1151))*100=34,15%. If we lower our threshold for predicting positive subscription to 0.2,
# then the accuracy decreases to 88.30%, but false negatives are reduces so the accuracy for predicting subscriptions goes up to
# (1155(593+1155))*100=66.08%, which is a much better value. This may make more sense for the bank since they can spend their effort
# on contacting clients that have a higher probability of buying the finantial product.
# It is also important to mention that lowering the threshold will increment the number of false positives, in this case from
# 345 to 1170, so we need more domain knowledge to know if this is acceptable or not by the bank
table(glm.pred, original.test.data$subscribed)
mean(glm.pred==original.test.data$subscribed)
mean(glm.pred!=original.test.data$subscribed)

glm.pred[glm.probs>.2]="yes"
table(glm.pred, original.test.data$subscribed)
mean(glm.pred==original.test.data$subscribed)
mean(glm.pred!=original.test.data$subscribed)

# Now we fit the model without the variables discarded before
glm.fit = glm(subscribed~.-age-job-marital-default-balance-poutcome, data=original.learn.data, family="binomial")

glm.probs=predict(glm.fit, original.test.data, type="response")
glm.pred=rep("no", length(glm.probs))

# The total accuracy decreases to 87.19% and for positive subscriptions is (1120/(1120+628))*100=64.07%,
# a slightly higher test error, so using less variables makes our model a bit less accurate
# so we will use a model taking into account all variables. If we have too many variables and computation time is important,
# we can also see that removing the ones we selected won't affect so much our model prediction
glm.pred[glm.probs>.2]="yes"
table(glm.pred, original.test.data$subscribed)
mean(glm.pred==original.test.data$subscribed)
mean(glm.pred!=original.test.data$subscribed)

# PRE-PROCESSED DATA

glm.fit = glm(subscribed~., data=learn.data, family="binomial")

glm.probs=predict(glm.fit, test.data, type="response")
glm.pred=rep("no", length(glm.probs))

glm.pred[glm.probs>.2]="yes"

# We have a total accuracy of 86.97%, which is a bit lower than with not pre-processed data, 
# but for positive subscriptions it's of (1252/(1252+496))*100=71.62%, which is significantly higher than
# the previous one, so our model is much more precise detecting clients that will probably buy the finantial product of the bank
table(glm.pred, test.data$subscribed)
mean(glm.pred==test.data$subscribed)
mean(glm.pred!=test.data$subscribed)

# DUDAS

# Deberíamos hacer el análisis LDA y QDA para comparar?
# LDA ??
# QDA ??

# Robert: estos comentarios significa que el preprocesado lo deja tal cual? O que hay que hacaer algo al respecto?
## what to do with 'pdays' and 'previous'? it is not clear ... we leave as it is
## the 'unknown' and "-1" may need some treatment
## The rest seem OK (it would take a careful analysis, and a lot of domain knowledge)
## Preparing dataset for modeling: we should consider taking log10 of other variables, scaling the continuous ones (after the eventual log10s),
## treat the '999' and 'unknown', balance the errors (if need be) ...
# Robert: no se a que se refiere con 999, he estado mirando y no veo problemas al respecto en los datos., igual es el tema de -1
# Robert: Cómo tratamos los 'unknown'?
# Robert: Qué es el escalado de variables? Que es lo que debemos hacer en concreto?

































