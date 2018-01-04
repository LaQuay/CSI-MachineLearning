####################################################################
#################### Project 1: the Bank Marketing Data Set
#################### Robert Carausu & Marc Vila, 2018
####################################################################

## Direct marketing campaigns (phone calls) of a Portuguese banking institution. 
## The classification goal is to predict if the client will subscribe a term deposit

## Getting the dataset
set.seed (6046)
deposit <- read.table(file="./data/bank.csv", header=TRUE, stringsAsFactors=TRUE, sep=";")
attach(deposit)

# 45211 observations and 17 different variables 
# (9 categorical: job, marital, education, default, housing, loan, contanct, month, poutcome and y)
dim(deposit) 
summary(deposit)

# We rename the target variable
colnames(deposit)[ncol(deposit)] <- "subscribed"

## This dataset needs a lot of pre-processing ... also it displays a good mixture of categorical and numeric variables: LDA/QDA may not the best choice (better use LogReg)

table(deposit$job)
table(deposit$education)
table(deposit$month)

# The duration of the contact duration is highy skewed so we fix it applying log
hist(deposit$duration)
hist(log(deposit$duration))
deposit$duration = log(deposit$duration)

# pdays has most of values -1 (not contacted previously). Maybe we can make a categorical value with contacted or not contacted previously
# x = deposit[!(deposit$pdays==-1),]
# hist(x$pdays)
# hist(log(x$pdays))

#### Fixing "unknown" values

summary(deposit[(deposit$job=="unknown" & deposit$subscribed=="yes"),]) # There are 34 yes for unknown job, maybe we can remove these observations?
# deposit2=deposit[!(deposit$job=="unknown"),]

# deposit$duration <- log(deposit$duration+0.001) # Robert: porque le suma 0.001?? Lo descarto porque no lo entiendo

# what to do with 'pdays' and 'previous'? it is not clear ... we leave as it is

## the 'unknown' and "-1" may need some treatment

## The rest seem OK (it would take a careful analysis, and a lot of domain knowledge)


## Let's have a visual inspection of the continuous variables

library(reshape2)
library(ggplot2)

 d.cont <- melt(deposit[,c("age","balance","duration","campaign","pdays","previous")])
 ggplot(d.cont,aes(x = value)) + 
  facet_wrap(~variable,scales = "free") + geom_histogram()

## Let's have a visual inspection of the factor variables

 d.categ <- melt(deposit, measure.vars=c("job","marital","education","housing","loan","contact","default", "poutcome"))
 ggplot(d.categ,aes(x = value)) + facet_wrap(~variable,scales = "free") + geom_bar()

## Preparing dataset for modeling: we should consider taking log10 of other variables, scaling the continuous ones (after the eventual log10s),
## treat the '999' and 'unknown', balance the errors (if need be) ...
# Robert: no se a que se refiere con 999, he estado mirando y no veo problemas al respecto en los datos., igual es el tema de -1
# Los unknown sí que habría que pensar como tratarlos.

## Since we want to use different methods, we need CV and a separate test set:


N <- nrow(deposit)
all.indexes <- 1:N

learn.indexes <- sample(1:N, round(2*N/3))
test.indexes <- all.indexes[-learn.indexes]

learn.data <- deposit[learn.indexes,]
test.data <- deposit[test.indexes,]

nlearn <- length(learn.indexes)
ntest <- N - nlearn

#attach(deposit)

glm.fit = glm(subscribed~., data=learn.data, family=binomial)
# Según el artículo del estudio, que podemos mencionar, hay 6 valores que destacan por encima de los demás:
# duration, month, previous, pdays, poutcome (First contact duration no lo tenemos en el dataset)
glm.fit = glm(subscribed~duration+month+previous+pdays+poutcome, data=learn.data, family=binomial)

# Si miramos las variables que tienen el coeficiente positivo y el p-value pequeño...
glm.fit = glm(subscribed~marital+education+default+balance+month+previous+poutcome, data=learn.data, family=binomial)
summary(glm.fit)

glm.probs=predict(glm.fit, test.data, type="response")
glm.pred=rep("no", length(glm.probs))

glm.pred[glm.probs>.5]="yes"

summary(test.data$subscribed) # 1748 yes answers
table(glm.pred, test.data$subscribed) # 554 correct yes answers predicted by our model
298/(1194+298) # 31.70% correct prediction for positive answers
mean(glm.pred==test.data$subscribed) # 89.91% overall accuracy
mean(glm.pred!=test.data$subscribed) # 10.09% overall test error. 
# The last two numbers are so high because we take into account also the subscribe==no answer, is it a valid model?

# DUDAS
# Preprocesar datos?
# Como tratar los unknowns y el -1? Qué hacer con los log?
# Hacer los otros métodos también para comparar?
































