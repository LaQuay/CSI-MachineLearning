####################################################################
#################### Project 1: the Bank Marketing Data Set
####################################################################

set.seed (6046)

## Direct marketing campaigns (phone calls) of a Portuguese banking institution. The classification goal is to predict if the client will subscribe a term deposit

## Getting the dataset
deposit <- read.table(file="./data/bank.csv", header=TRUE, stringsAsFactors=TRUE, sep=";")

dim(deposit)

summary(deposit)

## This dataset needs a lot of pre-processing ... also it displays a good mixture of categorical and numeric variables: LDA/QDA may not the best choice (better use LogReg)

# age seems OK
# job has 12 values, let's check their frequency

# seems OK
table(deposit$job)

# education has 4 values, let's check their frequency

# seems OK
table(deposit$education)

# month looks very suspicious ... but is OK

table(deposit$month)


# Robert checking poutcome
table(deposit$poutcome) 

hist(deposit$duration) # highly skewed ...
hist(log(deposit$duration)) # fixing the skew?

deposit$duration <- log(deposit$duration+0.001)

# what to do with 'pdays' and 'previous'? it is not clear ... we leave as it is

## the 'unknown' and "-1" may need some treatment

## The rest seem OK (it would take a careful analysis, and a lot of domain knowledge)

# I rename the target ...

colnames(deposit)[ncol(deposit)] <- "subscribed"

dim(deposit)

summary(deposit)

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

nlearn <- length(learn.indexes)
ntest <- N - nlearn


