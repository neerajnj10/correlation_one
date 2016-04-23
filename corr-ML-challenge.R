
#read the data
library(readr)
base_path <- getwd()
data_corr <- read_csv(paste0(base_path, "/stock_returns_base150.csv"))


#separate train data with test.

training <- subset(data_corr, data_corr$S1 != "NA")
sub<-as.numeric(rownames(training)) # because rownames() returns character
test <- data_corr[-sub,]

#remove trailing white spaces in data observations. 
test <- subset(test, test$date != "NA")

#load the library
library(glmnet)
set.seed(123)

#extract the target variable.
target <- training$S1




#applying elastic net regularization model to include both type of penalties(lasso and ridge regression).

Var <- cv.glmnet(x = as.matrix(training[,c(3:11)]),
                 y = target,
                 family = "gaussian",
                 alpha = 0.8, #elastic net to include both l1 and l2 penalty for regularization.
                 #dfmax = 150, 
                 nfolds = 5,
                 type.gaussian = "covariance",
                 type.measure="deviance")

#Now the main part.

################Variable Selection##################

#we can select lambda.min or lambda.1se for it, lambda.min provides the coefficients for best model, while lambda.1se provides for "simple model", I chose lambda.min after carefully considering the requirements.


### using lambda.min parameter.
c1 <- coef.cv.glmnet(Var,s="lambda.min",exact=T)


#only those coefficient values that are non-zero.
inds <- which(c1!=0)
variables <-  row.names(c1)[inds]
`%ni%` <- Negate(`%in%`)


#final variables.
variables1 <- variables[variables %ni% '(Intercept)']
variables1


#now using the above obtained list of variables to subset the training set predictors to smaller number but retained predicting power. 
newtrain <- training[,variables1]
Date <- training$date
newtrain <- as.data.frame(cbind(target, Date, newtrain))


# we will use deep learning method provided in R in the form `h2o`
# it is a very powerful tool today, that makes it easier to apply and perform machine learning on variety of dataset.


library(h2o)
###
h2o.init()
###

#convert date format

newtrain$Date <-format(as.POSIXct(newtrain$Date,format='%m/%d/%Y'),format='%m/%d/%Y')
test$date <- format(as.POSIXct(test$date,format='%m/%d/%Y'),format='%m/%d/%Y')



#convert to accessible h2o format.

train.hex <- as.h2o(newtrain)
test.hex <- as.h2o(test)


###
## building gbm and learning model, and ensembling at the end by giving weights to each mode and make them a powerful piece togethe.



gbmF_model_1 = h2o.gbm( x= c(2:8),
                        y = 1,
                        training_frame =train.hex ,
                        #validation_frame =testHex ,
                        max_depth = 3,
                        distribution = "gaussian",
                        ntrees =500,
                        learn_rate = 0.05,
                        nbins_cats = 50,
                        nfolds = 5
)

gbmF_model_2 = h2o.gbm( x=c(2:8),
                        y = 1,
                        training_frame =train.hex ,
                        #validation_frame =testHex ,
                        max_depth = 3,
                        distribution = "gaussian",
                        ntrees =430,
                        learn_rate = 0.03,
                        nbins_cats = 60
)


dl_model_1 = h2o.deeplearning( x=c(2:8),
                               y = 1,
                               training_frame =train.hex ,
                               #validation_frame =testHex ,
                               activation="Rectifier",
                               hidden=6,
                               epochs=60,
                               adaptive_rate =T
)


dl_model_2 = h2o.deeplearning( x=c(2:8),
                               # x=feature,
                               y = 1,
                               nfolds = 5,
                               fold_assignment = "Modulo",
                               training_frame =train.hex ,
                               #validation_frame =testHex ,
                               activation="TanhWithDropout",
                               hidden=60,
                               epochs=40,
                               adaptive_rate =F,
                               input_dropout_ratio = 0.02
)


dl_model_3 = h2o.deeplearning( x=c(2:8),
                               y = 1,
                               training_frame =train.hex ,
                               #validation_frame =testHex ,
                               activation="Rectifier",
                               hidden=6,
                               epochs=120,
                               adaptive_rate =F,
                               stopping_metric = "deviance",
                               single_node_mode = T,
                               hidden_dropout_ratios = 0.4
)



#creating a submission file.

MySubmission = test[, c("date", "S1")]


#Making the predictions on test set.


test_gbm_1 = as.data.frame(h2o.predict(gbmF_model_1, newdata = test.hex) )
test_gbm_2 = as.data.frame(h2o.predict(gbmF_model_2, newdata = test.hex) )

test_dl_model_1 = as.data.frame(h2o.predict(dl_model_1, newdata = test.hex) )
test_dl_model_2 = as.data.frame(h2o.predict(dl_model_2, newdata = test.hex) )
test_dl_model_3 = as.data.frame(h2o.predict(dl_model_3, newdata = test.hex) )




# ensembling and giving weights to our model.

MySubmission$S1=0.3*(test_dl_model_1$predict)+
  0.15*(test_dl_model_2$predict)+
  0.25*(test_dl_model_3$predict)+
  0.2*(test_gbm_1$predict)+
  0.1*(test_gbm_2$predict)


head(MySubmission)


####
## few alteration to colnames to fit the requirements.

## making final submission.

colnames(MySubmission) <- c("Date", "Value")

write.csv(MySubmission, "predictions.csv", row.names = F)




> Does S1 go up or down cumulatively (on an open-to-close basis) over this period?


We can see from the plot that unknown predicted value for S1 begins with a neggative value `-0.410200461` and 
from there there has been *almost* constant movement in its value for successive days, that is, it is down then 
goes up and then followed by, moving up again, with many peaks going up close to or above 1. 
Cumulativey there has been a **trend** here and it ends with S1 going **down** on the last day hitting a 
low of `-1.129400837`.





## plotting to identify whether the S1 value for test set was going up or down.

plot(MySubmission$Value,col= "BLUE", type="l", ylab="Value")
