library(tidyverse) #Required for dplyr and ggplot2
library(dplyr) #Required for data manipulation
library(ggplot2) #Required for data visualization 
library(GGally) #For pairwise plots
library(factoextra) #Required for PCA visualization
library(MASS) 
library(reshape2) #Required for data transformation
library(memisc) #Required for finding missing values
library(glmnet) #Required for regularization models
library(gbm) #Required for boosting model
library(caret) #Required for train-test partition and pre-processing
library(xgboost) #Required for Extreme Gradient Boosting model
library(randomForest) #Required for Random Forest model
library(pls) #Required for Principal component regression

#Importing csv file 
df <- read.csv(file = '/Users/robertocolangelo/Desktop/Data Analysis final project/WineQuality.csv')

#Taking a glimpse at the first 6 instances 
head(df)

# Investigating on the dataset dimension and size
dim(df)#4898 instances and 12 attributes/features

#Let's take a look at the dataset structure and typology of each attribute (numeric,integer,character,etc...)
str(df) #All numeric attributes

#let's compute some statistics (Quartiles,min and max values)
summary(df)

#Removing duplicates
df<-unique(df)
dim(df) #Now 3961 instances.There were 937 duplicates

#Checking null values 
is.null(df) #return FALSE -> means that there are no missing values

#defining a function for finding any type of unary column in the dataframe
Nonunary <- function(x) length(unique(x))>1

#Applying it to our dataframe
df[sapply(df,Nonunary)] 
dim(df)
#No unary columns --> returns a filtered dataframe with the same number of attributes as before

#Data exploration/visualization 

#Correlation plot between this 6 variables 
varsubset<-c('quality', 'alcohol','fixed.acidity',
               'density', 'pH','residual.sugar')
ggpairs(df[varsubset], aes(alpha=0.3))
#Alcohol seems to be one of the most correlated,together with density, with quality

#Correlation matrix between all variables 
df[,-c(12)]%>%
  cor()%>%
  melt()%>%
  ggplot(aes(Var1,Var2, fill=value))+
  scale_fill_gradient2(low = "#075AFF",
                       mid = "#FFFFCC",
                       high = "#FF0000")+
  geom_tile(color='black')+
  geom_text(aes(label=paste(round(value,2))),size=2, color='black')+
  theme(axis.text.x = element_text(vjust=0.5, angle = 65))+
  labs(title='Correlation between variables',
       x='',y='')

#Relationship between alcohol and residual sugar
ggplot(aes(y = alcohol, x = sulphates), data = df) + 
  geom_point(alpha = 0.35, pch = 1) + 
  geom_smooth(method = loess) + 
  labs(y= 'Alcohol (% by volume)',
       x= 'Sulphates',
       title= 'Relationship of Alcohol Vs. Sulphates')
#Not significant, just denser area of wines with alcoholic rate between 9 to 13 and sulphates between 0.3 and 0.6

dfplot<-df
#Violinplot of Wine quality level vs alcohol level
#Creating 4 tiers of wine quality level as a new column of the dataframe
dfplot$quality.levels <- cut(df$quality,c(2,4,6,8,10), 
                         labels = c('Low','Medium', 'High','Excellent'))
#From 2 to 4 'Low'
#From 4 to 6 'Medium'
#From 6 to 8 'High'
#From 9 on 'Excellent'
ggplot(aes(x= quality.levels, y= alcohol), data = dfplot) +
  geom_violin(trim=FALSE, fill="gray") + 
  geom_jitter( alpha = 0.3, color = 'orange ') + 
  stat_summary(fun.y = "mean", geom = "point", color = "blue", size = 4) +
  labs(x= 'Quality level',
       y= 'Alcohol (% by volume)',
       title= 'Alcohol Vs. Quality level (Violin plot)')
#Just a few instances in the top tier and while most are concentrated in the medium one (lot of wines with quality between 4 and 6)
#The blue point in the graph represents the mean value

#Plotting correlations of pH level with respect to the target 'quality' splitted in buckets
ggplot(aes(x= quality.levels, y= pH), data = dfplot) +
  geom_jitter( alpha = .3) +
  geom_boxplot( alpha = .5,color = 'red')+
  stat_summary(fun.y = "mean", geom = "point", color = "darkblue", 
               shape = 1, size = 4) +
  labs(x= 'Quality (in levels)',
       y= 'pH level',
       title= 'pH levels Vs. Quality')

#Distribution of target variable
ddata<- df$quality
distrplot<-hist(ddata, breaks=5, col="red", xlab="Quality of the wine",
                main="Distribution of Wine Quality")
xfit<-seq(min(ddata),max(ddata),length=40)
yfit<-dnorm(xfit,mean=mean(ddata),sd=sd(ddata))
yfit <- yfit*diff(distrplot$mids[1:2])*length(ddata)
lines(xfit, yfit, col="blue", lwd=2)
#Close but not equal to a Gaussian distribution. A little bit skewed on the left
#Do not really need to transform it with log function 

#MODELLING --> PREDICTING TARGET 'quality' AS CONTINUOUS 

# Train - test split
#Setting seed 
set.seed(45)
training <- df$quality %>%
  createDataPartition(p = 0.75, list = FALSE)
trainset  <- df[training, ]
xtrain<-trainset[,-12]
ytrain<-trainset[,12]
testset<- df[-training, ]
xtest<-trainset[,-12]
ytest<-trainset[,12]
#Scaling variables
xtrain <- xtrain %>% scale()
xtest <- xtest %>% scale(center=attr(xtrain, "scaled:center"), 
                            scale=attr(xtrain, "scaled:scale"))

#BASELINE MODEL
# Fitting the model on the training data
base <- lm(quality~ alcohol, data = trainset)
summary(base) 
preds<-predict(base,testset[,-12])
mean((testset$quality-preds)^2) #MSE ==0.57 terrible-->but  alcohol seems to be statistically significant p-value <0.05
#Let's see the fit
preds<-base$fitted.values
plot_baseline_regression <- tibble(Yobs=trainset$quality, YPreds = preds) %>% 
  ggplot() + 
  geom_point(aes(x = Yobs, y = YPreds, 
                 fill = I("black")), shape = 21, color = I("tomato"), size = 2) +
  geom_abline(intercept = 2, slope = 0.8, linetype = "dashed", size = 1.05, color = I("firebrick")) +
  labs(x = "Observed values", y = "Fitted values", title = "Goodness of fit") +
  theme_bw() +
  theme(text = element_text(size = 10))
plot_baseline_regression
#Clearly poor fit since  the result we want should not be continuous but discrete 

#full model (ALL INPUT VARIABLES)
lr <- lm(quality~., data = trainset)
summary(lr)#full model has adjusted Rsquared of 0.298 --> too low 
preds<-predict(lr,testset[,-12])
mean((preds-testset[,12])^2)  #MSE still 0.52

#Trying with StepAIC
#Both
modelstepaicboth<- step( lr, direction = "both", trace = F)
summary(modelstepaicboth)
mean(modelstepaicboth$residuals^2) #MSE 0.57 -> improved
#Back
modelstepaicback<- step( lr, direction = "backward", trace = F)
summary(modelstepaicback)
mean(modelstepaicback$residuals^2) #MSE 0.57
#Forward
modelstepaicforward<- step( lr, direction = "forward", trace = F)
summary(modelstepaicforward)
mean(modelstepaicforward$residuals^2) #MSE 0.57 

#It's the same with all types of steps

#total.sulfur dioxide doesn't seem to be statistically significant 
#Citric.acid has a p-value of 0.03, Let's try to remove it 
train.dataint<-trainset[,c(-7,-3)]
test.dataint<-testset[,c(-7,-3)]

#Trying interaction moodels with a subset of variables ,later we'll reuse them all
modinteraction<-lm(quality~ alcohol+I(density*alcohol)+ residual.sugar+free.sulfur.dioxide+sulphates+pH+density+volatile.acidity+fixed.acidity,data=train.dataint)
summary(modinteraction) #Adjusted Rsquared 0.2976 interaction between density and alcohol doesn't really impact 
mean(modinteraction$residuals^2) #MSE 0.57 

modinteraction2<- lm(quality~ alcohol+ residual.sugar+free.sulfur.dioxide+sulphates+pH+density+volatile.acidity+I(pH*volatile.acidity),data=train.dataint)
summary(modinteraction2)#Adjusted Rsquared 0.30
mean(modinteraction2$residuals^2) #MSE 0.57 

#Regularization models 
folds <- 5
ctrl <- trainControl(method="cv", number=folds)

#Ridge
nlambdas <- 1000
lambdas <- seq(0.001, 2, length.out = nlambdas)
trainmatrix<-as.matrix(trainset)
testmatrix<-as.matrix(testset)
x<-trainmatrix[,-12]
y<-trainmatrix[,12]
modridge <- cv.glmnet(x, y,nfolds=folds, alpha=0)
modridge <- train(x, y, method = "glmnet", trControl = ctrl,
                  tuneGrid = expand.grid(alpha = 0, 
                                         lambda = lambdas))
preds<-predict(modridge,as.matrix(testmatrix[,-12]))
mean((preds - testmatrix[,12])^2 )#MSE 0.524 it starts improving

#Lasso
lambdas <- 10^seq(2, -3, by = -.1)
nalphas <- 20
alphas <- seq(0, 1, length.out=nalphas)
ctrl <- trainControl(method="cv", number=folds)
modlasso <- train(x, y, method = "glmnet", trControl = ctrl,
                     tuneGrid = expand.grid(alpha = alphas, 
                                            lambda = lambdas))
modlasso$bestTune$alpha#Best parameters 
modlasso$bestTune$lambda
preds<-predict(modlasso,as.matrix(testmatrix [,-12]))
mean((preds - testmatrix[,12])^2) #MSE 0.522 

#Elastic net
lambda.grid<-10^seq(2,-2,length=100)
alpha.grid<-seq(0,1,length=10)
grid<-expand.grid(alpha=alpha.grid,lambda=lambda.grid)

Elnet<-train(x, y, method = "glmnet", trControl = ctrl,tuneGrid = grid)
Elnet$bestTune #Best parameters
preds<-predict(Elnet,as.matrix(testmatrix [,-12]))
mean((preds - testmatrix[,12])^2) #MSE 0.523

#Random Forest 
Rf <- randomForest(
  formula = quality~ .,
  data= trainset,
  importance=TRUE,
  ntree=1000 #Setting number of trees to train Random Forest model.The higher the number the more the time to execute the code
)
preds<-predict(Rf,testset[-12])
mean((preds - testset[,12])^2) #MSE 0.453 ,the best until now

# Gradient Boosting
boost = gbm(quality ~.,
                data = trainset,
                distribution='gaussian',
                cv.folds = 5,
                shrinkage = .1,
                n.minobsinnode = 10,
                n.trees = 500 )
summary(boost)
predboost<-predict(boost,testset[,-12])
mean((predboost - testset[,12])^2) #0.497 Random forest performed better

#Xgboost 
modxgboost = xgboost(data = trainmatrix[,-12], label = trainmatrix[,12], nrounds = 150, 
                 objective = "reg:squarederror", eval_metric = "error")

err_xg_tr = modxgboost$evaluation_log$train_error
predxgboost<-predict(modxgboost,testmatrix[,-12])
mean((predxgboost - testset[,12])^2) #0.547 

#PCA 
#Visualizing number of components needed 
pca <- princomp(trainset[,-12], 
                cor = TRUE,
                score = TRUE)
pca$loadings
summary(pca)
fviz_eig(pca) #By using elbow method it is clear that 4 components are more than enough

#Principal component regression 
pcrmodel <- pcr(quality~., data = trainset, scale = TRUE, validation = "CV")
pcrpred <- predict(pcrmodel, testset[,-12], ncomp = 4)
mean((pcrpred - testset[,12])^2) #0.60 

#Comparing results
baseline.predicted<-predict(base,testset[,-12])
lr.predicted<-predict(lr,testset[,-12])
step.forward.predicted <- predict(modelstepaicforward, testset[,-12])
step.backward.predicted <- predict(modelstepaicback, testset[,-12])
step.both.predicted <- predict(modelstepaicboth, testset[,-12])
lrinteraction1.predicted<-predict(modinteraction,test.dataint[,-12])
lrinteraction2.predicted<-predict(modinteraction2,test.dataint[,-12])
pcr.predicted <- predict(pcrmodel, testset[,-12], ncomp = 4)
lasso.predicted <- predict(modlasso, testmatrix[,-12])
Elasticnet.predicted<-predict(Elnet,testmatrix[,-12])
ridge.predicted <- predict(modridge, testmatrix[,-12])
randomforest.predicted<-predict(Rf,testset[,-12])
boost.predicted<- predict (boost,testset[,-12])
xgboost.predicted<-predict(modxgboost,testmatrix[,-12])
final_results <- data.frame(baseline.predicted,lr.predicted,step.forward.predicted, step.backward.predicted, step.both.predicted,lrinteraction1.predicted,lrinteraction2.predicted, pcr.predicted, lasso.predicted,Elasticnet.predicted, ridge.predicted, randomforest.predicted, boost.predicted,xgboost.predicted)
colnames(final_results) <- c('baseline','Multivariate LR', "Step_forwardAIC", "Step_backwardAIC", "Step_bothAIC",'Interaction LR1', 'Interaction LR2',"PCR", "Lasso",'Elastic Net',"Ridge", "RandomForest", "Boosting",'XGBoost')
head(final_results)
MSE = c()
for(i in 1:14){
  MSE <- rbind(MSE, c(names(final_results)[i], mean((testset$quality - final_results[,i])^2)))
}
MSE <- data.frame(Model = MSE[,1], MSE = as.numeric(MSE[,2]))
MSE[order(MSE$MSE),]

#Conclusions, premised that predicting the quality of wines was not a regression task but most likely a multiclass classification task,among all the models we tried the best one seems to be Random Forest with MSE =0.456, together with boosting with 0.497
#Regularization methods don't seem to impact that much on the performance of the model
#PCR is the worst because it obviously loses some of the variability and information of the dataset but we decided to do it for academic and studying purposes

#TASK 2 BAD/GOOD WINES
#Using the best model which is Random Forest
#If wine quality predicted is higher than 6 is a good wine otherwise is a bad one 
#Predicting on the whole dataset 
winepred<- ifelse(predict(Rf, df[,-12]) >= 6 , "Good Wine", "Bad Wine")
df$Winequality<- winepred
df

#There are some wines with quality==6 but the prediction is probably smaller than 6 and are classified as bad 
#Looking at the most important variables for predicting wine quality in Rf model
Featureimportance <- varImp(Rf, conditional=TRUE)

 
Featureimportance <- Featureimportance %>% tibble::rownames_to_column("var") 
Featureimportance$var<- Featureimportance$var %>% as.factor()

#Plotting the bar chart for comparing feature importance
Featuresbar <- ggplot(data = Featureimportance) + 
  geom_bar(
    stat = "identity",#it leaves the data without count and bin
    mapping = aes(x = var, y=Overall, fill = var), 
    show.legend = FALSE,
    width = 1
  ) + 
  labs(x = NULL, y = NULL)
Featuresbar + coord_flip() + theme_minimal()

#The 5 most important variables for predicting the quality of the wine are (in ranks):
# 1) alcohol
# 2) free.sulfur.dioxide
# 3) volatile.acidity
# 4) density 
# 5) residual.sugar