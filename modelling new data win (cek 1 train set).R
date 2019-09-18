library(tidyverse)
library(caret)
library(Hmisc)
library(CatEncoders)
library(woeBinning)
library(smotefamily)
library(randomForest)
library(inTrees)
library(doSNOW)

#import training data
datafeb <- read.csv("D:/FOLDER AFGANTA/Revisit Anti Lapse/Modelling/20190228 data.csv")

#import test data
datamar <- read.csv("D:/FOLDER AFGANTA/Revisit Anti Lapse/Modelling/20190331 data.csv")
dataapr <- read.csv("D:/FOLDER AFGANTA/Revisit Anti Lapse/Modelling/20190430 data.csv")
datajul <- read.csv("D:/FOLDER AFGANTA/Revisit Anti Lapse/Modelling/20190731 data.csv")

#import agent segment
dataagent <- read.csv("D:/FOLDER AFGANTA/Revisit Anti Lapse/Modelling/agent segment 2019.csv")

#import agent behavior
dataagentb <- read.csv("D:/FOLDER AFGANTA/Revisit Anti Lapse/Modelling/Agent_Behavior.csv")
dataagentb <- rename(dataagentb,policynumber=PolicyNumber)
dataagentb$policynumber <- as.integer(dataagentb$policynumber)

#import pay x to lapse and status
payfeb <- read.csv("D:/FOLDER AFGANTA/Revisit Anti Lapse/Modelling/20190228 paytola and status.csv")
paymar <- read.csv("D:/FOLDER AFGANTA/Revisit Anti Lapse/Modelling/20190331 paytola and status.csv")
payjul <- read.csv("D:/FOLDER AFGANTA/Revisit Anti Lapse/Modelling/20190731 paytola.csv")

#import customer segmentation
segfeb <- read.csv("D:/FOLDER AFGANTA/Revisit Anti Lapse/Modelling/cust segmentation 20190228.csv")
segmar <- read.csv("D:/FOLDER AFGANTA/Revisit Anti Lapse/Modelling/cust segmentation 20190331.csv")
segjul <- read.csv("D:/FOLDER AFGANTA/Revisit Anti Lapse/Modelling/cust segmentation 20190731.csv")

#----A. PREPARATION----

#attach agent segment to training data
datafeb <- left_join(datafeb,dataagent,by='agentcode')
datafeb$agentclub <- impute(datafeb$agentclub,'NONE')

#attach agent segment to test data
datamar <- left_join(datamar,dataagent,by='agentcode')
datamar$agentclub <- impute(datamar$agentclub,'NONE')
datajul <- left_join(datajul,dataagent,by='agentcode')
datajul$agentclub <- impute(datajul$agentclub,'NONE')

#attach agent behavior to training data
datafeb <- left_join(datafeb,dataagentb,by='policynumber')
datafeb <- select(datafeb, -c(bookingagent))

#attach agent behavior to test data
datamar <- left_join(datamar,dataagentb,by='policynumber')
datamar <- select(datamar, -c(bookingagent))
datajul <- left_join(datajul,dataagentb,by='policynumber')
datajul <- select(datajul, -c(bookingagent))

#attach pay x to lapse to training data
datafeb <- left_join(datafeb,payfeb,by='policynumber')

#attach pay x to lapse to test data
datamar <- left_join(datamar,paymar,by='policynumber')
datajul <- left_join(datajul,payjul,by='policynumber')

#attach customer segmentation to training data
datafeb <- left_join(datafeb,segfeb,by='CIF_Number')

#attach customer segmentation to test data
datamar <- left_join(datamar,segmar,by='CIF_Number')
datajul <- left_join(datajul,segjul,by='CIF_Number')

#cleaning productname
datafeb$productname <- as.factor(gsub("Perfecto-5","Perfecto",datafeb$productname))
datafeb$productname <- as.factor(gsub("Perfecto-8","Perfecto",datafeb$productname))
datamar$productname <- as.factor(gsub("Perfecto-5","Perfecto",datamar$productname))
datamar$productname <- as.factor(gsub("Perfecto-8","Perfecto",datamar$productname))
datajul$productname <- as.factor(gsub("Perfecto-5","Perfecto",datajul$productname))
datajul$productname <- as.factor(gsub("Perfecto-8","Perfecto",datajul$productname))

#insert column id for data train & test
datafeb$id <- c("trainfeb")
datamar$id <- c("testmar")

#insert column policynumber & id for data train & test
datafeb$policynumberid <- paste(datafeb$policynumber,datafeb$id, sep = "")
datamar$policynumberid <- paste(datamar$policynumber,datamar$id, sep = "")

#Export full data train and test for cross ref
write.csv(datafeb, "D:/FOLDER AFGANTA/Revisit Anti Lapse/Modelling/datatrain.csv")
write.csv(datamar, "D:/FOLDER AFGANTA/Revisit Anti Lapse/Modelling/datatest.csv")
write.csv(datajul, "D:/FOLDER AFGANTA/Revisit Anti Lapse/Modelling/datapredjul.csv")

#drop unused columns
datafeb <- select(datafeb,-c(agentcode,policystatus,row))
datamar <- select(datamar,-c(agentcode,policystatus,row))
dataapr <- select(dataapr,-c(agentcode,policystatus,row))

#Combine all data train & test
dataall <- rbind(datafeb,datamar)


#----B. PREPROCESSING----

#----B.1. PREPROCESSING DATAALL----

#cleaning Null values & convert numeric
num <- c("avg_dpd","min_dpd","max_dpd","lastdpd","p1dpd","p2dpd","p3dpd","p1p2diff","p1p3diff","p2p3diff",
         "avg_trxattempt","min_trxattempt","max_trxattempt","ageatissued","paymenttobepaid","monthlypremium",
         "suminsured","agentlosmonth","coidebtmonthly","totalaum","outstandingcoi","payfreqtola",
         "AnnPremium","Total_SumInsured","Age","NoofPolicy","NoofRider","MonthlySalary")
for (x in num){
  tmp <- get("x")
  dataall[[tmp]] <- as.numeric(gsub("NULL",999999,dataall[[tmp]]))
  assign(x,tmp)
}

#binning policyage
dataall$policyagemonth <- if_else(dataall$policyagemonth <= 12,"<= 12",
                                  if_else(dataall$policyagemonth <= 24,"<= 24",
                                          "> 24"))
dataall$policyagemonth <- as.factor(dataall$policyagemonth)

#binning continous variable
num1 <- select(dataall, policynumberid, policynumber,lapse,avg_dpd,min_dpd,max_dpd,lastdpd,p1dpd,p2dpd,p3dpd,p1p2diff,p1p3diff,p2p3diff,
               avg_trxattempt,min_trxattempt,max_trxattempt,ageatissued,paymenttobepaid,monthlypremium,suminsured,agentlosmonth,
               coidebtmonthly,totalaum,outstandingcoi,payfreqtola,AnnPremium,Total_SumInsured,Age,NoofPolicy,NoofRider,MonthlySalary)
cutpoints <- woe.binning(num1, "lapse", c("avg_dpd","min_dpd","max_dpd","lastdpd","p1dpd","p2dpd","p3dpd","p1p2diff","p1p3diff","p2p3diff",
                                          "avg_trxattempt","min_trxattempt","max_trxattempt","ageatissued","paymenttobepaid","monthlypremium",
                                          "suminsured","agentlosmonth","coidebtmonthly","totalaum","outstandingcoi","payfreqtola",
                                          "AnnPremium","Total_SumInsured","Age","NoofPolicy","NoofRider","MonthlySalary"))
#plotbin <- woe.binning.plot(cutpoints)
dataall_woe <- woe.binning.deploy(num1, cutpoints, add.woe.or.dum.var = "woe")
dataall_woe <- select(dataall_woe, policynumberid,avg_dpd.binned,min_dpd.binned,max_dpd.binned,lastdpd.binned,p1dpd.binned,p2dpd.binned,p3dpd.binned,
                      p1p2diff.binned,p1p3diff.binned,p2p3diff.binned,avg_trxattempt.binned,min_trxattempt.binned,max_trxattempt.binned,ageatissued.binned,
                      paymenttobepaid.binned,monthlypremium.binned,suminsured.binned,agentlosmonth.binned,coidebtmonthly.binned,totalaum.binned,
                      outstandingcoi.binned,payfreqtola.binned,AnnPremium.binned,Total_SumInsured.binned,Age.binned,NoofPolicy.binned,NoofRider.binned,
                      MonthlySalary.binned)

#attach binning result to train data
dataall <- left_join(dataall,dataall_woe,by='policynumberid')

#remove unused & cont variables that already binned
dataall <- select(dataall,-c(phclientcode,policynumberid,policynumber,avg_dpd,min_dpd,max_dpd,lastdpd,p1dpd,p2dpd,p3dpd,p1p2diff,p1p3diff,p2p3diff,
                             avg_trxattempt,min_trxattempt,max_trxattempt,ageatissued,paymenttobepaid,monthlypremium,suminsured,agentlosmonth,
                             coidebtmonthly,totalaum,outstandingcoi,payfreqtola,AnnPremium,Total_SumInsured,Age,NoofPolicy,NoofRider,MonthlySalary,
                             CIF_Number))

#encode all categorical variables
catvar <- select(dataall,-c(occupation,lapse,id))
for (y in names(catvar)){
  tmp1 <- get("y")
  label <- LabelEncoder.fit(dataall[[tmp1]])
  assign(paste("label",y,sep = ""),label)
  dataall[[tmp1]] <- transform(label,dataall[[tmp1]])
  assign(y,tmp1)
}

#Separate data train & test
datatrain <- filter(dataall,id=="trainfeb")
datatrain <- select(datatrain,-c(id))
datatest <- filter(dataall,id=="testmar")
datatest <- select(datatest,-c(id))

#separate datatrain based on product & policyage
for(a in c(1:7)){
  for(b in c(1:3)){
    tmpa <- get("a")
    tmpb <- get("b")
    datatraina <- filter(datatrain,productname==tmpa,policyagemonth==tmpb)
    assign(paste("datatrain",a,b,sep = ""),datatraina)
  }
}

#separate datatest based on product & policyage
for(a in c(1:7)){
  for(b in c(1:3)){
    tmpa <- get("a")
    tmpb <- get("b")
    datatesta <- filter(datatest,productname==tmpa,policyagemonth==tmpb)
    assign(paste("datatest",a,b,sep = ""),datatesta)
  }
}


#----B.2. GENERATE OVERSAMPLING SMOTE DATA TRAIN----

#check member lapse to identify dup_size smote generation
summary(as.factor(datatrain11$lapse))

#generate oversampling because of unbalance data to every separated training data
set.seed(123456)
gen_data11 <- SMOTE(datatrain11,datatrain11$lapse,K=5,dup_size = 9)
smote_data11 <- gen_data11$data
smote_data11 <- select(smote_data11,-c(class))
smote_data11 <- lapply(smote_data11,floor)  #round down number from gen_data
smote_data11 <- lapply(smote_data11,as.factor)
smote_data11 <- as.data.frame(smote_data11)

gen_data12 <- SMOTE(datatrain12,datatrain12$lapse,K=5,dup_size = 11)
smote_data12 <- gen_data12$data
smote_data12 <- select(smote_data12,-c(class))
smote_data12 <- lapply(smote_data12,floor)  #round down number from gen_data
smote_data12 <- lapply(smote_data12,as.factor)
smote_data12 <- as.data.frame(smote_data12)

gen_data13 <- SMOTE(datatrain13,datatrain13$lapse,K=5,dup_size = 20)
smote_data13 <- gen_data13$data
smote_data13 <- select(smote_data13,-c(class))
smote_data13 <- lapply(smote_data13,floor)  #round down number from gen_data
smote_data13 <- lapply(smote_data13,as.factor)
smote_data13 <- as.data.frame(smote_data13)

gen_data31 <- SMOTE(datatrain31,datatrain31$lapse,K=5,dup_size = 3.3)
smote_data31 <- gen_data31$data
smote_data31 <- select(smote_data31,-c(class))
smote_data31 <- lapply(smote_data31,floor)  #round down number from gen_data
smote_data31 <- lapply(smote_data31,as.factor)
smote_data31 <- as.data.frame(smote_data31)

gen_data32 <- SMOTE(datatrain32,datatrain32$lapse,K=5,dup_size = 8)
smote_data32 <- gen_data32$data
smote_data32 <- select(smote_data32,-c(class))
smote_data32 <- lapply(smote_data32,floor)  #round down number from gen_data
smote_data32 <- lapply(smote_data32,as.factor)
smote_data32 <- as.data.frame(smote_data32)

gen_data33 <- SMOTE(datatrain33,datatrain33$lapse,K=5,dup_size = 15)
smote_data33 <- gen_data33$data
smote_data33 <- select(smote_data33,-c(class))
smote_data33 <- lapply(smote_data33,floor)  #round down number from gen_data
smote_data33 <- lapply(smote_data33,as.factor)
smote_data33 <- as.data.frame(smote_data33)

gen_data41 <- SMOTE(datatrain41,datatrain41$lapse,K=5,dup_size = 1.5)
smote_data41 <- gen_data41$data
smote_data41 <- select(smote_data41,-c(class))
smote_data41 <- lapply(smote_data41,floor)  #round down number from gen_data
smote_data41 <- lapply(smote_data41,as.factor)
smote_data41 <- as.data.frame(smote_data41)

gen_data42 <- SMOTE(datatrain42,datatrain42$lapse,K=5,dup_size = 1.5)
smote_data42 <- gen_data42$data
smote_data42 <- select(smote_data42,-c(class))
smote_data42 <- lapply(smote_data42,floor)  #round down number from gen_data
smote_data42 <- lapply(smote_data42,as.factor)
smote_data42 <- as.data.frame(smote_data42)

gen_data51 <- SMOTE(datatrain51,datatrain51$lapse,K=5,dup_size = 6)
smote_data51 <- gen_data51$data
smote_data51 <- select(smote_data51,-c(class))
smote_data51 <- lapply(smote_data51,floor)  #round down number from gen_data
smote_data51 <- lapply(smote_data51,as.factor)
smote_data51 <- as.data.frame(smote_data51)

gen_data71 <- SMOTE(datatrain71,datatrain71$lapse,K=2,dup_size = 1.5)
smote_data71 <- gen_data71$data
smote_data71 <- select(smote_data71,-c(class))
smote_data71 <- lapply(smote_data71,floor)  #round down number from gen_data
smote_data71 <- lapply(smote_data71,as.factor)
smote_data71 <- as.data.frame(smote_data71)

gen_data72 <- SMOTE(datatrain72,datatrain72$lapse,K=5,dup_size = 14)
smote_data72 <- gen_data72$data
smote_data72 <- select(smote_data72,-c(class))
smote_data72 <- lapply(smote_data72,floor)  #round down number from gen_data
smote_data72 <- lapply(smote_data72,as.factor)
smote_data72 <- as.data.frame(smote_data72)

gen_data73 <- SMOTE(datatrain73,datatrain73$lapse,K=5,dup_size = 18)
smote_data73 <- gen_data73$data
smote_data73 <- select(smote_data73,-c(class))
smote_data73 <- lapply(smote_data73,floor)  #round down number from gen_data
smote_data73 <- lapply(smote_data73,as.factor)
smote_data73 <- as.data.frame(smote_data73)


#----B.3. TRAINING MODEL DATATRAIN----

#----B.3.1. MODEL DATATRAIN11----

#Modelling with default par
summary(smote_data11)
set.seed(123456)
modelrf11 <- randomForest(lapse ~ ., data = smote_data11, importance = TRUE)
print(modelrf11)

#obtain optimum ntree
opntree11 <- which.min(modelrf11$err.rate)
print(opntree11)

#check features importance
imp11 <- varImp(modelrf11)
plotimp11 <- varImpPlot(modelrf11)
dfimp11 <- as.data.frame(plotimp11)
dfimp11 <- rownames_to_column(dfimp11, var = "variable")
dfimp11 <- select(dfimp11, -c(MeanDecreaseAccuracy))
dfimp11 <- arrange(dfimp11, desc(MeanDecreaseGini))
plotimportance11 <- ggplot(dfimp11, aes(x=reorder(variable,MeanDecreaseGini), weight=MeanDecreaseGini)) + 
  geom_bar(fill = "red2") + coord_flip() + ggtitle("Variable Importance Training Data from Random Forest Fit") + 
  xlab("Variables") + ylab("Variable Importance (Mean Decrease in Gini Index)") + theme(legend.position = "none")
  print(plotimportance11)

#drop unimportance features
dropvar11 <- filter(dfimp11, MeanDecreaseGini < 0.000005)
dropvar11 <- as.vector(dropvar11$variable)
smote_data11 <- select(smote_data11, -c(dropvar11))

#tuning for optimum mtry
pred <- select(smote_data11, -c(lapse))
rftune11 <- tuneRF(pred,smote_data11$lapse,1,ntreeTry = opntree11,stepFactor = 2,improve = 0.01,trace = TRUE,plot=TRUE,doBest = FALSE)
rftunedf11 <- as.data.frame(rftune11)
plottune11 <- ggplot(rftunedf11,aes(x=mtry,y=OOBError)) + geom_line(color="red") + geom_point(color="red4")
print(plottune11)

#train model with optimum par
set.seed(123456)
modelrfop11 <- randomForest(lapse ~ ., data = smote_data11, ntree = opntree11, mtry = 16, importance = TRUE)
print(modelrfop11)

#----B.3.2. MODEL DATATRAIN12----

#Modelling with default par
summary(smote_data12)
set.seed(123456)
modelrf12 <- randomForest(lapse ~ ., data = smote_data12, importance = TRUE)
print(modelrf12)

#obtain optimum ntree
opntree12 <- which.min(modelrf12$err.rate)
print(opntree12)

#check features importance
imp12 <- varImp(modelrf12)
plotimp12 <- varImpPlot(modelrf12)
dfimp12 <- as.data.frame(plotimp12)
dfimp12 <- rownames_to_column(dfimp12, var = "variable")
dfimp12 <- select(dfimp12, -c(MeanDecreaseAccuracy))
dfimp12 <- arrange(dfimp12, desc(MeanDecreaseGini))
plotimportance12 <- ggplot(dfimp12, aes(x=reorder(variable,MeanDecreaseGini), weight=MeanDecreaseGini)) + 
  geom_bar(fill = "red2") + coord_flip() + ggtitle("Variable Importance Training Data from Random Forest Fit") + 
  xlab("Variables") + ylab("Variable Importance (Mean Decrease in Gini Index)") + theme(legend.position = "none")
print(plotimportance12)

#drop unimportance features
dropvar12 <- filter(dfimp12, MeanDecreaseGini < 0.000005)
dropvar12 <- as.vector(dropvar12$variable)
smote_data12 <- select(smote_data12, -c(dropvar12))

#tuning for optimum mtry
pred <- select(smote_data12, -c(lapse))
rftune12 <- tuneRF(pred,smote_data12$lapse,1,ntreeTry = opntree12,stepFactor = 2,improve = 0.01,trace = TRUE,plot=TRUE,doBest = FALSE)
rftunedf12 <- as.data.frame(rftune12)
plottune12 <- ggplot(rftunedf12,aes(x=mtry,y=OOBError)) + geom_line(color="red") + geom_point(color="red4")
print(plottune12)

#train model with optimum par
set.seed(123456)
modelrfop12 <- randomForest(lapse ~ ., data = smote_data12, ntree = opntree12, mtry = 16, importance = TRUE)
print(modelrfop12)

#----B.3.3. MODEL DATATRAIN13----

#Modelling with default par
summary(smote_data13)
set.seed(123456)
modelrf13 <- randomForest(lapse ~ ., data = smote_data13, importance = TRUE)
print(modelrf13)

#obtain optimum ntree
opntree13 <- which.min(modelrf13$err.rate)
print(opntree13)

#check features importance
imp13 <- varImp(modelrf13)
plotimp13 <- varImpPlot(modelrf13)
dfimp13 <- as.data.frame(plotimp13)
dfimp13 <- rownames_to_column(dfimp13, var = "variable")
dfimp13 <- select(dfimp13, -c(MeanDecreaseAccuracy))
dfimp13 <- arrange(dfimp13, desc(MeanDecreaseGini))
plotimportance13 <- ggplot(dfimp13, aes(x=reorder(variable,MeanDecreaseGini), weight=MeanDecreaseGini)) + 
  geom_bar(fill = "red2") + coord_flip() + ggtitle("Variable Importance Training Data from Random Forest Fit") + 
  xlab("Variables") + ylab("Variable Importance (Mean Decrease in Gini Index)") + theme(legend.position = "none")
print(plotimportance13)

#drop unimportance features
dropvar13 <- filter(dfimp13, MeanDecreaseGini < 0.000005)
dropvar13 <- as.vector(dropvar13$variable)
smote_data13 <- select(smote_data13, -c(dropvar13))

#tuning for optimum mtry
pred <- select(smote_data13, -c(lapse))
rftune13 <- tuneRF(pred,smote_data13$lapse,1,ntreeTry = opntree13,stepFactor = 2,improve = 0.01,trace = TRUE,plot=TRUE,doBest = FALSE)
rftunedf13 <- as.data.frame(rftune13)
plottune13 <- ggplot(rftunedf13,aes(x=mtry,y=OOBError)) + geom_line(color="red") + geom_point(color="red4")
print(plottune13)

#train model with optimum par
set.seed(123456)
modelrfop13 <- randomForest(lapse ~ ., data = smote_data13, ntree = opntree13, mtry = 4, importance = TRUE)
print(modelrfop13)

#----B.3.4. MODEL DATATRAIN31----

#Modelling with default par
summary(smote_data31)
set.seed(123456)
modelrf31 <- randomForest(lapse ~ ., data = smote_data31, importance = TRUE)
print(modelrf31)

#obtain optimum ntree
opntree31 <- which.min(modelrf31$err.rate)
print(opntree31)

#check features importance
imp31 <- varImp(modelrf31)
plotimp31 <- varImpPlot(modelrf31)
dfimp31 <- as.data.frame(plotimp31)
dfimp31 <- rownames_to_column(dfimp31, var = "variable")
dfimp31 <- select(dfimp31, -c(MeanDecreaseAccuracy))
dfimp31 <- arrange(dfimp31, desc(MeanDecreaseGini))
plotimportance31 <- ggplot(dfimp31, aes(x=reorder(variable,MeanDecreaseGini), weight=MeanDecreaseGini)) + 
  geom_bar(fill = "red2") + coord_flip() + ggtitle("Variable Importance Training Data from Random Forest Fit") + 
  xlab("Variables") + ylab("Variable Importance (Mean Decrease in Gini Index)") + theme(legend.position = "none")
print(plotimportance31)
impperc31 <- as.data.frame(importance(modelrf31,type = 1))

#drop unimportance features
dropvar31 <- filter(dfimp31, MeanDecreaseGini < 0.000005)
dropvar31 <- as.vector(dropvar31$variable)
smote_data31 <- select(smote_data31, -c(dropvar31))

#tuning for optimum mtry
pred <- select(smote_data31, -c(lapse))
rftune31 <- tuneRF(pred,smote_data31$lapse,1,ntreeTry = opntree31,stepFactor = 2,improve = 0.01,trace = TRUE,plot=TRUE,doBest = FALSE)
rftunedf31 <- as.data.frame(rftune31)
plottune31 <- ggplot(rftunedf31,aes(x=mtry,y=OOBError)) + geom_line(color="red") + geom_point(color="red4")
print(plottune31)

#train model with optimum par
set.seed(123456)
modelrfop31 <- randomForest(lapse ~ ., data = smote_data31, ntree = opntree31, mtry = 8, importance = TRUE)
print(modelrfop31)

#----B.3.5. MODEL DATATRAIN32----

#Modelling with default par
summary(smote_data32)
set.seed(123456)
modelrf32 <- randomForest(lapse ~ ., data = smote_data32, importance = TRUE)
print(modelrf32)

#obtain optimum ntree
opntree32 <- which.min(modelrf32$err.rate)
print(opntree32)

#check features importance
imp32 <- varImp(modelrf32)
plotimp32 <- varImpPlot(modelrf32)
dfimp32 <- as.data.frame(plotimp32)
dfimp32 <- rownames_to_column(dfimp32, var = "variable")
dfimp32 <- select(dfimp32, -c(MeanDecreaseAccuracy))
dfimp32 <- arrange(dfimp32, desc(MeanDecreaseGini))
plotimportance32 <- ggplot(dfimp32, aes(x=reorder(variable,MeanDecreaseGini), weight=MeanDecreaseGini)) + 
  geom_bar(fill = "red2") + coord_flip() + ggtitle("Variable Importance Training Data from Random Forest Fit") + 
  xlab("Variables") + ylab("Variable Importance (Mean Decrease in Gini Index)") + theme(legend.position = "none")
print(plotimportance32)

#drop unimportance features
dropvar32 <- filter(dfimp32, MeanDecreaseGini < 0.000005)
dropvar32 <- as.vector(dropvar32$variable)
smote_data32 <- select(smote_data32, -c(dropvar32))

#tuning for optimum mtry
pred <- select(smote_data32, -c(lapse))
rftune32 <- tuneRF(pred,smote_data32$lapse,1,ntreeTry = opntree32,stepFactor = 2,improve = 0.01,trace = TRUE,plot=TRUE,doBest = FALSE)
rftunedf32 <- as.data.frame(rftune32)
plottune32 <- ggplot(rftunedf32,aes(x=mtry,y=OOBError)) + geom_line(color="red") + geom_point(color="red4")
print(plottune32)

#train model with optimum par
set.seed(123456)
modelrfop32 <- randomForest(lapse ~ ., data = smote_data32, ntree = opntree32, mtry = 8, importance = TRUE)
print(modelrfop32)

#----B.3.6. MODEL DATATRAIN33----

#Modelling with default par
summary(smote_data33)
set.seed(123456)
modelrf33 <- randomForest(lapse ~ ., data = smote_data33, importance = TRUE)
print(modelrf33)

#obtain optimum ntree
opntree33 <- which.min(modelrf33$err.rate)
print(opntree33)

#check features importance
imp33 <- varImp(modelrf33)
plotimp33 <- varImpPlot(modelrf33)
dfimp33 <- as.data.frame(plotimp33)
dfimp33 <- rownames_to_column(dfimp33, var = "variable")
dfimp33 <- select(dfimp33, -c(MeanDecreaseAccuracy))
dfimp33 <- arrange(dfimp33, desc(MeanDecreaseGini))
plotimportance33 <- ggplot(dfimp33, aes(x=reorder(variable,MeanDecreaseGini), weight=MeanDecreaseGini)) + 
  geom_bar(fill = "red2") + coord_flip() + ggtitle("Variable Importance Training Data from Random Forest Fit") + 
  xlab("Variables") + ylab("Variable Importance (Mean Decrease in Gini Index)") + theme(legend.position = "none")
print(plotimportance33)

#drop unimportance features
dropvar33 <- filter(dfimp33, MeanDecreaseGini < 0.000005)
dropvar33 <- as.vector(dropvar33$variable)
smote_data33 <- select(smote_data33, -c(dropvar33))

#tuning for optimum mtry
pred <- select(smote_data33, -c(lapse))
rftune33 <- tuneRF(pred,smote_data33$lapse,1,ntreeTry = opntree33,stepFactor = 2,improve = 0.01,trace = TRUE,plot=TRUE,doBest = FALSE)
rftunedf33 <- as.data.frame(rftune33)
plottune33 <- ggplot(rftunedf33,aes(x=mtry,y=OOBError)) + geom_line(color="red") + geom_point(color="red4")
print(plottune33)

#train model with optimum par
set.seed(123456)
modelrfop33 <- randomForest(lapse ~ ., data = smote_data33, ntree = opntree33, mtry = 8, importance = TRUE)
print(modelrfop33)

#----B.3.7. MODEL DATATRAIN41----

#Modelling with default par
summary(smote_data41)
set.seed(123456)
modelrf41 <- randomForest(lapse ~ ., data = smote_data41, importance = TRUE)
print(modelrf41)

#obtain optimum ntree
opntree41 <- which.min(modelrf41$err.rate)
print(opntree41)

#check features importance
imp41 <- varImp(modelrf41)
plotimp41 <- varImpPlot(modelrf41)
dfimp41 <- as.data.frame(plotimp41)
dfimp41 <- rownames_to_column(dfimp41, var = "variable")
dfimp41 <- select(dfimp41, -c(MeanDecreaseAccuracy))
dfimp41 <- arrange(dfimp41, desc(MeanDecreaseGini))
plotimportance41 <- ggplot(dfimp41, aes(x=reorder(variable,MeanDecreaseGini), weight=MeanDecreaseGini)) + 
  geom_bar(fill = "red2") + coord_flip() + ggtitle("Variable Importance Training Data from Random Forest Fit") + 
  xlab("Variables") + ylab("Variable Importance (Mean Decrease in Gini Index)") + theme(legend.position = "none")
print(plotimportance41)

#drop unimportance features
dropvar41 <- filter(dfimp41, MeanDecreaseGini < 0.000005)
dropvar41 <- as.vector(dropvar41$variable)
smote_data41 <- select(smote_data41, -c(dropvar41))

#tuning for optimum mtry
pred <- select(smote_data41, -c(lapse))
rftune41 <- tuneRF(pred,smote_data41$lapse,1,ntreeTry = opntree41,stepFactor = 2,improve = 0.01,trace = TRUE,plot=TRUE,doBest = FALSE)
rftunedf41 <- as.data.frame(rftune41)
plottune41 <- ggplot(rftunedf41,aes(x=mtry,y=OOBError)) + geom_line(color="red") + geom_point(color="red4")
print(plottune41)

#train model with optimum par
set.seed(123456)
modelrfop41 <- randomForest(lapse ~ ., data = smote_data41, ntree = opntree41, mtry = 16, importance = TRUE)
print(modelrfop41)

#----B.3.8. MODEL DATATRAIN42----

#Modelling with default par
summary(smote_data42)
set.seed(123456)
modelrf42 <- randomForest(lapse ~ ., data = smote_data42, importance = TRUE)
print(modelrf42)

#obtain optimum ntree
opntree42 <- which.min(modelrf42$err.rate)
print(opntree42)

#check features importance
imp42 <- varImp(modelrf42)
plotimp42 <- varImpPlot(modelrf42)
dfimp42 <- as.data.frame(plotimp42)
dfimp42 <- rownames_to_column(dfimp42, var = "variable")
dfimp42 <- select(dfimp42, -c(MeanDecreaseAccuracy))
dfimp42 <- arrange(dfimp42, desc(MeanDecreaseGini))
plotimportance42 <- ggplot(dfimp42, aes(x=reorder(variable,MeanDecreaseGini), weight=MeanDecreaseGini)) + 
  geom_bar(fill = "red2") + coord_flip() + ggtitle("Variable Importance Training Data from Random Forest Fit") + 
  xlab("Variables") + ylab("Variable Importance (Mean Decrease in Gini Index)") + theme(legend.position = "none")
print(plotimportance42)

#drop unimportance features
dropvar42 <- filter(dfimp42, MeanDecreaseGini < 0.000005)
dropvar42 <- as.vector(dropvar42$variable)
smote_data42 <- select(smote_data42, -c(dropvar42))

#tuning for optimum mtry
pred <- select(smote_data42, -c(lapse))
rftune42 <- tuneRF(pred,smote_data42$lapse,1,ntreeTry = opntree42,stepFactor = 2,improve = 0.01,trace = TRUE,plot=TRUE,doBest = FALSE)
rftunedf42 <- as.data.frame(rftune42)
plottune42 <- ggplot(rftunedf42,aes(x=mtry,y=OOBError)) + geom_line(color="red") + geom_point(color="red4")
print(plottune42)

#train model with optimum par
set.seed(123456)
modelrfop42 <- randomForest(lapse ~ ., data = smote_data42, ntree = opntree42, mtry = 8, importance = TRUE)
print(modelrfop42)

#----B.3.9. MODEL DATATRAIN51----

#Modelling with default par
summary(smote_data51)
set.seed(123456)
modelrf51 <- randomForest(lapse ~ ., data = smote_data51, importance = TRUE)
print(modelrf51)

#obtain optimum ntree
opntree51 <- which.min(modelrf51$err.rate)
print(opntree51)

#check features importance
imp51 <- varImp(modelrf51)
plotimp51 <- varImpPlot(modelrf51)
dfimp51 <- as.data.frame(plotimp51)
dfimp51 <- rownames_to_column(dfimp51, var = "variable")
dfimp51 <- select(dfimp51, -c(MeanDecreaseAccuracy))
dfimp51 <- arrange(dfimp51, desc(MeanDecreaseGini))
plotimportance51 <- ggplot(dfimp51, aes(x=reorder(variable,MeanDecreaseGini), weight=MeanDecreaseGini)) + 
  geom_bar(fill = "red2") + coord_flip() + ggtitle("Variable Importance Training Data from Random Forest Fit") + 
  xlab("Variables") + ylab("Variable Importance (Mean Decrease in Gini Index)") + theme(legend.position = "none")
print(plotimportance51)

#drop unimportance features
dropvar51 <- filter(dfimp51, MeanDecreaseGini < 0.000005)
dropvar51 <- as.vector(dropvar51$variable)
smote_data51 <- select(smote_data51, -c(dropvar51))

#tuning for optimum mtry
pred <- select(smote_data51, -c(lapse))
rftune51 <- tuneRF(pred,smote_data51$lapse,1,ntreeTry = opntree51,stepFactor = 2,improve = 0.01,trace = TRUE,plot=TRUE,doBest = FALSE)
rftunedf51 <- as.data.frame(rftune51)
plottune51 <- ggplot(rftunedf51,aes(x=mtry,y=OOBError)) + geom_line(color="red") + geom_point(color="red4")
print(plottune51)

#train model with optimum par
set.seed(123456)
modelrfop51 <- randomForest(lapse ~ ., data = smote_data51, ntree = opntree51, mtry = 4, importance = TRUE)
print(modelrfop51)

#----B.3.10. MODEL DATATRAIN71----

#Modelling with default par
summary(smote_data71)
set.seed(123456)
modelrf71 <- randomForest(lapse ~ ., data = smote_data71, importance = TRUE)
print(modelrf71)

#obtain optimum ntree
opntree71 <- which.min(modelrf71$err.rate)
print(opntree71)

#check features importance
imp71 <- varImp(modelrf71)
plotimp71 <- varImpPlot(modelrf71)
dfimp71 <- as.data.frame(plotimp71)
dfimp71 <- rownames_to_column(dfimp71, var = "variable")
dfimp71 <- select(dfimp71, -c(MeanDecreaseAccuracy))
dfimp71 <- arrange(dfimp71, desc(MeanDecreaseGini))
plotimportance71 <- ggplot(dfimp71, aes(x=reorder(variable,MeanDecreaseGini), weight=MeanDecreaseGini)) + 
  geom_bar(fill = "red2") + coord_flip() + ggtitle("Variable Importance Training Data from Random Forest Fit") + 
  xlab("Variables") + ylab("Variable Importance (Mean Decrease in Gini Index)") + theme(legend.position = "none")
print(plotimportance71)

#drop unimportance features
dropvar71 <- filter(dfimp71, MeanDecreaseGini < 0.000005)
dropvar71 <- as.vector(dropvar71$variable)
smote_data71 <- select(smote_data71, -c(dropvar71))

#tuning for optimum mtry
pred <- select(smote_data71, -c(lapse))
rftune71 <- tuneRF(pred,smote_data71$lapse,1,ntreeTry = opntree71,stepFactor = 2,improve = 0.01,trace = TRUE,plot=TRUE,doBest = FALSE)
rftunedf71 <- as.data.frame(rftune71)
plottune71 <- ggplot(rftunedf71,aes(x=mtry,y=OOBError)) + geom_line(color="red") + geom_point(color="red4")
print(plottune71)

#train model with optimum par
set.seed(123456)
modelrfop71 <- randomForest(lapse ~ ., data = smote_data71, ntree = opntree71, mtry = 4, importance = TRUE)
print(modelrfop71)

#----B.3.11. MODEL DATATRAIN72----

#Modelling with default par
summary(smote_data72)
set.seed(123456)
modelrf72 <- randomForest(lapse ~ ., data = smote_data72, importance = TRUE)
print(modelrf72)

#obtain optimum ntree
opntree72 <- which.min(modelrf72$err.rate)
print(opntree72)

#check features importance
imp72 <- varImp(modelrf72)
plotimp72 <- varImpPlot(modelrf72)
dfimp72 <- as.data.frame(plotimp72)
dfimp72 <- rownames_to_column(dfimp72, var = "variable")
dfimp72 <- select(dfimp72, -c(MeanDecreaseAccuracy))
dfimp72 <- arrange(dfimp72, desc(MeanDecreaseGini))
plotimportance72 <- ggplot(dfimp72, aes(x=reorder(variable,MeanDecreaseGini), weight=MeanDecreaseGini)) + 
  geom_bar(fill = "red2") + coord_flip() + ggtitle("Variable Importance Training Data from Random Forest Fit") + 
  xlab("Variables") + ylab("Variable Importance (Mean Decrease in Gini Index)") + theme(legend.position = "none")
print(plotimportance72)

#drop unimportance features
dropvar72 <- filter(dfimp72, MeanDecreaseGini < 0.000005)
dropvar72 <- as.vector(dropvar72$variable)
smote_data72 <- select(smote_data72, -c(dropvar72))

#tuning for optimum mtry
pred <- select(smote_data72, -c(lapse))
rftune72 <- tuneRF(pred,smote_data72$lapse,1,ntreeTry = opntree72,stepFactor = 2,improve = 0.01,trace = TRUE,plot=TRUE,doBest = FALSE)
rftunedf72 <- as.data.frame(rftune72)
plottune72 <- ggplot(rftunedf72,aes(x=mtry,y=OOBError)) + geom_line(color="red") + geom_point(color="red4")
print(plottune72)

#train model with optimum par
set.seed(123456)
modelrfop72 <- randomForest(lapse ~ ., data = smote_data72, ntree = opntree72, mtry = 2, importance = TRUE)
print(modelrfop72)

#----B.3.12. MODEL DATATRAIN73----

#Modelling with default par
summary(smote_data73)
set.seed(123456)
modelrf73 <- randomForest(lapse ~ ., data = smote_data73, importance = TRUE)
print(modelrf73)

#obtain optimum ntree
opntree73 <- which.min(modelrf73$err.rate)
print(opntree73)

#check features importance
imp73 <- varImp(modelrf73)
plotimp73 <- varImpPlot(modelrf73)
dfimp73 <- as.data.frame(plotimp73)
dfimp73 <- rownames_to_column(dfimp73, var = "variable")
dfimp73 <- select(dfimp73, -c(MeanDecreaseAccuracy))
dfimp73 <- arrange(dfimp73, desc(MeanDecreaseGini))
plotimportance73 <- ggplot(dfimp73, aes(x=reorder(variable,MeanDecreaseGini), weight=MeanDecreaseGini)) + 
  geom_bar(fill = "red2") + coord_flip() + ggtitle("Variable Importance Training Data from Random Forest Fit") + 
  xlab("Variables") + ylab("Variable Importance (Mean Decrease in Gini Index)") + theme(legend.position = "none")
print(plotimportance73)

#drop unimportance features
dropvar73 <- filter(dfimp73, MeanDecreaseGini < 0.000005)
dropvar73 <- as.vector(dropvar73$variable)
smote_data73 <- select(smote_data73, -c(dropvar73))

#tuning for optimum mtry
pred <- select(smote_data73, -c(lapse))
rftune73 <- tuneRF(pred,smote_data73$lapse,1,ntreeTry = opntree73,stepFactor = 2,improve = 0.01,trace = TRUE,plot=TRUE,doBest = FALSE)
rftunedf73 <- as.data.frame(rftune73)
plottune73 <- ggplot(rftunedf73,aes(x=mtry,y=OOBError)) + geom_line(color="red") + geom_point(color="red4")
print(plottune73)

#train model with optimum par
set.seed(123456)
modelrfop73 <- randomForest(lapse ~ ., data = smote_data73, ntree = opntree73, mtry = 8, importance = TRUE)
print(modelrfop73)


#----C. TESTING----

#----C.1. DROP UNIMPORTANCE FEATURES & SET SAME LEVEL FACTOR ON TEST DATA----

#drop unimportance features on test data based on importance data train
datatest11 <- select(datatest11, -c(dropvar11))
datatest11 <- lapply(datatest11,as.factor)
datatest11 <- as.data.frame(datatest11)

datatest12 <- select(datatest12, -c(dropvar12))
datatest12 <- lapply(datatest12,as.factor)
datatest12 <- as.data.frame(datatest12)

datatest13 <- select(datatest13, -c(dropvar13))
datatest13 <- lapply(datatest13,as.factor)
datatest13 <- as.data.frame(datatest13)

datatest31 <- select(datatest31, -c(dropvar31))
datatest31 <- lapply(datatest31,as.factor)
datatest31 <- as.data.frame(datatest31)

datatest32 <- select(datatest32, -c(dropvar32))
datatest32 <- lapply(datatest32,as.factor)
datatest32 <- as.data.frame(datatest32)

datatest33 <- select(datatest33, -c(dropvar33))
datatest33 <- lapply(datatest33,as.factor)
datatest33 <- as.data.frame(datatest33)

datatest41 <- select(datatest41, -c(dropvar41))
datatest41 <- lapply(datatest41,as.factor)
datatest41 <- as.data.frame(datatest41)

datatest42 <- select(datatest42, -c(dropvar42))
datatest42 <- lapply(datatest42,as.factor)
datatest42 <- as.data.frame(datatest42)

datatest51 <- select(datatest51, -c(dropvar51))
datatest51 <- lapply(datatest51,as.factor)
datatest51 <- as.data.frame(datatest51)

datatest71 <- select(datatest71, -c(dropvar71))
datatest71 <- lapply(datatest71,as.factor)
datatest71 <- as.data.frame(datatest71)

datatest72 <- select(datatest72, -c(dropvar72))
datatest72 <- lapply(datatest72,as.factor)
datatest72 <- as.data.frame(datatest72)

datatest73 <- select(datatest73, -c(dropvar73))
datatest73 <- lapply(datatest73,as.factor)
datatest73 <- as.data.frame(datatest73)

#set same nlevel on training & testing data
for (z in names(smote_data11)){
  tmp2 <- get("z")
  while (nlevels(smote_data11[[tmp2]]) < nlevels(datatest11[[tmp2]])){
    smote_data11[[tmp2]] <- factor(smote_data11[[tmp2]], levels = levels(datatest11[[tmp2]]))
    smote_data11[[tmp2]] <- smote_data11[[tmp2]]
  }
  assign(z,tmp2)
}
for (z in names(smote_data11)){
  tmp3 <- get("z")
  while (nlevels(datatest11[[tmp3]]) < nlevels(smote_data11[[tmp3]])){
    datatest11[[tmp3]] <- factor(datatest11[[tmp3]], levels = levels(smote_data11[[tmp3]]))
    datatest11[[tmp3]] <- datatest11[[tmp3]]
  }
  assign(z,tmp3)
}

for (z in names(smote_data12)){
  tmp2 <- get("z")
  while (nlevels(smote_data12[[tmp2]]) < nlevels(datatest12[[tmp2]])){
    smote_data12[[tmp2]] <- factor(smote_data12[[tmp2]], levels = levels(datatest12[[tmp2]]))
    smote_data12[[tmp2]] <- smote_data12[[tmp2]]
  }
  assign(z,tmp2)
}
for (z in names(smote_data12)){
  tmp3 <- get("z")
  while (nlevels(datatest12[[tmp3]]) < nlevels(smote_data12[[tmp3]])){
    datatest12[[tmp3]] <- factor(datatest12[[tmp3]], levels = levels(smote_data12[[tmp3]]))
    datatest12[[tmp3]] <- datatest12[[tmp3]]
  }
  assign(z,tmp3)
}

for (z in names(smote_data13)){
  tmp2 <- get("z")
  while (nlevels(smote_data13[[tmp2]]) < nlevels(datatest13[[tmp2]])){
    smote_data13[[tmp2]] <- factor(smote_data13[[tmp2]], levels = levels(datatest13[[tmp2]]))
    smote_data13[[tmp2]] <- smote_data13[[tmp2]]
  }
  assign(z,tmp2)
}
for (z in names(smote_data13)){
  tmp3 <- get("z")
  while (nlevels(datatest13[[tmp3]]) < nlevels(smote_data13[[tmp3]])){
    datatest13[[tmp3]] <- factor(datatest13[[tmp3]], levels = levels(smote_data13[[tmp3]]))
    datatest13[[tmp3]] <- datatest13[[tmp3]]
  }
  assign(z,tmp3)
}

for (z in names(smote_data31)){
  tmp2 <- get("z")
  while (nlevels(smote_data31[[tmp2]]) < nlevels(datatest31[[tmp2]])){
    smote_data31[[tmp2]] <- factor(smote_data31[[tmp2]], levels = levels(datatest31[[tmp2]]))
    smote_data31[[tmp2]] <- smote_data31[[tmp2]]
  }
  assign(z,tmp2)
}
for (z in names(smote_data31)){
  tmp3 <- get("z")
  while (nlevels(datatest31[[tmp3]]) < nlevels(smote_data31[[tmp3]])){
    datatest31[[tmp3]] <- factor(datatest31[[tmp3]], levels = levels(smote_data31[[tmp3]]))
    datatest31[[tmp3]] <- datatest31[[tmp3]]
  }
  assign(z,tmp3)
}

for (z in names(smote_data32)){
  tmp2 <- get("z")
  while (nlevels(smote_data32[[tmp2]]) < nlevels(datatest32[[tmp2]])){
    smote_data32[[tmp2]] <- factor(smote_data32[[tmp2]], levels = levels(datatest32[[tmp2]]))
    smote_data32[[tmp2]] <- smote_data32[[tmp2]]
  }
  assign(z,tmp2)
}
for (z in names(smote_data32)){
  tmp3 <- get("z")
  while (nlevels(datatest32[[tmp3]]) < nlevels(smote_data32[[tmp3]])){
    datatest32[[tmp3]] <- factor(datatest32[[tmp3]], levels = levels(smote_data32[[tmp3]]))
    datatest32[[tmp3]] <- datatest32[[tmp3]]
  }
  assign(z,tmp3)
}

for (z in names(smote_data33)){
  tmp2 <- get("z")
  while (nlevels(smote_data33[[tmp2]]) < nlevels(datatest33[[tmp2]])){
    smote_data33[[tmp2]] <- factor(smote_data33[[tmp2]], levels = levels(datatest33[[tmp2]]))
    smote_data33[[tmp2]] <- smote_data33[[tmp2]]
  }
  assign(z,tmp2)
}
for (z in names(smote_data33)){
  tmp3 <- get("z")
  while (nlevels(datatest33[[tmp3]]) < nlevels(smote_data33[[tmp3]])){
    datatest33[[tmp3]] <- factor(datatest33[[tmp3]], levels = levels(smote_data33[[tmp3]]))
    datatest33[[tmp3]] <- datatest33[[tmp3]]
  }
  assign(z,tmp3)
}

for (z in names(smote_data41)){
  tmp2 <- get("z")
  while (nlevels(smote_data41[[tmp2]]) < nlevels(datatest41[[tmp2]])){
    smote_data41[[tmp2]] <- factor(smote_data41[[tmp2]], levels = levels(datatest41[[tmp2]]))
    smote_data41[[tmp2]] <- smote_data41[[tmp2]]
  }
  assign(z,tmp2)
}
for (z in names(smote_data41)){
  tmp3 <- get("z")
  while (nlevels(datatest41[[tmp3]]) < nlevels(smote_data41[[tmp3]])){
    datatest41[[tmp3]] <- factor(datatest41[[tmp3]], levels = levels(smote_data41[[tmp3]]))
    datatest41[[tmp3]] <- datatest41[[tmp3]]
  }
  assign(z,tmp3)
}

for (z in names(smote_data42)){
  tmp2 <- get("z")
  while (nlevels(smote_data42[[tmp2]]) < nlevels(datatest42[[tmp2]])){
    smote_data42[[tmp2]] <- factor(smote_data42[[tmp2]], levels = levels(datatest42[[tmp2]]))
    smote_data42[[tmp2]] <- smote_data42[[tmp2]]
  }
  assign(z,tmp2)
}
for (z in names(smote_data42)){
  tmp3 <- get("z")
  while (nlevels(datatest42[[tmp3]]) < nlevels(smote_data42[[tmp3]])){
    datatest42[[tmp3]] <- factor(datatest42[[tmp3]], levels = levels(smote_data42[[tmp3]]))
    datatest42[[tmp3]] <- datatest42[[tmp3]]
  }
  assign(z,tmp3)
}

for (z in names(smote_data51)){
  tmp2 <- get("z")
  while (nlevels(smote_data51[[tmp2]]) < nlevels(datatest51[[tmp2]])){
    smote_data51[[tmp2]] <- factor(smote_data51[[tmp2]], levels = levels(datatest51[[tmp2]]))
    smote_data51[[tmp2]] <- smote_data51[[tmp2]]
  }
  assign(z,tmp2)
}
for (z in names(smote_data51)){
  tmp3 <- get("z")
  while (nlevels(datatest51[[tmp3]]) < nlevels(smote_data51[[tmp3]])){
    datatest51[[tmp3]] <- factor(datatest51[[tmp3]], levels = levels(smote_data51[[tmp3]]))
    datatest51[[tmp3]] <- datatest51[[tmp3]]
  }
  assign(z,tmp3)
}

for (z in names(smote_data71)){
  tmp2 <- get("z")
  while (nlevels(smote_data71[[tmp2]]) < nlevels(datatest71[[tmp2]])){
    smote_data71[[tmp2]] <- factor(smote_data71[[tmp2]], levels = levels(datatest71[[tmp2]]))
    smote_data71[[tmp2]] <- smote_data71[[tmp2]]
  }
  assign(z,tmp2)
}
for (z in names(smote_data71)){
  tmp3 <- get("z")
  while (nlevels(datatest71[[tmp3]]) < nlevels(smote_data71[[tmp3]])){
    datatest71[[tmp3]] <- factor(datatest71[[tmp3]], levels = levels(smote_data71[[tmp3]]))
    datatest71[[tmp3]] <- datatest71[[tmp3]]
  }
  assign(z,tmp3)
}

for (z in names(smote_data72)){
  tmp2 <- get("z")
  while (nlevels(smote_data72[[tmp2]]) < nlevels(datatest72[[tmp2]])){
    smote_data72[[tmp2]] <- factor(smote_data72[[tmp2]], levels = levels(datatest72[[tmp2]]))
    smote_data72[[tmp2]] <- smote_data72[[tmp2]]
  }
  assign(z,tmp2)
}
for (z in names(smote_data72)){
  tmp3 <- get("z")
  while (nlevels(datatest72[[tmp3]]) < nlevels(smote_data72[[tmp3]])){
    datatest72[[tmp3]] <- factor(datatest72[[tmp3]], levels = levels(smote_data72[[tmp3]]))
    datatest72[[tmp3]] <- datatest72[[tmp3]]
  }
  assign(z,tmp3)
}

for (z in names(smote_data73)){
  tmp2 <- get("z")
  while (nlevels(smote_data73[[tmp2]]) < nlevels(datatest73[[tmp2]])){
    smote_data73[[tmp2]] <- factor(smote_data73[[tmp2]], levels = levels(datatest73[[tmp2]]))
    smote_data73[[tmp2]] <- smote_data73[[tmp2]]
  }
  assign(z,tmp2)
}
for (z in names(smote_data73)){
  tmp3 <- get("z")
  while (nlevels(datatest73[[tmp3]]) < nlevels(smote_data73[[tmp3]])){
    datatest73[[tmp3]] <- factor(datatest73[[tmp3]], levels = levels(smote_data73[[tmp3]]))
    datatest73[[tmp3]] <- datatest73[[tmp3]]
  }
  assign(z,tmp3)
}


#----C.2. TESTING & PREDICTION----

#----C.2.1 TESTING & PREDICTION DATA11---- 

#Obtain optimum ntree for both training & testing
set.seed(123456)
predtr <- select(smote_data11, -c(lapse))
predte <- select(datatest11, -c(lapse))
modelrfop11 <- randomForest(x = predtr, y = smote_data11$lapse, xtest = predte, ytest = datatest11$lapse, importance = TRUE)
print(modelrfop11)
#plotntree <- plot(modelrfop11)
opntree11 <- which.min(modelrfop11$err.rate)
print(opntree11)

#Tuning optimum mtry for both training & testing
accuracytrain=c()
accuracytest=c()
i=1
for (i in 1:10) {
  modelrfop11 <- randomForest(lapse ~ ., data = smote_data11, ntree = opntree11, mtry = i, importance = TRUE)
  accuracytrain[i] <- as.vector(1 - (modelrfop11$err.rate[opntree11,1]))
  predtest1 <- predict(modelrfop11,datatest11)
  accuracytest[i] <- mean(predtest1 == datatest11$lapse)
}
err <- as.data.frame(cbind(accuracytrain, accuracytest))

plottune11 <- ggplot(err,aes(x=c(1:10))) + geom_line(aes(y=accuracytrain, color="Train  Accuracy Rate")) + geom_line(aes(y=accuracytest, color="Test Accuracy Rate")) +
  ggtitle("Optimum mtry for Both Training and Testing Data") + xlab("mtry") + ylab("Accuracy Rate") + scale_color_manual("Data Type ", values = c("Train  Accuracy Rate"="red", "Test Accuracy Rate"="blue")) + 
  scale_x_continuous(breaks = c(1:10))
print(plottune11)

#Model with optimum par for both training & testing
set.seed(123456)
predtr11 <- select(smote_data11, -c(lapse))
predte11 <- select(datatest11, -c(lapse))
modelrfop11 <- randomForest(x = predtr, y = smote_data11$lapse, xtest = predte, ytest = datatest11$lapse, 
                            ntree = 1194, mtry = 2, importance = TRUE)
print(modelrfop11)

#Model for prediction
modelrfop11 <- randomForest(lapse ~ ., data = smote_data11, ntree = 250, mtry = 2, importance = TRUE)
print(modelrfop11)

#Extract rule
treelist11 <-RF2List(modelrfop11)
extract11 <- extractRules(treelist11, predtr11,ntree = 250)
rulemetric11 <- getRuleMetric(extract11,predtr11,smote_data11$lapse)
rulemetric11 <- pruneRule(rulemetric11,predtr11,smote_data11$lapse,maxDecay = 0.05,typeDecay = 2)
rulemetric11 <- selectRuleRRF(rulemetric11,predtr11,smote_data11$lapse)
learner11 <- buildLearner(rulemetric11,predtr11,smote_data11$lapse)
sim_learner11 <- presentRules(rulemetric11,colnames(predtr11))

#Creating prediction output
pred11 <- predict(modelrfop11,datapred11,type = "prob")
pred11 <- as.data.frame(pred11)

summary(datapred11$lapse)

#Export prediction output & rule
write.csv(datatest11, "D:/FOLDER AFGANTA/Revisit Anti Lapse/Modelling/test11.csv")
write.csv(pred11, "D:/FOLDER AFGANTA/Revisit Anti Lapse/Modelling/Output/pred11.csv")
write.csv(sim_learner11, "D:/FOLDER AFGANTA/Revisit Anti Lapse/Modelling/Output/rule11.csv")

#----C.2.2 TESTING & PREDICTION DATA12---- 

#Obtain optimum ntree for both training & testing
set.seed(123456)
predtr <- select(smote_data12, -c(lapse))
predte <- select(datatest12, -c(lapse))
modelrfop12 <- randomForest(x = predtr, y = smote_data12$lapse, xtest = predte, ytest = datatest12$lapse, importance = TRUE)
print(modelrfop12)
#plotntree <- plot(modelrfop12)
opntree12 <- which.min(modelrfop12$err.rate)
print(opntree12)

#Tuning optimum mtry for both training & testing
accuracytrain=c()
accuracytest=c()
i=1
for (i in 1:10) {
  modelrfop12 <- randomForest(lapse ~ ., data = smote_data12, ntree = opntree12, mtry = i, importance = TRUE)
  accuracytrain[i] <- as.vector(1 - (modelrfop12$err.rate[opntree12,1]))
  predtest1 <- predict(modelrfop12,datatest12)
  accuracytest[i] <- mean(predtest1 == datatest12$lapse)
}
err <- as.data.frame(cbind(accuracytrain, accuracytest))

plottune12 <- ggplot(err,aes(x=c(1:10))) + geom_line(aes(y=accuracytrain, color="Train  Accuracy Rate")) + geom_line(aes(y=accuracytest, color="Test Accuracy Rate")) +
  ggtitle("Optimum mtry for Both Training and Testing Data") + xlab("mtry") + ylab("Accuracy Rate") + scale_color_manual("Data Type ", values = c("Train  Accuracy Rate"="red", "Test Accuracy Rate"="blue")) + 
  scale_x_continuous(breaks = c(1:10))
print(plottune12)

#Model with optimum par for both training & testing
set.seed(123456)
predtr12 <- select(smote_data12, -c(lapse))
predte12 <- select(datatest12, -c(lapse))
modelrfop12 <- randomForest(x = predtr, y = smote_data12$lapse, xtest = predte, ytest = datatest12$lapse, 
                            ntree = 500, mtry = 2, importance = TRUE)
print(modelrfop12)

#Model for prediction
modelrfop12 <- randomForest(lapse ~ ., data = smote_data12, ntree = 500, mtry = 2, importance = TRUE)
print(modelrfop12)

#Extract rule
treelist12 <-RF2List(modelrfop12)
extract12 <- extractRules(treelist12, predtr12)
rulemetric12 <- getRuleMetric(extract12,predtr12,smote_data12$lapse)
rulemetric12 <- pruneRule(rulemetric12,predtr12,smote_data12$lapse,maxDecay = 0.05,typeDecay = 2)
rulemetric12 <- selectRuleRRF(rulemetric12,predtr12,smote_data12$lapse)
learner12 <- buildLearner(rulemetric12,predtr12,smote_data12$lapse)
sim_learner12 <- presentRules(rulemetric12,colnames(predtr12))

#Creating prediction output
pred12 <- predict(modelrfop12,datapred12,type = "prob")
pred12 <- as.data.frame(pred12)

summary(datapred12$lapse)

#Export prediction output & rule
write.csv(datatest12, "D:/FOLDER AFGANTA/Revisit Anti Lapse/Modelling/test12.csv")
write.csv(pred12, "D:/FOLDER AFGANTA/Revisit Anti Lapse/Modelling/Output/pred12.csv")
write.csv(sim_learner12, "D:/FOLDER AFGANTA/Revisit Anti Lapse/Modelling/Output/rule12.csv")

#----C.2.3 TESTING & PREDICTION DATA13---- 

#Obtain optimum ntree for both training & testing
set.seed(123456)
predtr <- select(smote_data13, -c(lapse))
predte <- select(datatest13, -c(lapse))
modelrfop13 <- randomForest(x = predtr, y = smote_data13$lapse, xtest = predte, ytest = datatest13$lapse, importance = TRUE)
print(modelrfop13)
#plotntree <- plot(modelrfop13)
opntree13 <- which.min(modelrfop13$err.rate)
print(opntree13)

#Tuning optimum mtry for both training & testing
accuracytrain=c()
accuracytest=c()
i=1
for (i in 1:10) {
  modelrfop13 <- randomForest(lapse ~ ., data = smote_data13, ntree =  opntree13, mtry = i, importance = TRUE)
  accuracytrain[i] <- as.vector(1 - (modelrfop13$err.rate[opntree13,1]))
  predtest1 <- predict(modelrfop13,datatest13)
  accuracytest[i] <- mean(predtest1 == datatest13$lapse)
}
err <- as.data.frame(cbind(accuracytrain, accuracytest))

plottune13 <- ggplot(err,aes(x=c(1:10))) + geom_line(aes(y=accuracytrain, color="Train  Accuracy Rate")) + geom_line(aes(y=accuracytest, color="Test Accuracy Rate")) +
  ggtitle("Optimum mtry for Both Training and Testing Data") + xlab("mtry") + ylab("Accuracy Rate") + scale_color_manual("Data Type ", values = c("Train  Accuracy Rate"="red", "Test Accuracy Rate"="blue")) + 
  scale_x_continuous(breaks = c(1:10))
print(plottune13)

#Model with optimum par for both training & testing
set.seed(123456)
predtr13 <- select(smote_data13, -c(lapse))
predte13 <- select(datatest13, -c(lapse))
modelrfop13 <- randomForest(x = predtr, y = smote_data13$lapse, xtest = predte, ytest = datatest13$lapse, 
                            ntree = 500, mtry = 2, importance = TRUE)
print(modelrfop13)

#Model for prediction
modelrfop13 <- randomForest(lapse ~ ., data = smote_data13, ntree = 500, mtry = 2, importance = TRUE)
print(modelrfop13)

#Extract rule
treelist13 <-RF2List(modelrfop13)
extract13 <- extractRules(treelist13, predtr13)
rulemetric13 <- getRuleMetric(extract13,predtr13,smote_data13$lapse)
rulemetric13 <- pruneRule(rulemetric13,predtr13,smote_data13$lapse,maxDecay = 0.05,typeDecay = 2)
rulemetric13 <- selectRuleRRF(rulemetric13,predtr13,smote_data13$lapse)
learner13 <- buildLearner(rulemetric13,predtr13,smote_data13$lapse)
sim_learner13 <- presentRules(rulemetric13,colnames(predtr13))

#Creating prediction output
pred13 <- predict(modelrfop13,datapred13,type = "prob")
pred13 <- as.data.frame(pred13)

summary(datapred13$lapse)

#Export prediction output & rule
write.csv(datatest13, "D:/FOLDER AFGANTA/Revisit Anti Lapse/Modelling/test13.csv")
write.csv(pred13, "D:/FOLDER AFGANTA/Revisit Anti Lapse/Modelling/Output/pred13.csv")
write.csv(sim_learner13, "D:/FOLDER AFGANTA/Revisit Anti Lapse/Modelling/Output/rule13.csv")

#----C.2.4 TESTING & PREDICTION DATA31---- 

#Obtain optimum ntree for both training & testing
set.seed(123456)
predtr <- select(smote_data31, -c(lapse))
predte <- select(datatest31, -c(lapse))
modelrfop31 <- randomForest(x = predtr, y = smote_data31$lapse, xtest = predte, ytest = datatest31$lapse, importance = TRUE)
print(modelrfop31)
#plotntree <- plot(modelrfop31)
opntree31 <- which.min(modelrfop31$err.rate)
print(opntree31)

#Tuning optimum mtry for both training & testing
accuracytrain=c()
accuracytest=c()
i=1
for (i in 1:10) {
  modelrfop31 <- randomForest(lapse ~ ., data = smote_data31, ntree = opntree31, mtry = i, importance = TRUE)
  accuracytrain[i] <- as.vector(1 - (modelrfop31$err.rate[opntree31,1]))
  predtest1 <- predict(modelrfop31,datatest31)
  accuracytest[i] <- mean(predtest1 == datatest31$lapse)
}
err <- as.data.frame(cbind(accuracytrain, accuracytest))

plottune31 <- ggplot(err,aes(x=c(1:10))) + geom_line(aes(y=accuracytrain, color="Train  Accuracy Rate")) + geom_line(aes(y=accuracytest, color="Test Accuracy Rate")) +
  ggtitle("Optimum mtry for Both Training and Testing Data") + xlab("mtry") + ylab("Accuracy Rate") + scale_color_manual("Data Type ", values = c("Train  Accuracy Rate"="red", "Test Accuracy Rate"="blue")) + 
  scale_x_continuous(breaks = c(1:10))
print(plottune31)

#Model with optimum par for both training & testing
set.seed(123456)
predtr31 <- select(smote_data31, -c(lapse))
predte31 <- select(datatest31, -c(lapse))
modelrfop31 <- randomForest(x = predtr31, y = smote_data31$lapse, xtest = predte31, ytest = datatest31$lapse, 
                            ntree = 903, mtry = 6, importance = TRUE)
print(modelrfop31)

#Model for prediction
modelrfop31 <- randomForest(lapse ~ ., data = smote_data31, ntree = 903, mtry = 6, importance = TRUE)
print(modelrfop31)

#Extract rule
treelist31 <-RF2List(modelrfop31)
extract31 <- extractRules(treelist31, predtr31)
rulemetric31 <- getRuleMetric(extract31,predtr31,smote_data31$lapse)
rulemetric31 <- pruneRule(rulemetric31,predtr31,smote_data31$lapse,maxDecay = 0.05,typeDecay = 2)
rulemetric31 <- selectRuleRRF(rulemetric31,predtr31,smote_data31$lapse)
learner31 <- buildLearner(rulemetric31,predtr31,smote_data31$lapse)
sim_learner31 <- presentRules(rulemetric31,colnames(predtr31))

#Creating prediction output for testdata
test31 <- predict(modelrfop31,datatest31,type = "prob")
test31 <- as.data.frame(test31)

#Creating prediction output
pred31 <- predict(modelrfop31,datapred31,type = "prob")
pred31 <- as.data.frame(pred31)

summary(datapred31$lapse)

#Export prediction output & rule
write.csv(datatest31, "D:/FOLDER AFGANTA/Revisit Anti Lapse/Modelling/datatest31.csv")
write.csv(pred31, "D:/FOLDER AFGANTA/Revisit Anti Lapse/Modelling/Output/pred31.csv")
write.csv(test31, "D:/FOLDER AFGANTA/Revisit Anti Lapse/Modelling/test31.csv")
write.csv(sim_learner31, "D:/FOLDER AFGANTA/Revisit Anti Lapse/Modelling/Output/rule31.csv")

#----C.2.5 TESTING & PREDICTION DATA32---- 

#Obtain optimum ntree for both training & testing
set.seed(123456)
predtr <- select(smote_data32, -c(lapse))
predte <- select(datatest32, -c(lapse))
modelrfop32 <- randomForest(x = predtr, y = smote_data32$lapse, xtest = predte, ytest = datatest32$lapse, importance = TRUE)
print(modelrfop32)
#plotntree <- plot(modelrfop32)
opntree32 <- which.min(modelrfop32$err.rate)
print(opntree32)

#Tuning optimum mtry for both training & testing
accuracytrain=c()
accuracytest=c()
i=1
for (i in 1:10) {
  modelrfop32 <- randomForest(lapse ~ ., data = smote_data32, ntree = opntree32, mtry = i, importance = TRUE)
  accuracytrain[i] <- as.vector(1 - (modelrfop32$err.rate[opntree32,1]))
  predtest1 <- predict(modelrfop32,datatest32)
  accuracytest[i] <- mean(predtest1 == datatest32$lapse)
}
err <- as.data.frame(cbind(accuracytrain, accuracytest))

plottune32 <- ggplot(err,aes(x=c(1:10))) + geom_line(aes(y=accuracytrain, color="Train  Accuracy Rate")) + geom_line(aes(y=accuracytest, color="Test Accuracy Rate")) +
  ggtitle("Optimum mtry for Both Training and Testing Data") + xlab("mtry") + ylab("Accuracy Rate") + scale_color_manual("Data Type ", values = c("Train  Accuracy Rate"="red", "Test Accuracy Rate"="blue")) + 
  scale_x_continuous(breaks = c(1:10))
print(plottune32)

#Model with optimum par for both training & testing
set.seed(123456)
predtr32 <- select(smote_data32, -c(lapse))
predte32 <- select(datatest32, -c(lapse))
modelrfop32 <- randomForest(x = predtr, y = smote_data32$lapse, xtest = predte, ytest = datatest32$lapse, 
                            ntree = 776, mtry = 4, importance = TRUE)
print(modelrfop32)

#Model for prediction
modelrfop32 <- randomForest(lapse ~ ., data = smote_data32, ntree = 500, mtry = 3, importance = TRUE)
print(modelrfop32)

#Extract rule
treelist32 <-RF2List(modelrfop32)
extract32 <- extractRules(treelist32, predtr32)
rulemetric32 <- getRuleMetric(extract32,predtr32,smote_data32$lapse)
rulemetric32 <- pruneRule(rulemetric32,predtr32,smote_data32$lapse,maxDecay = 0.05,typeDecay = 2)
rulemetric32 <- selectRuleRRF(rulemetric32,predtr32,smote_data32$lapse)
learner32 <- buildLearner(rulemetric32,predtr32,smote_data32$lapse)
sim_learner32 <- presentRules(rulemetric32,colnames(predtr32))

#Creating prediction output
pred32 <- predict(modelrfop32,datapred32,type = "prob")
pred32 <- as.data.frame(pred32)

summary(datapred32$lapse)

#Export prediction output & rule
write.csv(datatest32, "D:/FOLDER AFGANTA/Revisit Anti Lapse/Modelling/test32.csv")
write.csv(pred32, "D:/FOLDER AFGANTA/Revisit Anti Lapse/Modelling/Output/pred32.csv")
write.csv(sim_learner32, "D:/FOLDER AFGANTA/Revisit Anti Lapse/Modelling/Output/rule32.csv")

#----C.2.6 TESTING & PREDICTION DATA33---- 

#Obtain optimum ntree for both training & testing
set.seed(123456)
predtr <- select(smote_data33, -c(lapse))
predte <- select(datatest33, -c(lapse))
modelrfop33 <- randomForest(x = predtr, y = smote_data33$lapse, xtest = predte, ytest = datatest33$lapse, importance = TRUE)
print(modelrfop33)
#plotntree <- plot(modelrfop33)
opntree33 <- which.min(modelrfop33$err.rate)
print(opntree33)

#Tuning optimum mtry for both training & testing
accuracytrain=c()
accuracytest=c()
i=1
for (i in 1:10) {
  modelrfop33 <- randomForest(lapse ~ ., data = smote_data33, ntree = 500, mtry = i, importance = TRUE)
  accuracytrain[i] <- as.vector(1 - (modelrfop33$err.rate[500,1]))
  predtest1 <- predict(modelrfop33,datatest33)
  accuracytest[i] <- mean(predtest1 == datatest33$lapse)
}
err <- as.data.frame(cbind(accuracytrain, accuracytest))

plottune33 <- ggplot(err,aes(x=c(1:10))) + geom_line(aes(y=accuracytrain, color="Train  Accuracy Rate")) + geom_line(aes(y=accuracytest, color="Test Accuracy Rate")) +
  ggtitle("Optimum mtry for Both Training and Testing Data") + xlab("mtry") + ylab("Accuracy Rate") + scale_color_manual("Data Type ", values = c("Train  Accuracy Rate"="red", "Test Accuracy Rate"="blue")) + 
  scale_x_continuous(breaks = c(1:10))
print(plottune33)

#Model with optimum par for both training & testing
set.seed(123456)
predtr33 <- select(smote_data33, -c(lapse))
predte33 <- select(datatest33, -c(lapse))
modelrfop33 <- randomForest(x = predtr, y = smote_data33$lapse, xtest = predte, ytest = datatest33$lapse, 
                            ntree = 100, mtry = 3, importance = TRUE)
print(modelrfop33)

#Model for prediction
modelrfop33 <- randomForest(lapse ~ ., data = smote_data33, ntree = 100, mtry = 3, importance = TRUE)
print(modelrfop33)

#Extract rule
treelist33 <-RF2List(modelrfop33)
extract33 <- extractRules(treelist33, predtr33,ntree = 100)
rulemetric33 <- getRuleMetric(extract33,predtr33,smote_data33$lapse)
rulemetric33 <- pruneRule(rulemetric33,predtr33,smote_data33$lapse,maxDecay = 0.05,typeDecay = 2)
rulemetric33 <- selectRuleRRF(rulemetric33,predtr33,smote_data33$lapse)
learner33 <- buildLearner(rulemetric33,predtr33,smote_data33$lapse)
sim_learner33 <- presentRules(rulemetric33,colnames(predtr33))

#Creating prediction output
pred33 <- predict(modelrfop33,datapred33,type = "prob")
pred33 <- as.data.frame(pred33)

summary(datapred33$lapse)

#Export prediction output & rule
write.csv(datatest33, "D:/FOLDER AFGANTA/Revisit Anti Lapse/Modelling/test33.csv")
write.csv(pred33, "D:/FOLDER AFGANTA/Revisit Anti Lapse/Modelling/Output/pred33.csv")
write.csv(sim_learner33, "D:/FOLDER AFGANTA/Revisit Anti Lapse/Modelling/Output/rule33.csv")

#----C.2.7 TESTING & PREDICTION DATA41---- 

#Add new factor level
levels(predtr$bank) <- c(levels(predtr$bank), "10")
levels(predte$bank) <- c(levels(predte$bank), "8")

#Obtain optimum ntree for both training & testing
set.seed(123456)
predtr <- select(smote_data41, -c(lapse))
predte <- select(datatest41, -c(lapse))

#Add new factor level
levels(predtr$bank) <- c(levels(predtr$bank), "10")
levels(predte$bank) <- c(levels(predte$bank), "8")

modelrfop41 <- randomForest(x = predtr, y = smote_data41$lapse, xtest = predte, ytest = datatest41$lapse, importance = TRUE)
print(modelrfop41)
#plotntree <- plot(modelrfop41)
opntree41 <- which.min(modelrfop41$err.rate)
print(opntree41)

#Tuning optimum mtry for both training & testing
accuracytrain=c()
accuracytest=c()
i=1
for (i in 1:10) {
  modelrfop41 <- randomForest(lapse ~ ., data = smote_data41, ntree = opntree41, mtry = i, importance = TRUE)
  accuracytrain[i] <- as.vector(1 - (modelrfop41$err.rate[opntree41,1]))
  predtest1 <- predict(modelrfop41,datatest41)
  accuracytest[i] <- mean(predtest1 == datatest41$lapse)
}
err <- as.data.frame(cbind(accuracytrain, accuracytest))

plottune41 <- ggplot(err,aes(x=c(1:10))) + geom_line(aes(y=accuracytrain, color="Train  Accuracy Rate")) + geom_line(aes(y=accuracytest, color="Test Accuracy Rate")) +
  ggtitle("Optimum mtry for Both Training and Testing Data") + xlab("mtry") + ylab("Accuracy Rate") + scale_color_manual("Data Type ", values = c("Train  Accuracy Rate"="red", "Test Accuracy Rate"="blue")) + 
  scale_x_continuous(breaks = c(1:10))
print(plottune41)

#Add new factor level
levels(smote_data41$bank) <- c(levels(smote_data41$bank), "10")
levels(datapred41$bank) <- c(levels(datapred41$bank), "8")

#Model with optimum par for both training & testing
set.seed(123456)
predtr41 <- select(smote_data41, -c(lapse))
predte41 <- select(datatest41, -c(lapse))
modelrfop41 <- randomForest(x = predtr, y = smote_data41$lapse, xtest = predte, ytest = datatest41$lapse, 
                            ntree = 250, mtry = 8, importance = TRUE)
print(modelrfop41)

#Model for prediction
modelrfop41 <- randomForest(lapse ~ ., data = smote_data41, ntree = 250, mtry = 8, importance = TRUE)
print(modelrfop41)

#Extract rule
treelist41 <-RF2List(modelrfop41)
extract41 <- extractRules(treelist41, predtr41, ntree = 250)
rulemetric41 <- getRuleMetric(extract41,predtr41,smote_data41$lapse)
rulemetric41 <- pruneRule(rulemetric41,predtr41,smote_data41$lapse,maxDecay = 0.05,typeDecay = 2)
rulemetric41 <- selectRuleRRF(rulemetric41,predtr41,smote_data41$lapse)
learner41 <- buildLearner(rulemetric41,predtr41,smote_data41$lapse)
sim_learner41 <- presentRules(rulemetric41,colnames(predtr41))

#Creating prediction output
pred41 <- predict(modelrfop41,datapred41,type = "prob")
pred41 <- as.data.frame(pred41)

summary(datapred41$lapse)

#Export prediction output & rule
write.csv(datatest41, "D:/FOLDER AFGANTA/Revisit Anti Lapse/Modelling/test41.csv")
write.csv(pred41, "D:/FOLDER AFGANTA/Revisit Anti Lapse/Modelling/Output/pred41.csv")
write.csv(sim_learner41, "D:/FOLDER AFGANTA/Revisit Anti Lapse/Modelling/Output/rule41.csv")

#----C.2.8 TESTING & PREDICTION DATA42---- 

#Obtain optimum ntree for both training & testing
set.seed(123456)
predtr <- select(smote_data42, -c(lapse))
predte <- select(datatest42, -c(lapse))
modelrfop42 <- randomForest(x = predtr, y = smote_data42$lapse, xtest = predte, ytest = datatest42$lapse, importance = TRUE)
print(modelrfop42)
#plotntree <- plot(modelrfop42)
opntree42 <- which.min(modelrfop42$err.rate)
print(opntree42)

#Tuning optimum mtry for both training & testing
accuracytrain=c()
accuracytest=c()
i=1
for (i in 1:10) {
  modelrfop42 <- randomForest(lapse ~ ., data = smote_data42, ntree = opntree42, mtry = i, importance = TRUE)
  accuracytrain[i] <- as.vector(1 - (modelrfop42$err.rate[opntree42,1]))
  predtest1 <- predict(modelrfop42,datatest42)
  accuracytest[i] <- mean(predtest1 == datatest42$lapse)
}
err <- as.data.frame(cbind(accuracytrain, accuracytest))

plottune42 <- ggplot(err,aes(x=c(1:10))) + geom_line(aes(y=accuracytrain, color="Train  Accuracy Rate")) + geom_line(aes(y=accuracytest, color="Test Accuracy Rate")) +
  ggtitle("Optimum mtry for Both Training and Testing Data") + xlab("mtry") + ylab("Accuracy Rate") + scale_color_manual("Data Type ", values = c("Train  Accuracy Rate"="red", "Test Accuracy Rate"="blue")) + 
  scale_x_continuous(breaks = c(1:10))
print(plottune42)

#Model with optimum par for both training & testing
set.seed(123456)
predtr42 <- select(smote_data42, -c(lapse))
predte42 <- select(datatest42, -c(lapse))
modelrfop42 <- randomForest(x = predtr, y = smote_data42$lapse, xtest = predte, ytest = datatest42$lapse, 
                            ntree = 500, mtry = 6, importance = TRUE)
print(modelrfop42)

#Model for prediction
modelrfop42 <- randomForest(lapse ~ ., data = smote_data42, ntree = 500, mtry = 6, importance = TRUE)
print(modelrfop42)

#Extract rule
treelist42 <-RF2List(modelrfop42)
extract42 <- extractRules(treelist42, predtr42, ntree = 500)
rulemetric42 <- getRuleMetric(extract42,predtr42,smote_data42$lapse)
rulemetric42 <- pruneRule(rulemetric42,predtr42,smote_data42$lapse,maxDecay = 0.05,typeDecay = 2)
rulemetric42 <- selectRuleRRF(rulemetric42,predtr42,smote_data42$lapse)
learner42 <- buildLearner(rulemetric42,predtr42,smote_data42$lapse)
sim_learner42 <- presentRules(rulemetric42,colnames(predtr42))

#Creating prediction output
pred42 <- predict(modelrfop42,datapred42,type = "prob")
pred42 <- as.data.frame(pred42)

summary(datapred42$lapse)

#Export prediction output & rule
write.csv(datatest42, "D:/FOLDER AFGANTA/Revisit Anti Lapse/Modelling/test42.csv")
write.csv(pred42, "D:/FOLDER AFGANTA/Revisit Anti Lapse/Modelling/Output/pred42.csv")
write.csv(sim_learner42, "D:/FOLDER AFGANTA/Revisit Anti Lapse/Modelling/Output/rule42.csv")

#----C.2.9 TESTING & PREDICTION DATA51---- 

#Obtain optimum ntree for both training & testing
set.seed(123456)
predtr <- select(smote_data51, -c(lapse))
predte <- select(datatest51, -c(lapse))
modelrfop51 <- randomForest(x = predtr, y = smote_data51$lapse, xtest = predte, ytest = datatest51$lapse, importance = TRUE)
print(modelrfop51)
#plotntree <- plot(modelrfop51)
opntree51 <- which.min(modelrfop51$err.rate)
print(opntree51)

#Tuning optimum mtry for both training & testing
accuracytrain=c()
accuracytest=c()
i=1
for (i in 1:10) {
  modelrfop51 <- randomForest(lapse ~ ., data = smote_data51, ntree = 501, mtry = i, importance = TRUE)
  accuracytrain[i] <- as.vector(1 - (modelrfop51$err.rate[501,1]))
  predtest1 <- predict(modelrfop51,datatest51)
  accuracytest[i] <- mean(predtest1 == datatest51$lapse)
}
err <- as.data.frame(cbind(accuracytrain, accuracytest))

plottune51 <- ggplot(err,aes(x=c(1:10))) + geom_line(aes(y=accuracytrain, color="Train  Accuracy Rate")) + geom_line(aes(y=accuracytest, color="Test Accuracy Rate")) +
  ggtitle("Optimum mtry for Both Training and Testing Data") + xlab("mtry") + ylab("Accuracy Rate") + scale_color_manual("Data Type ", values = c("Train  Accuracy Rate"="red", "Test Accuracy Rate"="blue")) + 
  scale_x_continuous(breaks = c(1:10))
print(plottune51)

#Model with optimum par for both training & testing
set.seed(123456)
predtr51 <- select(smote_data51, -c(lapse))
predte51 <- select(datatest51, -c(lapse))
modelrfop51 <- randomForest(x = predtr, y = smote_data51$lapse, xtest = predte, ytest = datatest51$lapse, 
                            ntree = 100, mtry = 6, importance = TRUE)
print(modelrfop51)

#Model for prediction
modelrfop51 <- randomForest(lapse ~ ., data = smote_data51, ntree = 100, mtry = 6, importance = TRUE)
print(modelrfop51)

#Extract rule
treelist51 <-RF2List(modelrfop51)
extract51 <- extractRules(treelist51, predtr51, ntree = 100)
rulemetric51 <- getRuleMetric(extract51,predtr51,smote_data51$lapse)
rulemetric51 <- pruneRule(rulemetric51,predtr51,smote_data51$lapse,maxDecay = 0.05,typeDecay = 2)
rulemetric51 <- selectRuleRRF(rulemetric51,predtr51,smote_data51$lapse)
learner51 <- buildLearner(rulemetric51,predtr51,smote_data51$lapse)
sim_learner51 <- presentRules(rulemetric51,colnames(predtr51))

#Creating prediction output
pred51 <- predict(modelrfop51,datapred51,type = "prob")
pred51 <- as.data.frame(pred51)

summary(datapred51$lapse)

#Export prediction output & rule
write.csv(datatest51, "D:/FOLDER AFGANTA/Revisit Anti Lapse/Modelling/test51.csv")
write.csv(pred51, "D:/FOLDER AFGANTA/Revisit Anti Lapse/Modelling/Output/pred51.csv")
write.csv(sim_learner51, "D:/FOLDER AFGANTA/Revisit Anti Lapse/Modelling/Output/rule51.csv")

#----C.2.10 TESTING & PREDICTION DATA71---- 

#Obtain optimum ntree for both training & testing
set.seed(123456)
predtr <- select(smote_data71, -c(lapse))
predte <- select(datatest71, -c(lapse))
modelrfop71 <- randomForest(x = predtr, y = smote_data71$lapse, xtest = predte, ytest = datatest71$lapse, importance = TRUE)
print(modelrfop71)
#plotntree <- plot(modelrfop71)
opntree71 <- which.min(modelrfop71$err.rate)
print(opntree71)

#Tuning optimum mtry for both training & testing
accuracytrain=c()
accuracytest=c()
i=1
for (i in 1:10) {
  modelrfop71 <- randomForest(lapse ~ ., data = smote_data71, ntree = 1069, mtry = i, importance = TRUE)
  accuracytrain[i] <- as.vector(1 - (modelrfop71$err.rate[1069,1]))
  predtest1 <- predict(modelrfop71,datatest71)
  accuracytest[i] <- mean(predtest1 == datatest71$lapse)
}
err <- as.data.frame(cbind(accuracytrain, accuracytest))

plottune71 <- ggplot(err,aes(x=c(1:10))) + geom_line(aes(y=accuracytrain, color="Train  Accuracy Rate")) + geom_line(aes(y=accuracytest, color="Test Accuracy Rate")) +
  ggtitle("Optimum mtry for Both Training and Testing Data") + xlab("mtry") + ylab("Accuracy Rate") + scale_color_manual("Data Type ", values = c("Train  Accuracy Rate"="red", "Test Accuracy Rate"="blue")) + 
  scale_x_continuous(breaks = c(1:10))
print(plottune71)

#Model with optimum par for both training & testing
set.seed(123456)
predtr71 <- select(smote_data71, -c(lapse))
predte71 <- select(datatest71, -c(lapse))
modelrfop71 <- randomForest(x = predtr, y = smote_data71$lapse, xtest = predte, ytest = datatest71$lapse, 
                            ntree = 500, mtry = 6, importance = TRUE)
print(modelrfop71)

#Model for prediction
modelrfop71 <- randomForest(lapse ~ ., data = smote_data71, ntree = 500, mtry = 6, importance = TRUE)
print(modelrfop71)

#Extract rule
treelist71 <-RF2List(modelrfop71)
extract71 <- extractRules(treelist71, predtr71, ntree = 500)
rulemetric71 <- getRuleMetric(extract71,predtr71,smote_data71$lapse)
rulemetric71 <- pruneRule(rulemetric71,predtr71,smote_data71$lapse,maxDecay = 0.05,typeDecay = 2)
rulemetric71 <- selectRuleRRF(rulemetric71,predtr71,smote_data71$lapse)
learner71 <- buildLearner(rulemetric71,predtr71,smote_data71$lapse)
sim_learner71 <- presentRules(rulemetric71,colnames(predtr71))

#Creating prediction output
pred71 <- predict(modelrfop71,datapred71,type = "prob")
pred71 <- as.data.frame(pred71)

summary(datapred71$lapse)

#Export prediction output & rule
write.csv(datatest71, "D:/FOLDER AFGANTA/Revisit Anti Lapse/Modelling/test71.csv")
write.csv(pred71, "D:/FOLDER AFGANTA/Revisit Anti Lapse/Modelling/Output/pred71.csv")
write.csv(sim_learner71, "D:/FOLDER AFGANTA/Revisit Anti Lapse/Modelling/Output/rule71.csv")

#----C.2.11 TESTING & PREDICTION DATA72---- 

#Obtain optimum ntree for both training & testing
set.seed(123456)
predtr <- select(smote_data72, -c(lapse))
predte <- select(datatest72, -c(lapse))
modelrfop72 <- randomForest(x = predtr, y = smote_data72$lapse, xtest = predte, ytest = datatest72$lapse, importance = TRUE)
print(modelrfop72)
#plotntree <- plot(modelrfop72)
opntree72 <- which.min(modelrfop72$err.rate)
print(opntree72)

#Tuning optimum mtry for both training & testing
accuracytrain=c()
accuracytest=c()
i=1
for (i in 1:10) {
  modelrfop72 <- randomForest(lapse ~ ., data = smote_data72, ntree = 1072, mtry = i, importance = TRUE)
  accuracytrain[i] <- as.vector(1 - (modelrfop72$err.rate[1072,1]))
  predtest1 <- predict(modelrfop72,datatest72)
  accuracytest[i] <- mean(predtest1 == datatest72$lapse)
}
err <- as.data.frame(cbind(accuracytrain, accuracytest))

plottune72 <- ggplot(err,aes(x=c(1:10))) + geom_line(aes(y=accuracytrain, color="Train  Accuracy Rate")) + geom_line(aes(y=accuracytest, color="Test Accuracy Rate")) +
  ggtitle("Optimum mtry for Both Training and Testing Data") + xlab("mtry") + ylab("Accuracy Rate") + scale_color_manual("Data Type ", values = c("Train  Accuracy Rate"="red", "Test Accuracy Rate"="blue")) + 
  scale_x_continuous(breaks = c(1:10))
print(plottune72)

#Model with optimum par for both training & testing
set.seed(123456)
predtr72 <- select(smote_data72, -c(lapse))
predte72 <- select(datatest72, -c(lapse))
modelrfop72 <- randomForest(x = predtr, y = smote_data72$lapse, xtest = predte, ytest = datatest72$lapse, 
                            ntree = 250, mtry = 6, importance = TRUE)
print(modelrfop72)

#Model for prediction
modelrfop72 <- randomForest(lapse ~ ., data = smote_data72, ntree = 250, mtry = 6, importance = TRUE)
print(modelrfop72)

#Extract rule
treelist72 <-RF2List(modelrfop72)
extract72 <- extractRules(treelist72, predtr72, ntree = 250)
rulemetric72 <- getRuleMetric(extract72,predtr72,smote_data72$lapse)
rulemetric72 <- pruneRule(rulemetric72,predtr72,smote_data72$lapse,maxDecay = 0.05,typeDecay = 2)
rulemetric72 <- selectRuleRRF(rulemetric72,predtr72,smote_data72$lapse)
learner72 <- buildLearner(rulemetric72,predtr72,smote_data72$lapse)
sim_learner72 <- presentRules(rulemetric72,colnames(predtr72))

#Creating prediction output
pred72 <- predict(modelrfop72,datapred72,type = "prob")
pred72 <- as.data.frame(pred72)

summary(datapred72$lapse)

#Export prediction output & rule
write.csv(datatest72, "D:/FOLDER AFGANTA/Revisit Anti Lapse/Modelling/test72.csv")
write.csv(pred72, "D:/FOLDER AFGANTA/Revisit Anti Lapse/Modelling/Output/pred72.csv")
write.csv(sim_learner72, "D:/FOLDER AFGANTA/Revisit Anti Lapse/Modelling/Output/rule72.csv")

#----C.2.12 TESTING & PREDICTION DATA73---- 

#Obtain optimum ntree for both training & testing
set.seed(123456)
predtr <- select(smote_data73, -c(lapse))
predte <- select(datatest73, -c(lapse))
modelrfop73 <- randomForest(x = predtr, y = smote_data73$lapse, xtest = predte, ytest = datatest73$lapse, importance = TRUE)
print(modelrfop73)
#plotntree <- plot(modelrfop73)
opntree73 <- which.min(modelrfop73$err.rate)
print(opntree73)

#Tuning optimum mtry for both training & testing
accuracytrain=c()
accuracytest=c()
i=1
for (i in 1:10) {
  modelrfop73 <- randomForest(lapse ~ ., data = smote_data73, ntree = 726, mtry = i, importance = TRUE)
  accuracytrain[i] <- as.vector(1 - (modelrfop73$err.rate[726,1]))
  predtest1 <- predict(modelrfop73,datatest73)
  accuracytest[i] <- mean(predtest1 == datatest73$lapse)
}
err <- as.data.frame(cbind(accuracytrain, accuracytest))

plottune73 <- ggplot(err,aes(x=c(1:10))) + geom_line(aes(y=accuracytrain, color="Train  Accuracy Rate")) + geom_line(aes(y=accuracytest, color="Test Accuracy Rate")) +
  ggtitle("Optimum mtry for Both Training and Testing Data") + xlab("mtry") + ylab("Accuracy Rate") + scale_color_manual("Data Type ", values = c("Train  Accuracy Rate"="red", "Test Accuracy Rate"="blue")) + 
  scale_x_continuous(breaks = c(1:10))
print(plottune73)

#Model with optimum par for both training & testing
set.seed(123456)
predtr73 <- select(smote_data73, -c(lapse))
predte73 <- select(datatest73, -c(lapse))
modelrfop73 <- randomForest(x = predtr, y = smote_data73$lapse, xtest = predte, ytest = datatest73$lapse, 
                              ntree = 500, mtry = 6, importance = TRUE)
print(modelrfop73)

#Model for prediction
modelrfop73 <- randomForest(lapse ~ ., data = smote_data73, ntree = 500, mtry = 6, importance = TRUE)
print(modelrfop73)

#Extract rule
treelist73 <-RF2List(modelrfop73)
extract73 <- extractRules(treelist73, predtr73, ntree = 500)
rulemetric73 <- getRuleMetric(extract73,predtr73,smote_data73$lapse)
rulemetric73 <- pruneRule(rulemetric73,predtr73,smote_data73$lapse,maxDecay = 0.05,typeDecay = 2)
rulemetric73 <- selectRuleRRF(rulemetric73,predtr73,smote_data73$lapse)
learner73 <- buildLearner(rulemetric73,predtr73,smote_data73$lapse)
sim_learner73 <- presentRules(rulemetric73,colnames(predtr73))

#Creating prediction output
pred73 <- predict(modelrfop73,datapred73,type = "prob")
pred73 <- as.data.frame(pred73)

summary(datatest73$lapse)

#Export prediction output & rule
write.csv(datatest73, "D:/FOLDER AFGANTA/Revisit Anti Lapse/Modelling/test73.csv")
write.csv(pred73, "D:/FOLDER AFGANTA/Revisit Anti Lapse/Modelling/Output/pred73.csv")
write.csv(sim_learner73, "D:/FOLDER AFGANTA/Revisit Anti Lapse/Modelling/Output/rule73.csv")

#----D. PREDICTION UNSEEN DATA----

#----D.1. PREPROCESSING DATAPRED----

write.csv(dataall, "D:/FOLDER AFGANTA/Revisit Anti Lapse/Modelling/dataall.csv")
write.csv(datatrain11, "D:/FOLDER AFGANTA/Revisit Anti Lapse/Modelling/datatrain.csv")
write.csv(smote_data11, "D:/FOLDER AFGANTA/Revisit Anti Lapse/Modelling/datatrainsmote.csv")
write.csv(dataapr, "D:/FOLDER AFGANTA/Revisit Anti Lapse/Modelling/datapred.csv")

datapred <- read.csv("D:/FOLDER AFGANTA/Revisit Anti Lapse/Modelling/datapred.csv")
datapred <- select(datapred, -c(policynumber))

#separate datapred based on product & policyage
for(a in c(1:7)){
  for(b in c(1:3)){
    tmpa <- get("a")
    tmpb <- get("b")
    datapreda <- filter(datapred,productname==tmpa,policyagemonth==tmpb)
    assign(paste("datapred",a,b,sep = ""),datapreda)
  }
}


#----D.2. DROP UNIMPORTANCE FEATURES & SET SAME LEVEL FACTOR ON UNSEEN DATA----

#drop unimportance features on test data based on importance data pred
datapred11 <- select(datapred11, -c(dropvar11))
datapred11 <- lapply(datapred11,as.factor)
datapred11 <- as.data.frame(datapred11)

datapred12 <- select(datapred12, -c(dropvar12))
datapred12 <- lapply(datapred12,as.factor)
datapred12 <- as.data.frame(datapred12)

datapred13 <- select(datapred13, -c(dropvar13))
datapred13 <- lapply(datapred13,as.factor)
datapred13 <- as.data.frame(datapred13)

datapred31 <- select(datapred31, -c(dropvar31))
datapred31 <- lapply(datapred31,as.factor)
datapred31 <- as.data.frame(datapred31)

datapred32 <- select(datapred32, -c(dropvar32))
datapred32 <- lapply(datapred32,as.factor)
datapred32 <- as.data.frame(datapred32)

datapred33 <- select(datapred33, -c(dropvar33))
datapred33 <- lapply(datapred33,as.factor)
datapred33 <- as.data.frame(datapred33)

datapred41 <- select(datapred41, -c(dropvar41))
datapred41 <- lapply(datapred41,as.factor)
datapred41 <- as.data.frame(datapred41)

datapred42 <- select(datapred42, -c(dropvar42))
datapred42 <- lapply(datapred42,as.factor)
datapred42 <- as.data.frame(datapred42)

datapred51 <- select(datapred51, -c(dropvar51))
datapred51 <- lapply(datapred51,as.factor)
datapred51 <- as.data.frame(datapred51)

datapred71 <- select(datapred71, -c(dropvar71))
datapred71 <- lapply(datapred71,as.factor)
datapred71 <- as.data.frame(datapred71)

datapred72 <- select(datapred72, -c(dropvar72))
datapred72 <- lapply(datapred72,as.factor)
datapred72 <- as.data.frame(datapred72)

datapred73 <- select(datapred73, -c(dropvar73))
datapred73 <- lapply(datapred73,as.factor)
datapred73 <- as.data.frame(datapred73)

#set same nlevel on training & prediction data
for (z in names(smote_data11)){
  tmp2 <- get("z")
  while (nlevels(smote_data11[[tmp2]]) < nlevels(datapred11[[tmp2]])){
    smote_data11[[tmp2]] <- factor(smote_data11[[tmp2]], levels = levels(datapred11[[tmp2]]))
    smote_data11[[tmp2]] <- smote_data11[[tmp2]]
  }
  assign(z,tmp2)
}
for (z in names(smote_data11)){
  tmp3 <- get("z")
  while (nlevels(datapred11[[tmp3]]) < nlevels(smote_data11[[tmp3]])){
    datapred11[[tmp3]] <- factor(datapred11[[tmp3]], levels = levels(smote_data11[[tmp3]]))
    datapred11[[tmp3]] <- datapred11[[tmp3]]
  }
  assign(z,tmp3)
}

for (z in names(smote_data12)){
  tmp2 <- get("z")
  while (nlevels(smote_data12[[tmp2]]) < nlevels(datapred12[[tmp2]])){
    smote_data12[[tmp2]] <- factor(smote_data12[[tmp2]], levels = levels(datapred12[[tmp2]]))
    smote_data12[[tmp2]] <- smote_data12[[tmp2]]
  }
  assign(z,tmp2)
}
for (z in names(smote_data12)){
  tmp3 <- get("z")
  while (nlevels(datapred12[[tmp3]]) < nlevels(smote_data12[[tmp3]])){
    datapred12[[tmp3]] <- factor(datapred12[[tmp3]], levels = levels(smote_data12[[tmp3]]))
    datapred12[[tmp3]] <- datapred12[[tmp3]]
  }
  assign(z,tmp3)
}

for (z in names(smote_data13)){
  tmp2 <- get("z")
  while (nlevels(smote_data13[[tmp2]]) < nlevels(datapred13[[tmp2]])){
    smote_data13[[tmp2]] <- factor(smote_data13[[tmp2]], levels = levels(datapred13[[tmp2]]))
    smote_data13[[tmp2]] <- smote_data13[[tmp2]]
  }
  assign(z,tmp2)
}
for (z in names(smote_data13)){
  tmp3 <- get("z")
  while (nlevels(datapred13[[tmp3]]) < nlevels(smote_data13[[tmp3]])){
    datapred13[[tmp3]] <- factor(datapred13[[tmp3]], levels = levels(smote_data13[[tmp3]]))
    datapred13[[tmp3]] <- datapred13[[tmp3]]
  }
  assign(z,tmp3)
}

for (z in names(smote_data31)){
  tmp2 <- get("z")
  while (nlevels(smote_data31[[tmp2]]) < nlevels(datapred31[[tmp2]])){
    smote_data31[[tmp2]] <- factor(smote_data31[[tmp2]], levels = levels(datapred31[[tmp2]]))
    smote_data31[[tmp2]] <- smote_data31[[tmp2]]
  }
  assign(z,tmp2)
}
for (z in names(smote_data31)){
  tmp3 <- get("z")
  while (nlevels(datapred31[[tmp3]]) < nlevels(smote_data31[[tmp3]])){
    datapred31[[tmp3]] <- factor(datapred31[[tmp3]], levels = levels(smote_data31[[tmp3]]))
    datapred31[[tmp3]] <- datapred31[[tmp3]]
  }
  assign(z,tmp3)
}

for (z in names(smote_data32)){
  tmp2 <- get("z")
  while (nlevels(smote_data32[[tmp2]]) < nlevels(datapred32[[tmp2]])){
    smote_data32[[tmp2]] <- factor(smote_data32[[tmp2]], levels = levels(datapred32[[tmp2]]))
    smote_data32[[tmp2]] <- smote_data32[[tmp2]]
  }
  assign(z,tmp2)
}
for (z in names(smote_data32)){
  tmp3 <- get("z")
  while (nlevels(datapred32[[tmp3]]) < nlevels(smote_data32[[tmp3]])){
    datapred32[[tmp3]] <- factor(datapred32[[tmp3]], levels = levels(smote_data32[[tmp3]]))
    datapred32[[tmp3]] <- datapred32[[tmp3]]
  }
  assign(z,tmp3)
}

for (z in names(smote_data33)){
  tmp2 <- get("z")
  while (nlevels(smote_data33[[tmp2]]) < nlevels(datapred33[[tmp2]])){
    smote_data33[[tmp2]] <- factor(smote_data33[[tmp2]], levels = levels(datapred33[[tmp2]]))
    smote_data33[[tmp2]] <- smote_data33[[tmp2]]
  }
  assign(z,tmp2)
}
for (z in names(smote_data33)){
  tmp3 <- get("z")
  while (nlevels(datapred33[[tmp3]]) < nlevels(smote_data33[[tmp3]])){
    datapred33[[tmp3]] <- factor(datapred33[[tmp3]], levels = levels(smote_data33[[tmp3]]))
    datapred33[[tmp3]] <- datapred33[[tmp3]]
  }
  assign(z,tmp3)
}

for (z in names(smote_data41)){
  tmp2 <- get("z")
  while (nlevels(smote_data41[[tmp2]]) < nlevels(datapred41[[tmp2]])){
    smote_data41[[tmp2]] <- factor(smote_data41[[tmp2]], levels = levels(datapred41[[tmp2]]))
    smote_data41[[tmp2]] <- smote_data41[[tmp2]]
  }
  assign(z,tmp2)
}
for (z in names(smote_data41)){
  tmp3 <- get("z")
  while (nlevels(datapred41[[tmp3]]) < nlevels(smote_data41[[tmp3]])){
    datapred41[[tmp3]] <- factor(datapred41[[tmp3]], levels = levels(smote_data41[[tmp3]]))
    datapred41[[tmp3]] <- datapred41[[tmp3]]
  }
  assign(z,tmp3)
}

for (z in names(smote_data42)){
  tmp2 <- get("z")
  while (nlevels(smote_data42[[tmp2]]) < nlevels(datapred42[[tmp2]])){
    smote_data42[[tmp2]] <- factor(smote_data42[[tmp2]], levels = levels(datapred42[[tmp2]]))
    smote_data42[[tmp2]] <- smote_data42[[tmp2]]
  }
  assign(z,tmp2)
}
for (z in names(smote_data42)){
  tmp3 <- get("z")
  while (nlevels(datapred42[[tmp3]]) < nlevels(smote_data42[[tmp3]])){
    datapred42[[tmp3]] <- factor(datapred42[[tmp3]], levels = levels(smote_data42[[tmp3]]))
    datapred42[[tmp3]] <- datapred42[[tmp3]]
  }
  assign(z,tmp3)
}

for (z in names(smote_data51)){
  tmp2 <- get("z")
  while (nlevels(smote_data51[[tmp2]]) < nlevels(datapred51[[tmp2]])){
    smote_data51[[tmp2]] <- factor(smote_data51[[tmp2]], levels = levels(datapred51[[tmp2]]))
    smote_data51[[tmp2]] <- smote_data51[[tmp2]]
  }
  assign(z,tmp2)
}
for (z in names(smote_data51)){
  tmp3 <- get("z")
  while (nlevels(datapred51[[tmp3]]) < nlevels(smote_data51[[tmp3]])){
    datapred51[[tmp3]] <- factor(datapred51[[tmp3]], levels = levels(smote_data51[[tmp3]]))
    datapred51[[tmp3]] <- datapred51[[tmp3]]
  }
  assign(z,tmp3)
}

for (z in names(smote_data71)){
  tmp2 <- get("z")
  while (nlevels(smote_data71[[tmp2]]) < nlevels(datapred71[[tmp2]])){
    smote_data71[[tmp2]] <- factor(smote_data71[[tmp2]], levels = levels(datapred71[[tmp2]]))
    smote_data71[[tmp2]] <- smote_data71[[tmp2]]
  }
  assign(z,tmp2)
}
for (z in names(smote_data71)){
  tmp3 <- get("z")
  while (nlevels(datapred71[[tmp3]]) < nlevels(smote_data71[[tmp3]])){
    datapred71[[tmp3]] <- factor(datapred71[[tmp3]], levels = levels(smote_data71[[tmp3]]))
    datapred71[[tmp3]] <- datapred71[[tmp3]]
  }
  assign(z,tmp3)
}

for (z in names(smote_data72)){
  tmp2 <- get("z")
  while (nlevels(smote_data72[[tmp2]]) < nlevels(datapred72[[tmp2]])){
    smote_data72[[tmp2]] <- factor(smote_data72[[tmp2]], levels = levels(datapred72[[tmp2]]))
    smote_data72[[tmp2]] <- smote_data72[[tmp2]]
  }
  assign(z,tmp2)
}
for (z in names(smote_data72)){
  tmp3 <- get("z")
  while (nlevels(datapred72[[tmp3]]) < nlevels(smote_data72[[tmp3]])){
    datapred72[[tmp3]] <- factor(datapred72[[tmp3]], levels = levels(smote_data72[[tmp3]]))
    datapred72[[tmp3]] <- datapred72[[tmp3]]
  }
  assign(z,tmp3)
}

for (z in names(smote_data73)){
  tmp2 <- get("z")
  while (nlevels(smote_data73[[tmp2]]) < nlevels(datapred73[[tmp2]])){
    smote_data73[[tmp2]] <- factor(smote_data73[[tmp2]], levels = levels(datapred73[[tmp2]]))
    smote_data73[[tmp2]] <- smote_data73[[tmp2]]
  }
  assign(z,tmp2)
}
for (z in names(smote_data73)){
  tmp3 <- get("z")
  while (nlevels(datapred73[[tmp3]]) < nlevels(smote_data73[[tmp3]])){
    datapred73[[tmp3]] <- factor(datapred73[[tmp3]], levels = levels(smote_data73[[tmp3]]))
    datapred73[[tmp3]] <- datapred73[[tmp3]]
  }
  assign(z,tmp3)
}
