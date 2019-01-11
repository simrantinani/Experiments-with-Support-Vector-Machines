####################################
# # Project: Experiments with soft-margin Support Vector Machines using the US Postal Service Zip Code data set
# # Author: Simran Tinani
# # Date: 11-01-2019
# # Remarks: Inspired by the problems of homework assignment 8 of the Learning from Data telecourse 
#     by Yaser Abu-Mostafa (CalTech University): https://work.caltech.edu/homework/hw8.pdf
#####################################

# install.packages("e1071")
# install.packages("dplyr")
# install.packages("caret")

library('e1071') # libsvm, the library that will perform the algorithm
library('dplyr')
library(caret)

features.train<-read.table("features.train.txt")
features.test<-read.table("features.test.txt")
# .txt files that contain intensity and symmetry features
# downloaded from http://www.amlbook.com/data/zip/features.train, http://www.amlbook.com/data/zip/features.test

labels <- features.train[,1]
train <- features.train[,-1]

dataset <-data.frame(label=as.factor(labels), train)
n<-nrow(dataset)

# We experiment with the one-vs-all and one-vs-one binary classifiers
# Throughout, the error used is binary classification error

###########################################
# ONE V.S. ALL CLASSIFICATION

A<-matrix(0, nrow=10, ncol=2) # "A" stores the results (number of support vectors and in-sample error) of running the one-vs-all classifier for each of the 10 digits

for(r in 0:9){ # loop over all digits
  new<-features.train
  new$V1<-as.character(new$V1)
  for(i in 1:n){
    if(new$V1[i]!=paste(r)){new$V1[i]="-1"} else{new$V1[i]="1"}
  } # the label is 1 for the digit r and -1 for all other digits, making it a binary problem
  new$V1<-as.integer(new$V1)
  newlabels <- new[,1]
  newtrain <- new[,-1]
  dataset_r =data.frame(label=as.factor(newlabels), newtrain)
  model_r = svm(label ~., data = dataset_r, kernel = "polynomial",  cost = 0.01, degree = 2)
  predictedY_r <- predict(model_r, dataset_r)
  Y_r <- as.numeric(as.matrix(dataset_r[,1]))
  
  i <- 1
  E_in_r <- 0 # in-sample error
  while(i<=n){
    if(Y_r[i] != as.numeric(as.matrix(predictedY_r)[i])) {E_in_r = E_in_r + 100/n}
    i = i + 1
  }
  E_in_r # percentage in-sample error (binary classification error)
  
  A[r+1,]<-c(E_in_r, model_r$tot.nSV) # storing the results
  message(paste("The number of support vectors for the one vs all classifier with digit", r, "is", model_r$tot.nSV, "and the total in-sample error is", round(E_in_r,2), sep=" " ))
  
}

plot(data=dataset_r,model_r) # visualizing the results for 9 vs all

######################################################
# ONE-V.S.-ONE CLASSIFIER: We use the digits 1 and 5

new<-features.train
newtest<-features.test
new$V1<-as.character(new$V1)
newtest$V1<-as.character(newtest$V1)
new <- filter(new, V1 == "1"| V1 =="5") # Extract rows with digits 1 or 5
newtest <- filter(newtest, V1 == "1"| V1 =="5")

new$V1<-as.integer(new$V1)
newtest$V1<-as.integer(newtest$V1)

newlabels <- new[,1]
newtestlabels<-newtest[,1]
newtrain <- new[,-1]
dataset15 <-data.frame(label=as.factor(newlabels), newtrain)
m<-nrow(dataset15)
l<-nrow(newtest)

# Experimenting with the regularization parameter C (the slack allowed in the margin distances)

for(c in c(0.001,0.01,0.1)){
model15 <- svm(label ~., data = dataset15, kernel = "polynomial",  cost = c, degree = 2)
predictedY15 <- predict(model15, dataset15)

Y15 <- as.numeric(paste(as.matrix(dataset15[,1])))

i <- 1
E_in_15 <- 0 # in-sample error for 1 vs 5 classification
while(i<=nrow(dataset15)){
  if(Y15[i]!= as.numeric(as.matrix(predictedY15)[i])) {E_in_15<- E_in_15 + 100/m}
  i = i + 1
}

message(paste("The number of support vectors for the 1 vs 5 classifier with C =", c, "is", model15$tot.nSV, "and the total in-sample error is", round(E_in_15,2), sep=" " ))
}

plot(data=dataset15, model15)
 # shift the plot function inside the for loop to visualize the results for all the values of C

# The number of support vectors for the 1 vs 5 classifier with C = 0.001 is 1088 and the total in-sample error is 27.48
# The number of support vectors for the 1 vs 5 classifier with C = 0.01 is 828 and the total in-sample error is 16.46
# The number of support vectors for the 1 vs 5 classifier with C = 0.1 is 598 and the total in-sample error is 11.4

# Experimenting with the polynomial degree Q

for(Q in c(2,5)){
  
for(c in c(0.0001, 0.001,0.01,0.1,1)){
  model15 <- svm(label ~., data = dataset15, kernel = "polynomial",  cost = c, degree = Q)
  predictedY15 <- predict(model15, dataset15)
  
  Y15 <- as.numeric(paste(as.matrix(dataset15[,1])))
  
  i <- 1
  E_in_15 <- 0
  while(i<=nrow(dataset15)){
    if(Y15[i]!= as.numeric(as.matrix(predictedY15)[i])) {E_in_15= E_in_15 + 100/m}
    i = i + 1
  }
  
  Z <- as.numeric(paste(as.matrix(newtest[,1])))
  predictedZ <- predict(model15, newdata = newtest)
  i <- 1
  Eout15 <- 0
  while(i<=nrow(newtest)){
    if((Z[i] == 1 || Z[i] == 5) && Z[i]!= as.numeric(as.matrix(predictedZ)[i])) {Eout15 = Eout15 + 100/l}
    i = i + 1
  }
  message(paste("The number of support vectors for the 1 vs 5 classifier with C =", c, "and degree Q =", Q, "is", model15$tot.nSV, "the total in-sample error is", round(E_in_15,2), "and the total out-of-sample error is", round(Eout15,2), sep=" " ))
}
}
plot(data=dataset15, model15)

# The number of support vectors for the 1 vs 5 classifier with C = 1e-04 and degree Q = 2 is 1112 the total in-sample error is 35.62 and the total out-of-sample error is 37.74
# The number of support vectors for the 1 vs 5 classifier with C = 0.001 and degree Q = 2 is 1088 the total in-sample error is 27.48 and the total out-of-sample error is 25.47
# The number of support vectors for the 1 vs 5 classifier with C = 0.01 and degree Q = 2 is 828 the total in-sample error is 16.46 and the total out-of-sample error is 18.63
# The number of support vectors for the 1 vs 5 classifier with C = 0.1 and degree Q = 2 is 598 the total in-sample error is 11.4 and the total out-of-sample error is 12.97
# The number of support vectors for the 1 vs 5 classifier with C = 1 and degree Q = 2 is 445 the total in-sample error is 9.1 and the total out-of-sample error is 11.32
# The number of support vectors for the 1 vs 5 classifier with C = 1e-04 and degree Q = 5 is 906 the total in-sample error is 24.79 and the total out-of-sample error is 24.06
# The number of support vectors for the 1 vs 5 classifier with C = 0.001 and degree Q = 5 is 686 the total in-sample error is 17.81 and the total out-of-sample error is 19.1
# The number of support vectors for the 1 vs 5 classifier with C = 0.01 and degree Q = 5 is 497 the total in-sample error is 12.81 and the total out-of-sample error is 12.74
# The number of support vectors for the 1 vs 5 classifier with C = 0.1 and degree Q = 5 is 360 the total in-sample error is 8.14 and the total out-of-sample error is 9.91
# The number of support vectors for the 1 vs 5 classifier with C = 1 and degree Q = 5 is 241 the total in-sample error is 5.64 and the total out-of-sample error is 7.78


###############################################
# RADIAL BASIS FUNCTION KERNEL

for(c in c(0.01, 1,100, 10^4, 10^6)){
  model15 <- svm(label ~., data = dataset15, kernel = "radial",  cost = c)
  predictedY15 <- predict(model15, dataset15)
  
  Y15 <- as.numeric(paste(as.matrix(dataset15[,1])))
  
  i <- 1
  E_in_15 <- 0
  while(i<=nrow(dataset15)){
    if(Y15[i]!= as.numeric(as.matrix(predictedY15)[i])) {E_in_15= E_in_15 + 100/m}
    i = i + 1
  }
  
  Z <- as.numeric(paste(as.matrix(newtest[,1])))
  predictedZ <- predict(model15, newdata = newtest)
  i <- 1
  Eout15 <- 0
  while(i<=nrow(newtest)){
    if((Z[i] == 1 || Z[i] == 5) && Z[i]!= as.numeric(as.matrix(predictedZ)[i])) {Eout15 = Eout15 + 100/l}
    i = i + 1
  }
  message(paste("The number of support vectors for the 1 vs 5 classifier with a Radial Basis Kernel and C =", c, "is", model15$tot.nSV, "the total in-sample error is", round(E_in_15,2), "and the total out-of-sample error is", round(Eout15,2), sep=" " ))
}

plot(data=dataset15, model15)

######################################################################
# PERFORMING 100-fold CROSS VALIDATION TO GET THE OPTIMUM COST VALUE C

new<-features.train
new$V1<-as.character(new$V1)
new <- filter(new, V1 == "1"| V1 =="5")# Extract rows with 1 and 5
new$V1<-as.integer(new$V1)
newlabels <- new[,1]
newtrain <- new[,-1]
dataset0 <-data.frame(label=as.factor(newlabels), newtrain)
folds <- cut(seq(1,nrow(dataset0)),breaks=10,labels=FALSE) # Create 10 equally size folds

l <- 1
j1 = matrix(nrow=100, ncol = 10) # matrix corresponding to C = 0.0001
j2 = matrix(nrow=100, ncol = 10) # matrix corresponding to C = 0.001
j3 = matrix(nrow=100, ncol = 10) # matrix corresponding to C = 0.01
j4 = matrix(nrow=100, ncol = 10) # matrix corresponding to C = 0.1
j5 = matrix(nrow=100, ncol = 10) # matrix corresponding to C = 1
means = matrix(nrow = 100, ncol = 5) # mean over the test error of all digits, 5 columns correspond to 5 different values of C
means[,]=0
j1[,]=0; j2[,]=0; j3[,]=0; j4[,]=0; j5[,]=0

while(l <101){
  dataset0 =data.frame(label=as.factor(newlabels), newtrain)
  dataset0<-dataset0[sample(nrow(dataset0)),]  # Randomly shuffle the data
  
  for(i in 1:10){ # Perform 100 steps of 10-fold cross validation
    testIndices <- which(folds==i,arr.ind=TRUE)    
    testData <- dataset0[testIndices, ]
    trainData <- dataset0[-testIndices, ]
    p<-nrow(testData)
    # C = 0.0001
    model1 = svm(label ~., data = trainData, kernel = "polynomial", cost = 0.0001, degree = 2)
    Y<- as.numeric(paste(as.matrix(testData[,1])))
    predictedY <- predict(model1, newdata = testData)
    k = 1
    while(k<157){
      if(Y[k]!= as.numeric(as.matrix(predictedY)[k])) {j1[l,i] = j1[l,i] + 100/p}
      k = k + 1
    }
    
    # C = 0.001
    model2 = svm(label ~., data = trainData, kernel = "polynomial",  cost = 0.001, degree = 2)
    predictedY <- predict(model2, newdata = testData)
    
    k = 1
    while(k<157){
      if(Y[k]!= as.numeric(as.matrix(predictedY)[k])) {j2[l,i] = j2[l,i] + 100/p}
      k = k + 1
    }
    
    # C = 0.01
    model3 = svm(label ~., data = trainData, kernel = "polynomial",  cost = 0.01, degree = 2)
    predictedY <- predict(model3, newdata = testData)
    k = 1
    while(k<157){
      if(Y[k]!= as.numeric(as.matrix(predictedY)[k])) {j3[l,i] = j3[l,i] + 100/p}
      k = k + 1
    }
    
    # C = 0.1
    model4 = svm(label ~., data = trainData, kernel = "polynomial", cost = 0.1, degree = 2)
    predictedY <- predict(model4, newdata = testData)
    k = 1
    while(k<157){
      if(Y[k]!= as.numeric(as.matrix(predictedY)[k])) {j4[l,i] = j4[l,i] + 100/p}
      k = k + 1
    }
    
    # C = 1
    model5 = svm(label ~., data = trainData, kernel = "polynomial", cost = 1, degree = 2)
    Y<- as.numeric(paste(as.matrix(testData[,1])))
    predictedY <- predict(model5, newdata = testData)
    k = 1
    while(k<157){
      if(Y[k]!= as.numeric(as.matrix(predictedY)[k])) {j5[l,i] = j5[l,i] + 100/p}
      k = k + 1
    }
    means[l,] = c(mean(j1[l,]), mean(j2[l,]), mean(j3[l,]), mean(j4[l,]), mean(j5[l,]))
    # take the mean over the validation error of all digits, 5 columns correspond to the 5 possible values of C
  }
  l = l +1
}

v<-vector()

for(i in 1:100){
  a<-as.numeric(which.min(means[i,]))
  print(paste(a, 10^(-5+a)))
  v<-c(v,10^(-5+a))
  # index and value of C gave the lowest mean error over all digits
}
best_C<- v[as.numeric(sort(table(v),decreasing=TRUE)[1])]
# 0.1
index_best_C<- log10(best_C)+5
E_val <- mean(means[,index_best_C])*100 # mean % validation error for the best value of C over all digits
# 0.3064103

message(paste("The overall best cost value is found to be", best_C, "and its mean % validation error over all digits is", round(E_val,2), "%",sep=" "))
