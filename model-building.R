# Practical Machine Learning Course Project
# Eduardo Rodríguez
# October 2015

library(caret)
library(ggplot2)
set.seed(314761)

# First, download the data

trainURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

setwd("~/Dropbox/eduardo/datascience/coursera/8-predmachlearn/CP/PredMachLearn_CourseProject/")

trainFile <- "pml-training.csv"
if (!file.exists(trainFile)) {
    download.file(url = trainURL, destfile = trainFile, method = "curl")
}

testFile <- "pml-testing.csv"
if (!file.exists(testFile)) {
    download.file(url = testURL, destfile = testFile, method = "curl")
}

# Try na.strings = "#DIV/0!" (also interprets empty cells as NA)
# Try stringsAsFactors = FALSE

cc <- c("integer",            # 1 X
        "character",          # 2 user_name
        "integer",            # 3 raw_timestamp_part_1
        "integer",            # 4 raw_timestamp_part_2
        "character",          # 5 cvtd_timestamp
        "character",          # 6 new_window
        "integer",            # 7 num_window
        rep("numeric", 152),  # the 152 numerical variables we care about
        "factor")             # the class variable we want to predict

pmlTrain <- read.csv(trainFile, nrows = 19630, stringsAsFactors = FALSE)
pmlTest <- read.csv(testFile)

# Preprocessing

# Extract class variable

pmlY <- pmlTrain$classe

# Eliminate uninteresting columns/variables

pmlTrain <- pmlTrain[, -160]
pmlTrain <- pmlTrain[, -(1:7)]

# Find columns with way too many NAs and eliminate them

nc <- ncol(pmlTrain)
nna <- numeric(nc)
for (c in 1:nc) {
    nna[c] <- sum(is.na(as.numeric(pmlTrain[, c])))
}
nacols <- which(nna > 19000)
pmlTrain <- pmlTrain[, -nacols]

# Create training and testing subsets (from the original training data)

inTrain <- createDataPartition(pmlY, p = 0.7, list = FALSE)
trainX <- pmlTrain[inTrain,]
testX <- pmlTrain[-inTrain,]

trainY <- pmlY[inTrain]
testY <- pmlY[-inTrain]

# Beware: This may take a long time

m1 <- train(trainX, trainY, method = "rf")

# Save model to a file

save(m1, file = "m1.RData")

# Beware: this may crash your R session

m2 <- train(trainX, trainY, method = "RRF", proximity = TRUE)

# Save model to a file

save(m2, file = "m2.RData")

###
###
###
###
###
###
###
###
###
###
###
###
###
###
###
###
###
###
###
###
