## install required packages

if(!require(BiocManager)){
  install.packages("BiocManager")
  library(Biocmanager)
}

if(!require(RBGL)){
  install.packages("RBGL")
  library(RBGL)
}

if(!require(graph)){
  install.packages("graph")
  library(graph)
}

if(!require(gRbase)){
  install.packages("gRbase")
  library(gRbase)
}

if(!require(grid)){
  install.packages("grid")
  library(grid)
}

if(!require(psych)){
  install.packages("psych")
  library(psych)
}

if(!require(igraph)){
  install.packages("igraph")
  library(igraph)
}

if(!require(bnlearn)){
  install.packages("bnlearn")
  library(bnlearn)
}

if(!require(ggm)){
  install.packages("ggm")
  library(ggm)
}

if(!require(gRain)){
  install.packages("gRain")
  library(gRain)
}

if(!require(crop)){
  install.packages("crop")
  library(crop)
}

if(!require(catnet)){
  install.packages("catnet")
  library(catnet)
}


if(!require(Hmisc)){
  install.packages("Hmisc")
  library(Hmisc)
}


if(!require(pcalg)){
  install.packages("pcalg")
  library(pcalg)
}


if(!require(tidyr)){
  install.packages("tidyr")
  library(tidyr)
}


if(!require(stringr)){
  install.packages("stringr")
  library(stringr)
}


if(!require(dplyr)){
  install.packages("dplyr")
  library(dplyr)
}


if(!require(tidyverse)){
  install.packages("tidyverse")
  library(tidyverse)
}


if(!require(data.table)){
  install.packages("data.table")
  library(data.table)
}


if(!require(lubridate)){
  install.packages("lubridate")
  library(lubridate)
}


if(!require(naniar)){
  install.packages("naniar")
  library(naniar)
}


# install.packages("BiocManager")
# BiocManager::install("Rgraphviz")
# BiocManager::install("pcalg")
# BiocManager::install("gRbase")


## set universal path
path <- '.'
dir <- paste0(path, "/data/dataset.csv")

## load data
data <- read.csv(dir, sep = ";")

## see data
summary(data)

## prepare data for analysis
data <- data[,-1]
cols <- names(data)
## convert data to factors
data[cols] <- lapply(data[cols], factor)
sapply(data, class)

## prepare data for classification
data[data == ""] <- NA
## split the dataset into train and test
train <- data %>% drop_na()
test <- data[is.na(data$Perception),]
test <- test[,-31]

####################################################################

## CLASSIFICATION
## try Naive Bayes and Tree Augmented and evaluate performance
nbcl <- naive.bayes(train, training = "Perception")
nbcl.trained <- bn.fit(nbcl, train)
graphviz.plot(nbcl)
tan <- tree.bayes(train, training = "Perception")
tan.trained <- bn.fit(tan, train)
graphviz.plot(tan)

## cross-validation on both models to evaluate the performances
cv.nb <- bn.cv(nbcl, data = train, method = "k-fold", runs = 10)
cv.tan <- bn.cv(tan, data = train, method = "k-fold", runs = 10)
plot(cv.nb, cv.tan, xlab = c("NaiveBayes", "Tree-Augmented"))

## predicting using NaiveBayes, who had better performance
pred <- predict(nbcl.trained, test)
test$Perception <- pred
## joining the data imputing the predicted missing values
data <- rbind(train, test)
## check data
summary(data)

###################################################################

## BLACKLISTING
block <- c(3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2,2,2,1,1,1,3,1,1,4)
blM <- matrix(0, nrow=31, ncol=31)
rownames(blM) <- colnames(blM) <- names(data)
for (b in 2:4) blM[block==b, block<b] <- 1
blackL <- data.frame(get.edgelist(as(blM, "igraph"))) 
names(blackL) <- c("from", "to")

###################################################################

## LEARNING ALGORITHMS
## score based algorithms
## Hill-Climbing
hc <- hc(data, blacklist = blackL)
plot(as(amat(hc), "graphNEL"))


## constraint based algorithms
## Incremental Association
iamb <- iamb(data, blacklist = blackL, undirected = FALSE)
plot(as(amat(iamb), "graphNEL"))
fiamb <- fast.iamb(data, blacklist = blackL, undirected = FALSE)
plot(as(amat(fiamb), "graphNEL"))
intiamb <- inter.iamb(data, blacklist = blackL, undirected = FALSE)
plot(as(amat(intiamb), "graphNEL"))


## constraint based local discovery algorithms
## max-min parents and children
mmpc <- mmpc(data, blacklist = blackL, undirected = FALSE)
plot(as(amat(mmpc), "graphNEL"))
## Semi-Interleaved HITON-PC
hit <- si.hiton.pc(data, blacklist = blackL, undirected = FALSE)
plot(as(amat(hit), "graphNEL"))

## hybrid-structure
## Sparse Candidate
sc <- rsmax2(data, blacklist = blackL)
plot(as(amat(sc), "graphNEL"))
## Max-Min Hill Climbing
mm <- mmhc(data, blacklist = blackL)
plot(as(amat(mm), "graphNEL"))
## Hybrid HPC
h2pc <- h2pc(data, blacklist = blackL)
plot(as(amat(h2pc), "graphNEL"))

## fit (Bayesian method preferred)
## Maximum Likelihood Estimator
mle <- bn.fit(h2pc, data, method = "mle")


## Bayesian estimation
bay <- bn.fit(h2pc, data, method = "bayes")

################################################################

## INFERENCE
## prepare for inference
data_gr <- as.grain(bay)
jtree <- compile(data_gr)
plot(jtree, type = "jt", main = "Junction Tree")

## Creation of functions facilitating the visual output

## exploration of beliefs updates of discrepancy of needs on denial of needs P(Q1.1|Q1.2)
denial_needs <- function (x) {
  x <- unlist(querygrain(setFinding(jtree, nodes="Q1.2", states = c("yes")),nodes=c("Q1.1"), type = "marginal"))
  x <- rbind(x,  unlist(querygrain(setFinding(jtree, nodes="Q1.2", states = c("no")),nodes=c("Q1.1"), type = "marginal")))
  x <- rbind(x,  unlist(querygrain(setFinding(jtree, nodes="Q1.2", states = c("dnk")),nodes=c("Q1.1"), type = "marginal")))
  x <- x[,-1]
  rownames(x) <- c("Yes", "No", "Dnk")
  print(x)
}

## exploration of complex registration on discrepancy of needs P(Q1.2|Q3.1)
discrepancy_needs <- function (x) {
  x <- unlist(querygrain(setFinding(jtree, nodes="Q3.1", states = c("yes")),nodes=c("Q1.2"), type = "marginal"))
  x <- rbind(x,  unlist(querygrain(setFinding(jtree, nodes="Q3.1", states = c("no")),nodes=c("Q1.2"), type = "marginal")))
  x <- rbind(x,  unlist(querygrain(setFinding(jtree, nodes="Q3.1", states = c("dnk")),nodes=c("Q1.2"), type = "marginal")))
  x <- x[,-1]
  rownames(x) <- c("Yes", "No", "Dnk")
  print(x)
}

## exploration of admin requirements to assistance on complex registration P(Q3.1|Q2.2)
burocracy <- function (x) {
  x <- unlist(querygrain(setFinding(jtree, nodes="Q2.2", states = c("yes")),nodes=c("Q3.1"), type = "marginal"))
  x <- rbind(x,  unlist(querygrain(setFinding(jtree, nodes="Q2.2", states = c("no")),nodes=c("Q3.1"), type = "marginal")))
  x <- rbind(x,  unlist(querygrain(setFinding(jtree, nodes="Q2.2", states = c("dnk")),nodes=c("Q3.1"), type = "marginal")))
  x <- x[,-1]
  rownames(x) <- c("Yes", "No", "Dnk")
  print(x)
}

## violence on interference P(Q5.1|Q7.1)
interference_violence <- function (x) {
  x <- unlist(querygrain(setFinding(jtree, nodes="Q7.1", states = c("yes")),nodes=c("Q5.1"), type = "marginal"))
  x <- rbind(x,  unlist(querygrain(setFinding(jtree, nodes="Q7.1", states = c("no")),nodes=c("Q5.1"), type = "marginal")))
  x <- rbind(x,  unlist(querygrain(setFinding(jtree, nodes="Q7.1", states = c("dnk")),nodes=c("Q5.1"), type = "marginal")))
  x <- x[,-1]
  rownames(x) <- c("Yes", "No", "Dnk")
  print(x)
}

## interference on denial right to assistance P(Q1.3|Q5.1)
interference_assistance <- function (x) {
    x <- unlist(querygrain(setFinding(jtree, nodes="Q5.1", states = c("yes")),nodes=c("Q1.3"), type = "marginal"))
  x <- rbind(x,  unlist(querygrain(setFinding(jtree, nodes="Q5.1", states = c("no")),nodes=c("Q1.3"), type = "marginal")))
  x <- rbind(x,  unlist(querygrain(setFinding(jtree, nodes="Q5.1", states = c("dnk")),nodes=c("Q1.3"), type = "marginal")))
  x <- x[,-1]
  rownames(x) <- c("Yes", "No", "Dnk")
  print(x)
}

## denial on forced displacement P(Q2.3|Q1.3)
denial_displacement <- function (x) {
  x <- unlist(querygrain(setFinding(jtree, nodes="Q1.3", states = c("yes")),nodes=c("Q2.3"), type = "marginal"))
  x <- rbind(x,  unlist(querygrain(setFinding(jtree, nodes="Q1.3", states = c("no")),nodes=c("Q2.3"), type = "marginal")))
  x <- rbind(x,  unlist(querygrain(setFinding(jtree, nodes="Q1.3", states = c("dnk")),nodes=c("Q2.3"), type = "marginal")))
  x <- x[,-1]
  rownames(x) <- c("Yes", "No", "Dnk")
  print(x)
}

## violence affecting movement of people on agencies on hold P(Q4.5|Q7.1)
onhold_violence <- function (x) {
  x <- unlist(querygrain(setFinding(jtree, nodes="Q7.1", states = c("yes")),nodes=c("Q4.5"), type = "marginal"))
  x <- rbind(x,  unlist(querygrain(setFinding(jtree, nodes="Q7.1", states = c("no")),nodes=c("Q4.5"), type = "marginal")))
  x <- rbind(x,  unlist(querygrain(setFinding(jtree, nodes="Q7.1", states = c("dnk")),nodes=c("Q4.5"), type = "marginal")))
  x <- x[,-1]
  rownames(x) <- c("Yes", "No", "Dnk")
  print(x)
}

## violence towards civilians on counterterrorism measures P(Q5.2|Q7.2)
violence_counterterrorism <- function (x) {
  x <- unlist(querygrain(setFinding(jtree, nodes="Q7.2", states = c("yes")),nodes=c("Q5.2"), type = "marginal"))
  x <- rbind(x,  unlist(querygrain(setFinding(jtree, nodes="Q7.2", states = c("no")),nodes=c("Q5.2"), type = "marginal")))
  x <- rbind(x,  unlist(querygrain(setFinding(jtree, nodes="Q7.2", states = c("dnk")),nodes=c("Q5.2"), type = "marginal")))
  x <- x[,-1]
  rownames(x) <- c("Yes", "No", "Dnk")
  print(x)
}

## violence towards civilians on relocation or suspension staff P(Q7.3|Q7.2)
violence_operations <- function (x) {
  x <- unlist(querygrain(setFinding(jtree, nodes="Q7.2", states = c("yes")),nodes=c("Q7.3"), type = "marginal"))
  x <- rbind(x,  unlist(querygrain(setFinding(jtree, nodes="Q7.2", states = c("no")),nodes=c("Q7.3"), type = "marginal")))
  x <- rbind(x,  unlist(querygrain(setFinding(jtree, nodes="Q7.2", states = c("dnk")),nodes=c("Q7.3"), type = "marginal")))
  x <- x[,-1]
  rownames(x) <- c("Yes", "No", "Dnk")
  print(x)
}
 
## logistical constraints on imports and visas P(Q3.3|Q9.3)
log_imports <- function (x) {
  x <- unlist(querygrain(setFinding(jtree, nodes="Q9.3", states = c("yes")),nodes=c("Q3.3"), type = "marginal"))
  x <- rbind(x,  unlist(querygrain(setFinding(jtree, nodes="Q9.3", states = c("no")),nodes=c("Q3.3"), type = "marginal")))
  x <- rbind(x,  unlist(querygrain(setFinding(jtree, nodes="Q9.3", states = c("dnk")),nodes=c("Q3.3"), type = "marginal")))
  x <- x[,-1]
  rownames(x) <- c("Yes", "No", "Dnk")
  print(x)
}

## imports on checkpoints P(Q4.3|Q3.3)
imports_checkpoints <- function (x) {
  x <- unlist(querygrain(setFinding(jtree, nodes="Q3.3", states = c("yes")),nodes=c("Q4.3"), type = "marginal"))
  x <- rbind(x,  unlist(querygrain(setFinding(jtree, nodes="Q3.3", states = c("no")),nodes=c("Q4.3"), type = "marginal")))
  x <- rbind(x,  unlist(querygrain(setFinding(jtree, nodes="Q3.3", states = c("dnk")),nodes=c("Q4.3"), type = "marginal")))
  x <- x[,-1]
  rownames(x) <- c("Yes", "No", "Dnk")
  print(x)
}

## landmine contamination on casualties P(Q8.2|Q8.1)
landmine_casualties <- function (x) {
  x <- unlist(querygrain(setFinding(jtree, nodes="Q8.1", states = c("suspected")),nodes=c("Q8.2"), type = "marginal"))
  x <- rbind(x,  unlist(querygrain(setFinding(jtree, nodes="Q8.1", states = c("confirmed")),nodes=c("Q8.2"), type = "marginal")))
  x <- rbind(x,  unlist(querygrain(setFinding(jtree, nodes="Q8.1", states = c("no")),nodes=c("Q8.2"), type = "marginal")))
  x <- x[,-1]
  rownames(x) <- c("Suspected", "Confirmed", "Dnk")
  print(x)
}

## exploration of beliefs updates of perception P(Perception|Q4.1)
perception_territory <- function (x) {
  x <- unlist(querygrain(setFinding(jtree, nodes="Q4.1", states = c("yes")),nodes=c("Perception"), type = "marginal"))
  x <- rbind(x,  unlist(querygrain(setFinding(jtree, nodes="Q4.1", states = c("no")),nodes=c("Perception"), type = "marginal")))
  x <- rbind(x,  unlist(querygrain(setFinding(jtree, nodes="Q4.1", states = c("dnk")),nodes=c("Perception"), type = "marginal")))
  x <- x[,-1]
  rownames(x) <- c("Yes", "No", "Dnk")
  print(x)
}

## violence on perception variable P(Perception|Q7.1)
perception_violence <- function (x) {
  x <- unlist(querygrain(setFinding(jtree, nodes="Q7.1", states = c("yes")),nodes=c("Perception"), type = "marginal"))
  x <- rbind(x,  unlist(querygrain(setFinding(jtree, nodes="Q7.1", states = c("no")),nodes=c("Perception"), type = "marginal")))
  x <- rbind(x,  unlist(querygrain(setFinding(jtree, nodes="Q7.1", states = c("dnk")),nodes=c("Perception"), type = "marginal")))
  x <- x[,-1]
  rownames(x) <- c("Yes", "No", "Dnk")
  print(x)
}

#############################################################
#############################################################

## ACAPS SCORING METHOD
## Cleaning

toreplace <-  c("Q1.1", "Q1.2", "Q1.3",
               "Q2.1", "Q2.2","Q2.3",
               "Q3.1", "Q3.2", "Q3.3", "Q3.4",
               "Q4.1", "Q4.2", "Q4.3", "Q4.4", "Q4.5",
               "Q5.1", "Q5.2", "Q5.3",
               "Q6.1", "Q6.2", "Q6.3",
               "Q7.1", "Q7.2", "Q7.3",
               "Q8.1", "Q8.2",
               "Q9.1", "Q9.2", "Q9.3", "Q9.4")

## Replace "yes" with 1 / "no" with 0 / "dnk" with 0 / "confirmed" with 1 / "suspected" with 0.5
root <- data %>%
  mutate_at(vars(all_of(toreplace)), ~ str_replace(., "yes", "1"))
root <- root %>%
  mutate_at(vars(all_of(toreplace)), ~ str_replace(., "no", "0"))
root <- root %>%
  mutate_at(vars(all_of(toreplace)), ~ str_replace(., "confirmed", "1"))
root <- root %>%
  mutate_at(vars(all_of(toreplace)), ~ str_replace(., "suspected", "0.5"))
root <- root %>%
  mutate_at(vars(all_of(toreplace)), ~ str_replace(., "dnk", "0"))
root[toreplace] <- sapply(root[toreplace], as.numeric)

q_scores <- root

## score each column according to their relative weight
q_scores$Q1.1 <- q_scores$Q1.1 * 0.3
q_scores$Q1.2 <- q_scores$Q1.2 * 0.3
q_scores$Q1.3 <- q_scores$Q1.3 * 0.4
q_scores$Q2.1 <- q_scores$Q2.1 * 0.4
q_scores$Q2.2 <- q_scores$Q2.2 * 0.3
q_scores$Q2.3 <- q_scores$Q2.3 * 0.3
q_scores$Q3.1 <- q_scores$Q3.1 * 0.1
q_scores$Q3.2 <- q_scores$Q3.2 * 0.25
q_scores$Q3.3 <- q_scores$Q3.3 * 0.1
q_scores$Q3.4 <- q_scores$Q3.4 * 0.55
q_scores$Q4.1 <- q_scores$Q4.1 * 0.3
q_scores$Q4.2 <- q_scores$Q4.2 * 0.2
q_scores$Q4.3 <- q_scores$Q4.3 * 0.1
q_scores$Q4.4 <- q_scores$Q4.4 * 0.2
q_scores$Q4.5 <- q_scores$Q4.5 * 0.2
q_scores$Q5.1 <- q_scores$Q5.1 * 0.3
q_scores$Q5.2 <- q_scores$Q5.2 * 0.3
q_scores$Q5.3 <- q_scores$Q5.3 * 0.4
q_scores$Q6.1 <- q_scores$Q6.1 * 0.4
q_scores$Q6.2 <- q_scores$Q6.2 * 0.3
q_scores$Q6.3 <- q_scores$Q6.3 * 0.3
q_scores$Q7.1 <- q_scores$Q7.1 * 0.25
q_scores$Q7.2 <- q_scores$Q7.2 * 0.25
q_scores$Q7.3 <- q_scores$Q7.3 * 0.5
q_scores$Q8.1 <- q_scores$Q8.1 * 0.45
q_scores$Q8.2 <- q_scores$Q8.2 * 0.55
q_scores$Q9.1 <- q_scores$Q9.1 * 0.25
q_scores$Q9.2 <- q_scores$Q9.2 * 0.25
q_scores$Q9.3 <- q_scores$Q9.3 * 0.25
q_scores$Q9.4 <- q_scores$Q9.4 * 0.25

## add column with results per indicator
q_scores <- mutate(q_scores, I1 = q_scores$Q1.1 + q_scores$Q1.2 + q_scores$Q1.3)
q_scores <- mutate(q_scores, I2 = q_scores$Q2.1 + q_scores$Q2.2 + q_scores$Q2.3)
q_scores <- mutate(q_scores, I3 = q_scores$Q3.1 + q_scores$Q3.2 + q_scores$Q3.3 + q_scores$Q3.4)
q_scores <- mutate(q_scores, I4 = q_scores$Q4.1 + q_scores$Q4.2 + q_scores$Q4.3 + q_scores$Q4.4 + q_scores$Q4.5)
q_scores <- mutate(q_scores, I5 = q_scores$Q5.1 + q_scores$Q5.2 + q_scores$Q5.3)
q_scores <- mutate(q_scores, I6 = q_scores$Q6.1 + q_scores$Q6.2 + q_scores$Q6.3)
q_scores <- mutate(q_scores, I7 = q_scores$Q7.1 + q_scores$Q7.2 + q_scores$Q7.3)
q_scores <- mutate(q_scores, I8 = q_scores$Q8.1 + q_scores$Q8.2)
q_scores <- mutate(q_scores, I9 = q_scores$Q9.1 + q_scores$Q9.2 + q_scores$Q9.3 + q_scores$Q9.4)

## scoring indicators according to thresholds
q_scores$I1 <- sapply(q_scores$I1, function(x) if (x <= 0) {
  0
} else if (x <= 0.3) {
  1
} else if (x <= 0.7) {
  2
} else {
  3
})

q_scores$I2 <- sapply(q_scores$I2, function(x) if (x <= 0) {
  0
} else if (x <= 0.3) {
  1
} else if (x <= 0.7) {
  2
} else {
  3
})

q_scores$I3 <- sapply(q_scores$I3, function(x) if (x <= 0) {
  0
} else if (x <= 0.25) {
  1
} else if (x <= 0.45) {
  2
} else {
  3
})


q_scores$I4 <- sapply(q_scores$I4, function(x) if (x <= 0) {
  0
} else if (x <= 0.3) {
  1
} else if (x <= 0.7) {
  2
} else {
  3
})

q_scores$I5 <- sapply(q_scores$I5, function(x) if (x <= 0) {
  0
} else if (x <= 0.3) {
  1
} else if (x <= 0.7) {
  2
} else {
  3
})

q_scores$I6 <- sapply(q_scores$I6, function(x) if (x <= 0) {
  0
} else if (x <= 0.3) {
  1
} else if (x <= 0.7) {
  2
} else {
  3
})

q_scores$I7 <- sapply(q_scores$I7, function(x) if (x <= 0) {
  0
} else if (x <= 0.3) {
  1
} else if (x <= 0.7) {
  2
} else {
  3
})

q_scores$I8 <- sapply(q_scores$I8, function(x) if (x <= 0) {
  0
} else if (x <= 0.225) {
  1
} else if (x <= 0.45) {
  2
} else {
  3
})

q_scores$I9 <- sapply(q_scores$I9, function(x) if (x <= 0) {
  0
} else if (x <= 0.3) {
  1
} else if (x <= 0.7) {
  2
} else {
  3
})

## new variable
scores <- q_scores

## summing and scoring pillars
scores$P1 <- scores$I1 + scores$I2
scores$P2 <- scores$I3 + scores$I4 + scores$I5 + scores$I6
scores$P3 <- scores$I7 + scores$I8 + scores$I9

scores$P1 <- sapply(scores$P1, function(x) if (x == 0) {
  0
} else if (x == 1) {
  1
} else if (x < 3) {
  2
} else if (x < 4) {
  3
} else if (x < 5) {
  4
} else {
  5
})

scores$P2 <- sapply(scores$P2, function(x) if (x == 0) {
  0
} else if (x < 2) {
  1
} else if (x < 6) {
  2
} else if (x < 8) {
  3
} else if (x < 10) {
  4
} else {
  5
})

scores$P3 <- sapply(scores$P3, function(x) if (x == 0) {
  0
} else if (x == 1) {
  1
} else if (x < 3) {
  2
} else if (x < 5) {
  3
} else if (x < 7) {
  4
} else {
  5
})


## removing not useful columns
scores <- subset(scores, select = -c(Q1.1, Q1.2, Q1.3, Q2.1, Q2.2, Q2.3, Q3.1, Q3.2, Q3.3, Q3.4, Q4.1, Q4.2, Q4.3, Q4.4, Q4.5, Q5.1, Q5.2, Q5.3, Q6.1, Q6.2, Q6.3, Q7.1, Q7.2, Q7.3, Q8.1, Q8.2, Q9.1, Q9.2, Q9.3, Q9.4))

## mean pillars
scores$MEAN <- round(rowMeans(select(scores, c(P1, P2, P3))))

## giving final scores
scores$FINAL <- ifelse(scores$I3 == 3, 5, scores$MEAN)

## removing not useful columns
scores <- subset(scores, select = -c(MEAN))

############################################################
############################################################

## SCORE EDA
hist(scores$Perception)
hist(scores$FINAL)
boxplot(scores$Perception)
boxplot(scores$FINAL)

## SCORING COMPARISON (SPEARMAN)

## coercing "perception" in numbers for comparison with score
scores$Perception <- sapply(scores$Perception, function(x) if (x == "high") {
  3
} else if (x == "medium") {
  2
} else {
  1
})

## running Spearman comparison
cor.test(scores$Perception, scores$FINAL, method = "spearman")

## plot results
set.seed(3)
ggplot(scores, aes(FINAL, Perception, color = Perception))+
  geom_point(size=1.5, position=position_jitter(h=0.25,w=0.25))

