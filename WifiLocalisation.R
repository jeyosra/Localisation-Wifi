head(wifi)
summary(wifi)
plot(wifi)

plot(wifi[, c(1:7, 8)])
options(max.print=16000)
is.na(wifi) #No Missed value

n <- nrow(wifi) #sample size
n

ptrain <- 0.7   #proportion d'apprentissage
Itrain <- sample(n,round(ptrain*n)) #echantillon d'apprentissage
Ytest <- wifi[-Itrain,8] 
Ytrain <- wifi[Itrain,8]
Xtest <- wifi[-Itrain,1:7]
Xtrain <- wifi[Itrain,1:7]

#Decision Tree
library("tree")
tree.fit <- tree(V8~.,data.frame(wifi),subset=Itrain,method="class")
ypred1<-Predict(tree.fit,newdata=Xtest)
ypred1
tree.fit
summary(tree.fit)
plot(tree.fit)
text(tree.fit)
Itrain2 <- sample(n,round(ptrain*n))
tree.fit2 <- tree(V8~.,data.frame(wifi),subset=Itrain2,type="class")
summary(tree.fit2)

par(mfrow=1:2, mar=c(4,4,2,1))
plot(tree.fit)
text(tree.fit)
plot(tree.fit2)
text(tree.fit2)


#Random Forest
library("party")
RF.fit <- cforest(V8~.,data.frame(wifi),subset=Itrain)
Ytestpred.RF <- predict(RF.fit,newdata=Xtest,type="response")
risqueclassif.RF = mean(Ytestpred.RF!=Ytest)


RF.fit = cforest(V8~.,wifi,subset=Itrain, controls=
                   cforest_control(ntree=100))
RF.fit.vimp <- varimp(RF.fit, mincriterion = 1e-5)
barplot(sort(RF.fit.vimp[abs(RF.fit.vimp) > 4e-5]))

Ytestpred.RF = predict(RF.fit,newdata=Xtest)
risqueclassif.RF = mean(Ytestpred.RF!=Ytest)

print(RF.fit)


#Neural Network

#redefinition of plot without dev.new ()
plotnn <- function (x, rep = NULL, x.entry = NULL, x.out = NULL, radius = 0.15, 
                    arrow.length = 0.2, intercept = TRUE, intercept.factor = 0.4, 
                    information = TRUE, information.pos = 0.1, col.entry.synapse = "black", 
                    col.entry = "black", col.hidden = "black", col.hidden.synapse = "black", 
                    col.out = "black", col.out.synapse = "black", col.intercept = "blue", 
                    fontsize = 12, dimension = 6, show.weights = TRUE, file = NULL, 
                    ...) 
{
  net <- x
  if (is.null(net$weights)) 
    stop("weights were not calculated")
  if (!is.null(file) && !is.character(file)) 
    stop("'file' must be a string")
  if (is.null(rep)) {
    for (i in 1:length(net$weights)) {
      if (!is.null(file)) 
        file.rep <- paste(file, ".", i, sep = "")
      else file.rep <- NULL
      #grDevices::dev.new()
      neuralnet:::plot.nn(net, rep = i, x.entry, x.out, radius, arrow.length, 
                          intercept, intercept.factor, information, information.pos, 
                          col.entry.synapse, col.entry, col.hidden, col.hidden.synapse, 
                          col.out, col.out.synapse, col.intercept, fontsize, 
                          dimension, show.weights, file.rep, ...)
    }
  }
  else {
    if (is.character(file) && file.exists(file)) 
      stop(sprintf("%s already exists", sQuote(file)))
    result.matrix <- t(net$result.matrix)
    if (rep == "best") 
      rep <- as.integer(which.min(result.matrix[, "error"]))
    if (rep > length(net$weights)) 
      stop("'rep' does not exist")
    weights <- net$weights[[rep]]
    if (is.null(x.entry)) 
      x.entry <- 0.5 - (arrow.length/2) * length(weights)
    if (is.null(x.out)) 
      x.out <- 0.5 + (arrow.length/2) * length(weights)
    width <- max(x.out - x.entry + 0.2, 0.8) * 8
    radius <- radius/dimension
    entry.label <- net$model.list$variables
    out.label <- net$model.list$response
    neuron.count <- array(0, length(weights) + 1)
    neuron.count[1] <- nrow(weights[[1]]) - 1
    neuron.count[2] <- ncol(weights[[1]])
    x.position <- array(0, length(weights) + 1)
    x.position[1] <- x.entry
    x.position[length(weights) + 1] <- x.out
    if (length(weights) > 1) 
      for (i in 2:length(weights)) {
        neuron.count[i + 1] <- ncol(weights[[i]])
        x.position[i] <- x.entry + (i - 1) * (x.out - 
                                                x.entry)/length(weights)
      }
    y.step <- 1/(neuron.count + 1)
    y.position <- array(0, length(weights) + 1)
    y.intercept <- 1 - 2 * radius
    information.pos <- min(min(y.step) - 0.1, 0.2)
    if (length(entry.label) != neuron.count[1]) {
      if (length(entry.label) < neuron.count[1]) {
        tmp <- NULL
        for (i in 1:(neuron.count[1] - length(entry.label))) {
          tmp <- c(tmp, "no name")
        }
        entry.label <- c(entry.label, tmp)
      }
    }
    if (length(out.label) != neuron.count[length(neuron.count)]) {
      if (length(out.label) < neuron.count[length(neuron.count)]) {
        tmp <- NULL
        for (i in 1:(neuron.count[length(neuron.count)] - 
                     length(out.label))) {
          tmp <- c(tmp, "no name")
        }
        out.label <- c(out.label, tmp)
      }
    }
    grid::grid.newpage()
    for (k in 1:length(weights)) {
      for (i in 1:neuron.count[k]) {
        y.position[k] <- y.position[k] + y.step[k]
        y.tmp <- 0
        for (j in 1:neuron.count[k + 1]) {
          y.tmp <- y.tmp + y.step[k + 1]
          result <- calculate.delta(c(x.position[k], 
                                      x.position[k + 1]), c(y.position[k], y.tmp), 
                                    radius)
          x <- c(x.position[k], x.position[k + 1] - result[1])
          y <- c(y.position[k], y.tmp + result[2])
          grid::grid.lines(x = x, y = y, arrow = grid::arrow(length = grid::unit(0.15, 
                                                                                 "cm"), type = "closed"), gp = grid::gpar(fill = col.hidden.synapse, 
                                                                                                                          col = col.hidden.synapse, ...))
          if (show.weights) 
            draw.text(label = weights[[k]][neuron.count[k] - 
                                             i + 2, neuron.count[k + 1] - j + 1], x = c(x.position[k], 
                                                                                        x.position[k + 1]), y = c(y.position[k], 
                                                                                                                  y.tmp), xy.null = 1.25 * result, color = col.hidden.synapse, 
                      fontsize = fontsize - 2, ...)
        }
        if (k == 1) {
          grid::grid.lines(x = c((x.position[1] - arrow.length), 
                                 x.position[1] - radius), y = y.position[k], 
                           arrow = grid::arrow(length = grid::unit(0.15, 
                                                                   "cm"), type = "closed"), gp = grid::gpar(fill = col.entry.synapse, 
                                                                                                            col = col.entry.synapse, ...))
          draw.text(label = entry.label[(neuron.count[1] + 
                                           1) - i], x = c((x.position - arrow.length), 
                                                          x.position[1] - radius), y = c(y.position[k], 
                                                                                         y.position[k]), xy.null = c(0, 0), color = col.entry.synapse, 
                    fontsize = fontsize, ...)
          grid::grid.circle(x = x.position[k], y = y.position[k], 
                            r = radius, gp = grid::gpar(fill = "white", 
                                                        col = col.entry, ...))
        }
        else {
          grid::grid.circle(x = x.position[k], y = y.position[k], 
                            r = radius, gp = grid::gpar(fill = "white", 
                                                        col = col.hidden, ...))
        }
      }
    }
    out <- length(neuron.count)
    for (i in 1:neuron.count[out]) {
      y.position[out] <- y.position[out] + y.step[out]
      grid::grid.lines(x = c(x.position[out] + radius, 
                             x.position[out] + arrow.length), y = y.position[out], 
                       arrow = grid::arrow(length = grid::unit(0.15, 
                                                               "cm"), type = "closed"), gp = grid::gpar(fill = col.out.synapse, 
                                                                                                        col = col.out.synapse, ...))
      draw.text(label = out.label[(neuron.count[out] + 
                                     1) - i], x = c((x.position[out] + radius), x.position[out] + 
                                                      arrow.length), y = c(y.position[out], y.position[out]), 
                xy.null = c(0, 0), color = col.out.synapse, fontsize = fontsize, 
                ...)
      grid::grid.circle(x = x.position[out], y = y.position[out], 
                        r = radius, gp = grid::gpar(fill = "white", col = col.out, 
                                                    ...))
    }
    if (intercept) {
      for (k in 1:length(weights)) {
        y.tmp <- 0
        x.intercept <- (x.position[k + 1] - x.position[k]) * 
          intercept.factor + x.position[k]
        for (i in 1:neuron.count[k + 1]) {
          y.tmp <- y.tmp + y.step[k + 1]
          result <- calculate.delta(c(x.intercept, x.position[k + 
                                                                1]), c(y.intercept, y.tmp), radius)
          x <- c(x.intercept, x.position[k + 1] - result[1])
          y <- c(y.intercept, y.tmp + result[2])
          grid::grid.lines(x = x, y = y, arrow = grid::arrow(length = grid::unit(0.15, 
                                                                                 "cm"), type = "closed"), gp = grid::gpar(fill = col.intercept, 
                                                                                                                          col = col.intercept, ...))
          xy.null <- cbind(x.position[k + 1] - x.intercept - 
                             2 * result[1], -(y.tmp - y.intercept + 2 * 
                                                result[2]))
          if (show.weights) 
            draw.text(label = weights[[k]][1, neuron.count[k + 
                                                             1] - i + 1], x = c(x.intercept, x.position[k + 
                                                                                                          1]), y = c(y.intercept, y.tmp), xy.null = xy.null, 
                      color = col.intercept, alignment = c("right", 
                                                           "bottom"), fontsize = fontsize - 2, ...)
        }
        grid::grid.circle(x = x.intercept, y = y.intercept, 
                          r = radius, gp = grid::gpar(fill = "white", 
                                                      col = col.intercept, ...))
        grid::grid.text(1, x = x.intercept, y = y.intercept, 
                        gp = grid::gpar(col = col.intercept, ...))
      }
    }
    if (information) 
      grid::grid.text(paste("Error: ", round(result.matrix[rep, 
                                                           "error"], 6), "   Steps: ", result.matrix[rep, 
                                                                                                     "steps"], sep = ""), x = 0.5, y = information.pos, 
                      just = "bottom", gp = grid::gpar(fontsize = fontsize + 
                                                         2, ...))
    if (!is.null(file)) {
      weight.plot <- grDevices::recordPlot()
      save(weight.plot, file = file)
    }
  }
}

## ------------------------------------------------------------------------
library("neuralnet")

ntrain = length(Itrain)
ntest = n - ntrain
donnees_RN = data.frame(Xtrain,V8=Ytrain)



#f <- as.formula(paste("V8 ~", paste(names(wifi)[1:7], collapse = " + ")))

#1 hidden layer

#RN.fit1 = neuralnet(f,donnees_RN,linear.output = FALSE)
RN.fit1 = neuralnet(V8~V1+V2+V3+V4+V5+V6+V7,donnees_RN,linear.output = FALSE)

#RN.fit1 = neuralnet(f,donnees_RN,linear.output = FALSE)
plotnn(RN.fit1)

prob.RN.fit1 = compute(RN.fit1,Xtest)$net.result

errorRN<-mean(round(prob.RN.fit1,digit=0)!=Ytest)

#3 Hidden layers
RN.fit3 =neuralnet(V8~V1+V2+V3+V4+V5+V6+V7,donnees_RN,linear.output = FALSE,hidden=3)
plotnn(RN.fit3)
Ytestpred.RN.fit3 = compute(RN.fit3,Xtest)$net.result

risqueclassif.RN.fit3 = mean(round(Ytestpred.RN.fit3,digit=0)!=Ytest)

"-------------------------------------------------------------------------------"
# kNN
library(class)

wifi.sc = data.frame(wifi) 
wifi.sc[,-8] = scale(wifi[,-8]) # donnees centrees reduites

Xtest.sc = wifi.sc[-Itrain,-8]
Xtrain.sc = wifi.sc[Itrain,-8]

Ytestpred.1NN = knn(Xtrain.sc,Xtest.sc,Ytrain,k=1)
risqueclassif.1NN = mean(Ytestpred.1NN!=Ytest)

Ytestpred.3NN = knn(Xtrain.sc, Xtest.sc, Ytrain,k=3)
risqueclassif.3NN = mean(Ytestpred.3NN!=Ytest)

Ytestpred.5NN = knn(Xtrain.sc,Xtest.sc,Ytrain,k=5)
risqueclassif.5NN = mean(Ytestpred.5NN!=Ytest)

library('class')
n <- nrow(wifi)
ptrain <- 0.7
Itrain <- sample(n,round(ptrain*n))

Ytest <- wifi[-Itrain,8]
Ytrain <- wifi[Itrain,8]
Xtest <- wifi[-Itrain,1:7]
Xtrain <- wifi[Itrain,1:7]

knn1 = knn(Xtrain,Xtest,Ytrain, k=5)
mean(knn1!=Ytest)
knn2 = knn(Xtrain,Xtest,Ytrain, k=20)
mean(knn2!=Ytest)

knnCV <- function(xtrain,ytrain,kvect){
  ntrain = nrow(xtrain)
  nbk = length(kvect)
  risque = rep(NA,nbk)
  for (j in 1:nbk){
    preds = rep(NA,ntrain)
    for (i in 1:ntrain){
      preds[i] = knn(xtrain[-i,],xtrain[i,],ytrain[-i],k=kvect[j])
    }
    risque[j] = mean(preds!=ytrain)
  }
  khat <- kvect[which.min(risque)]
  print(cbind(k=kvect, risk=risque))
  plot(kvect, risque, type="b", pch=16)
  points(khat, min(risque), col="red")
  khat
}

K <- 50
kvect1 <- 1:K
khat1 <- knnCV(Xtrain, Ytrain, kvect1)
res1 <- knn(Xtrain,Xtest,Ytrain,khat1)
# Classification Error 
mean(res1!=Ytest)

knn3 = knn(Xtrain,Xtest,Ytrain, k=3)
mean(knn3!=Ytest)

"-----------------------------svm multiclasse----------------------------------"
svmlin = svm(V8~., data=data.frame(Xtrain,V8=Ytrain),type="C-classification" ,kernel="linear")
Ytestpred.svmlin=predict(svmlin ,Xtest)
risqueclassif.svmlin= mean(Ytestpred.svmlin!=Ytest)
print(svmlin)
summary(svmlin)
