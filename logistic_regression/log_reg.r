library(aod)
library(ggplot2)
library(glmtoolbox)

plot_logistic_curve = function(log_mod,file){
  mod_frame = model.frame(log_mod)
  var_names = names(mod_frame)
  newdat = setNames(
    data.frame(seq(min(mod_frame[[2]]), 
    max(mod_frame[[2]]), len=100)), 
    var_names[2])
  newdat[var_names[1]] = predict(log_mod, newdata = newdat, type="response")
  write.table(newdat,file,row.names = FALSE,col.names = FALSE)
  plot(mod_frame[[1]] ~ mod_frame[[2]], 
        col = "red4", 
        xlab = "Number of Formica colonies", 
        ylab = "Polyergus survival",
        cex=1.5)
  lines(newdat[[var_names[2]]], newdat[[var_names[1]]], col = "green4", lwd = 4)

} 

tab <- read.table("../data/logistic_regression/poisson-table-score-full.out",sep=' ',col.names=c('R0','R1','R2','survive'))

mylogit <- glm(survive ~ R1, data=tab, family=binomial("logit"))
summary(mylogit)
hltest(mylogit)
plot_logistic_curve(mylogit, '../data/logistic_regression/pois-pred-score-full.out')



dens <- read.table("../data/logistic_regression/sweep.out",sep=' ',col.names=c('survive','m'))

mylogit <- glm(survive ~ m, data=dens, family=binomial("logit"))
summary(mylogit)
hltest(mylogit)

plot_logistic_curve(mylogit,'../data/logistic_regression/sweep-pred.out')
