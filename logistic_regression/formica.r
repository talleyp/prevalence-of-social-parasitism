library(aod)
library(ggplot2)
library(glmtoolbox)
library(dplyr)

make_table <- function(files,R0,R1) {
    N <- length(files)
    cols <- c('NF','Fend','survive')
    df <- data.frame(matrix(nrow = 0, ncol = length(cols))) 
    colnames(df) <- cols
    for (file in files) {
        Z <- read.csv(file,header = FALSE)
        Z <- Z$V1
        N <- length(Z)-1
        p <- sum(Z[length(Z)]>5000)
        Fend <- sum(Z[1:N]>300)
        df[nrow(df)+1, ] <- c(N, Fend, p)
    }
    return(df)
}

## load data
folder0 <- '../landscape/data/poisson/'
z0 <- list.files(path=folder0,include.dirs = TRUE, pattern = '*-Z.out')
z0 <- paste(folder0,z0,sep='')
folder1 <- '../landscape/data/poisson-2/'
z1 <- list.files(path=folder1,include.dirs = TRUE, pattern = '*-Z.out')
z1 <- paste(folder1,z1,sep='')
z <- c(z0,z1)

## get survival densities
F_df <- make_table(z)

write.csv(F_df,"../data/formica_density/df.csv")

# F_df %>%
#     filter( survive==1 ) %>%
#     ggplot( aes(x=1/Fend)) +
#         geom_density(fill="#69b3a2", color="#e9ecef", alpha=0.9) +
#         xlab("Final local abundance where Polyergus survived") + 
#         xlim(0,0.15)
#         ggtitle("") #+


