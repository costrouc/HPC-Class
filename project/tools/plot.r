data <- read.csv("data/results.txt", header=TRUE, sep='\t')

library(ggplot2)

q1 <- qplot(data=data, x=n, y=GFLOPS, group=factor(P), colour=factor(P), geom="line")
ggsave("img/GFLOPS.png", q1)

