data <- read.csv("data/results.txt", header=TRUE, sep='\t')

library(ggplot2)

q1 <- qplot(data=data, x=Block.Size, y=MFLOPS, geom="line")
ggsave("img/MFLOPS.png", q1)

q3 <- qplot(data=data, x=Block.Size, y=Flops.Cycle, geom="line")
ggsave("img/FlopsPerCycle.png", q3)