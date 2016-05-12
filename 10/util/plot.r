data <- read.csv("data/results.txt", header=TRUE, sep='\t')

library(ggplot2)

q1 <- qplot(data=data, x=Size, y=MFLOPS, group=Method, colour=Method, geom="line")
ggsave("img/MFLOPS.png", q1)

q2 <- qplot(data=data, x=Size, y=Relative.Error, group=Method, colour=Method, geom="line")
ggsave("img/Error.png", q2)

q3 <- qplot(data=data, x=Size, y=Flops.Cycle, group=Method, colour=Method, geom="line")
ggsave("img/FlopsPerCycle.png", q3)