library(ggplot2)

data1 <- read.table("ex2_test.txt", header=TRUE, sep='\t')

p1 <- qplot(data=data1, x = factor(Size), y = Time, group=Solver, geom="line", colour=Solver, facets=~Processors)

p2 <- qplot(data=data1, x = factor(Size), y = Error, group=Solver, geom="line", colour=Solver, facets=~Processors)

p3 <- qplot(data=data1, x = factor(Size), y = Itterations, group=Solver, geom="line", colour=Solver, facets=~Processors)

ggsave("Time.png", plot = p1)
ggsave("Error.png", plot = p2)
ggsave("Itterations.png", plot = p3)