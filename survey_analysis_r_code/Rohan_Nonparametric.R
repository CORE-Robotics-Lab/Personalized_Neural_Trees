library('lme4')
library('dplyr')
library('car')
library('multcomp')
library('FSA')

setwd("C:\\Users\\rohan\\Desktop")

data <- read.csv(file = 'survey_responses_23_pairwise_only_fried.csv',header=TRUE)
#data <- read.csv(file = 'survey_responses_23_pairwise_only.csv',header=TRUE)

#######################################

#y = data$Tree.Decisions.Correct/data$Tree.Decisions.Possible 
#y = data$Tree.Decisions.Correct
y = data$Time
#y = data$Overall.Correct/data$Tree.Decisions.Possible
#y = data$Schore
#y = data$Schore
#y = data$Schore

#y = data$Overall.Correct 
#######################################

# Run Model
ID = factor(data$ï..Response.Id)
Type = factor(data$Type)

#m <- friedman.test(y ~ Type | ID,data = data)
#m <- friedman.test(y ~ Type|ID,data = data)
m <- wilcox.test(y~ Type, alternative = "less")

if (m$p.value < 0.05){
	print('Significant!')
} else {
	print('Not significant :(')
}
print(m$p.value)

Za = qnorm(m$p.value/2)
print(Za)


# Pairwise Tests

PT = dunnTest(y ~ Type, data=data, method="bh")
m <- lmer(y ~ Type + (1|ID),data = data)
summary(glht(m, linfct=mcp("Type"="Tukey")))

