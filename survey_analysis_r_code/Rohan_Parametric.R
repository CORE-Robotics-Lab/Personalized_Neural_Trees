library('lme4')
library('dplyr')
library('car')
library('multcomp')



setwd("C:\\Users\\rohan\\Desktop")



data <- read.csv(file = 'survey_responses_23_pairwise_only_fried.csv',header=TRUE)
str(data)
#######################################

#y = data$Likert.Tree
#y = data$Likert.Overall
#y = data$Time
y = data$Tree.Decisions.Correct/data$Tree.Decisions.Possible 


#######################################

# Run Model
ID = factor(data$ï..Response.Id)
Type = factor(data$Type)
m <- lmer(y ~ Type + (1|ID),data = data)

stats <- anova(m)

studentized_residuals <- rstudent(m)
shapiro_stats <- shapiro.test(studentized_residuals)
#qqnorm(studentized_residuals); 
#qqline(studentized_residuals, col = 2)

p <- shapiro_stats$p.value
p
if (p < 0.05){
	print(cat('Not Normally Distributed -- BAD (' ,p, ')'))
} else {
	print(cat('Normally Distributed-- GOOD (',p,')'))
}

levenes <- leveneTest(y  ~ Type, data = data)
p <- levenes$'Pr(>F)'[1]
p
if (p < 0.05){
	print(cat('Heteroscedastic -- BAD (' ,p, ')'))
} else {
	print(cat('Homoscedastic -- GOOD (',p,')'))
}

res <- t.test(y~Type, paired = TRUE, alternative='greater')
res
#summary(glht(m, linfct=mcp("Type"="Tukey")))

