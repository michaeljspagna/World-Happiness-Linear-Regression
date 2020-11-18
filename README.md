# World-Happiness-Linear-Regression

Created a model using that accurately predict's a countries 'Happiness Score'

A Linear Regression using Gradient Descent for Multiple Variables was used to train this model

Gradient Descent: 
  Repeat until converge{
    1. Use Hypothesis to make prediction on given feature data
    2. Compute the cost of your function using the true y values
    3. Update Theta values to improve prediction
  }
  
 Hypothesis(Vectorized): h_theta(X) = X dot thetas
    - X: mxn matrix of feature values
    - thetas: nx1 vector of feature weights
 
 Cost(Vecorized) J(theta)

Data retrieved from Kaggle.com
https://www.kaggle.com/unsdsn/world-happiness
License - CC0: Public Domain

All csv files were updated to match format of 2018/2019.csv. 
'Family' and 'Social support' were treated as the same column and named 'Social support'

Below Information taken directly from Kaggle about the dataset

Context
The World Happiness Report is a landmark survey of the state of global happiness. The first report was published in 2012, the second in 2013, the third in 2015, and the fourth in the 2016 Update. The World Happiness 2017, which ranks 155 countries by their happiness levels, was released at the United Nations at an event celebrating International Day of Happiness on March 20th. The report continues to gain global recognition as governments, organizations and civil society increasingly use happiness indicators to inform their policy-making decisions. Leading experts across fields – economics, psychology, survey analysis, national statistics, health, public policy and more – describe how measurements of well-being can be used effectively to assess the progress of nations. The reports review the state of happiness in the world today and show how the new science of happiness explains personal and national variations in happiness.

Content
The happiness scores and rankings use data from the Gallup World Poll. The scores are based on answers to the main life evaluation question asked in the poll. This question, known as the Cantril ladder, asks respondents to think of a ladder with the best possible life for them being a 10 and the worst possible life being a 0 and to rate their own current lives on that scale. The scores are from nationally representative samples for the years 2013-2016 and use the Gallup weights to make the estimates representative. The columns following the happiness score estimate the extent to which each of six factors – economic production, social support, life expectancy, freedom, absence of corruption, and generosity – contribute to making life evaluations higher in each country than they are in Dystopia, a hypothetical country that has values equal to the world’s lowest national averages for each of the six factors. They have no impact on the total score reported for each country, but they do explain why some countries rank higher than others.


