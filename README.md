# Exploring the Effect of Diet Health on Recipe Ratings

Exploring the Effect of Diet Health on Recipe Ratings is a data science project conducted at UCSD. This project investigates the relationship of ratings of online recipes and their health benefits through methods such as exploratory data analysis, hypothesis testing, and predictive modeling. The following is a comprehensive report of this investigation. 

Author: Kai Arizabal

## Introduction

Having a balanced diet is essential for maintaining a healthy life. Proper diets provide necessary nutrition, ultimately boosting immunity against chronic diseases and improving physical and emotional health. Although the best diets vary between individuals, all healthy diets follow basic principles. Sustaining a healthy diet requires balancing nutrients, reducing sodium and sugar intake, and avoiding processed foods. Using datasets containing recipes and ratings from [food.com](https://www.food.com/) (originally scraped by Prasad Majumder et al. for their research into generating personalized recipes), this project investigates public preferences of health and diet. 

The research question guiding this project is: **Do healthier diets tend to have higher ratings?** 

Exploring this question provides insight into popular diets, particularly if diet-specific foods are rated based on their health benefits, taste, or convenience, therefore elucidating the most accessible yet effective food diets for a healthy life.

The first dataset `recipes` has 83782 rows and 12 columns, containing the following information:

| COLUMN | DESCRIPTION |
| ----------- | ----------- |
| `'name'` | Name of the recipe |
| `'id'` | Unique ID given to the recipe |
| `'minutes'` | Number of minutes to prepare the recipe |
| `'contributer_id'` | The ID of individual who submitted the recipe |
| `'submitted'` | Date on which recipe was submitted |
| `'tags'` | Food.com tags to categorize the recipe |
| `'nutrition'` | The following nutritional information as percent daily values (PDV): `'['calories (#), total fat (PDV), sugar (PDV), sodium (PDV), protein (PDV), saturated fat (PDV), carbohydrates (PDV)]'` |
| `'n_steps'` | Number of steps |
| `'steps'` | Steps to preparing the recipe, in order |
| `'description'` | A user-provided description |
| `'ingredients'` | All of the ingredients used |
| `'n_ingredients'` | Number of ingredients |

The second dataset `interactions` has 731927 rows of user ratings and 5 columns:

| COLUMN | DESCRIPTION |
| ----------- | ----------- |
| `'user_id'` | User ID of an individual |
| `'recipe_id'` | Recipe ID that this user interacted with |
| `'date'` | Date of their interaction |
| `'rating'` | Rating given |
| `'review'` | Review given by user |

Using the combined data provided by both datasets will allow for this exploration into the relationship of ratings and diets.

## Data Cleaning and Exploratory Data Analysis

### Data Cleaning

Before any exploration of the data, cleaning the data is necessary for efficient and easy analysis.

1. For convenience, I left merge the datasets on their keys, `'id'` (of `recipes`) and `'recipe_id'` (of `interactions`) . This merge ensures that all data is contained in one DataFrame.

2. I perform data quality checks, including checking the datatypes each column stores.

3. I fill all `'rating'` of `0.0` with `np.nan`. The scale of `'rating'` is `1, 2, 3, 4, 5`. Any `0` rating indicates that no star rating was given, which may disproportionately skew the data. Therefore, this replacement corrects this skew.

4. Afterwards, I compute the average rating for each recipe. The resulting series is concatenated to the larger DataFrame as `'avg rating'` column.

5. The columns `'submitted'` and `'date'` are converted to DateTime objects.

6. Cleaning the `'nutrition'` column, which was stored as a string object. I split the information into the following individual columns (as floats): `'calories (#)', 'total fat (PDV)', 'sugar (PDV)', 'sodium (PDV)', 'protein (PDV)', 'saturated fat (PDV)', 'carbohydrates (PDV)'`

7. For the nutrients `'total fat (PDV)', 'sugar (PDV)', 'protein (PDV)', 'saturated fat (PDV)', 'carbohydrates (PDV)'`, I compute their proportions to calories. This normalizes the nutrition relative to their calories, allowing for easy comparability and interpretability. These are stored in `'prop total fat', 'prop sugar', 'prop saturated fat', 'prop protein', 'prop carbohydrates'`. The process involved:

    1. I divide these nutrients by 100 so they are expressed as decimals rather than percentages. 
    
    2. Using the recommended daily values from the [Food and Drug Administration (FDA)](https://www.fda.gov/food/nutrition-facts-label/daily-value-nutrition-and-supplement-facts-labels), I multiply these decimals by their respective daily values for each nutrient. This conversion gives the nutrient content in grams.

    3. I multiply the grams by their calories per gram to get the number of calories contributed by each nutrient. Finally, I divide by the `'calories (#)'` column to obtain the proportions to calories.

    4. For proportions > 1.0, I cap them so that they are = 1.0.

    **NOTE:** Sodium has no caloric content.

8. Using the `'tags'` column, I add the column `'diet'` that specifies if and what diet the recipe is appropriate for. If the recipe is not apparently diet-specific, then the value is `np.nan`. This process involved:

    1. Cleaning the `'tags'` column, which was originally stored as a string object. After cleaning, the `'tags'` column contain lists of strings.

    2. Choosing and categorizing diets based on a specific tag. The diets chosen for this category are based on popularity and sustainability. Their corresponding tags are chosen from only the unique strings in the `'tags'` column based on best (general) descriptor(s). The following diets and their specific tags are:

        | DIET | TAG(S) |
        | ----------- | ----------- |
        | `'vegan'` | `'vegan'` |
        | `'vegetarian'` | `'vegetarian'` |
        | `'mediterranean'` | `'high-fiber'` or `'low-saturated-fat'` |
        | `'general_health'` | `'low-sodium'` , `'low-calorie'`, `'low-carb'`, `'low-cholesterol'`, or `'low-fat'`|

        Any recipe may be appropriate for multiple diets.

9. Using the values of the column `'diet'`, I add an additional column `'is_diet_specific'` based on whether the value stored in `'diet'` is `np.nan`. This column is a boolean that indicates if the recipe is diet-specific (pertaining to a diet) or not.

After this cleaning process, I choose only the relevant columns for hypothesis testing and predictive modeling. Below is a random sample (`n=5`) of the cleaned DataFrame, consisting only of said relevant columns:

| name                                                     |     id |   minutes | submitted           |   rating |   avg rating |   calories (#) |   prop total fat |   prop sugar |   prop saturated fat |   prop protein |   prop carbohydrates | diet                             | is_diet_specific   |
|:---------------------------------------------------------|-------:|----------:|:--------------------|---------:|-------------:|---------------:|-----------------:|-------------:|---------------------:|---------------:|---------------------:|:---------------------------------|:-------------------|
| chili salsa                                              | 305557 |        10 | 2008-05-28 00:00:00 |        5 |            5 |          112.4 |         0.624555 |    0.391459  |            0.0800712 |      0.0533808 |             0.391459 | ['vegetarian', 'vegan']          | True               |
| fassolakia freska me domata   greek green beans   tomato | 307592 |        65 | 2008-06-05 00:00:00 |        5 |            4 |          604.2 |         0.975968 |    0.115856  |            0.113208  |      0.0430321 |             0.163853 | ['general_health', 'vegetarian'] | True               |
| spicy ginger tuna patties                                | 320487 |        22 | 2008-08-20 00:00:00 |        2 |            2 |           86.9 |         0.161565 |    0.0920598 |            0.0207135 |      0.598389  |             0.126582 | nan                              | False              |
| silver palate carrot cake                                | 352606 |        65 | 2009-01-28 00:00:00 |        5 |            4 |          899.3 |         0.546425 |    0.787279  |            0.146114  |      0.0400311 |             0.464806 | nan                              | False              |
| one pot cheesy chicken and noodles                       | 334688 |        25 | 2008-11-03 00:00:00 |        5 |            5 |          562.1 |         0.424622 |    0.042697  |            0.153709  |      0.25974   |             0.332681 | ['general_health']               | True               |

**NOTE**: Should any columns require further cleaning for the hypothesis testing or predictive modeling specifically, the adjustments will be included at those steps. 

## Exploratory Data Analysis (EDA)

### Univariate Analysis

To begin my EDA, I first look into the distribution of average ratings, which provides an overall measure of public preference for a recipe.

<iframe src='assets/avg_ratings_distribution.html' width='800' height='500' frameborder='0' ></iframe>

The distribution of average ratings is highly skewed. There are significantly more higher ratings (4s and 5s) than lower ratings, indicating a selection bias among the recipes that received these high ratings. Typically, users are more likely to rate recipes they enjoyed or preferred. Therefore, throughout this project, I explore how additional information such as nutritional composition or diet specificity may contribute to this imbalance.

### Bivariate Analysis

I now explore the distribution of average ratings based on diet specificity.

<iframe src='assets/avg_ratings_distribution_on_diet.html' width='800' height='500' frameborder='0' ></iframe>

This probability distribution suggests that users are more likely to give higher ratings, thus corroborating a selection bias among ratings. However, it also appears that diet-specific recipes tend to get rated higher more often than nonspecific recipes. 

For each nutrient besides sodium, I made a box plot visualization by diet. The most interesting visuals are shown below:

<iframe src='assets/box_prop_carbs.html' width='800' height='500' frameborder='0' ></iframe>

As depicted above, the specific diets of mediterranean, vegetarian, and vegan typically have higher proportions of carbohydrates in their recipes. Higher content of carbohydrates suggest more unprocessed ingredients, such as whole grains or vegetables. This discrepancy among diets and general diets may be relevant to the high average ratings of diet-specific recipes.

<iframe src='assets/box_prop_total_fat.html' width='800' height='500' frameborder='0' ></iframe>

Although there is no obvious difference between non-diet recipes and diet-specific recipes, diet-specific recipes tend to have a wider variety of total fats content—notably, 50% of diet-specific recipes may have similar or lower proportions of total fats than nonspecific recipes. Additionally, the mediterranean diet has a considerable lower proportion of total fats—interestingly, the mediterranean diet prefers lower total fats.

### Interesting Aggregates

To see if there were trends in the nutritional compositions of specific ratings and diets, I grouped by columns such as `'rating'`, `'avg rating'`, `'diet'`, or `'is_diet_specific'` and took the mean nutritional content per group. Below is one of the results:

|   avg rating |   calories (#) |   prop total fat |   prop protein |   prop carbohydrates |   prop saturated fat |   prop sugar |
|-------------:|---------------:|-----------------:|---------------:|---------------------:|---------------------:|-------------:|
|            1 |        463.345 |            0.463 |          0.137 |                0.394 |                0.145 |        0.351 |
|            2 |        432.541 |            0.452 |          0.152 |                0.392 |                0.148 |        0.333 |
|            3 |        459.762 |            0.462 |          0.153 |                0.38  |                0.146 |        0.33  |
|            4 |        428.508 |            0.467 |          0.166 |                0.365 |                0.148 |        0.3   |
|            5 |        413.724 |            0.488 |          0.157 |                0.351 |                0.154 |        0.301 |

This table provides the mean nutritional content for every `'avg rating'`. The most obvious trend is the decrease in `'calories (#)'` as rating increases. However, there are more subtle trends, such as a small increase of `'prop protein'` and `'prop total fat'` or slight decline in `'prop carbohydrates'` and `'prop sugar'` as rating increases.

Here is another interesting aggregate:

| is_diet_specific   |   avg rating |   calories (#) |   prop total fat |   prop protein |   prop carbohydrates |   prop saturated fat |   prop sugar |
|:-------------------|-------------:|---------------:|-----------------:|---------------:|---------------------:|---------------------:|-------------:|
| False              |        4.694 |        452.197 |            0.502 |          0.157 |                0.338 |                0.165 |        0.295 |
| True               |        4.708 |        380.738 |            0.458 |          0.16  |                0.376 |                0.137 |        0.31  |

This table demonstrates the differences of nutrition between diets and nonspecific recipes. Diet-specific recipes tend to be lower in calories and saturated fats but are higher in carbohydrates and sugar, possibly because of plant-based ingredients. However, diet-specific recipes have slightly higher average ratings, suggesting they may be more popular than nonspecific recipes.

## Assessment of Missingness

In the dataset, there are four columns containing a significant number of missing values: `'description'`, `'rating'`, `'avg rating'`, and `'diet'`.

### NMAR Analysis

Of these columns, the most likely to be not missing at random (NMAR) is `'description'`. NMAR occurs when the missingness of the value depends on the data itself. The description may be missing because the contributer may have not provided any additional information if the recipe is straight-forward or self-explanatory. Therefore, if the contributer decided to not include a description, there would be missing values in the dataset. Since the missingness depends on if a description is provided or not, `'description'` may be NMAR.

### Missingness Dependency

Among the remaining three columns, I have the least information on the `'rating'` column. Therefore, I will test the missingness dependency of `'rating'` on columns `'prop protein'` and `'prop carbohydrates'`.

The test statistic is the absolute difference in means, and the significance level is 0.05. For both tests, I ran 1000 permutation tests.

I will first test the dependency of `'rating'` on `'prop protein'`.

> **Null Hypothesis:** The distributions of `''prop protein'` are the same whether `'rating'` is missing or not. 

> **Alternative Hypothesis:** The distributions of `'prop protein'` are **not** the same if `'rating'` is missing or not. 

Below are analyses of the observed distribution of `'prop protein'` when `'rating'` is missing and not missing. 

<iframe src='assets/protein-dependency-kde.html' width='800' height='500' frameborder='0'></iframe>

The observed means of the above distribution:

| rating_missing   |   prop protein |
|:-----------------|---------------:|
| False            |       0.155073 |
| True             |       0.158982 |

The observed statistic is 0.004.

**The p-value is 0.00**, so I **reject** the null at a 0.05 significance level. Therefore, the missingness of `'rating'` may be dependent on `'prop protein'`. Below is the empirical distribution of the test statistics (and observed statistic):

<iframe src='assets/protein-dependency-test.html' width='800' height='500' frameborder='0' ></iframe>

Now, I will test the dependency of `'rating'` on `'prop carbohydrates'`.

> **Null Hypothesis:** The distributions of `'prop carbohydrates'` are the same whether `'rating'` is missing or not. 

> **Alternative Hypothesis:** The distributions of `'prop carbohydrates'` are **not** the same if `'rating'` is missing or not. 

Below are analyses of the observed distribution of `'prop carbohydrates'` when `'rating'` is missing and not missing. 

<iframe src='assets/carbs-dependency-kde.html' width='800' height='500' frameborder='0' ></iframe>

The observed means of the above distribution:

| rating_missing   |   prop carbohydrates |
|:-----------------|---------------------:|
| False            |             0.358671 |
| True             |             0.355406 |

The observed statistic is 0.003.

**The p-value is 0.091**, therefore I **fail to reject** the null at a 0.05 significance level. Accordingly, the missingness of `'rating'` may not be dependent on `'prop carbohydrates'`. Below is the empirical distribution of the test statistics (and observed statistic):

<iframe src='assets/carbs-dependency-test.html' width='800' height='500' frameborder='0' ></iframe>

## Hypothesis Testing

For the research question **do healthier diets tend to have higher ratings?**, I test whether there is a signficant difference in the ratings between diet-specific recipes or non-diet recipes. This hypothesis test will therefore provide basic insight into the public preferences of food diets, specifically into those of healthy diets. The alternative hypothesis claims that diet-specific recipes may be rated **higher** than non-diet recipes, under this perception of **increased health benefits** because of careful nutrient intake.

> **Null Hypothesis:** There is no difference between the average ratings for diet-specific and non-specific recipes.

> **Alternative Hypothesis:** Diet-specific recipes have higher average ratings compared to nonspecific recipes. 

> **Test Statistic:** The difference in mean average ratings between the diet-specific and nonspecific recipes (Diet-Specific - Nonspecific).

> **Significance Level:** 0.05

The observed statistic is **0.014** (rounded to three decimals). 

I test under the null using a permutation test of 5000 simulations. **The resulting p-value is 0.00**, and so I **reject** the null at a 0.05 significance level. This result indicates that the average ratings of diet-specific recipes tend to be higher than nonspecific recipes. Ultimately, this may be attributed to the health benefits of maintaining a health-related diet, whether it be for boosted immunity, overall health, weight loss, muscle gain, etc. 

The plot below is the histogram containing the distribution of mean differences computed for the test, including the observed difference:

<iframe src='assets/hypothesis-test.html' width='800' height='500' frameborder='0' ></iframe>

## Framing a Prediction Problem

Based on the results of previous sections, there is a likely correlation between the (average) ratings of recipes and dietary health benefits. To further this investigation into public preferences for healthy diets, I will **predict average ratings** based on diet-related metrics. This prediction problem also addresses the research question of the previous sections: Do healthier diets tend to have higher ratings?.

This prediction problem requires a multiclass classification model. The response variable is `'avg rating'`, as it provides more comprehensive insights into public preferences, and the results of the hypothesis test suggest that average ratings may be tied to diet specificity. Since this variable is an qualitative ordinal variable of values `1, 2, 3, 4, 5`, this predictive model should be able to predict any average rating in this range.

As seen in the EDA, the distribution of average ratings tend to be highly skewed. Therefore, I will use both accuracy and F1 score to evaluate the model's performance: accuracy to assess the model performance overall, and F1 to assess the model performance per class—those being `1, 2, 3, 4, 5`. Additionally, I will split the dataset into 75% training and 25% testing data to reduce overfitting and improve generalization. 

At the time of prediction, the data I would know is only the information recorded in the `recipes` dataset. These data are recorded when the recipe is submitted and published online. The most relevant data for this predictive model is their nutritional values (`'nutrition'`) and descriptive tags (`'tags'`) for identifying diet specificity. 

**NOTE:** 2777 average ratings are missing. I will drop these rows since this missing data is only a fraction of the entire dataset.

## Baseline Model

For the baseline model, I used a Random Forest Classifier with the features `'calories (#)'` and `'diet'`. These data consist of quantitative numerical and qualitative nomial values, respectively. 

To transform `'calories (#)'`, I use a `RobustScaler` to address the outliers in the data. For `'diet'`, I one hot encode the diet types. Since many recipes are appropriate for multiple diets (stored as lists), I use a count vectorizer to perform this transformation. Since `CountVectorizer` is commonly used on text string data, I also use a `FunctionTransformer` to clean the data, joining the lists into a string or an empty string, if `np.nan`. This ensures that the vectorizer works as intended.

For this model, the accuracy is **0.812**, and the F1 scores for each rating are: **0.124, 0.3  , 0.355, 0.601, 0.882**. Even though the accuracy is 81%, this metric does not consider the imbalance of higher average ratings. The scores for lower average ratings, `1, 2, 3`, are relatively low compared to `4` and `5`, where the model's performance improves significantly because of this skew.

## Final Model

The final model is a Random Forest Classifier using the features `'calories (#)'`, `'diet'`, `'submitted'`, `'prop total fat'`, `'prop protein'`, `'prop carbohydrates'`, and `'prop sugar'`.

- `'calories (#)'` gives the number of calories in a recipe. For many diets, caloric intake is significant because it provides the body energy, increases or decreases body weight, and determines nutrient density. As seen in the **Interesting Aggregates** section, the number of calories tend to decrease as rating increases. This suggests a considerable relationship between calories and average ratings, and this trend is consistent with healthy diets. Therefore, I use this feature in the baseline and final model. As described above, I use `'RobustScaler'` to address the skewed data.

- `'diet'` lists the approriate diets for a recipe. The results of the hypothesis test suggest a significant relationship between diet specificity and average ratings, so I use `'diet'` in the model. I chose `'diet'` over `'is_diet_specific'` because both are related—the data of `'is_diet_specific'` is derived from `'diet'`—but `'diet'` provides more insight than the latter. I one hot encoded the diet categories of this feature, as described above.

- `'submitted'` states the date a recipe was submitted. Although this data is not inherently tied to diet-related metrics, the age of the recipe correlates to popularity. The older the recipe, the higher or more stable its rating may be. Additionally, several health-related diets may have differing popularities across the years. I believe `'submitted'` provides additional information of recipe popularity trends over the years, so I include this feature and extract only the year using a `FunctionTransformer`. 

- `'prop total fat'`, `'prop protein'`, `'prop carbohydrates'`, and `'prop sugar'` give the caloric proportion of these nutrients of a recipe. Not only are these important to a balanced diet, but as seen in the **EDA** such as the table from **Interesting Aggregates**, there may be a correlation between these measures and average rating that I believe will improve the model. I use these information in my model as they are: all feature scaling or cleaning were done during the **Data Cleaning** stages, most notably, capping the proportions at 1.0.

For the hyperparameters of the model, I ultimately chose the default hyperparameters. After several rounds of `GridSearchCV`, the best hyperparameters only provided a marginal—sometimes worse—improvement to the model's performance. Additionally, at times, the tuned hyperparameters performed worse for lower rating classes than the default hyperparameters. Given this, I opted for the default settings to maintain a balanced and computationally efficient model.

The accuracy is now **0.926**, which is a 0.114 increase from the baseline model. Additionallty, the F1 scores for all classes also increased signficantly: **0.356, 0.649, 0.70 , 0.858, 0.952**. Since both metrics increased, the final model has improved from the baseline model's performance.

## Fairness Analysis

To assess the fairness of the model, I will answer the following question: **Does my model perform better for recipes that are diet-specific than it does for recipes without a specific diet?** To answer this question, I will perform a permutation test of 1000 simulations, using the difference in weighted F1 scores as the test statistic. This metric not only combines the precision and recall metrics, but also provides an overall measure of model performance that addresses class imbalance.

> **Null Hypothesis:** The model is fair. Its F1 score for diet-specific and nonspecific recipes are roughly the same, and any differences are due to random chance.

> **Alternative Hypothesis:** The model is unfair. Its F1 score for diet-specific recipes is higher than the F1 score for nonspecific recipes.

> **Test Statistic:** The difference in weighted F1 scores between the diet-specific and nonspecific recipes (Diet-Specific - Nonspecific).

> **Significance Level:** 0.05

<iframe src='assets/fairness-analysis.html' width='800' height='500' frameborder='0' ></iframe>

The observed statistic is **0.007** (rounded to three decimals). 

**The resulting p-value is 0.0002**, so I **reject** the null at a 0.05 significance level. This indicates that the model may be unfair, as it performs better for diet-specific recipes than for nonspecific recipes. This result may suggest that the feature `'diet'` may cause the model to perform significantly better for diet recipes. This bias can stem from a greater representation or greater predictive power of diet-specific recipes in the training dataset. 