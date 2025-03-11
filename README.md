# Exploring the Effect of Diet Health on Recipe Ratings

Exploring the Effect of Diet Health on Recipe Ratings is a data science project conducted at UCSD. This project investigates the relationship of ratings of online published recipes and their health benefits through methods such as exploratory data analysis, hypothesis testing, and predictive modeling. The following provides a report of this investigation. 

Author(s): Kai Arizabal

## Introduction

Having a balanced diet is essential for maintaining a healthy life. Proper diets provide necessary nutrition, ultimately boosting immunity against chronic diseases and improving physical and emotional health. Although the best diets vary between individuals, all healthy diets follow basic principles. Sustaining a healthy diet requires balancing nutrients, reducing sodium and sugar intake, and avoiding processed foods. Using datasets containing recipes and ratings from [food.com](https://www.food.com/) (originally scraped by Prasad Majumder et al. for their research into generating personalized recipes), this project investigates public preferences of health and diet. 

The research question guiding this project is: **Do healthier diets tend to have higher ratings?** 

Exploring this question provides insight into popular diets, particularly if diet-specific foods are rated based on their health benefits, taste, or convenience, therefore elucidating the most accessible and effective diets for a healthy life.

The first dataset `recipes` has 83782 rows and 12 columns, containing the following information:

| COLUMN | DESCRIPTION |
| ----------- | ----------- |
| `'name'` | name of the recipe |
| `'id'` | unique ID given to the recipe |
| `'minutes'` | number of minutes to prepare the recipe |
| `'contributer_id'` | the ID of individual who submitted the recipe |
| `'submitted'` | date on which recipe was submitted |
| `'tags'` | food.com tags to categorize the recipe |
| `'nutrition'` | the following nutritional information as percent daily values (PDV): `'['calories (#), total fat (PDV), sugar (PDV), sodium (PDV), protein (PDV), saturated fat (PDV), carbohydrates (PDV)]'` |
| `'n_steps'` | number of steps |
| `'steps'` | steps to preparing the recipe, in order |
| `'description'` | a user-provided description |
| `'ingredients'` | all of the ingredients used |
| `'n_ingredients'` | number of ingredients |

The second dataset `interactions` has 731927 rows of user ratings and 5 columns:

| COLUMN | DESCRIPTION |
| ----------- | ----------- |
| `'user_id'` | user ID of an individual |
| `'recipe_id'` | recipe ID that this user interacted with |
| `'date'` | date of their interaction |
| `'rating'` | rating given |
| `'review'` | review given by user |

Using the combined data provided by both datasets will allow for this exploration into the relationship of ratings and diets.

## Data Cleaning and Exploratory Data Analysis

### Data Cleaning

Before any exploration of the data, cleaning the data is necessary for efficient and easy analysis.

1. For convenience, I left merge the datasets on their keys, `'id'` (of `recipes`) and `'recipe_id'` (of `interactions`) . This merge ensures that all data is contained in one DataFrame.

2. I perform data quality checks, including checking the datatypes each column stores.

3. I fill all `'rating'` of `0.0` with `np.nan`. The scale of `'rating'` is `1, 2, 3, 4, 5`. Any `0` rating indicates that no star rating was given, which may disproportionately skew the data. Therefore, this replacement corrects this skew.

4. Afterwards, I compute the average rating for each recipe. The resulting series is concatenated to the larger DataFrame as `'avg_rating'` column.

5. The columns `'submitted'` and `'date'` are converted to DateTime objects.

6. Cleaning the `'nutrition'` column, which was stored as a string object. I split the information into the following individual columns (as floats): `'calories (#)', 'total fat (PDV)', 'sugar (PDV)', 'sodium (PDV)', 'protein (PDV)', 'saturated fat (PDV)', 'carbohydrates (PDV)'`

    1. I also divide the nutrients recorded in PDV by 100 to get them as decimals. This makes their data more representative of percentages offering more consistency and interpretability.

7. Using the `'tags'` column, I add the column `'diet'` that specifies if and what diet the recipe is most appropriate for. If the recipe is not apparently diet-specific, then the value is `np.nan`. This process involved:

    1. Cleaning the `'tags'` column, which was originally stored as a string object. After cleaning, the `'tags'` column contain lists of strings.

    2. Choosing and categorizing diets based on a specific tag. The diets chosen for this category are based on popularity or longevity or sustainability. Their corresponding tags are chosen from only the unique strings of the `'tags'` column based on best (general) descriptor(s). The following diets and their specific tags are:

        | DIET | TAG(S) |
        | ----------- | ----------- |
        | `'vegan'` | `'vegan'` |
        | `'vegetarian'` | `'vegetarian'` |
        | `'mediterranean'` | `'high-fiber'` or `'low-saturated-fat'` |
        | `'keto'` | `'low-carb'` |
        | `'high-protein'` | `'high-protein'` |
        | `'low-sodium'` | `'general health'` |
        
        If any recipe contains multiple string tags for different diets, the most restrictive or specific diet is chosen.

8. Using the values of the column `'diet'`, I add an additional column `'is_diet_specific'` based on whether the value stored in `'diet'` is `np.nan`. This column is a boolean that indicates if the recipe is diet-specific (pertaining to a diet) or not.

After this cleaning process, I choose only the relevant columns for hypothesis testing and predictive modeling. Below is the head of the cleaned DataFrame, consisting only of said relevant columns:

| name                                 |     id |   minutes |   rating |   avg_rating |   calories (#) |   total fat (PDV) |   sugar (PDV) |   sodium (PDV) |   protein (PDV) |   saturated fat (PDV) |   carbohydrates (PDV) |   diet | is_diet_specific   |
|:-------------------------------------|-------:|----------:|---------:|-------------:|---------------:|------------------:|--------------:|---------------:|----------------:|----------------------:|----------------------:|-------:|:-------------------|
| 1 brownies in the world    best ever | 333281 |        40 |        4 |            4 |          138.4 |              0.1  |          0.5  |           0.03 |            0.03 |                  0.19 |                  0.06 |    nan | False              |
| 1 in canada chocolate chip cookies   | 453467 |        45 |        5 |            5 |          595.1 |              0.46 |          2.11 |           0.22 |            0.13 |                  0.51 |                  0.26 |    nan | False              |
| 412 broccoli casserole               | 306168 |        40 |        5 |            5 |          194.8 |              0.2  |          0.06 |           0.32 |            0.22 |                  0.36 |                  0.03 |    nan | False              |
| 412 broccoli casserole               | 306168 |        40 |        5 |            5 |          194.8 |              0.2  |          0.06 |           0.32 |            0.22 |                  0.36 |                  0.03 |    nan | False              |
| 412 broccoli casserole               | 306168 |        40 |        5 |            5 |          194.8 |              0.2  |          0.06 |           0.32 |            0.22 |                  0.36 |                  0.03 |    nan | False              |

**NOTE**: Should any columns require further cleaning for the hypothesis testing or predictive modeling specifically, the adjustments will be included at those steps. 

### Univariate Analysis

### Bivariate Analysis

### Interesting Aggregates

## Assessment of Missingness

In the dataset, there are four columns containing a significant number of missing values: `'description'`, `'rating'`, `'avg_rating'`, and `'diet'`.

### NMAR Analysis

Of these columns, the most likely to be not missing at random (NMAR) is `'description'`. NMAR occurs when the missingness of the value depends on the data itself. The description may be missing because the contributer may have not provided any additional information if the recipe is straight-forward or self-explanatory. Therefore, if the contributer decided to not include a description, there would be missing values in the dataset. Since the missingness depends on if a description is provided or not, `'description'` may be NMAR.

### Missingness Dependency

Among the remaining three columns, I have the least information on the `'rating'` column. Therefore, I will test the missingness dependency of `'rating'` on columns `'calories (#)'` and `'minutes'`.

The test statistic is the absolute difference in means, and the significance level is 0.05. For both tests, I ran 1000 permutation tests.

I will first test the dependency of `'rating'` on `'calories (#)'`.

> **Null Hypothesis:** The distributions of `'calories (#)'` are the same whether `'rating'` is missing or not. 

> **Alternative Hypothesis:** The distributions of `'calories (#)'` are **not** the same if `'rating'` is missing or not. 

Below are analyses of the observed distribution of `'calories (#)'` when `'rating'` is missing and not missing. 

<iframe
  src='assets/calories-dependency-kde.html'
  width='800'
  height='500'
  frameborder='0'
></iframe>

The observed means of the above distribution:

| rating_missing   |   calories (#) |
|:-----------------|---------------:|
| False            |        484.11  |
| True             |        415.103 |

The observed statistic is 69.01.

**The p-value is 0.00**, so I **reject** the null under a 0.05 significance level. Therefore, the missingness of `'rating'` may be dependent on `'calories (#)'`. Below is the empirical distribution of the test statistics (and observed statistic):

<iframe
  src='assets/calories-dependency-test.html'
  width='800'
  height='500'
  frameborder='0'
></iframe>

Now, I will test the dependency of `'rating'` on `'minutes'`.

> **Null Hypothesis:** The distributions of `'minutes'` are the same whether `'rating'` is missing or not. 

> **Alternative Hypothesis:** The distributions of `'minutes'` are **not** the same if `'rating'` is missing or not. 

Below are analyses of the observed distribution of `'minutes'` when `'rating'` is missing and not missing. 

<iframe
  src='assets/minutes-dependency-kde.html'
  width='800'
  height='500'
  frameborder='0'
></iframe>

The observed means of the above distribution:

| rating_missing   |   minutes |
|:-----------------|----------:|
| False            |   154.942 |
| True             |   103.49  |

The observed statistic is 51.45.

**The p-value is 0.125**, therefore I **fail to reject** the null under a 0.05 significance level. Accordingly, the missingness of `'rating'` may not be dependent on `'minutes'`. Below is the empirical distribution of the test statistics (and observed statistic):

<iframe
  src='assets/minutes-dependency-test.html'
  width='800'
  height='500'
  frameborder='0'
></iframe>

## Hypothesis Testing

For the research question **do healthier diets tend to have higher ratings?**, I test whether there is a signficant difference in the ratings between diet-specific recipes or non-diet recipes. This hypothesis test will therefore provide basic insight into the public preferences of food diets, specifically into those of healthy diets. The alternative hypothesis claims that diet-specific recipes may be rated **higher** than non-diet recipes, under this perception of **increased health benefits** because of careful nutrient intake.

> **Null Hypothesis:** There is no difference between the average ratings for diet-specific and non-specific recipes.

> **Alternative Hypothesis:** Diet-specific recipes have higher average ratings compared to nonspecific recipes. 

> **Test Statistic:** The difference in mean average ratings between the diet-specific and nonspecific recipes. 

> **Significance Level:** 0.05

The observed statistic is **0.013** (rounded to three decimals). 

I test under the null using a permutation test of 5000 simulations. **The resulting p-value is 0.00**, and so I **reject** the null under a 0.05 significance level. This result indicates that the average ratings of diet-specific recipes tend to be higher than nonspecific recipes. Ultimately, this may be attributed to the health benefits of maintaining a health-related diet, whether it be for boosted immunity, improved health, weight loss, muscle gain, etc. 

The plot below is the histogram containing the distribution of mean differences computed for the test, including the observed difference:

<iframe
  src='assets/hypothesis-test.html'
  width='800'
  height='500'
  frameborder='0'
></iframe>

## Framing a Prediction Problem