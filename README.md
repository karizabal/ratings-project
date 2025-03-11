# Exploring the Effect of Diet Health on Recipe Ratings

Exploring the Effect of Diet Health on Recipe Ratings is a data science project conducted at UCSD. This project investigates the relationship of ratings of online-published recipes and their perceived dietary health through methods such as exploratory data analysis, hypothesis testing, and predictive modeling. The following provides a comprehensive report of this investigation. 

Author(s): Kai Arizabal

## Introduction

Maintaining a balanced diet is essential to good health. Proper nutrition reduces malnutrition and provides greater immunity to chronic diseases. However, in today's era of over- production and consumption, sustaining a healthy diet requires more involvement, such as learning to balance nutrients, reduce sodium and sugar intake, and avoid processed foods. Although the best diet varies between individuals, there are basic principles to all healthy diets. Understanding these specificities is made easier through online resources. 

Using datasets containing recipes and ratings from [food.com](https://www.food.com/) (originally scraped by Prasad Majumder et al. for their research into generating personalized recipes), this project investigates public preferences for health and diet. The research question guiding this project is: **Do healthier diets tend to have higher ratings?** 

Exploring this question provides insight into public perception of food choices, especially if healthy diets are rated based on a bias of perceived health or taste.

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

1. For convenience, I left merged the datasets on their keys, `'id'` (of `recipes`) and `'recipe_id'` (of `interactions`) . This merge ensures that all data is contained in one DataFrame.

2. I performed data quality checks, including checking the datatypes each column stores.

3. I filled all `'rating'` of `0.0` with `np.nan`. The scale of `'rating'` is `1, 2, 3, 4, 5`. Any `0` rating indicates that no star rating was given, which may disproportionately skew the data. Therefore, this replacement corrects this skew.

4. Afterwards, I computed the average rating for each recipe. The resulting series was concatenated to the larger DataFrame as `'avg_rating'` column.

5. The columns `'submitted'` and `'date'` were converted to DateTime objects.

6. Cleaning the `'nutrition'` column, which was stored as a string object. I split the information into the following individual columns (as floats): `'calories (#)', 'total fat (PDV)', 'sugar (PDV)', 'sodium (PDV)', 'protein (PDV)', 'saturated fat (PDV)', 'carbohydrates (PDV)'`

7. Using the `'tags'` column, I added the column `'diet'` that specifies if and what diet the recipe is most appropriate for. If the recipe is not apparently diet-specific, then the value is `np.nan`. This process involved:

    1. Cleaning the `'tags'` column, which was originally stored as a string object. After cleaning, the `'tags'` column contained lists of strings.

    2. Choosing and categorizing diets based on a specific tag. The diets chosen for this category were based on popularity or longevity or sustainability. Their corresponding tags were chosen from only the unique strings of the `'tags'` column based on best (general) descriptor. The following diets and their specific tags are:

        | DIET | TAG(S) |
        | ----------- | ----------- |
        | `'vegan'` | `'vegan'` |
        | `'vegetarian'` | `'vegetarian'` |
        | `'mediterranean'` | `'high-fiber'` or `'low-saturated-fat'` |
        | `'keto'` | `'low-carb'` |
        | `'high-protein'` | `'high-protein'` |
        | `'low-sodium'` | `'general health'` |
        
        If any recipe contained multiple string tags for different diets, the most restrictive or specific diet was chosen.

8. Using the values of the column `'diet'`, I added an additional column `'is_diet_specific'` based on whether the value stored in `'diet'` is `np.nan`. This column is a boolean that indicates if the recipe is diet-specific (pertaining to a diet) or not.

After this cleaning process, I chose only the relevant columns for hypothesis testing and predictive modeling. Below is the head of the cleaned DataFrame, consisting only of said relevant columns:

| name                                 |     id |   minutes |   rating |   avg_rating |   calories (#) |   total fat (PDV) |   sugar (PDV) |   sodium (PDV) |   protein (PDV) |   saturated fat (PDV) |   carbohydrates (PDV) |   diet | is_diet_specific   |
|:-------------------------------------|-------:|----------:|---------:|-------------:|---------------:|------------------:|--------------:|---------------:|----------------:|----------------------:|----------------------:|-------:|:-------------------|
| 1 brownies in the world    best ever | 333281 |        40 |        4 |            4 |          138.4 |                10 |            50 |              3 |               3 |                    19 |                     6 |    nan | False              |
| 1 in canada chocolate chip cookies   | 453467 |        45 |        5 |            5 |          595.1 |                46 |           211 |             22 |              13 |                    51 |                    26 |    nan | False              |
| 412 broccoli casserole               | 306168 |        40 |        5 |            5 |          194.8 |                20 |             6 |             32 |              22 |                    36 |                     3 |    nan | False              |
| 412 broccoli casserole               | 306168 |        40 |        5 |            5 |          194.8 |                20 |             6 |             32 |              22 |                    36 |                     3 |    nan | False              |
| 412 broccoli casserole               | 306168 |        40 |        5 |            5 |          194.8 |                20 |             6 |             32 |              22 |                    36 |                     3 |    nan | False              |

**NOTE**: Should any columns require further cleaning for the hypothesis testing or predictive modeling specifically, the adjustments will be included at these steps. 

### Univariate Analysis

### Bivariate Analysis

### Interesting Aggregates

## Assessment of Missingness

### NMAR Analysis