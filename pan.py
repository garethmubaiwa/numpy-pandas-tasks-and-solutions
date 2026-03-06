import pandas as pd
import numpy as np

'''
The dataset is compiled from the 1994 Census database and contains income data. 
Data Details: age: continuous. workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked. fnlwgt: continuous. 
education: Bachelor's, Some-college, 11th, High School-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, *Master's, 1st-4th, 10th, Doctorate, 5th-6th, Preschool. education-num: continuous.
marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse. 
occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces. 
relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried. 
race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black. 
sex: Female, Male. capital-gain: continuous. capital-loss: continuous. hours-per-week: continuous. 
native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holland-Netherlands. 
salary: >50K,<=50K
'''
df = pd.read_csv('adult.data.csv')

# 1. Calculate how many men and women (feature sex) are represented in this dataset.
gender_counts = df['sex'].value_counts()
print('Gender counts:')
print(gender_counts)

# 2. What is the average age of men (feature age) in the entire dataset?
average_age_men = df[df['sex'] == 'Male']['age'].mean()
print('Average age of men:', average_age_men)

# 3. What is the proportion of US citizens (feature native-country)?
proportion_us_citizens = (df['native-country']== 'United-States').mean()
print('Proportion of US citizens:', proportion_us_citizens)

# 4-5. Calculate the mean and standard deviation of the ages of those earning over 50,000 per year (feature salary) and those earning less than 50,000 per year.
mean_age_over_50k = df[df['salary'] == '>50K']['age'].mean()
std_age_over_50k = df[df['salary'] == '>50K']['age'].std()
mean_age_under_50k = df[df['salary'] == '<=50K']['age'].mean()
std_age_under_50k = df[df['salary'] == '<=50K']['age'].std()
print('Mean age of those earning >50K:', mean_age_over_50k)
print('Standard deviation of age of those earning >50K:', std_age_over_50k)
print('Mean age of those earning <=50K:', mean_age_under_50k)
print('Standard deviation of age of those earning <=50K:', std_age_under_50k)

# 6. Is it true that people earning over 50,000 have at least a college education? 
# (feature education – Bachelor's, Professional school, Assoc-acdm, Assoc-voc, Master's, or Doctorate)
education_levels = ['Bachelors', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', 'Masters', 'Doctorate']
over_50k_education = df[df['salary'] == '>50K']['education']
has_college_education = over_50k_education.isin(education_levels).all()
print('Do all people earning >50K have at least a college education?', has_college_education)

# 7. Print age statistics for each race (feature race) and each gender. 
# Use groupby and describe. Find the maximum age of Asian-Pacific-Islander men using this method.
age_stats = df.groupby(['race', 'sex'])['age'].describe()
print('Age statistics for each race and gender:')
print(age_stats)

max_age_asian_pacific_islander_men = age_stats.loc[('Asian-Pac-Islander', 'Male'), 'max']
print('Maximum age of Asian-Pacific-Islander men:', max_age_asian_pacific_islander_men)

# 8. Who has a higher proportion of high earners (>50K): married or single men (marital-status indicator)? 
# We consider those with a marital-status starting with "Married" (Married-civ-spouse, Married-spouse-absent, or Married-AF-spouse) to be married; the rest are considered single.
men = df[df['sex'] == 'Male']
married = ['Married-civ-spouse', 'Married-spouse-absent', 'Married-AF-spouse']
married_men = men[men['marital-status'].isin(married)]
single_men = men[~men['marital-status'].isin(married)]
proportion_married_high_earners = (married_men['salary'] == '>50K').mean()
proportion_single_high_earners = (single_men['salary'] == '>50K').mean()
print('Proportion of high earners among married men:', proportion_married_high_earners)
print('Proportion of high earners among single men:', proportion_single_high_earners)

# 9. What is the maximum number of hours a person works per week (hours-per-week indicator)? 
# How many people work that many hours, and what percentage of them are high earners?
max_hours_per_week = df['hours-per-week'].max()
people_working_max_hours = df[df['hours-per-week'] == max_hours_per_week]
percentage_high_earners_max_hours = (people_working_max_hours['salary'] == '>50K').mean()

print('Maximum hours per week:', max_hours_per_week)
print('Number of people working maximum hours:', len(people_working_max_hours))
print('Percentage of high earners among those working maximum hours:', percentage_high_earners_max_hours)

# 10. Calculate the average hours worked (hours-per-week) for low and high earners (salary) for each country (native-country).
average_hours_by_country_salary = df.groupby(['native-country', 'salary'])['hours-per-week'].mean()
print('Average hours worked by country and salary:')
print(average_hours_by_country_salary)

'''11. Group people by age groups: young, adult, and retired, 
where:
young corresponds to 16-35 years old
adult - 35-70 years old
retiree - 70-100 years old
Enter the name of the corresponding group for each person in the new AgeGroup column.'''
def age_group(age):
    if 16 <= age < 35:
        return 'young'
    elif 35 <= age < 70:
        return 'adult'
    elif 70 <= age <= 100:
        return 'retiree'

df['AgeGroup'] = df['age'].apply(age_group)
print('Data with AgeGroup column:')
print(df['AgeGroup'].head())

# 12-13. Determine the number of people earning >50K in each age group (AgeGroup column), 
# and also display the name of the age group in which people most often earn more than 50K (>50K).
high_earners_by_age_group = df[df['salary'] == '>50K']['AgeGroup'].value_counts()
print('Number of people earning >50K in each age group:')
print(high_earners_by_age_group)

# 14. Group people by employment type (occupation column) and determine the number of people in each group. 
# Then write a filter function, filter_func, 
# that will return only those groups in which the average age (age column) is 40 or less and in which all workers work more than 5 hours per week (hours-per-week column).
group = df.groupby('occupation')
group_counts = group.size()
print('Number of people in each employment type:')
print(group_counts)

def filter_func(group):
    return group['age'].mean() <= 40 and (group['hours-per-week'] > 5).all()

filtered_groups = group.filter(filter_func)
print('Filtered groups where average age is 40 or less and all workers work more than 5 hours per week:')
print(filtered_groups['occupation'].unique())

