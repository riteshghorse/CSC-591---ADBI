# PROJECT: Data Wrangling with Pandas and Regex: Learning Objectives
In this Project, I performed the following core Data Wrangling steps using
Python’s Pandas and Regex modules:

1. Data Acquisition
2. Data Cleansing
3. Data Understanding: Basics
4. Data Manipulation

Each step consists of objectives to be covered in the Project.

### Data Acquisition Objectives
1. How to import data in different formats? (read_excel, read_csv)
2. How to import multiples files for storage and access? (store filenames in array)
3. How are they read into by Pandas? (DataFrame)
4. How to have a peek at the data after import? (head/tail)

#### Methods used:
1. read_excel
2. df.columns
3. df.loc
4. df.head

### Data Cleansing Objectives
1. Check attributes of each file
2. Identify data types
3. Apply coercion if applicable
4. Check for corrupt/incorrect data
  1. Check for NA/missing data
  2. Check for data consistency (e.g., GPA cannot be less than 0)
5. Remove/replace corrupt data
6. Identify duplicate data
7. Identifying and removing outliers

#### Methods used:
1. df.dtypes
2. df.isnull
3. df.fillna
4. df.drop
5. df.drop_duplicates
6. df.sort_values
7. df.appendBasic 

### Data Understanding Objectives
1. Summary Statistics
2. Dimensionality

#### Methods used:
1. df.describe
2. df.shape

### Data Manipulation Objectives
1. Merge/Concatenate DataFrame
2. Filter to subset the data
3. Mapping to create a new attribute
4. Incorporate the use of multiple functions
5. Discretize data

#### Methods used:
1. pd.merge
2. pd.cut
3. unique

### Regular Expressions
1. Use regular expressions to find/match specific content
2. String manipulation via. substring and replace methods
3. Combine with data transformation methods to find, filter and manipulate data

#### Methods used:
1. re.search
2. re.sub
