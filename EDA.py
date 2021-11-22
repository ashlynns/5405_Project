import pandas as pd 
import seaborn as sns
import sklearn 
from sklearn.model_selection import train_test_split

data = pd.read_csv('train_data.csv') # load original dataset 

# split dataset into 75/25 train test split 
train, test = sklearn.model_selection.train_test_split(data, test_size=.25, random_state=42, shuffle=True, stratify=data['Label'])
train.to_csv('split_train_data.csv', index = False) # save train data for future use 
test.to_csv('split_test_data.csv', index = False) # save test data for future use 

# descriptive info 
shape = train.shape # shape of the training set 
column_names = train.columns # names of the columns 
nunique = train.nunique(axis = 0) # number of unique entries in each column 
describe = train.describe().apply(lambda s: s.apply(lambda x: format(x, 'f')))

# varriable relationships 
corr = train.corr()
#sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))
#sns.pairplot(train)