#imports necessary libraries
import pandas as pd
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

#loads dataset
dataset = pd.read_csv('car.data', header=None)
df = dataset.copy()

#assigns column names
df.columns = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]


# Testing to see whether the data is being read
print("\nRandom Data Check:")
print(df.head())
print("\nSummary Statistics:")
print(df.describe())
print("\nMissing Values:")
print(df.isnull().sum())
print(f"\nDataset Shape: {dataset.shape}\n")



def train_test_split(dataset):
    training_data = dataset.iloc[:50].reset_index(drop=True)  #uses first 50% of dataset for training
    testing_data = dataset.iloc[50:].reset_index(drop=True )  #uses last 50% of dataset for testing
    return training_data, testing_data

training_data, testing_data = train_test_split(df)  

#labelencoder changes categorical values into numerical
le = LabelEncoder(); df['buying'] = le.fit_transform(df['buying'])
le = LabelEncoder(); df['maint'] = le.fit_transform(df['maint'])
le = LabelEncoder(); df['safety'] = le.fit_transform(df['safety'])
le = LabelEncoder(); df['lug_boot'] = le.fit_transform(df['lug_boot'])
le = LabelEncoder(); df['doors'] = le.fit_transform(df['doors'])
le = LabelEncoder(); df['persons'] = le.fit_transform(df['persons'])
le = LabelEncoder(); df['class'] = le.fit_transform(df['class'])

print(df.head())


#calculates entropy
def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    entropy = np.sum([(-counts[i] / np.sum(counts)) * np.log2(counts[i] / np.sum(counts)) for i in range(len(elements))])
    return entropy
entropy_value = entropy(df['class'])
print(f"Entropy of the Classes: {entropy_value:.4f}")
#calculates information gain
def InfoGain(data, split_attribute_name, target_name='target_col'):
    
    total_entropy = entropy(data[target_name])
    
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)
    weighted_entropy = np.sum([(counts[i] / np.sum(counts)) * entropy(data.where(data[split_attribute_name] == vals[i]).dropna()[target_name]) for i in range(len(vals))])
    
    information_gain = total_entropy - weighted_entropy
    return information_gain

info_gains = {category: InfoGain(df, category, target_name='class') for category in df.columns[:-1]}
print("Info gain of all categories")
for category, gain in info_gains.items():
    print(f"{category}: {gain:.4f}")

    #main part of program (makes the decision tree)
def ID3(data, originaldata, features, target_attribute_name="class", parent_node_class=None):
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]
    elif len(data) == 0:
        return np.unique(originaldata[target_attribute_name])[np.argmax(np.unique(originaldata[target_attribute_name], return_counts=True)[1])]
    elif len(features) == 0:
        return parent_node_class
    else:
        parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]
        item_values = [InfoGain(data, feature, target_attribute_name) for feature in features]
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]
        
        tree = {best_feature: {}}
        features = [i for i in features if i != best_feature]
 
        for value in np.unique(data[best_feature]):
            sub_data = data.where(data[best_feature] == value).dropna()
            subtree = ID3(sub_data, originaldata, features, target_attribute_name, parent_node_class)
            tree[best_feature][value] = subtree
            
        return tree
    #predicts tree and checks if its formatted right
def predict(query, tree, default=None):
    if not isinstance(query, dict):
        print("Query is not formatted as a dictionary:", query)
        return default
    current_node = tree

    while isinstance(current_node, dict):
        feature = next(iter(current_node))
        value = query.get(feature)
        if value in current_node[feature]:
            current_node = current_node[feature][value]
        else:
            print(f"Value '{value}' for feature '{feature}' not found in the tree, returning default.")
            return default 

    return current_node
#measures accuracy 
def test(data, tree):
    queries = data.iloc[:, :-1].to_dict(orient="records")
    predicted = pd.DataFrame(columns=["predicted"]) 
    for i in range(len(data)):
        predicted.loc[i, "predicted"] = predict(queries[i], tree, None) 
    accuracy = (np.sum(predicted["predicted"] == data["class"]) / len(data)) * 100
    print('The prediction accuracy is: ', accuracy, '%')

# print prediction and tree
tree = ID3(training_data, training_data, training_data.columns[:-1])
pprint(tree)
test(testing_data, tree)


#labels variables for class
class_labels = {0: 'unacc', 1: 'acc', 2: 'good', 3: 'v-good'}
df['class'] = df['class'].map(class_labels)

#defines attribute values
attribute_values = {
    'buying': ['v-high', 'high', 'med', 'low'],
    'maint': ['v-high', 'high', 'med', 'low'],
    'doors': ['2', '3', '4', '5-more'],
    'persons': ['2', '4', 'more'],
    'lug_boot': ['small', 'med', 'big'],
    'safety': ['low', 'med', 'high']
}

#sets style for plots
sns.set(style="whitegrid")

#plots all the categories comparing them with class
def plot_categories(df, target):
    categorical_vars = df.select_dtypes(include=['int', 'object']).columns.tolist()
    

    if target in categorical_vars:
        categorical_vars.remove(target)


    for var in categorical_vars:
        plt.figure(figsize=(10, 8))
        sns.countplot(data=df, x=var, hue=target)


        if var in attribute_values:
            plt.xticks(ticks=range(len(attribute_values[var])), labels=attribute_values[var], rotation=45)

        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles, list(class_labels.values()), title=target)
        
        plt.title(f'Count plot of {var} by {target}')
        plt.xlabel(var)
        plt.ylabel('Count')
        plt.show()

#starts function
plot_categories(df, target='class')

#plots distrubution of class
def plot_class(df, target):
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x=target)
    plt.title('Distribution of Car Acceptability Classes')
    

    plt.xlabel('Class')
    plt.ylabel('Count')
    

    labels = list(class_labels.values()) 
    plt.xticks(ticks=range(len(labels)), labels=labels)
    
    plt.show()

#starts function
plot_class(df, target='class')

