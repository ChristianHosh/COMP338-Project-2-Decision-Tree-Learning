import os
import graphviz
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz

os.environ["PATH"] += os.pathsep + 'D:/Program File/Graphviz/bin/'


def main():
    # Load and prepare the data
    data = pd.read_csv('diabetesData.csv')

    # Separate features (features) and target variable (y)
    features = data.drop('Diabetic', axis=1)
    target = data['Diabetic']

    # Split the data into training and testing sets for the 2 models
    # Creating and training both models
    m1_y_pred, m1_y_test, model_1 = create_model(features, target, 0.3)
    m2_y_pred, m2_y_test, model_2 = create_model(features, target, 0.5)

    # Predicting new instance
    new_instance = [[6, 148, 72, 35, 0, 33.6, 0.627, 50]]
    prediction_m1 = model_1.predict(new_instance)
    prediction_m2 = model_2.predict(new_instance)

    # Printing the predications for both models
    print("Prediction of M1:", prediction_m1)
    print("Prediction of M2:", prediction_m2)

    # Generate the plot of the Decision Tree for both models
    graph_plot(features, model_1, 1)
    graph_plot(features, model_2, 2)

    # Printing statistics
    # Printing the main statistics of each of the attributes
    # (i.e., mean, median, standard deviation, min, and max values) in a proper table
    print("\n----------------")

    pd.set_option('display.max_columns', None)
    statistics = data.describe()
    print("Main Statistics of Attributes:")
    print(statistics)

    # Printing out the distribution of the target class
    # (i.e., the percentage of the positive to the negative class)
    print("\n----------------")

    class_distribution = data['Diabetic'].value_counts(normalize=True) * 100
    print("Class Distribution: ", class_distribution)

    # Printing the accuracy of the models
    print("\n----------------")
    accuracy_m1 = accuracy_score(m1_y_test, m1_y_pred)
    accuracy_m2 = accuracy_score(m2_y_test, m2_y_pred)
    print("M1 Accuracy: ", accuracy_m1)
    print("M2 Accuracy: ", accuracy_m2)


def create_model(features, target, size):
    # Splits the dataset into a training set and a testing set
    # x_* values have the features data
    # y_* values have the outcome data
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=size, random_state=42)

    # Creates the Decision Tree Model using sklearn python library
    # Trains the model bases on the training data sets
    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)

    # predicts the output for all test cases
    y_pred = model.predict(x_test)

    return y_pred, y_test, model


def graph_plot(features, model, num):
    # Uses sklearn export to graphviz to create tree plots for the data models
    dot_data = export_graphviz(model, out_file=None,
                               feature_names=features.columns,
                               class_names=['Negative', 'Positive'],
                               filled=True,
                               rounded=True,
                               special_characters=True)

    # Uses graphviz to generate the tree plot for the data model with feature names
    graph = graphviz.Source(dot_data)
    graph.render(f'decision_tree_plot_m{num}', format='png')


if __name__ == "__main__":
    main()
