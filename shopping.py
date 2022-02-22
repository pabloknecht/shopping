import csv
import sys
import datetime

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4

def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    filename = "shopping.csv"


    evidence = list()
    labels = list()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):

            # Evidence
            Administrative = int(row["Administrative"])
            Administrative_Duration = float(row["Administrative_Duration"])
            Informational = int(row["Informational"])
            Informational_Duration = float(row["Informational_Duration"])
            ProductRelated = int(row["ProductRelated"])
            ProductRelated_Duration = float(row["ProductRelated_Duration"])
            BounceRates = float(row["BounceRates"])
            ExitRates = float(row["ExitRates"])
            PageValues = float(row["PageValues"])
            SpecialDay = float(row["SpecialDay"])

            if row["Month"] == "June":
                tmp_month = "Jun"
            else:
                tmp_month = row["Month"]

            datetime_object = datetime.datetime.strptime(tmp_month, "%b")
            Month = datetime_object.month -1

            OperatingSystems = int(row["OperatingSystems"])
            Browser = int(row["Browser"])
            Region = int(row["Region"])
            TrafficType = int(row["TrafficType"])

            if row["VisitorType"] == "Returning_Visitor":
                VisitorType = 1
            else:
                VisitorType = 0
                
            if row["Weekend"] == "TRUE":
                Weekend = 1
            else:
                Weekend = 0

            evidence.append([Administrative, Administrative_Duration, Informational, Informational_Duration, ProductRelated, ProductRelated_Duration, BounceRates, ExitRates, PageValues, SpecialDay, Month,OperatingSystems, Browser, Region, TrafficType, VisitorType, Weekend])
            
            # Labels

            Revenue = row["Revenue"] == "TRUE"
            labels.append([Revenue])
            
    return tuple((evidence, labels))

  

def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    x = 1
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(evidence, labels)

    return neigh


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """

    # Sensitivity
    true_positives = 0
    true_positives_identified = 0
    for i, label in enumerate(labels):
        if label == 1:
            true_positives = true_positives + 1 # sum of all true positives
            if predictions[i] == 1:
                true_positives_identified = true_positives_identified + 1 # true positives accurately identified

    sensitivity = true_positives_identified / true_positives

    # Specificity
    true_negatives = 0
    true_negatives_identified = 0
    for i, label in enumerate(labels):
        if label == 0:
            true_negatives = true_negatives + 1 # sum of all true negatives
            if predictions[i] == 0:
                true_negatives_identified = true_negatives_identified + 1 # true negatives accurately identified

    specificity = true_negatives_identified / true_negatives


    return sensitivity, specificity


if __name__ == "__main__":
    main()
