import numpy as np


def classificationRate(valid_labels, valid_prediction):
    correct = np.zeros(7)
    total = np.zeros(7)
    for i in range (valid_labels.shape[0]):
        if (valid_labels[i] == 1):
            if (valid_prediction[i] == valid_labels[i]):
                correct[0] = correct[0] + 1
            total[0] = total[0] + 1
        elif (valid_labels[i] == 2):
            if (valid_prediction[i] == valid_labels[i]):
                correct[1] = correct[1] + 1
            total[1] = total[1] + 1
        elif (valid_labels[i] == 3):
            if (valid_prediction[i] == valid_labels[i]):
                correct[2] = correct[2] + 1
            total[2] = total[2] + 1
        elif (valid_labels[i] == 4):
            if (valid_prediction[i] == valid_labels[i]):
                correct[3] = correct[3] + 1
            total[3] = total[3] + 1
        elif (valid_labels[i] == 5):
            if (valid_prediction[i] == valid_labels[i]):
                correct[4] = correct[4] + 1
            total[4] = total[4] + 1
        elif (valid_labels[i] == 6):
            if (valid_prediction[i] == valid_labels[i]):
                correct[5] = correct[5] + 1
            total[5] = total[5] + 1
        else:
            if (valid_prediction[i] == valid_labels[i]):
                correct[6] = correct[6] + 1
            total[6] = total[6] + 1
    classification = correct/total
    print classification
    return classification