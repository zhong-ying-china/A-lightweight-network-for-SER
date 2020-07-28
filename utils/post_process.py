import numpy as np


def get_confusion_matrix(gt, pr, classes):
    LEN = len(classes)
    matrix = np.zeros((LEN, LEN))
    for i, j in zip(gt, pr):
        matrix[i, j] += 1
    return matrix, classes


def print_csv_confustion_matrix(gt, pr, classes, file=None):
    total_acc = np.sum(gt == pr) / len(gt)
    matrix, classes = get_confusion_matrix(gt, pr, classes)
    print(file=file)
    print('  a\\p', end='\t', file=file)
    for c in classes:
        print(c, end='\t', file=file)
    print(file=file)
    for i in range(len(classes)):
        print(' ', classes[i], end='\t', file=file)
        for ele in matrix[i]:
            print(ele, end='\t', file=file)
        print(file=file)
    print(file=file)

    sum_1 = np.sum(matrix, axis=1)
    matrix2 = matrix / sum_1.reshape((-1, 1))
    print('  a\\p', end='\t', file=file)
    for c in classes:
        print(' ', c, end='\t', file=file)
    print(file=file)
    for i in range(len(classes)):
        print(' ', classes[i], end='\t', file=file)
        for ele in matrix2[i]:
            print('%.4f' % ele, end='\t', file=file)
        print(file=file)
    print(file=file)

    avg = 0
    for i in range(len(classes)):
        avg += matrix2[i, i]
    print('  average(unweighted) accuracy is %.4f' % (avg / len(classes)), file=file)
    print('  total(weighted) accuracy is %.4f' % float(total_acc), file=file)
    print(file=file)
    return matrix, classes

