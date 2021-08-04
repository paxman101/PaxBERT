import re
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

parser = argparse.ArgumentParser(description="Analyze predictions from prediction file")
parser.add_argument("prediction_file", type=str)
parser.add_argument("-r", "--is-regression", dest="is_regression", default=False)

args = parser.parse_args()
prediction_file = args.prediction_file
is_regression = args.is_regression

with open(prediction_file) as file:
    inference_line = file.readline()
    index = inference_line.index("(0,")
    inferences = inference_line[:index]
    test_dataset = inference_line[index:]
    test_dataset += file.read()

y_infer = re.sub(r'[\[\]]', '', inferences).split(', ')
if is_regression:
    y_infer = [round(float(y) * 4) for y in y_infer]
    y_infer = np.asarray(y_infer).astype(int)

    y_true = re.findall(r'\d\.\d*', test_dataset)
    y_true = np.asarray(y_true).astype(float)[:len(y_infer)]
    y_true = (y_true * 4).astype(int)
else:
    y_infer = [int(y) for y in y_infer]
    y_infer = np.asarray(y_infer)

    y_true = re.findall(r'(\d)\)\)', test_dataset)
    y_true = np.asarray(y_true).astype(int)[:len(y_infer)]

class_report = classification_report(y_true + 1, y_infer + 1, output_dict=True)
class_report_map = sns.heatmap(pd.DataFrame(class_report).iloc[:-1, :].T, annot=True)
plt.show()
class_report_map.get_figure().savefig('class_report.png', dpi=200, bbox_inches='tight')

confus_matrix = confusion_matrix(y_true, y_infer)
confusion_map = sns.heatmap(pd.DataFrame(confus_matrix, range(1, 6), range(1, 6)), annot=True, fmt="d", robust=True)
plt.ylabel('True Rating')
plt.xlabel('Predicted Rating')
plt.show()
confusion_map.get_figure().savefig('confusion_matrix.png', dpi=200, bbox_inches='tight')
