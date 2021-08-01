import re
import numpy as np
from sklearn.metrics import classification_report

with open("./output/test_results2.txt") as file:
    inference_line = file.readline()
    index = inference_line.index("(0,")
    inferences = inference_line[:index]
    test_dataset = inference_line[index:]
    test_dataset += file.read()

y_infer = re.sub(r'[\[\]]', '', inferences).split(', ')
y_infer = [round(float(y)*4) for y in y_infer]
y_infer = np.asarray(y_infer).astype(int)
y_true = re.findall(r'\d\.\d*', test_dataset)
y_true = np.asarray(y_true).astype(float)[:len(y_infer)]
y_true = (y_true*4).astype(int)

class_report = classification_report(y_true, y_infer)
print(class_report)