# confusion_matrix_cv
Apply Confusion Matrix to a cross-validated model.


## Installation guide
```
git clone https://github.com/UrbanoFonseca/confusion_matrix_cv
cd confusion_matrix_cv
pip install .
```


## Implementation
```
from confusion_matrix_cv.confusion_matrix_cv import ConfusionMatrixCV

logit = LogisticRegression()

cm_cv = ConfusionMatrixCV()

cm_cv.cross_validate(logit, X, Y, cv=StratifiedKFold(n_splits=10))

# Show the metrics
cm_cv.accuracy
cm_cv.sensitivity
cm_cv.precision
cm_cv.ROC
```
