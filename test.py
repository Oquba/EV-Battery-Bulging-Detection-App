from sklearn.tree import export_text
import joblib

model = joblib.load("SVM_RBF.joblib")
print(model)

treeText = export_text(model)
print(treeText)