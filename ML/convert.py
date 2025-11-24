import joblib
import coremltools as ct

model = joblib.load("Models/Saved/LogisticRegression.joblib")

mlmodel = ct.converters.sklearn.convert(model)

mlmodel.save("Models/Conveted/LRClassifier.mlmodel")