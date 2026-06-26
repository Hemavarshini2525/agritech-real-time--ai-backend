import joblib

model = joblib.load("fertilizer_model.pkl")
target_encoder = joblib.load("target_encoder.pkl")
label_encoders = joblib.load("label_encoder_fertilizer.pkl")

print("Model classes (what the model can predict):")
print(model.classes_)

print("\nTarget encoder's known classes (what it can decode):")
print(target_encoder.classes_)

print("\nMissing from target encoder:")
missing = set(model.classes_) - set(target_encoder.classes_)
print(missing)