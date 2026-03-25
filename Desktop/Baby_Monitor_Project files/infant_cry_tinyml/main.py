from src.predict import predict_file

file = "C:/Users/Divine/Desktop/Baby_Monitor_Project files/NonCry_test/noncry_4.wav"

result = predict_file(file)

print("Result:", result)