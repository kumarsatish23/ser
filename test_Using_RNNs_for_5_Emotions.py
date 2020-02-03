from deep_emotion_recognition import DeepEmotionRecognizer

deeprec = DeepEmotionRecognizer(emotions=['angry', 'sad', 'neutral', 'ps', 'happy'], n_rnn_layers=2, n_dense_layers=2, rnn_units=128, dense_units=128)

deeprec.train()

print(deeprec.test_score())

prediction = deeprec.predict('data/validation/Actor_10/03-02-05-02-02-02-10_angry.wav')
print(f"Prediction: {prediction}")
print("Predicting probabilities")
print(deeprec.predict_proba("data/emodb/wav/16a01Wb.wav"))
print("Confusion Matrix")
print(deeprec.confusion_matrix(percentage=True, labeled=True))
