from emotion_recognition import EmotionRecognizer
from sklearn.svm import SVC

my_model = SVC()

rec = EmotionRecognizer(model=my_model, emotions=['sad', 'neutral', 'happy'], balance=True, verbose=0)

rec.train()

print("Test score:", rec.test_score())

print("Train score:", rec.train_score())
