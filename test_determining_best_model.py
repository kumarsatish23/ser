from emotion_recognition import EmotionRecognizer
from sklearn.svm import SVC

my_model = SVC()

rec = EmotionRecognizer(model=my_model, emotions=['sad', 'neutral', 'happy'], balance=True, verbose=0)

rec.train()
rec.determine_best_model(train=True)

print(rec.model.__class__.__name__, "is the best")

print("Test score:", rec.test_score())
