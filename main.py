import preprocessor
import classifiers

preprocess = False
train = True
predict = False


def main(name):
    # print("HELLO")
    if preprocess:
        print("Preprocessing input dataset, may take a while...")
        preprocessor.vectorize_text("spam_ham_dataset.csv")
    if train:
        print("Invoking classifiers.")
        classifiers.classifiers_main("vectorized_dataset.csv")
    if predict:
        sample_text = "Subject: hpl nom for january 9 , 2001 ( see attached file : hplnol 09 . xls ) - hplnol 09 . xls"
        print("Mail for prediction: \n=== START OF MAIL ===")
        print(sample_text)
        print("=== END OF MAIL ===\n")

        classifiers.predict_load_model(sample_text, "gNB.sav")
        classifiers.predict_load_model(sample_text, "kNN.sav")
        classifiers.predict_load_model(sample_text, "RF.sav")


if __name__ == '__main__':
    main('PyCharm')

