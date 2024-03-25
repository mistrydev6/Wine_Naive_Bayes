import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.metrics import auc
from sklearn.preprocessing import label_binarize
import sys

args = sys.argv[1:]
if len(args) != 1 or (int(args[0])< 20 or int(args[0]) > 80):
    TRAIN_SIZE = 80/100
    a=80
     
else:
    TRAIN_SIZE = (int(args[0])/100)
    a = int(args[0])


print("Mistry, DevPrashant, A20561645")
print("Training Size: ", a)



data = pd.read_csv('data.csv')
stop = stopwords.words('english')
x = data.description
y = data.points



corpus = []
IGNORE = False
for i in range(len(x)):
    if IGNORE == False:
        review = x[i].split(" ")
        review = [word for word in review if word not in stop]
        review = ' '.join(review)
        corpus.append(review)
    else:
        review = [i].split(" ")
        review = [word for word in review]
        review = ' '.join(review)
        corpus.append(review)    



df = pd.DataFrame({"Description": corpus, "Points":y})


bins = [80, 84, 90, 94, 100]
labels = [1, 2, 3, 4]


df['Points_Class'] = pd.cut(df['Points'], bins=bins, labels=labels, include_lowest=True)




def create_binary_bag_of_words(documents):
    vocabulary = set()
    binary_bow = []

    # build vocabulary
    for doc in documents:
        words = doc.split()
        for word in words:
            vocabulary.add(word)

    # create binary bag of words
    for doc in documents:
        binary_vector = []
        words = doc.split()
        for word in vocabulary:
            if word in words:
                binary_vector.append(1)
            else:
                binary_vector.append(0)
        binary_bow.append(binary_vector)

    return binary_bow, list(vocabulary)



df2 = pd.DataFrame({"Text": corpus, "Points":df["Points_Class"]})

print("Training Classifier")

X_train, X_test, y_train, y_test = train_test_split(df2["Text"], df2["Points"], test_size=1-TRAIN_SIZE, random_state=651694207)

binary_bag_train, vocabulary_train = create_binary_bag_of_words(X_train)
X_train = binary_bag_train


one = np.count_nonzero(y_train == 1) / len(y_train)
two = np.count_nonzero(y_train == 2) / len(y_train)
three = np.count_nonzero(y_train == 3) / len(y_train)
four = np.count_nonzero(y_train == 4) / len(y_train)



total_one_words = 0
total_two_words = 0
total_three_words = 0
total_four_words = 0


for i in range(len(X_train)):
    if y_train.iloc[i] == 1:
        for number in X_train[i]:
            total_one_words += number
    elif y_train.iloc[i] == 2:
        for number in X_train[i]:
            total_two_words += number
    elif y_train.iloc[i] == 3:
        for number in X_train[i]:
            total_three_words += number
    elif y_train.iloc[i] == 4:
        for number in X_train[i]:
            total_four_words += number


V_size = len(vocabulary_train)

one_prob = {}
two_prob = {}
three_prob = {}
four_prob = {}

for i in range(len(X_train)):
    if y_train.iloc[i] == 1:
        # print("one")
        for j in range(len(X_train[i])):
            if X_train[i][j] == 1:
                if vocabulary_train[j] not in one_prob:
                    one_prob[vocabulary_train[j]] = 2/(total_one_words+(1*V_size))
                else:
                    one_prob[vocabulary_train[j]] = one_prob[vocabulary_train[j]] + 1/(total_one_words+(1*V_size))
                
                if vocabulary_train[j] not in two_prob:
                    two_prob[vocabulary_train[j]] = 1/(total_two_words+(1*V_size))
                if vocabulary_train[j] not in three_prob:
                    three_prob[vocabulary_train[j]] = 1/(total_three_words+(1*V_size))
                if vocabulary_train[j] not in four_prob:
                    four_prob[vocabulary_train[j]] = 1/(total_four_words+(1*V_size))
                

    elif y_train.iloc[i] == 2:
        for j in range(len(X_train[i])):
            # print("two")
            if X_train[i][j] == 1:
                if vocabulary_train[j] not in two_prob:
                    two_prob[vocabulary_train[j]] = 2/(total_two_words+(1*V_size))
                else:
                    two_prob[vocabulary_train[j]] = two_prob[vocabulary_train[j]] + 1/(total_two_words+(1*V_size))
                
                if vocabulary_train[j] not in one_prob:
                    one_prob[vocabulary_train[j]] = 1/(total_one_words+(1*V_size))
                if vocabulary_train[j] not in three_prob:
                    three_prob[vocabulary_train[j]] = 1/(total_three_words+(1*V_size))
                if vocabulary_train[j] not in four_prob:
                    four_prob[vocabulary_train[j]] = 1/(total_four_words+(1*V_size))

    elif y_train.iloc[i] == 3:
        for j in range(len(X_train[i])):
            # print("three")
            if X_train[i][j] == 1:
                if vocabulary_train[j] not in three_prob:
                    three_prob[vocabulary_train[j]] = 2/(total_three_words+(1*V_size))
                else:
                    three_prob[vocabulary_train[j]] = three_prob[vocabulary_train[j]] + 1/(total_three_words+(1*V_size))
                
                if vocabulary_train[j] not in one_prob:
                    one_prob[vocabulary_train[j]] = 1/(total_one_words+(1*V_size))
                if vocabulary_train[j] not in two_prob:
                    two_prob[vocabulary_train[j]] = 1/(total_two_words+(1*V_size))
                if vocabulary_train[j] not in four_prob:
                    four_prob[vocabulary_train[j]] = 1/(total_four_words+(1*V_size))

    else:
        for j in range(len(X_train[i])):
            # print("four")
            if X_train[i][j] == 1:
                if vocabulary_train[j] not in four_prob:
                    four_prob[vocabulary_train[j]] = 2/(total_four_words+(1*V_size))
                else:
                    four_prob[vocabulary_train[j]] = four_prob[vocabulary_train[j]] + 1/(total_four_words+(1*V_size))
                
                if vocabulary_train[j] not in one_prob:
                    one_prob[vocabulary_train[j]] = 1/(total_one_words+(1*V_size))
                if vocabulary_train[j] not in two_prob:
                    two_prob[vocabulary_train[j]] = 1/(total_two_words+(1*V_size))
                if vocabulary_train[j] not in three_prob:
                    three_prob[vocabulary_train[j]] = 1/(total_three_words+(1*V_size))


print("Testing Classifier")

test_df = pd.DataFrame({"Text": X_test, "Points":y_test}).reset_index()
predictions = []
for row in test_df['Text']:
    one_x = np.log(one)
    two_x = np.log(two)
    three_x = np.log(three)
    four_x = np.log(four)

    # pos = Prob_pos
    # neg = Prob_neg
    for word in row.split():
        if word in vocabulary_train:
            one_x += np.log(one_prob[word])
            two_x += np.log(two_prob[word])
            three_x += np.log(three_prob[word])
            four_x += np.log(four_prob[word])
            # pos *= pos_prob[word]
            # neg *= neg_prob[word]
    if max(one_x, two_x, three_x, four_x) == one_x:
        predictions.append(1)
    elif max(one_x, two_x, three_x, four_x) == two_x:
        predictions.append(2)
    elif max(one_x, two_x, three_x, four_x) == three_x:
        predictions.append(3)
    elif max(one_x, two_x, three_x, four_x) == four_x:
        predictions.append(4)



def create_metrics(actual, predicted):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(actual)):
        if actual[i] == 3:
            if actual[i] == predicted[i]:
                TP += 1
            else:
                FN += 1
        else:
            if actual[i] == predicted[i]:
                TN += 1
            else:
                FP += 1
    recall = TP/(TP+FN)
    specificity = TN/(TN+FP)
    precision = TP/(TP+FP)
    negative_predictive_value = TN/(TN+FN)
    accuracy = (TP+TN)/(TN+TP+FP+FN)
    F_score= 2*((recall*precision)/(recall+precision))

    return TP, TN, FP, FN, recall, specificity, precision, negative_predictive_value, accuracy, F_score



def predict(sentence):
    onex = np.log(one)
    twox = np.log(two)
    threex = np.log(three)
    fourx = np.log(four)

    for word in sentence.split():
        if word in vocabulary_train:
            onex += np.log(one_prob[word])
            twox += np.log(two_prob[word])
            threex += np.log(three_prob[word])
            fourx += np.log(four_prob[word])
    return onex, twox ,threex, fourx


TP, TN, FP, FN, recall, specificity, precision, negative_predictive_value, accuracy, F_score = create_metrics(test_df['Points'], predictions)


print("Test Results/Metrics")
print(f'Number of true positives: {TP}')
print(f'Number of true negatives: {TN}')
print(f'Number of false positives: {FP}')
print(f'Number of false negatives: {FN}')
print(f'Sensitivity (recall): {recall}')
print(f'Specificity: {specificity}')
print(f'Precision: {precision}')
print(f'Negative predictive value: {negative_predictive_value}')
print(f'Accuracy: {accuracy}')
print(f'F-score: {F_score}')


running=True

while running:
    S = input('Enter your sentence: ')
    print(f'Sentence S: \n{S}')
    onex, twox ,threex, fourx = predict(S)
    if onex > twox and onex > threex and onex > fourx:
        classification = 'Class 1 (Least)'
    elif twox > onex and twox > threex and twox > fourx:
        classification = 'Class 2 (Lower Mid)'
    elif threex > onex and threex > twox and threex > fourx:
        classification = 'Class 3 (Upper Mid)'
    elif fourx > onex and fourx > twox and fourx > threex:
        classification = 'Class 4 (Most)'

    print(f'Was classified as {classification}.')
    print(f'P(Class 1 | S) = {np.e**onex}')
    print(f'P(Class 2 | S) = {np.e**twox}')
    print(f'P(Class 3 | S) = {np.e**threex}')
    print(f'P(Class 4 | S) = {np.e**fourx}')

    run = input('Do you want to continue? (Y/N): ')
    if run != 'Y' and run != 'y':
        running = False
        break






y_test_binarized=label_binarize(y_test,classes=np.unique(y_test))

fpr = {}
tpr = {}
thresh ={}
roc_auc = dict()

n_class = y_test_binarized.shape[1]

for i in range(n_class):
    fpr[i], tpr[i], thresh[i] = roc_curve(y_test_binarized[:, i], predictions)
    roc_auc[i] = auc(fpr[i], tpr[i])
 
    plt.plot(fpr[i], tpr[i])


plt.plot([0,1],[0,1],'b--')
plt.xlim([0,1])
plt.ylim([0,1.05])
plt.title('Multiclass ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(labels)
plt.show()


labels = [4,3,2,1]

cm = confusion_matrix(test_df['Points'], predictions)

cmap = LinearSegmentedColormap.from_list('custom_cmap', ["#2D398B", "#C14169"])

sns.set(font_scale=1.4)
sns.heatmap(cm, annot=True, annot_kws={"size": 16}, cmap=cmap, fmt='d', xticklabels=labels, yticklabels=labels)


fig = plt.gcf()
fig.patch.set_facecolor('none')
fig.patch.set_alpha(0.0)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion matrix')
plt.show()
