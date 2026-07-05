import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the dataset directly from the internet
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])

print("First 5 rows of data:")
print(df.head())
print()

# Step 2: Check how many spam vs ham messages
print("Label counts:")
print(df['label'].value_counts())
print()

# Step 3: Convert text messages into numbers
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['message'])
y = df['label']

# Step 4: Split into training data and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 6: Test the model's accuracy
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
print()
print("Detailed performance report:")
print(classification_report(y_test, predictions))

# Step 7: Try your own custom messages
def predict_message(msg):
    vec = vectorizer.transform([msg])
    return model.predict(vec)[0]

test1 = "Congratulations! You won a free iPhone, click here now!"
test2 = "Hey, are we still meeting for lunch today?"

print(f"'{test1}' --> {predict_message(test1)}")
print(f"'{test2}' --> {predict_message(test2)}")

