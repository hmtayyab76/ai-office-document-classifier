import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from colorama import init, Fore, Style

# Initialize colorama
init(autoreset=True)

# Load documents and labels
def load_documents(data_path):
    print(Fore.CYAN + "\n🔍 Loading documents from:", data_path)
    texts = []
    labels = []

    for filename in os.listdir(data_path):
        file_path = os.path.join(data_path, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                texts.append(f.read())

            label = ''.join([c for c in filename if not c.isdigit()]).replace('.txt', '')
            labels.append(label)

    print(Fore.GREEN + f"✅ Loaded {len(texts)} documents.\n")
    return texts, labels

# Train model
def train_model(texts, labels):
    print(Fore.CYAN + "🧠 Training the model...")
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(Fore.YELLOW + "\n📊 Model Evaluation:")
    print(Fore.WHITE + classification_report(y_test, predictions))
    
    return model

# Save model
def save_model(model, filename='model.pkl'):
    joblib.dump(model, filename)
    print(Fore.GREEN + f"\n💾 Model has been saved as '{filename}' (this is a serialized file).")

# Predict category
def predict(model_path, new_text):
    print(Fore.CYAN + f"\n🔍 Loading model from '{model_path}'...")
    model = joblib.load(model_path)
    prediction = model.predict([new_text])
    return prediction[0]

# Entry point
if __name__ == "__main__":
    print(Fore.MAGENTA + "📁 AI Office Document Classifier – Python Project")
    print(Style.BRIGHT + "-" * 50)

    data_path = './data'

    texts, labels = load_documents(data_path)
    model = train_model(texts, labels)
    save_model(model)

    # Predict sample
    sample_text = """Dear Hiring Manager, I am writing to express my interest in the Software Engineer position."""
    predicted = predict("model.pkl", sample_text)
    print(Fore.BLUE + f"\n📄 Sample Text Prediction: {Fore.GREEN}{predicted}")
    print(Style.BRIGHT + "\n✅ All steps completed successfully!\n")
