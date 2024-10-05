from flask import Flask, request, jsonify, render_template, send_file
import os
import nltk
import math
from collections import defaultdict, Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import json

app = Flask(__name__)

LEMMATIZER = WordNetLemmatizer()
STOPWORDS = set(stopwords.words('english'))

# Load and clean documents
def load_documents(directory):
    documents = {}
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r') as file:
                documents[filename] = file.read()
    return documents

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    tokens = word_tokenize(text)
    tokens = [LEMMATIZER.lemmatize(token) for token in tokens if token not in STOPWORDS]
    return tokens

# Vocabulary and TF-IDF Computation
def create_vocabulary(cleaned_documents):
    vocabulary = set()
    for tokens in cleaned_documents.values():
        vocabulary.update(tokens)
    return vocabulary

def compute_term_frequency(cleaned_documents):
    term_frequency = defaultdict(Counter)
    for filename, tokens in cleaned_documents.items():
        term_frequency[filename] = Counter(tokens)
    return term_frequency

def compute_inverse_document_frequency(cleaned_documents):
    num_documents = len(cleaned_documents)
    df = Counter()
    for tokens in cleaned_documents.values():
        unique_tokens = set(tokens)
        for token in unique_tokens:
            df[token] += 1
    idf = {token: math.log(num_documents / df[token]) for token in df}
    return idf

def compute_tf_idf(term_frequency, idf):
    tf_idf = defaultdict(dict)
    for filename, tf in term_frequency.items():
        for term, freq in tf.items():
            tf_idf[filename][term] = freq * idf[term]
    return tf_idf

def cosine_similarity(vec_a, vec_b):
    intersection = set(vec_a) & set(vec_b)
    numerator = sum(vec_a[x] * vec_b[x] for x in intersection)
    sum_a = sum(vec_a[x] ** 2 for x in vec_a)
    sum_b = sum(vec_b[x] ** 2 for x in vec_b)
    denominator = math.sqrt(sum_a) * math.sqrt(sum_b)
    return 0.0 if not denominator else numerator / denominator

def compute_query_tf_idf(query, idf):
    tokens = clean_text(query)
    tf = Counter(tokens)
    query_tf_idf = {term: freq * idf.get(term, 0) for term, freq in tf.items()}
    return query_tf_idf

# Load and process the documents
documents = load_documents('documents')
cleaned_documents = {filename: clean_text(content) for filename, content in documents.items()}
vocabulary = create_vocabulary(cleaned_documents)
term_frequency = compute_term_frequency(cleaned_documents)
idf = compute_inverse_document_frequency(cleaned_documents)
tf_idf = compute_tf_idf(term_frequency, idf)

# Map each document to an image
image_paths = {
    'Aloo Achaar (Spicy Potato Salad).txt': '/static/photos/Aloo Achaar (Spicy Potato Salad).jpg',
    'Aloo Tama (Potato and Bamboo Shoots Curry).txt': '/static/photos/Aloo Tama (Potato and Bamboo Shoots Curry).jpg',
    'Chatamari (Nepali Pizza).txt': '/static/photos/Chatamari (Nepali Pizza).jpg',
    'Chatpate (Spicy Puffed Rice Snack).txt': '/static/photos/Chatpate (Spicy Puffed Rice Snack).jpg',
    'Chhurpi (Fermented Yak Cheese).txt': '/static/photos/Chhurpi (Fermented Yak Cheese).jpg',
    'Dal Bhat (Lentils Rice).txt': '/static/photos/Dal Bhat (Lentils Rice).jpg',
    'Dhido.txt': '/static/photos/Dhido.jpg',
    'Gundruk (Fermented Leafy Greens).txt': '/static/photos/Gundruk (Fermented Leafy Greens).jpg',
    'Juju Dhau (King Yogurt from Bhaktapur).txt': '/static/photos/Juju Dhau (King Yogurt from Bhaktapur).jpg',
    'Kheer (Rice Pudding).txt': '/static/photos/Kheer (Rice Pudding).jpg',
    'Momo (Nepali Dumplings).txt': '/static/photos/Momo (Nepali Dumplings).jpg',
    'Sandheko Bhatmas(Spicy Salad).txt': '/static/photos/Sandheko Bhatmas(Spicy Salad).jpg',
    'Sel Roti.txt': '/static/photos/Sel Roti.jpg',
    'Thukpa (Nepali Noodle Soup).txt': '/static/photos/Thukpa (Nepali Noodle Soup).jpg',
    'Yomari (Sweet Dumpling).txt': '/static/photos/Yomari (Sweet Dumpling).jpg',
}

# Main page route
@app.route("/", methods=["GET"])
def welcome():
    return render_template("index.html")

# Results route for computing cosine similarity and returning ranked results with photos
@app.route("/results", methods=["POST"])
def results():
    data = request.get_json()
    ingredients = data.get('ingredients', '')

    # Compute TF-IDF for the query (ingredients entered by the user)
    query_tf_idf = compute_query_tf_idf(ingredients, idf)

    # Compute cosine similarities between the query and the documents
    cosine_similarities = []
    for doc_name, doc_vector in tf_idf.items():
        similarity = cosine_similarity(query_tf_idf, doc_vector)
        
        # Only include documents with non-zero similarity
        if similarity > 0.0:
            cosine_similarities.append({
                "doc_name": doc_name,
                "similarity": similarity,
                "image_path": image_paths.get(doc_name, '/static/photos/default.jpg')  # Use a default image if not found
            })
        
    # Check if there are no results with non-zero similarity
    if not cosine_similarities:
        return jsonify(message="Sorry, no recipes found."), 404  # Return a 404 with the message

    # Sort the documents based on similarity scores
    ranked_results = sorted(cosine_similarities, key=lambda x: x["similarity"], reverse=True)

    # Save results to results_soniya.txt
    with open('results_soniya.txt', 'w') as f:
        json.dump(ranked_results, f, indent=2)


    # Return the ranked results with photos as JSON
    return jsonify(results=ranked_results)

# Route to download a recipe
@app.route("/download/<filename>", methods=["GET"])
def download_recipe(filename):
    # Set the path to the 'documents' folder
    file_path = os.path.join('documents', filename)
    
    # Check if the file exists
    if os.path.exists(file_path):
        # Send the file as an attachment for download
        return send_file(file_path, as_attachment=True)
    else:
        return jsonify({"message": "File not found!"}), 404

if __name__ == "__main__":
    app.run(debug=True)