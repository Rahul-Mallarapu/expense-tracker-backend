from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import numpy as np


app = Flask(__name__)
CORS(app)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    transactions = data.get('transactions', [])

    if len(transactions) < 2:
        return jsonify({'error': 'Not enough transactions to cluster. Need at least 2.'}), 400

    purposes = [t['purpose'] for t in transactions]
    amounts = [abs(t['amount']) for t in transactions]

    le = LabelEncoder()
    encoded_purposes = le.fit_transform(purposes).tolist()

    X = np.array(list(zip(encoded_purposes, amounts)))

    max_k = min(10, len(X))
    wcss = []
    for i in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    if len(wcss) >= 3:
        diff1 = np.abs(np.diff(wcss))
        ratios = diff1[:-1] / (diff1[1:] + 1e-9) 
        elbow_point = int(np.argmax(ratios)) + 2
    else:
        elbow_point = max_k 

    elbow_point = max(2, min(elbow_point, max_k))
    kmeans_final = KMeans(n_clusters=elbow_point, init='k-means++', random_state=42, n_init=10)
    labels = kmeans_final.fit_predict(X).tolist()
    centroids = kmeans_final.cluster_centers_.tolist()
    label_mapping = {int(le.transform([cls])[0]): cls for cls in le.classes_}

    return jsonify({
        'encoded_purposes': encoded_purposes,
        'amounts': amounts,
        'purposes': purposes,
        'labels': labels,
        'centroids': centroids,
        'elbow_k': elbow_point,
        'wcss': wcss,
        'label_mapping': label_mapping
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)