# -------------------------------------------------------------------------
# AUTHOR: Brandon Tang
# FILENAME: similarity.py
# SPECIFICATION: Computes the pairwise cosine similarity between documents in a CSV file
# FOR: CS 4440 (Data Mining) - Assignment #1
# TIME SPENT: 1 hour
# -----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy or pandas.
# You have to work here only with standard dictionaries, lists, and arrays

# Importing some Python libraries
import csv
import math
from sklearn.metrics.pairwise import cosine_similarity

documents = []

# reading the documents in a csv file
with open("cleaned_documents.csv", "r") as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:  # skipping the header
            documents.append(row)

# Building the document-term matrix by using binary encoding.
# You must identify each distinct word in the collection using the white space as your character delimiter.
# --> add your Python code here
docTermMatrix = []
docIDs = []

for doc_id, text in documents:
    tokens = [w for w in text.split() if w]
    docTermMatrix.append(set(tokens))  # store as a set of unique words
    docIDs.append(doc_id)


# Compare the pairwise cosine similarities and store the highest one
# Use cosine_similarity([X], [Y]) to calculate the similarities between 2 vectors
# --> Add your Python code here
def cosine_similarity(A, B):
    setA = A[0]
    setB = B[0]
    inter = len(setA & setB)
    if inter == 0:
        return [[0.0]]
    return [[inter / math.sqrt(len(setA) * len(setB))]]


best_sim = -1.0
best_i = -1
best_j = -1

n = len(docTermMatrix)
for i in range(n):
    for j in range(i + 1, n):
        sim = cosine_similarity([docTermMatrix[i]], [docTermMatrix[j]])[0][0]
        if sim > best_sim:
            best_sim = sim
            best_i = i
            best_j = j

# Print the highest cosine similarity following the information below
# The most similar documents are document 10 and document 100 with cosine similarity = x
# --> Add your Python code here
print(
    f"The most similar documents are document {docIDs[best_i]} and document {docIDs[best_j]} with cosine similarity = {best_sim:.6f}."
)
