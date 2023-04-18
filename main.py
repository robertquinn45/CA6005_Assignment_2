# Vector Space Model to retrieve full url of images for a given query. Output is a dataframe/csv file of results
# change input file of images as required at location"images_data="
# next work on prep-processing the query and documents
#from google.colab import drive

#drive.mount('/content/drive')
#import os

#os.chdir("/content/drive/My Drive/CA6005_Mechanics_of_Search")
import os
#import pip
#pip install ntlk
from nltk.tokenize import word_tokenize
#from nltk.corpus import stopwords
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import PorterStemmer
import pandas as pd
import logging
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import streamlit as st
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

#drive_path = "C:\\Users\\patri\\OneDrive\\Documents\\dcu\\2023_semester_3\\CA6005_Mechanics_of_Search\\assignment_2\\prod"

#"C:\\"
#"C:\\Users\\patri\\OneDrive\\Documents\\dcu\\2023_semester_3\\CA6005_Mechanics_of_Search\\assignment_2\\prod"
# List the files in the root directory of the C:\ drive
#files = os.listdir(drive_path)

# Print the list of files
#print(files)

#file_path = "images_acquired_and_object_detected_test_small_amount.csv"  # Replace with the actual path to your file
# Load the CSV file into a DataFrame
df_images_data = pd.read_csv("images_acquired_and_object_detected_test_small_amount.csv")
# Print the DataFrame
print(df_images_data)

#with open(file_path, "r") as file:
#    df_images_data = file.read()
#    print(df_images_data)

#dataset_path = '/content/drive/My Drive/CA6005_Mechanics_of_Search/ceacht_a_do/image_object_detection_prediction'
#images_data = "C:\\Users\\patri\\OneDrive\\Documents\\dcu\\2023_semester_3\\CA6005_Mechanics_of_Search\\assignment_2\\prod\\images_acquired_and_object_detected_test_small_amount.csv"

#df_images_data = pd.read_csv(images_data)

# Print the dataframe
df_images_data = df_images_data.drop('src', axis=1)
df_images_data = df_images_data.drop('title', axis=1)
df_images_data = df_images_data.drop('images', axis=1)
df_images_data = df_images_data.drop('filename', axis=1)

# print('df_images count is:',df_images_data.shape[0])
df_images_data = df_images_data.drop_duplicates(subset=['full_url'])
# print('df_images count after duplicate removal is:',df_images_data.shape[0])
# print('printing df_images_data after duplicate removal',df_images_data)
# reset the index to create a new column with incremental IDs
df_images_data.reset_index(inplace=True)
# rename the new column as "id"
df_images_data.rename(columns={'index': 'image_id'}, inplace=True)

# print('printing df_images_data after dropping columns')
print(df_images_data)
# create the ordered list from the datframe
alt_desc_column = df_images_data['alt']
alt_desc_list_of_strings = alt_desc_column.tolist()
# print('alt_desc_list_of_strings is:',alt_desc_list_of_strings)

image_dictionary = df_images_data.set_index('image_id').to_dict()['alt']

#####################################
# Preprocess the documents by tokenizing, removing stop words, and stemming the words
nltk.download('stopwords')
nltk.download('punkt')

# Preprocess the documents and queries
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()


def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [stemmer.stem(token) for token in tokens if token not in stop_words and token.isalpha()]
    return tokens


processed_image_descriptions = {}
for image_id, alt in image_dictionary.items():
    # print('printing alt',alt)
    if alt is not None:
        processed_image_descriptions[image_id] = preprocess(str(alt))
    else:
        logging.info('alt not processed as it is None is:' + str(alt))

logging.info('processed_image_descriptions is:' + str(processed_image_descriptions))

# Create a dictionary of all the unique terms in the documents and queries, and assign an index to each term
term_dict = {}
for image_id, alt in processed_image_descriptions.items():
    # print ('image_id is:',image_id)
    # print ('alt is:',alt)
    for token in alt:
        if token not in term_dict:
            term_dict[token] = len(term_dict)

# logging.info('term_dict is:'+str(term_dict))
# term_dict is all the unqiue words listed once across each docmuent, with an index number beside it
# Create a term-document matrix, where each row represents a term and each column represents a document, and the value in each cell represents the frequency of the term in the document. You can use the following code to create the matrix:

# Create a term-document matrix - the term_doc_matrix calcs to have terms in the columns and documents on the rows, the cosine similarity computing in a section further down the script,
#   expects the columns to be the same number for both matrices, hence the term_doc_matrix has terms in the columns and documents on the rows as code below
term_doc_matrix = np.zeros((len(image_dictionary), len(term_dict)))
for image_id, alt_tokens in processed_image_descriptions.items():
    for token in alt_tokens:
        term_doc_matrix[int(image_id) - 1, term_dict[token]] += 1

# logging.info('term_doc_matrix is:'+str(term_doc_matrix))
# logging.info('len(term_dict) is:'+str(len(term_dict)))

# Compute the TF-IDF weights for the term-document matrix

# Compute the TF-IDF weights
# norm: The normalization scheme to use for the TF-IDF weights.
# 'l2' normalization scales the weights so that the sum of the squares of each row is 1
# smooth_idf: a smoothing factor to the IDF (inverse document frequency) weights to prevent division by zero. If True, a smoothing factor of 1 is added to the IDF weights, otherwise no smoothing is applied
# use_idf: use IDF weighting in addition to TF (term frequency) weighting. If True, the TF-IDF weights are computed as tf * idf, whereas tf is the raw term frequency in the document and idf is the inverse document frequency of the term.
tfidf = TfidfTransformer(norm='l2', smooth_idf=True, use_idf=True)
tfidf.fit(term_doc_matrix)
tfidf_weights = tfidf.transform(term_doc_matrix).toarray()
logging.info('tfidf_weights is:' + str(tfidf_weights))
# Compute the cosine similarity between each query and document using the query-term matrix and the TF-IDF weights

# Transform the query into a TF-IDF vector.
# from sklearn.feature_extraction.text import CountVectorizer
# vectorizer = CountVectorizer()

# Create a text input field for the user to enter their query
query = st.text_input("Enter your query:")

# Create a button that the user can click to submit their query
submit_button = st.button("Submit")

# If the user has submitted their query, display the results
if submit_button:

    #query = "what flag does a person like?"
    #print('query is', query)
    preprocessed_query = preprocess(query)
    print('preprocessed_query is', preprocessed_query)

    # Create bag-of-words representation
    bow_query = {}
    for token in preprocessed_query:
        if token in bow_query:
            bow_query[token] += 1  # Increment count if word already exists in dictionary
        else:
            bow_query[token] = 1  # Add word to dictionary if it doesn't exist

    # print('bow_query is:',bow_query)
    # Preprocessing steps
    vectorizer = CountVectorizer(stop_words="english")
    vectorizer.fit_transform(alt_desc_list_of_strings)
    # Represent query as vector
    query_vec = vectorizer.transform([query])
    # Calculate cosine similarity
    similarity_scores = cosine_similarity(query_vec, vectorizer.transform(alt_desc_list_of_strings))
    ###############
    # print('similarity score type is:')
    # print(type(similarity_scores))
    # print('print similarity_scores')
    # print(similarity_scores)
    # print('convert the numpy.ndarray to a dataframe')
    df = pd.DataFrame(similarity_scores)
    print(df)
    # print('convert the numpy.ndarray to a dataframe printed')
    # print('transposing the dafatrame')
    # df_transposed=df.T
    df_transposed = df.transpose().set_axis(['similarity_score'], axis=1)
    print('printing df_transposed')
    # print(df_transposed)
    # print('df_transposed printed')
    df_images_similarity = df_transposed.join(df_images_data)
    print('printing df_images_similarity')
    # print(df_images_similarity)
    # print('sort the image_id column by the similarity scores descending')
    # sort the dataframe by column similarity_score in descending order
    df_full_url_sorted_similarity_score_desc = df_images_similarity['full_url'].loc[
        df_images_similarity['similarity_score'].sort_values(ascending=False).index]
    print('URL image results ranked are:')
    print(df_full_url_sorted_similarity_score_desc)
    #print('about to output search results to local file')
    # output DataFrame to CSV file with directory path
    #output_dir = r'C:\Users\patri\OneDrive\Documents\dcu\2023_semester_3\CA6005_Mechanics_of_Search\assignment_2\prod\\'
    #output_file = 'image_results_ranked_pycharm_v1.csv'
    #df_full_url_sorted_similarity_score_desc.to_csv(output_dir + output_file, index=False)
    #df_full_url_sorted_similarity_score_desc.to_csv(
    #    'image_object_detection_prediction/image_results_ranked_pycharm_v1.csv',
    #    index=False)
    print('output of results to disk completed')

    results = df_full_url_sorted_similarity_score_desc.values.tolist()
    print(results)

    #results = ["Result 1", "Result 2", "Result 3"]
    st.write("Results:")
    for result in results:
        st.write(result)

    # Print results in descending order
    # results = [(similarity_scores[0][i], document) for i, document in enumerate(alt_desc_list_of_strings)]
    # results.sort(reverse=True)
    # print('i and score and document are:')
    # for score, document in results:
    # print(score, document)
