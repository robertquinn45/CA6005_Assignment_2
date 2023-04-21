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
nltk.download('punkt')
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer                                  
import pandas as pd
import logging
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer                                                           
import streamlit as st
import string             
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

#drive_path = "C:\\Users\\patri\\OneDrive\\Documents\\dcu\\2023_semester_3\\CA6005_Mechanics_of_Search\\assignment_2\\prod"
def preprocess_documents(documents):
    # Create an empty list to store the processed documents
    processed_documents = []

    # Loop through each document in the list
    for document in documents:
        # some documents may be in float format, convert these documnets to string, before lowercasing in the following step, as lower function does not work on float
        document= str(document)
        # Lowercase the document
        document = document.lower()

        # Tokenize the document
        tokens = word_tokenize(document)

        # Remove stop words
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]

        # Stemming/Lemmatization
        stemmer = SnowballStemmer("english")
        tokens = [stemmer.stem(token) for token in tokens]

        # Remove special characters
        tokens = [token.translate(str.maketrans('', '', string.punctuation)) for token in tokens]

        # Rejoin the tokens
        processed_document = " ".join(tokens)

        # Append the processed document to the list
        processed_documents.append(processed_document)

    return processed_documents

def preprocess_query(query):
    # Lowercase the query
    query = query.lower()

    # Tokenize the query
    tokens = word_tokenize(query)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Stemming/Lemmatization
    stemmer = SnowballStemmer("english")
    tokens = [stemmer.stem(token) for token in tokens]

    # Remove special characters
    tokens = [token.translate(str.maketrans('', '', string.punctuation)) for token in tokens]

    # Rejoin the tokens
    processed_query = " ".join(tokens)

    return processed_query                            

#"C:\\"
#"C:\\Users\\patri\\OneDrive\\Documents\\dcu\\2023_semester_3\\CA6005_Mechanics_of_Search\\assignment_2\\prod"
# List the files in the root directory of the C:\ drive
#files = os.listdir(drive_path)

# Print the list of files
#print(files)

#file_path = "images_acquired_and_object_detected_test_small_amount.csv"  # Replace with the actual path to your file
# Load the CSV file into a DataFrame
df_images_data = pd.read_csv("images_acquired_and_object_detected.csv")
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

processed_documents = preprocess_documents(alt_desc_list_of_strings)
# Use TfidfVectorizer to create a vector representation of the documents and query
vectorizer = TfidfVectorizer()                                        
# Create a text input field for the user to enter their query
query = st.text_input("Enter your query:")

# Create a button that the user can click to submit their query
submit_button = st.button("Submit")

# If the user has submitted their query, display the results
if submit_button:

    #query = "what flag does a person like?"
    #print('query is', query)
    preprocessed_query = preprocess_query(query)
    print('preprocessed_query is', preprocessed_query)
    # Use TfidfVectorizer to create a vector representation of the documents and query
    tfidf_matrix = vectorizer.fit_transform(processed_documents + [preprocessed_query])

    # Extract the vector representation of the query
    query_vector = tfidf_matrix[-1]

    # Compute the cosine similarity between the query vector and the document vectors
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix[:-1])
    
    df = pd.DataFrame(similarity_scores)
    print(df)
    print('dataframe of the similarity_scores printed')
    # print('convert the numpy.ndarray to a dataframe printed')
    # print('transposing the dafatrame')
    # df_transposed=df.T
    df_transposed = df.transpose().set_axis(['similarity_score'], axis=1)
    print('printing df_transposed')
    print(df_transposed)
    # print('df_transposed printed')
    df_images_similarity = df_transposed.join(df_images_data)
    print('printing df_images_similarity')
    # print(df_images_similarity)
    filtered = df_images_similarity['similarity_score'] > 0
    # use the filter with loc to create a new DataFrame
    df_images_similarity_no_zero = df_images_similarity.loc[filtered]
    print('printing  df_images_similarity_no_zero without similarity scores of zero')
    print(df_images_similarity_no_zero)                                                                                                                                                                                                                                                       
    # print('sort the image_id column by the similarity scores descending')
    # sort the dataframe by column similarity_score in descending order
    #df_full_url_sorted_similarity_score_desc = df_images_similarity_no_zero['full_url'].loc[df_images_similarity_no_zero['similarity_score'].sort_values(ascending=False).index]
    df_full_url_sorted_similarity_score_desc = df_images_similarity_no_zero.loc[df_images_similarity_no_zero['similarity_score'].sort_values(ascending=False).index,['full_url','similarity_score']]
    print('URL image results ranked are:')
    print(df_full_url_sorted_similarity_score_desc)
    df_full_url_sorted_similarity_score_desc['similarity_score'] = df_full_url_sorted_similarity_score_desc['similarity_score'].multiply(100)
    # Reorder the columns
    df_full_url_sorted_similarity_score_desc = df_full_url_sorted_similarity_score_desc.reindex(columns=['similarity_score', 'full_url'])
    # Round column 'similarity_score' to a whole number
    df_full_url_sorted_similarity_score_desc['similarity_score'] = df_full_url_sorted_similarity_score_desc['similarity_score'].round(0)
    # Convert column 'similarity_score' to integer data type
    df_full_url_sorted_similarity_score_desc['similarity_score'] = df_full_url_sorted_similarity_score_desc['similarity_score'].astype(int)
    df_full_url_sorted_similarity_score_desc['similarity_score'] = df_full_url_sorted_similarity_score_desc[
        'similarity_score'].astype(str)
    # define a lambda function to add percentage symbol to each value in the 'similarity_score' column
    add_percent = lambda x: str(x) + '%'
    # apply the lambda function to the 'Percentage' column using the 'apply()' method
    df_full_url_sorted_similarity_score_desc['similarity_score'] = df_full_url_sorted_similarity_score_desc['similarity_score'].apply(add_percent)

    # Rename column 'A' to 'X'
    df_full_url_sorted_similarity_score_desc = df_full_url_sorted_similarity_score_desc.rename(columns={'similarity_score': 'relevancy %'})
    print('output of results to disk completed')

    results = df_full_url_sorted_similarity_score_desc.values.tolist()
    print(results)

    #results = ["Result 1", "Result 2", "Result 3"]
    # st.write("Results:")
    # for result in results:
        # st.write(result)
        # display the data
    for i, row in df_full_url_sorted_similarity_score_desc.iterrows():
        st.write(f"{row['relevancy %']}<br>{row['full_url']}", unsafe_allow_html=True)
