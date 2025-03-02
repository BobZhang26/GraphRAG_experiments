import json
import gspread  # type: ignore
from oauth2client.service_account import ServiceAccountCredentials  # type: ignore
from sklearn.metrics import precision_score, recall_score, accuracy_score  # type: ignore
from dotenv import load_dotenv  # type: ignore
import os
import pandas as pd  # type: ignore
from libs import enhanced_chunk_finder


def connect2Googlesheet():
    load_dotenv()
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if creds_path is None:
        raise ValueError("The environment variable GOOGLE_SHEET_CREDS is not set.")
    creds = ServiceAccountCredentials.from_json_keyfile_name(creds_path, scope)
    client = gspread.authorize(creds)
    sheet_url = "https://docs.google.com/spreadsheets/d/1GRStLZdPvZTN-DDcxND3xaYaHGGl-wVCQxWdsFRhsqs/edit?gid=408470182#gid=408470182"
    # Open the Google Sheet (replace with your sheet name or URL)
    spreadsheet = client.open_by_url(sheet_url)  # Or use .open_by_url('URL')
    return spreadsheet


# Function to get ANNOTATED relevant documents for a specific question
def get_relevant_documents(df, question):
    relevant_docs = df[df[question] == 1]["Document"].tolist()
    return set(relevant_docs)


# Function to retrieve relevant documents for a specific question
def retrieval_rel_docs(
    graph, questions, top_k=5, limit=20, similarity_threshold=0.8, max_hops=1
):
    top_k_questions = questions.head(top_k)
    # Initialize a list to store the results
    results = []
    # Iterate over the top k questions
    for index, row in top_k_questions.iterrows():
        question_number = index + 1  # Assuming the question number is the index + 1
        question = row[
            "Question"
        ]  # Replace 'Question' with the actual column name for questions in df_MedQ

        # Generate response for the question
        # context = context_builder(graph, question, method="vector")
        filenames, output = enhanced_chunk_finder(
            graph, question, limit=20, similarity_threshold=0.8, max_hops=1
        )
        # Extract relevant documents from the response content
        # docs = response.choices[0].message.content  # Adjust this based on the actual response structure
        # Iterate over the output to extract chunk details
        # save output to a json file

        for chunk in output:
            file_name, chunk_text, page_number, position, similarity = chunk
            # Append the result to the list
            results.append(
                {
                    "Question number": question_number,
                    "Question": question,
                    "Retrieved FileName": file_name,
                    "Chunk Text": chunk_text,
                    "Page Number": page_number,
                    "Position": position,
                    "Similarity": similarity,
                }
            )

    # Convert the results to a DataFrame
    results_df = pd.DataFrame(
        results,
        columns=[
            "Question number",
            "Question",
            "Retrieved FileName",
            "Chunk Text",
            "Page Number",
            "Position",
            "Similarity",
        ],
    )
    # with open('./outputs/all_retrieval_results.json', 'w') as f:
    #         json.dump(results, f)
    results_df.to_csv("./outputs/all_retrieval_results.csv", index=False)
    # fileNames_df = pd.DataFrame(list(filenames), columns=['FileName'])
    return results_df


# Function to clean and format the text
def clean_text(text):
    if not isinstance(text, list):
        return ""
    # Remove unwanted characters and whitespace
    text = text.strip()
    # Remove '[]' and " " characters
    #text = text.replace("[", "").replace("]", "").replace("'", "").replace('"', "")
    # Remove commas
    text = text.replace(",", "")
    # remove ending with .pdf
    text = text.replace(".pdf", "")
    # Convert to uppercase for consistency
    text = text.upper()
    # Split into tokens, remove duplicates, and sort
    tokens = set(text.split())
    # Join tokens back into a string
    cleaned_text = " ".join(tokens)
    return cleaned_text


# Function to get the concatenated DataFrame
def get_concatenate_df(results_df, relevant_docs_df, topk):
    try:
        # Validate input DataFrames
        if not isinstance(results_df, pd.DataFrame) or not isinstance(
            relevant_docs_df, pd.DataFrame
        ):
            raise ValueError(
                "Both results_df and relevant_docs_df must be pandas DataFrames."
            )

        # Validate topk
        if not isinstance(topk, int) or topk <= 0:
            raise ValueError("topk must be a positive integer.")

        # Ensure the DataFrames have the necessary columns
        if (
            "Retrieved Files" not in results_df.columns
            or "Relevant Docs" not in relevant_docs_df.columns
        ):
            raise ValueError(
                "Both DataFrames must contain the necessary columns: 'Generated Docs' and 'Relevant Docs'."
            )

        # Concatenate results_df with relevant_docs_df side by side based on their index
        concatenated_df = pd.concat(
            [results_df.iloc[:topk], relevant_docs_df.iloc[:topk]], axis=1
        )

        # Ensure the concatenated DataFrame has the necessary columns
        if (
            "Question" not in concatenated_df.columns
            or "Relevant Docs" not in concatenated_df.columns
            or "Retrieved Files" not in concatenated_df.columns
        ):
            raise ValueError(
                "The concatenated DataFrame must contain the necessary columns: 'Question', 'Relevant Docs', and 'Generated Docs'."
            )

        # Clean the text
        concatenated_df["Generated Docs"] = concatenated_df["Generated Docs"].apply(
            clean_text
        )
        concatenated_df["Relevant Docs"] = concatenated_df["Relevant Docs"].apply(
            clean_text
        )

        concatenated_df = concatenated_df[
            ["Question", "Relevant Docs", "Generated Docs"]
        ]

        return concatenated_df

    except Exception as e:
        print(f"Error in get_concatenate_df: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of error


# Function to calculate accuracy, precision, and recall
def calculate_metrics(reference, candidate):
    reference_tokens = set(reference.split())
    candidate_tokens = set(candidate.split())

    true_positives = len(reference_tokens & candidate_tokens)
    false_positives = len(candidate_tokens - reference_tokens)
    false_negatives = len(reference_tokens - candidate_tokens)

    accuracy = true_positives / (true_positives + false_positives + false_negatives)
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0
    )

    return accuracy, precision, recall


def apply_metric(concatenated_df):
    metrics = concatenated_df.apply(
        lambda row: calculate_metrics(
            str(row["Relevant Docs"]), str(row["Generated Docs"])
        ),
        axis=1,
    )
    concatenated_df["Accuracy"] = metrics.apply(lambda metric: metric[0])
    concatenated_df["Precision"] = metrics.apply(lambda metric: metric[1])
    concatenated_df["Recall"] = metrics.apply(lambda metric: metric[2])
    return concatenated_df


def get_avg_similarity_df(df):
    # Concatenate the retrieved files for each question
    avg_similarity_df = (
        df.groupby(["Question number", "Question"])
        .agg(
            {
                "Retrieved FileName": lambda x: list(set(x)),  # Get unique filenames
                "Similarity": "mean",  # Average similarity score
            }
        )
        .reset_index()
    )
    avg_similarity_df.columns = [
        "Question Number",
        "Question",
        "Retrieved Files",
        "Avg Similarity",
    ]
    return avg_similarity_df
