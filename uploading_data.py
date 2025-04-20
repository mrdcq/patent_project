import re
import difflib
from datetime import datetime
from dateutil import parser
import datefinder
import os
import re
import pandas as pd
import datefinder
from dateutil import parser
from datetime import datetime

# --- Helper Functions ---

def closest_accurate_date(date_str):
    """
    Given a token (e.g. "Lug."), use fuzzy matching against valid month abbreviations
    and return the closest match if above a similarity threshold.
    """
    valid_months = ["Jan.", "Feb.", "Mar.", "Apr.", "May.", "Jun.",
                    "Jul.", "Aug.", "Sep.", "Oct.", "Nov.", "Dec."]
    if not date_str:
        return date_str
    parts = date_str.split()
    if parts:
        matches = difflib.get_close_matches(parts[0], valid_months, n=1, cutoff=0.7)
        if matches:
            parts[0] = matches[0]
    return " ".join(parts)

def is_valid_date(dt):
    """
    Returns True if the date is between 1800 and 2025.
    """
    if dt is None:
        return False
    return 1800 < dt.year < 2025

# --- Methods for Extracting Dates ---

def get_date_from_csv(df_patents, csv_filename):
    """
    Reads the historical_masterfile file to extract dates and merges with the patent DataFrame.
    
    Parameters:
      - df_patents: DataFrame containing the patent data (needs to have patent_id column)
      - csv_filename: Path to the historical_masterfile that contains columns 'patent' and 'disp_dt'
        (disp_dt should have date in format the format "20oct1908").
    """

    df_csv = pd.read_csv(csv_filename, usecols=["patent", "disp_dt"], dtype={"patent": str, "disp_dt": str})
    df_csv["patent"] = df_csv["patent"].astype(str).str.lstrip("0")

    df_csv["date"] = pd.to_datetime(df_csv["disp_dt"], format="%d%b%Y", errors="coerce")

    merged_df = pd.merge(df_patents, df_csv, left_on='patent_id', right_on='patent', how='left')
    
    merged_df.drop(columns=['disp_dt', 'patent'], inplace=True)
    
    return merged_df

    """
    Use datefinder to scan the (cleaned) text for any potential dates.
    Return the first candidate.
    """
    candidates = list(datefinder.find_dates(text))
    valid_candidates = [d for d in candidates if is_valid_date(d)]
    return valid_candidates[0] if valid_candidates else None

# --- Big Method to Determine the Date ---

def process_final_patent_date(df_patents, csv_filename):
    """
    Process patent data by merging with historical_masterfile and then determining a final 
    date for each patent by applying the methods in this order:
      1. Use the CSV date if available.
      2. Try to extract a date using the regex method on 'corpus' - pattern recognition.
      3. Use datefinder to scan 'corpus' - also cleaning the OCRization.
      
    Adds a 'date_method' column indicating which method was used.
    Returns:
      A merged DataFrame with new columns 'date' and 'date_method'.
    """
    # Extract the dates from the historical_masterfile
    merged_df = get_date_from_csv(df_patents, csv_filename)
    
    # If the CSV did not obatin a date; use the other methods
    def get_final_date_and_method(row):
        if pd.notnull(row['csv_date']):
            return pd.Series([row['csv_date'], "CSV"])
        # Try regex pattern recognition from the corpus.
        dt = get_date_from_regex(row['corpus'])
        if dt is not None:
            return pd.Series([dt, "Regex"])
        # Try datefinder w/ cleaning otherwise
        dt = get_date_from_datefinder(row['corpus'])
        if dt is not None:
            return pd.Series([dt, "Datefinder"])
        return pd.Series([None, "None"])
    
    
    merged_df[['date', 'date_method']] = merged_df.apply(get_final_date_and_method, axis=1)
    merged_df.drop(columns=['csv_date'], inplace=True)

    return merged_df

# --- Method to create the DataFrame ---

def extract_info_from_file(filepath):
    """
    Function to extract all the data from a .txt file (patent_id, date (None for now), corpus)
    """
    with open(filepath, "r", encoding="utf-8") as file:
        content = file.read()
    
    patent_id = os.path.splitext(os.path.basename(filepath))[0]
    
    return {
        "patent_id": patent_id,
        "date": None,
        "corpus": content
    }

def extract_info_from_directory(directory):
    """
    This function is extracting all the data from the .txt files within the directory, 
    and returns a DataFrame containing all the informations.
    
    The returned DataFrame has the columns:
      - 'patent_id'
      - 'date' (initially None)
      - 'corpus'
    """
    file_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.txt')]
    records = [extract_info_from_file(fp) for fp in file_paths]
    df = pd.DataFrame(records)
    return df

def upload_patents(directory, historical_masterfile = "data/historical_masterfile.csv"):
    df_no_date = extract_info_from_directory(directory)
    df = process_final_patent_date(df_no_date, historical_masterfile)
    return df
