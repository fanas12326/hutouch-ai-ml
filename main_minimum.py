from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords, wordnet
from nltk import pos_tag
import re
from nltk.tokenize import word_tokenize
import time
import json
from datetime import datetime
import requests
import httpx  # HTTP client for making requests
import copy
from token_count import TokenCount
from openai import OpenAI
from collections import Counter
import gdown  # Import gdown to use it for downloading files from Google Drive
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

class Item(BaseModel):
    id: int
    prompt: str
    data: str

@app.post("/calculate-task/")
async def calculate_task(item: Item):
        user_id = item.id
        
        # Set up the API endpoint URL
        
        url = f"http://35.85.112.192/api/ai_fetchapi_data?userID={user_id}"
        # Headers as specified
        headers = {
            'Accept': 'application/json',
            'X-API-KEY': 'JGIp4AWFmI',
            'Content-Type': 'application/json',
        }

        # Make the POST request
        response = requests.post(url, headers=headers)

        # Print the response from the server
        print(response.text)
        print(type(response.text))

        json_user_data = json.loads(response.text)
        print(json_user_data)
        
        for item in json_user_data['data']:
            print(item)
        
        # Create a DataFrame with only the needed columns
        df = pd.DataFrame(json_user_data['data'])[['appName', 'data']].rename(columns={'appName': 'App name ', 'data': 'BLOB data'})

        df
        
        # Write to Excel
        df.to_excel('apps_data.xlsx', index=False)

        print("Excel file created successfully with specified columns.")
        
        df_user = pd.read_excel('apps_data.xlsx')
        df_user 
        
        df_local = pd.read_csv('dev data (2).csv')
        df_local
        
        # Fill NaN values with an empty string before concatenating
        df_user['App name '] = df_user['App name '].fillna('')
        df_user['BLOB data'] = df_user['BLOB data'].fillna('')

        df_user['all_data']=df_user['App name ']+' '+df_user['BLOB data']

        # If you want to remove extra spaces caused by empty values in the middle
        df_user['all_data'] = df_user['all_data'].apply(lambda x: ' '.join(x.split()))

        df_user
        
        # converting the column 'all_data' into a list and then merging it to create a string
        user_merged_data = df_user['all_data'].tolist()
        user_merged_data = " ".join(user_merged_data)
        user_merged_data
        
        
        #function that help to preprocess the data

        def clean_text(text):
            # Split into words
            words = text.lower()
            tokens = word_tokenize(words)

            # Remove Stopwords
            english_stopwords = stopwords.words('english')
            filtered_words = [word for word in tokens if word not in english_stopwords]

            # Remove special characters and numbers
            cleaned_words = [re.sub(r'[^A-Za-z]', '', word) for word in filtered_words if word.isalnum()]

            # Remove empty string from list
            cleaned_words = list(filter(None,cleaned_words))

            return cleaned_words
        
        cleaned_user_data = clean_text(user_merged_data)
        print(cleaned_user_data)
        
        def find_best_match(df, cleaned_user_data):
            max_count = 0
            best_match = None

            for index, row in df.iterrows():
                tags = row['Tags'].lower().split(',')
                matches = [tag for tag in tags if tag in cleaned_user_data]
                for item in matches:
                    print(row['Developer']+": "+item)
                count = len(matches)

                if count > max_count:
                    max_count = count
                    best_match = row['Developer']

            return best_match

        user_role = find_best_match(df_local, cleaned_user_data)
        print("Role of the developer is " + user_role + " developer")
        return {"id":user_id,"role":user_role}
            
# #hard code
# @app.post("/identify-task/")
# async def identify_task(item: Item):
#     id = item.id
    
#     # Download the CSV file using gdown
#     url = 'https://drive.google.com/uc?id=1Ivswt6nFooeh1dcKAjel-ELepOjf6e1C'
#     output = 'application_data.csv'
#     gdown.download(url, output, quiet=False)

#     # Download necessary NLTK resources
#     nltk.download('punkt', quiet=True)
#     nltk.download('stopwords', quiet=True)
#     nltk.download('averaged_perceptron_tagger', quiet=True)
#     nltk.download('wordnet', quiet=True)

#     # Define tasks and their specific steps
#     tasks = {
#         'UI/UX Design': ['Implement responsive design'],
#         'API Integration': ['Implement HTTP client', 'Parse JSON data', 'Handle errors and exceptions', 'Cache data', 'Implement pull-to-refresh'],
#         'Testing': ['Write unit tests', 'Implement widget tests', 'Fix issues'],
#         'Debugging': ['Review logs', 'Use breakpoints', 'Apply fixes'],
#         'Performance Optimization': ['Minimize layout rebuilds', 'Use lazy loading', 'Optimize database queries'],
#         'Maintaining Codebase': ['Refactor code for readability', 'Update dependencies', 'Resolve merge conflicts'],
#         'Adapting to Platform-Specific Features': ['Implement platform channels']
#     }

#     def load_data(filename):
#         # Adjust the path to the location where the file is actually downloaded
#         return pd.read_csv(filename)

        
#     def extract_keywords(user_prompt):
#         stop_words = set(stopwords.words('english'))
#         return set(word_tokenize(user_prompt.lower())) - stop_words
    
#     def extract_relevant_task(user_prompt):
#         prompt_keywords = extract_keywords(user_prompt)
#         max_overlap = 0
#         best_match = ("No specific task matched", [])
#         for task, steps in tasks.items():
#             task_keywords = set(task.lower().split())
#             overlap = len(prompt_keywords & task_keywords)
#             if overlap > max_overlap:
#                 max_overlap = overlap
#                 best_match = (task, steps)
#         return best_match

#     def compare_tags(input_tags, app_tags):
#         input_count = Counter(input_tags)
#         app_count = Counter(app_tags)
#         return sum(min(input_count[tag], app_count[tag]) for tag in input_count if tag in app_count)

#     def find_top_apps(df, task_name, steps, top_n=2):
#         task_keywords = set(task_name.lower().split())
#         steps_keywords = set(word.lower() for step in steps for word in word_tokenize(step))
#         input_tags = task_keywords.union(steps_keywords)
#         df['score'] = df['tags'].apply(lambda tags: compare_tags(input_tags, extract_keywords(tags)))
#         return df.sort_values(by='score', ascending=False).head(top_n)['application_name']
    
#     user_prompt = item.prompt
#     filename = 'application_data.csv'  # This is the filename where gdown downloaded the file
#     df = load_data(filename)
#     task_name, steps = extract_relevant_task(user_prompt)
#     top_apps = find_top_apps(df, task_name, steps)
#     print("Identified Task:", task_name)
#     print("Steps:", ', '.join(steps))
#     print("\nTop Matching Applications:")
#     print(top_apps.to_string(index=False))
#     return {"status" : "success", "task": task_name, "steps": steps, "top_apps": top_apps.to_list()}

@app.post("/identify-task/")
async def identify_task(item: Item):
    task_data = item.data
    task_obj = None
    tasks_dict = {}

    if task_data:
        try:
            task_obj = json.loads(task_data)
            tasks_string = task_obj.get("task", "")
            for task_entry in tasks_string.split("\\n"):
                parts = task_entry.split(":")
                task_name = parts[0]
                steps_str = ":".join(parts[1:])
                steps = steps_str.strip("[]").split(",")
                tasks_dict[task_name] = steps
        except json.JSONDecodeError:
            print("Invalid JSON data for tasks. Skipping task extraction.")
    def compare_tags(input_tags, app_tags):
        input_count = Counter(input_tags)
        app_count = Counter(app_tags)
        return sum(min(input_count[tag], app_count[tag]) for tag in input_count if tag in app_count)

    def normalize_text(text):
        return re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())

    def load_data(filename):
        return pd.read_csv(filename)

    def extract_keywords(user_prompt):
        tokens = word_tokenize(user_prompt)  
        stop_words = set(stopwords.words('english'))  
        custom_stop_words = stop_words.union({'give', 'code', 'me',"task", 'for', 'in', 'a'})
        keywords = [word for word in tokens if word.lower() not in custom_stop_words and word.isalpha()]
        return keywords

    def find_top_apps_and_filter_keywords(df, user_prompt):
        initial_keywords = set(extract_keywords(user_prompt))
        df['score'] = df['tags'].apply(lambda tags: compare_tags(initial_keywords, extract_keywords(tags)))
        sorted_apps = df.sort_values(by='score', ascending=False)
        top_apps = sorted_apps.head(2)['application_name']
        # Filter out any app names from the keywords
        filtered_keywords = {kw for kw in initial_keywords if kw not in top_apps.values}
        return top_apps, filtered_keywords
    
    user_prompt = item.prompt
    filename = 'application_data.csv'
    df = load_data(filename)
    matching_apps, filtered_keywords = find_top_apps_and_filter_keywords(df, user_prompt)

    filtered_keywords_str = ', '.join(filtered_keywords)
    matching_apps_str = ', '.join(matching_apps)

    output = []
    output.append("Extracted Keywords: " + filtered_keywords_str)
    output.append("Matching Applications: " + matching_apps_str)
    for task, steps in tasks_dict.items():
        output.append(f"Task: {task}, Steps: {', '.join(steps)}")

    return {"success": True, "output": output}