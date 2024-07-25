from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import RedirectResponse
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
import base64
from token_count import TokenCount
from openai import OpenAI
from collections import Counter
import gdown  # Import gdown to use it for downloading files from Google Drive
from dotenv import load_dotenv
import os
from PIL import Image
from io import BytesIO
import spacy
import asyncio
import shutil
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator

load_dotenv()

app = FastAPI()

class Item(BaseModel):
    id: int
    prompt: str
    data: str

openai_api_key = os.getenv('OPENAI_API_KEY')
print('Open ai api key: '+openai_api_key)
client = OpenAI(api_key=openai_api_key)


def create_user_directory(user_id):
  try:
    path = f"./users_temp_data/user_{str(user_id)}"
    os.makedirs(path)
    print(f"Folder {path} created successfully")
    return path
  except FileExistsError:
    print(f"Folder '{path}' already exists.")
    path = f"./users_temp_data/user_{str(user_id)}"
    return path
  except Exception as e:
        print(f"An error occurred: {e}")
        return ""

def delete_folder_recursive(path):
    try:
        shutil.rmtree(path)
        print(f"Folder '{path}' and all its contents deleted successfully.")
    except FileNotFoundError:
        print(f"Folder '{path}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

def get_project_data(user_id):
  api_url = f'http://35.85.112.192/api/get-project-data/{user_id}'

  # Define the headers
  headers = {
      'Accept': 'application/json',
      'X-API-KEY': 'JGIp4AWFmI',
      'Content-Type': 'application/json'
  }

  # Make the GET request
  response = requests.get(api_url, headers=headers)

  # Check if the request was successful
  if response.status_code == 200:
      # Parse the JSON response
      data = response.json()
      # print(data)
      return {"status":"success","data":data}
  else:
      print(f"Failed to retrieve data: {response.status_code} - {response.text}")
      return {"status":"failed"}

def store_project_data_locally(user_id,dir_path):
  project_data = get_project_data(user_id)

  if project_data["status"]=="success":
    print("Data is successfully retrieved")
    print(project_data["data"]["data"]["user_nature"])
    print(type(project_data["data"]["data"]["user_nature"]))
    try:
      with open(dir_path+"/project_style_data.json", 'w') as file:
        data = json.loads(project_data["data"]["data"]["user_nature"])
        json.dump(data, file)
        return {"status":"success"}
    except:
      print("error while creating the file")
      return {"status":"failed"}
  else:
    print("Failed to retrieve data")
    return {"status":"failed"}

def store_error(id,func_name,error):
  url = 'http://35.85.112.192/api/ai-store-error'   
  # Define the headers
  headers = {
      'Accept': 'application/json',
      'X-API-KEY': 'JGIp4AWFmI',
      'Content-Type': 'application/json'
  }

  # Define the body
  body = {
      "user_id": id,
      "data": [
          {
              "func_name": func_name,
              "error": error
          }
      ]
  }
  # Make the POST request
  response = requests.post(url, headers=headers, json=body)
  return response

def get_response(threadID,assistantID,payload):
    message = client.beta.threads.messages.create(
        thread_id = threadID,
        role = "user",
        content = payload
    )
    print(message)
    #run the assistant
    run = client.beta.threads.runs.create(
        thread_id = threadID,
        assistant_id = assistantID
    )
    print(run)
    # Waits for the run to be completed
    while True:
        run_status = client.beta.threads.runs.retrieve(thread_id = threadID, run_id = run.id)
        if run_status.status == "completed":
            break
        elif run_status.status == "failed":
            print("Run failed: ",run_status.last_error)
            break
            
    if run_status.status == "completed":
        messages = client.beta.threads.messages.list(
            thread_id = threadID
        )

        # Prints the messages with the latest message at the bottom
        number_of_messages = len(messages.data)
        print( f'Number of messages: {number_of_messages}')

        for message in reversed(messages.data):
            role = message.role
            for content in message.content:
                if content.type == 'text':
                    response = content.text.value
    else:
        print("Something went wrong")
        response = 'Failed'
    return response

@app.post("/random-prompts/")
async def random_prompts(item: Item):
    data = json.loads(item.data)
    print(data)
    
    user_id = item.id
    prompt = item.prompt
    thread_id = data["thread_id"]
    assistant_id = data["assistant_id"]
    print("prompt: ",prompt)
    print("thread_id: ",thread_id)
    print("assistant_id: ",assistant_id)
    
    
    
    if thread_id == "" and assistant_id == "":
        assistant_name = data["assistant_name"]
        model_name = data["model_name"]
        instruction = data["instruction"]

        my_assistant = client.beta.assistants.create(
            instructions=instruction,
            name=assistant_name,
            tools=[{"type": "file_search"}],
            model=model_name,
        )
        print(my_assistant)
        assistant_id = my_assistant.id

        thread = client.beta.threads.create()
        thread_id = thread.id
        print(thread_id)

        curr_payload = prompt
        response = get_response(thread_id,assistant_id,curr_payload)
        print(response)

    else:
        curr_payload = prompt
        response = get_response(thread_id,assistant_id,curr_payload)
        print(response)
    
    if response == "Failed":
        store_error(user_id,"/random-prompts/","assistant api failed to generate response")    
        return {"status":"failed"}
    else:
        return  {"status":"success","response":response,"thread_id":thread_id,"assistant_id":assistant_id}
 
@app.post("/upload-file/")
async def upload_file(item: Item):
    
    file_id = item.prompt
    assistant_id = item.data
    
    openai_api_key = os.getenv('OPENAI_API_KEY')
    print('Open ai api key: '+openai_api_key)
    client = OpenAI(api_key=openai_api_key)
    
    vector_store = client.beta.vector_stores.create(
        name="Uploaded Files"
    )
    print(vector_store)
    
    vector_id = vector_store.id
    print("vector_id: ",vector_id)
    
    vector_store_file = client.beta.vector_stores.files.create(
        vector_store_id=vector_id,
        file_id=file_id
    )
    print("vector_store_file: ",vector_store_file)
    

    def update_assistant(vectorId):
        assistant = client.beta.assistants.update(
            assistant_id=assistant_id,
            tool_resources={"file_search": {"vector_store_ids": [vectorId]}},
        )   
        return assistant

    updated_assistant = update_assistant(vector_id)
    print("updated_assistant: ",updated_assistant)
    
    vector_store_files = client.beta.vector_stores.files.list(
        vector_store_id=vector_id
    )
    
    return {"status": "success", "response": updated_assistant, "vector_id": vector_id}


@app.post("/delete-uploaded-file/")
async def delete_uploaded_file(item: Item):
    openai_api_key = os.getenv('OPENAI_API_KEY')
    print('Open ai api key: '+openai_api_key)
    client = OpenAI(api_key=openai_api_key)
    
    vector_id = item.prompt
    vector_store_files = client.beta.vector_stores.files.list(
        vector_store_id=vector_id
    )
    print(vector_store_files)

    total_files = len(vector_store_files.data)
    
    return {"status":"success","total_files":total_files}    

@app.post("/calculate-task/")
async def calculate_task(item: Item):
        user_id = item.id
        
        # drive.mount('/content/drive')
        
        # Set up the API endpoint URL

        url = f"http://35.85.112.192/api/ai_profiles?user_id={user_id}"
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


        user_profile_data = json.loads(response.text)
        user_profile_data

        actual_user_apps = json.loads(user_profile_data['apps'])
        print(actual_user_apps)
        print(type(actual_user_apps))
        
        for i in range(len(actual_user_apps)):
            if actual_user_apps[i] == 'vscode' or actual_user_apps[i] == 'vs code':
                actual_user_apps[i] = 'Visual Studio Code'

        print(actual_user_apps)
        
        cleaned_list = [item.split('.')[0] for item in actual_user_apps]
        cleaned_list
        
        # Assuming 'data.csv' is your dataset file with 'id', 'parent_id', and 'data' columns
        df = pd.read_csv('formed_data (20).csv')
        df
        # Replace NaN in 'parent_id' with 0 to denote root nodes
        df['parent_id'].fillna(0, inplace=True)
        df
        
        # Create a directed graph
        G = nx.DiGraph()

        # Add nodes with the node attribute 'label' equal to 'data'
        for index, row in df.iterrows():
            G.add_node(row['id'], label=row['data'])

        # Add edges from parent to child
        for index, row in df.iterrows():
            if row['parent_id'] != 0:
                G.add_edge(row['parent_id'], row['id'])
        
        # Function to traverse the graph and return all paths for a given app as JSON objects
        def get_all_paths_for_app_as_json(graph, app):
            # Convert app parameter to lower case for case-insensitive comparison
            app_lower = app.lower()

            # Find all nodes that contain the app (loosely comparing)
            app_nodes = [node for node, data in graph.nodes(data=True) if app_lower in data.get('label', '').lower()]
            all_paths = []

            # Traverse from each app node to the root and build the JSON object
            for app_node in app_nodes:
                path = {}
                current_node = app_node
                path_nodes = []

                # Collect nodes up to the root
                while current_node != 0:
                    path_nodes.append(current_node)
                    predecessors = list(graph.predecessors(current_node))
                    current_node = predecessors[0] if predecessors else 0

                # Assign labels correctly from the root down to the node
                labels = ["type", "developer", "task", "steps", "apps"]
                label_index = 0

                for node in reversed(path_nodes):
                    if labels:
                        label = labels.pop(0)
                        path[label] = graph.nodes[node]['label']
                        if label == "apps":
                            # Collect all children of the current app node, these are considered as "parameters"
                            path["parameters"] = [graph.nodes[child]['label'] for child in graph.successors(node)]

                all_paths.append(path)

            return all_paths

        # Initialize an empty list to store results
        all_app_paths = []

        # Loop over each app and collect the paths
        for app in cleaned_list:
            # Call the function and append the result to the list
            app_paths = get_all_paths_for_app_as_json(G, app)
            if app_paths:
                all_app_paths.extend(app_paths)  # Extend the list with the paths of the current app
            else:
                print(f"No path found for {app}")

        # Check the results and print them or handle them as needed
        if all_app_paths:
            # Print each path or handle them as needed
            for path in all_app_paths:
                print(path)
        else:
            print("No data available for any apps.")
            
        filtered_data = [record for record in all_app_paths if 'apps' in record]
        all_app_paths = filtered_data
        all_app_paths
        
        # Convert the list to a JSON formatted string
        json_formatted_str = json.dumps(all_app_paths, indent=4)
        print("JSON formatted string of all app paths:")
        print(json_formatted_str)

        json_object = json.loads(json_formatted_str)

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
        
        # Function to select a path based on the developer tag
        def select_path(paths, developer_tag):
            # Filter paths by developer tag
            filtered_paths = [path for path in paths if path['developer'] == developer_tag]

            # If there's only one path, return it
            if len(filtered_paths) == 1:
                return json.dumps(filtered_paths[0])  # Return a JSON string

            # If there are multiple paths, ask the user to select one
            elif len(filtered_paths) > 1:
                print(f"Multiple paths found for the developer: {developer_tag}")
                for i, path in enumerate(filtered_paths):
                    print(f"{i+1}: {path}")
                return json.dumps(filtered_paths)    # Return a JSON string

            # If no paths are found, return a message
            else:
                return json.dumps("No paths found for the specified developer.")  # Return a JSON string

        if len(json_object) > 1:
            # User input for the developer tag
            user_input_developer = user_role
            # Call the function and print the selected path
            dev_json_data = select_path(json_object, user_input_developer)
            print(dev_json_data)
        elif len(json_object) < 1:
            print("No path found for specified app")
        else:
            print("No need, only one task is there")

        # Function to create a new list without the 'parameters' key
        def remove_parameters(json_list):
            new_list = []
            for item in json_list:
                if isinstance(item, dict):  # Check if the item is a dictionary
                    # Using dictionary comprehension to recreate each dictionary without 'parameters'
                    new_dict = {key: value for key, value in item.items() if key != 'parameters'}
                    new_list.append(new_dict)
                else:
                    new_list.append(item)  # Append the item unchanged if it's not a dictionary
            return new_list

        # Creating a new list without modifying the original data
        updated_data = remove_parameters(json.loads(dev_json_data))

        # Print the updated data
        updated_data
        
        dev_json_data
        
        tc = TokenCount(model_name="gpt-3.5-turbo")
        
        def excel_to_json_array(excel_file):
            # Read the Excel file
            df = pd.read_excel(excel_file)

            # Convert the DataFrame to a JSON array string
            json_data = df.to_json(orient='records')

            return json_data

        json_data = excel_to_json_array('apps_data.xlsx')
        print(json_data)  # Print the JSON array string
        
        print(type(json_data))
        
        json_object = json.loads(json_data)
        json_object
        
        def get_tokens_count(text):
            tokens = tc.num_tokens_from_string(text)
            print(f"Tokens in the string: {tokens}")
            return tokens
        
        def reduce_token_simp(json_data, max_tokens, reduced_by):
            # Create a deep copy of json_data to avoid modifying the original object
            data_1 = copy.deepcopy(json_data)

            # Convert the entire JSON data to a string and count tokens
            json_string_1 = json.dumps(data_1)
            total_tokens = get_tokens_count(json_string_1)
            print("Initial token count:", total_tokens)

            visited = [False] * len(data_1)  # Track if an item has been minimized
            count = 0  # Counter to track the number of minimized items

            # Reducing tokens if necessary
            while total_tokens > max_tokens:
                for index, item in enumerate(data_1):
                    if total_tokens <= max_tokens:
                        break

                    if len(item["BLOB data"]) >= reduced_by:
                        item["BLOB data"] = item["BLOB data"][:-reduced_by]  # Reduce by specified amount
                        json_string_1 = json.dumps(data_1)
                        total_tokens = get_tokens_count(json_string_1)
                    else:
                        # Only mark as visited and increment count if not already done
                        if not visited[index]:
                            visited[index] = True
                            count += 1

                # Break if all items are visited
                if count >= len(data_1):
                    print("All data has been minimized or cannot be reduced further.")
                    break

            return data_1
        
        print(get_tokens_count(json.dumps(json_object, indent=2)))
        
        reduced_data = reduce_token_simp(json_object,30000,200)
        reduced_data

        # Output the modified JSON data
        reduced_str = json.dumps(reduced_data, indent=2)
        print(get_tokens_count(reduced_str))
        print(reduced_str)
        
        # Specify the filename
        filename = 'user_data_file.json'

        # Writing JSON data to a file
        with open(filename, 'w') as file:
            json.dump(reduced_data, file)

        print(f"JSON data has been written to {filename}")
        
        # Specify the filename
        filename = 'public_data_file.json'

        # Writing JSON data to a file
        with open(filename, 'w') as file:
            json.dump(updated_data, file)

        print(f"JSON data has been written to {filename}")
        
        openai_api_key = os.getenv('OPENAI_API_KEY')
        print('Open ai api key: '+openai_api_key)
        client = OpenAI(api_key=openai_api_key)
        
        def upload_file_to_assistant(filePath1,filePath2):
            # Create a vector store caled "Financial Statements"
            vector_store = client.beta.vector_stores.create(name="Uploaded Files")

            # Ready the files for upload to OpenAI
            file_paths = [filePath1,filePath2]
            file_streams = [open(path, "rb") for path in file_paths]

            # Use the upload and poll SDK helper to upload the files, add them to the vector store,
            # and poll the status of the file batch for completion.
            file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
                vector_store_id=vector_store.id, files=file_streams
            )

            # You can print the status and the file counts of the batch to see the result of this operation.
            print(file_batch.status)
            print(file_batch.file_counts)
            print(vector_store.id)

            return vector_store.id

        vector_id = upload_file_to_assistant("user_data_file.json","public_data_file.json")
        
        assistant = client.beta.assistants.update(
        assistant_id="asst_gzeLQXC6s9aGZzrakZbuAho2",
        tool_resources={"file_search": {"vector_store_ids": [vector_id]}},
        model="gpt-3.5-turbo-0125"
        )
        
        def extract_json_using_brackets(text):
            # Find the first opening and last closing square brackets
            start_index = text.find('[')
            end_index = text.rfind(']') + 1  # +1 to include the closing bracket

            # Extract the substring containing JSON
            json_data = text[start_index:end_index]

            # Convert the string to a Python dictionary
            json_object = json.loads(json_data)

            return json_object
        
        content = f"Compare the files and generate output. Output must be a valid json array"
        print(content)
        
        def get_response():
            count = 0
            thread = client.beta.threads.create()
            message = client.beta.threads.messages.create(
                thread_id = thread.id,
                role = "user",
                content = f"Compare the files and generate output. Output must be a valid json array"
            )
            #run the assistant
            run = client.beta.threads.runs.create(
                thread_id = thread.id,
                assistant_id = 'asst_gzeLQXC6s9aGZzrakZbuAho2',
            )
            # Waits for the run to be completed
            while True:
                run_status = client.beta.threads.runs.retrieve(thread_id = thread.id, run_id = run.id)
                if run_status.status == "completed":
                    break
                elif run_status.status == "failed":
                    print("Run failed: ",run_status.last_error)
                    break
                
                time.sleep(2) # wait for 2 seconds before checking again
            if run_status.status == "completed":
                messages = client.beta.threads.messages.list(
                    thread_id = thread.id
                )

                # Prints the messages with the latest message at the bottom
                number_of_messages = len(messages.data)
                print( f'Number of messages: {number_of_messages}')

                for message in reversed(messages.data):
                    role = message.role
                    for content in message.content:
                        if content.type == 'text':
                            response = content.text.value
                            print(f'\n{role}: {response}')

            else:
                print("Something went wrong")
                response = 'Failed'

            # Extract and print JSON
            if response != 'Failed':
                json_output = extract_json_using_brackets(response)
                final_response = json.dumps(json_output, indent=4)
                print(type(final_response))
                print(final_response)
                return final_response

            else:
                return "Failed"
        
        flag = True
        while flag:
            get_updated_response = get_response()
            print(get_updated_response)
            if get_updated_response != "[]":
                flag=False

        if(get_updated_response != "Failed"):
            
            def format_task_steps(json_input):
                # Load data from JSON
                json_data = json.loads(json_input)

                # Dictionary to store task as key and list of steps as values
                task_steps_map = {}

                # Iterate through each item in the data
                for item in json_data:
                    task = item['task']
                    step = item['steps']
                    # If the task is already in the dictionary, append the step to its list
                    if task in task_steps_map:
                        task_steps_map[task].append(step)
                    else:
                        task_steps_map[task] = [step]

                # Build the output string
                output_string = ""
                for task, steps in task_steps_map.items():
                    output_string += f"{task} : [{', '.join(steps)}]\n"

                return output_string.strip()

            task_updated = format_task_steps(get_updated_response)
            print(task_updated)
            
            print(get_updated_response)
            len(get_updated_response)
            print(type(get_updated_response))

            # Load data from JSON string
            data = json.loads(get_updated_response)

            # Organizing data by tasks
            tasks = {}
            for item in data:
                task = item['task']
                step = item['steps']
                if task in tasks:
                    tasks[task].append(step)
                else:
                    tasks[task] = [step]

            all_tasks = ''

            # Generating sentences
            for task, steps in tasks.items():
                steps_formatted = ", ".join(steps[:-1]) + ", and " + steps[-1]
                all_tasks = all_tasks + (f"The task '{task}' involves the following steps: {steps_formatted}.\n")

            print(all_tasks)

            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"I wanted to generate instructions for training my AI assistant based which is designed for {user_role} developer. Sample example for React Native developer: You are ReactDev an assistant with the knowledge of React Native, Javascript and Redux. You are expert in developing codes and building logic, also you are expert in fixing faulty codes and provide assistance to the user. So, Generate a short template for the prompt don't provide any commentary or explanation for the "}
                ]
            )

            half_prompt = completion.choices[0].message.content

            print(half_prompt)

            final_prompt = half_prompt + '\n' + "You generally performs tasks like\n"+all_tasks
            print(final_prompt)

            dev_resp = json.loads(dev_json_data)
            dev_resp

            updated_resp = json.loads(get_updated_response)
            updated_resp

            # Create a function to check if an object from dev_json_data is present in get_updated_response
            def match_found(dev_obj, updated_response):
                for response_obj in updated_response:
                    if all(dev_obj[key] == response_obj.get(key) for key in response_obj.keys()):
                        return True
                return False

            # Create a new JSON object with matching objects
            matched_objects = [obj for obj in dev_resp if match_found(obj, updated_resp)]

            print(matched_objects)

            # Set up the API endpoint URL
            url = "http://35.85.112.192/api/ai_task_api"

            # Headers as specified
            headers = {
                'Accept': 'application/json',
                'X-API-KEY': 'JGIp4AWFmI',
                'Content-Type': 'application/json',
            }

            # Example data to be sent in the body of the POST request
            data = {
                'user_id': user_id,
                'rolename': user_role,
                'task': task_updated,
                'instruction': final_prompt
            }

            print('Data to be sent is ' , json.dumps(data))

            data

            # Make the POST request
            response = requests.post(url, json=data, headers=headers)

            # Print the response from the server
            print(response.text)
            
            return {"id": user_id,"user_role": user_role,"task":task_updated,"instruction":final_prompt,"status": "success"}
        
        else:
            print('No futher execution')
            return {"status":"failed"}
        
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

@app.post("/task-priority/")
async def task_priority(item: Item):
    user_id = item.id
    temp_data = json.loads(item.prompt)
    user_prompt = temp_data["prompt"]
    assistant_id = temp_data["assistant_id"]
    mode = temp_data["mode"]
    thread_id = temp_data["thread_id"]
    user_data = item.data
    
    dir_path = create_user_directory(user_id)
    print(dir_path)
    
    # uploading figma api data to vector store
    def upload_file_to_vector_store(filePath1):
        # Create a vector store caled "Financial Statements"
        vector_store = client.beta.vector_stores.create(name="Document Files")

        # Ready the files for upload to OpenAI
        file_paths = [filePath1]
        file_streams = [open(path, "rb") for path in file_paths]

        # Use the upload and poll SDK helper to upload the files, add them to the vector store,
        # and poll the status of the file batch for completion.
        file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vector_store.id, files=file_streams
        )

        # You can print the status and the file counts of the batch to see the result of this operation.
        print(file_batch.status)
        print(file_batch.file_counts)
        print(vector_store.id)

        return vector_store.id

    def update_assistant(vectorId,assistantID):
        assistant = client.beta.assistants.update(
            assistant_id=assistantID,
            tool_resources={"file_search": {"vector_store_ids": [vectorId]}},
        )
        return assistant
    
    if assistant_id =="":
        #Prompt2
        my_assistant = client.beta.assistants.create(
            instructions="You are Task Priority Assist, an AI assistant that will help user to find the priority task.",
            name="Task Priority Assist",
            tools=[{"type": "file_search"}],
            model="gpt-4o",
        )
        print("New assistant is created: ",my_assistant)
        assistant_id = my_assistant.id
    else:
        print("Assistant is already created")
    
    file_path = dir_path + "/sample.json"
    if user_data != "":
        user_json_obj = json.loads(user_data)
        print(user_json_obj)
        
        with open(file_path, 'w') as json_file:
            json.dump(user_json_obj, json_file, indent=4) 
            
        vector_id = upload_file_to_vector_store(file_path)
        updated_assistant = update_assistant(vector_id,assistant_id)
        print("New files are added: ",updated_assistant)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
            else:
                print(f"File not found: {file_path}")
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")
    else:
        print("No need to add files to assistant")

    if thread_id == "":
        thread = client.beta.threads.create()
        thread_id = thread.id
        print("New thread is created: ",thread_id)
    else:
        print("No need to create a thread")
    
    response = get_response(thread_id,assistant_id,user_prompt)
    print(response)
    
    def update_assistant(instruct):
        assistant = client.beta.assistants.update(
            assistant_id=assistant_id,
            instructions=instruct
        )   
        return assistant
    
    if user_prompt != "":
        if mode == "meet":
            inst = f"You are Task Priority Assist, an AI assistant that will fetch the meetings for the asked date. Format the response as follows:\n\nTitle: [Title of the meeting]\nDescription: [Description of the meeting (if provided, otherwise exclude this line)]\nTime: [Date and time of the meeting]\nRequirements for the meeting: [Details about what is required or the purpose of the meeting]\nLink: [Link to the meeting (if provided, otherwise exclude this line)]"
            updated_assistant = update_assistant(inst)
            print("updated_assistant: ",updated_assistant)
        else:
            inst = "You are Task Priority Assist, an AI assistant that will help user to find the priority task."
            updated_assistant = update_assistant(inst)
            print("updated_assistant: ",updated_assistant)
        
        if mode == "meet":     
            response = get_response(thread_id,assistant_id,user_prompt)
        else:
            prompt_1 = "Given the task data, structure each task with the following details one below the other:\n Title, Description : Summarize the description in short, Source : App name, Due date, Link to open task : Add the URL from the data."
            response_1 = get_response(thread_id,assistant_id,prompt_1)
            print(response_1)
            
            prompt_2 = "Only show tasks that have not been marked as completed, dev complete, test complete etc or the messages which contains tasks and does not have any replies indicating the completion of tasks."
            response_2 = get_response(thread_id,assistant_id,prompt_2)
            print(response_2)
            
            prompt_3 = "User's Prompt - "+ user_prompt + "\n Filter the tasks to display only those that have the due date specified in the user's prompt and priority is urgent or high, along with any overdue tasks.\n If no due date is found for any task, check the priority level. If no priority is found, then check the due date. \n If two or fewer tasks are found, include tasks from the next day or the next week or the tasks which do not have any due date or priority mentioned."
            response_3 = get_response(thread_id,assistant_id,prompt_2)
            print(response_3)
            
            response = response_3
        
        delete_folder_recursive(dir_path)
            
        helping_data = {"prompt":user_prompt,"mode":mode,"api_data":json.dumps(user_data,indent=4,ensure_ascii=False)}
    
        if response == "Failed":
            store_error(user_id,"/task-priority/","assistant api failed to generate response")
            return {"status":"failed","data":json.dumps(helping_data,indent=4,ensure_ascii=False)}
        return {"status":"success","assistant_id":assistant_id,"thread_id":thread_id,"response":response,"data":json.dumps(helping_data,indent=4,ensure_ascii=False)}
    else:
        store_error(user_id,"/task-priority/","Prompt is not entered")
        return {"status":"failed","exception":"Prompt is not entered"}
    
@app.post("/figma-custom-ui/")
async def figma_custom_ui(item: Item):
    user_id = item.id
    temp_api_data = json.loads(item.data)
    
    api_data = temp_api_data["figma_data"]
    
    added_requirements = temp_api_data["added_requirements"]
    
    
    temp_data = item.prompt
    data_json_obj = json.loads(temp_data)
    print(data_json_obj)
    
    image_url = data_json_obj["image_url"]
    user_role = data_json_obj["user_role"]
    
    print(image_url)
    api_json_data = json.loads(api_data)
    print(api_json_data)
    
    dir_path = create_user_directory(user_id)
    print(dir_path)
    
    openai_api_key = os.getenv('OPENAI_API_KEY')
    print('Open ai api key: '+openai_api_key)
    client = OpenAI(api_key=openai_api_key)

    def write_code_to_file(filename: str, code: str) -> None:
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(code)
    
    
    # Prompt 1
    response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Generate functional requirements of the UI having each component functionality explanation"},
                {
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                },
                },
            ],
        }
    ],
    max_tokens=500,
    )

    response_1 = response.choices[0].message.content
    print(response_1)

    
         
    #Prompt2
    my_assistant = client.beta.assistants.create(
        instructions="You are an helpful assistant",
        name="Good Assistant",
        tools=[{"type": "file_search"}],
        model="gpt-4o",
    )
    print(my_assistant)

    def upload_document_file_to_openai(filepath):
        uploaded_file = client.files.create(
            file=open(filepath, "rb"),
            purpose="assistants"
        )
        return uploaded_file.id
    
    # api_converted_data = get_analyzed_api_data(api_data,image_url)
    # Specify the filename
    # filename = 'figma_data_file.txt'

    # Writing JSON data to a file
    # with open(filename, 'w') as file:
    #     json.dump(api_converted_data, file)
    
    if api_data != "":
        try:
            with open(dir_path+"/figma_data_file.json", 'w') as file:
                json.dump(json.loads(api_data), file)
                print("API JSON data was successfully written to file")
        except:
            print("Some error occured while uploadng the data")

    # uploading figma api data to vector store
    def upload_file_to_vector_store(filePath1,filePath2):
        # Create a vector store caled "Financial Statements"
        vector_store = client.beta.vector_stores.create(name="Document Files")

        # Ready the files for upload to OpenAI
        if filePath2 == "":
            file_paths = [filePath1]
        else:
            file_paths = [filePath1,filePath2]
            
        file_streams = [open(path, "rb") for path in file_paths]

        # Use the upload and poll SDK helper to upload the files, add them to the vector store,
        # and poll the status of the file batch for completion.
        file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vector_store.id, files=file_streams
        )

        # You can print the status and the file counts of the batch to see the result of this operation.
        print(file_batch.status)
        print(file_batch.file_counts)
        print(vector_store.id)

        return vector_store.id
    
    if added_requirements != "":
        write_code_to_file(dir_path+"/More requirements.txt",added_requirements)
        print("More requirements was successfully made")
        vector_id = upload_file_to_vector_store(dir_path+"/figma_data_file.json",dir_path+"/More requirements.txt")
    else:
        vector_id = upload_file_to_vector_store(dir_path+"/figma_data_file.json","")

    assistant_id = my_assistant.id
    # print("file_id: ",file_id)
    print("assistant_id: ",assistant_id)
    print("vector_id: ",vector_id)
    
    thread = client.beta.threads.create()
    thread_id = thread.id
    print(thread_id)

    #storing the user styles data files
    status_project_code = store_project_data_locally(user_id,dir_path)
    
    styles_id = ""
    if status_project_code["status"] == "success":
        print("Uploading the styles file to vector store")
        created_file = client.files.create(
            file=open(dir_path+"/project_style_data.json", "rb"),
            purpose="assistants"
        )
        print("created_file: ",created_file)
        
        styles_id = created_file.id
        
        vector_store_file = client.beta.vector_stores.files.create(
            vector_store_id=vector_id,
            file_id=styles_id
        )
        print(vector_store_file)
        print("File was successfully uploaded")
        
    def update_assistant(vectorId):
        assistant = client.beta.assistants.update(
            assistant_id=assistant_id,
            tool_resources={"file_search": {"vector_store_ids": [vectorId]}},
        )   
        return assistant

    updated_assistant = update_assistant(vector_id)
    print(updated_assistant)
    
    figma_info = "absoluteBoundingBox: Describes the absolute position and size of the element in the frame, here position is given in the form  of x and y coordinated with respect to the screen, so place the components at proper positions"
    if user_role == "Flutter Developer":
        print("In flutter dev prompt mode")
        payload = [
            {
                "type": "text",
                "text": f"Generate a {user_role} code with MVVM architecture and proper State Management for the figma UI based on UI image, figma styling data(note: figma data is uploaded in figma_data_file.json file) and  Description of the UI: \" {response_1} \".\nMake separate files for reusable components, classes, and assets. Also maintain Colors and Strings as a reusable component. \n Some information about figma data is: \n  Note: The colors in figma API data is in the form of RGBA format so add accurate colors in code \n{figma_info} \n Ensure the code includes: 1. Proper error handling for each code file and method. 2. Proper commenting so that every non-coder can also understand the code. 3. Provide only the exact code with file and folder names. No need to generate code for Status bar showing battery, time, etc."
            },
            {
                "type": "image_url",
                "image_url": {"url": image_url}
            }
        ]
    else:
        print("Not in Flutter Dev Mode")
        if status_project_code["status"] == "success":
            print("Using the styles data")
            styles_prompt = "User coding styles data is present in project_style_data.json, create response according to it."
            payload = [{"type": "text", "text": f"Generate a {user_role} code for the figma UI based on UI image, figma styling data(note: figma data is uploaded in figma_data_file.json file) and  Description of the UI: \" {response_1} \".\nMake separate files for reusable components, classes and asssets.\n Some information about figma data is: \n  Note: The colors in figma api data is in the form of RGBA fromat so make so add accurate colors in code \n{figma_info} \n {styles_prompt}\n Ensure the code includes: 1. Proper error handling for each code file and method. 2. Proper commenting so that every non-coder can also understand the code. 3. Provide only the exact code with file and folder names. No need to generate code for Status bar showing battery, time, etc."},{"type": "image_url","image_url": {"url": image_url}}]
        else:
            print("Not using the styles data")
            payload = [{"type": "text", "text": f"Generate a {user_role} code for the figma UI based on UI image, figma styling data(note: figma data is uploaded in figma_data_file.json file) and  Description of the UI: \" {response_1} \".\nMake separate files for reusable components, classes and asssets.\n Some information about figma data is: \n  Note: The colors in figma api data is in the form of RGBA fromat so make so add accurate colors in code \n{figma_info} \n Ensure the code includes: 1. Proper error handling for each code file and method. 2. Proper commenting so that every non-coder can also understand the code. 3. Provide only the exact code with file and folder names. No need to generate code for Status bar showing battery, time, etc."},{"type": "image_url","image_url": {"url": image_url}}]
    # payload = [{"type": "text", "text": f"Generate a {user_role} code for the figma UI based on UI image, figma styling data(note: figma data is uploaded in figma_data_file.json file)\".\nMake separate files for reusable components, classes and asssets.\n Some information about figma data is: \n  Note: The colors in figma api data is in the form of RGBA fromat so make so add accurate colors in code \n{figma_info}"},{"type": "image_url","image_url": {"url": image_url}}]
    # store_error(user_id,"/figma-custom-ui/","assistant api failed to generate response")
    response_2 = get_response(thread_id,assistant_id,payload)
    print(response_2)

    # #deleting the uploaded image file
    def delete_openai_files(file_id):
        deleted_image_file = client.files.delete(file_id)
        return deleted_image_file
        
    vector_store_files = client.beta.vector_stores.files.list(
        vector_store_id=vector_id
    )
    print(vector_store_files)


    file_id = vector_store_files.data[0].id
    print(file_id)

    deleted_vector_store_file = client.beta.vector_stores.files.delete(
        vector_store_id=vector_id,
        file_id=file_id
    )
    print(deleted_vector_store_file)

    file_id = upload_document_file_to_openai("Common_Functionality.json")
    print(file_id)

    vector_store_file = client.beta.vector_stores.files.create(
        vector_store_id=vector_id,
        file_id=file_id
    )
    print(vector_store_file)
    
    # prompt 3
    
    print(thread_id)

    if added_requirements=="":
        payload = [{"type": "text", "text": f"Generate logic code for every interactable element and modify the code. The output must be fully functional. Generate logic for the code by yourself don't expect from user. I had uploaded some common functionality steps in the file you can refer from there to create functionality logic for component of UI. No need to add any additional functionality into the code, generate functionality for elements that are already present in the ui. Also see the uploaded ui image and correct the position of any component which is wrong. \n Ensure the code includes: 1. Proper error handling for each code file and method. 2. Proper commenting so that every non-coder can also understand the code. 3. Provide only the exact code with file and folder names 4. No need to generate code for Status bar showing battery, time, etc."},{"type": "image_url","image_url": {"url": image_url}}]
        print("Generating prompt 3 normally")
    else:
        payload = [{"type": "text", "text": f"Generate logic code for every interactable element and modify the code. The output must be fully functional. Generate logic for the code by yourself don't expect from user. I had uploaded some common functionality steps in the Common_Functionality.json and also some specific requirements in More requirements.txt file you can refer from there to create functionality logic for component of UI. No need to add any additional functionality into the code, generate functionality for elements that are already present in the ui. Also see the uploaded ui image and correct the position of any component which is wrong. \n Ensure the code includes: 1. Proper error handling for each code file and method. 2. Proper commenting so that every non-coder can also understand the code. 3. Provide only the exact code with file and folder names 4. No need to generate code for Status bar showing battery, time, etc."},{"type": "image_url","image_url": {"url": image_url}}]
        print("Generating prompt 3 for ui image")
    response_3 = get_response(thread_id,assistant_id,payload)
    print(response_3)

    def remove_locally_stored_files(file_path):
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
            else:
                print(f"File not found: {file_path}")
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")
        
    remove_locally_stored_files(dir_path+"/More requirements.txt")
    remove_locally_stored_files(dir_path+"/figma_data_file.json")
        
    deleted_document_file = delete_openai_files(file_id)
    print(deleted_document_file)
    
    if styles_id!="":
        deleted_document_file = client.files.delete(styles_id)
        print(deleted_document_file)
    
    #deleting the uploaded vector, so that i can create a new one
    deleted_vector_store = client.beta.vector_stores.delete(
    vector_store_id=vector_id
    )
    print(deleted_vector_store)

    response = client.beta.assistants.delete(assistant_id)
    print(response)
    
    helping_data = {"figma_data":api_data,"user_requirements_data":added_requirements}
    
    delete_folder_recursive(dir_path)
    
    if response_3=="Failed":
        store_error(user_id,"/figma-custom-ui/","assistant api failed to generate response")
        return {"status":"failed","data":json.dumps(helping_data,indent=4,ensure_ascii=False)}
    else:
        return {"status":"success","response":response_3,"data":json.dumps(helping_data,indent=4,ensure_ascii=False)}

    
@app.post("/new-functionalities/")
async def new_functionalities(item: Item):
    user_id = item.id
    code_data = item.data
    
    temp_data = item.prompt
    data_json_obj = json.loads(temp_data)
    print(data_json_obj)
    
    image_url = data_json_obj["image_url"]
    user_role = data_json_obj["user_role"]
    
    print(image_url)
    
    client = OpenAI(api_key="sk-proj-csMstUJ72UbVhyIBeE5ET3BlbkFJyTfXZ9fEZ4eC5N14i4X8")

    #Prompt 1
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
            "role": "user",
            "content": [
                {"type": "text", "text": "Generate functional requirements of the UI having each component functionality explanation with proper positioning"},
                {
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                },
                },
            ],
            }
        ],
        max_tokens=1000,
    )

    response_1 = response.choices[0].message.content
    print(response_1)

    # Download the image
    response = requests.get(image_url)
    if response.status_code == 200:
        # Open the image using Pillow
        image = Image.open(BytesIO(response.content))

        # Convert and save the image as PNG
        image.save("output.png", format="PNG")
        print("Image saved as output.png")
    else:
        print("Failed to retrieve the image")

    # uploading ui image to open ai
    def upload_image_file_to_openai(filepath):
        uploaded_file = client.files.create(
            file=open(filepath, "rb"),
            purpose="vision"
        )
        return uploaded_file.id
    
    file_id = upload_image_file_to_openai("output.png")
    print(file_id)

    image_file_id = file_id

    # Define the file path
    file_path = "code.txt"

    # Open the file in write mode and write the string
    with open(file_path, "w") as file:
        file.write(code_data)

    print(f"File created successfully at {file_path}")
    
    # uploading figma api data to vector store
    def upload_file_to_vector_store(filePath1):
        # Create a vector store caled "Financial Statements"
        vector_store = client.beta.vector_stores.create(name="Document Files")

        # Ready the files for upload to OpenAI
        file_paths = [filePath1]
        file_streams = [open(path, "rb") for path in file_paths]

        # Use the upload and poll SDK helper to upload the files, add them to the vector store,
        # and poll the status of the file batch for completion.
        file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vector_store.id, files=file_streams
        )

        # You can print the status and the file counts of the batch to see the result of this operation.
        print(file_batch.status)
        print(file_batch.file_counts)
        print(vector_store.id)

        return vector_store.id

    vector_id = upload_file_to_vector_store("code.txt")

    #Prompt2
    my_assistant = client.beta.assistants.create(
        instructions="You are an helpful assistant",
        name="Good Assistant",
        tools=[{"type": "file_search"}],
        model="gpt-4o",
    )
    print(my_assistant)

    assistant_id = my_assistant.id
    print("file_id: ",file_id)
    print("assistant_id: ",assistant_id)
    print("vector_id: ",vector_id)
    
    thread = client.beta.threads.create()
    thread_id = thread.id
    print(thread_id)

    def update_assistant(vectorId):
        assistant = client.beta.assistants.update(
            assistant_id=assistant_id,
            tool_resources={"file_search": {"vector_store_ids": [vectorId]}},
        )
        return assistant

    updated_assistant = update_assistant(vector_id)
    print(updated_assistant)
    

    payload = [{"type": "text", "text": f"Please modify the code inside the uploaded file to align with the design and layout specifications shown in the provided UI image. Ensure the following:\n\n1. Adapt the visual elements, colors, and layout as per the UI image.\n2. Verify that all interactive elements (buttons, forms, etc.) work correctly according to the new design.\n\nBelow is the functional description of the UI image {response_1}"},{"type": "image_file","image_file": {"file_id": file_id}}]
    response_2 = get_response(thread_id,assistant_id,payload)
    
    print(response_2)
    
    #deleting the uploaded image file
    def delete_openai_files(file_id):
        deleted_image_file = client.files.delete(file_id)
        return deleted_image_file

    deleted_image_file = delete_openai_files(image_file_id)
    print(deleted_image_file)

    #deleting the uploaded vector, so that i can create a new one
    deleted_vector_store = client.beta.vector_stores.delete(
        vector_store_id=vector_id
    )
    print(deleted_vector_store)

    response = client.beta.assistants.delete(assistant_id)
    print(response)
    
    if response_2=="Failed":
        store_error(user_id,"/new-functionalities/","assistant api failed to generate response")
        return {"status":"failed"}
    else:
        return {"status":"success","response":response_2}

@app.post("/detect-file-name/")
async def detect_file_name(item: Item):
    prompt = item.prompt
    
    # Load the spaCy model
    nlp = spacy.load('en_core_web_sm')
    
    def extract_file_name(statement):
        # Define a regex pattern to find file names with extensions
        file_name_pattern = r'\b\w+\.\w+\b'
        
        # Search for the pattern in the statement
        file_name_match = re.search(file_name_pattern, statement)
        
        if file_name_match:
            return file_name_match.group()
        else:
            # If no extension is found, use the current NLP logic to find potential file names
            doc = nlp(statement)
            possible_file_names = []

            for token in doc:
                # Collect proper nouns (potential file names without extensions)
                if token.pos_ == 'PROPN':
                    possible_file_names.append(token.text)
            
            # Check for keywords and the words following them
            keywords = ["file name", "document", "file"]
            words = statement.split()
            for i, word in enumerate(words):
                if any(keyword in word.lower() for keyword in keywords):
                    # Collect words following the keyword until a non-capitalized word, keyword, or stop word is encountered
                    file_name_parts = []
                    for j in range(i + 1, len(words)):
                        if any(keyword in words[j].lower() for keyword in keywords) or words[j].lower() in ["in", "which", "written", "is", "the"]:
                            break
                        file_name_parts.append(words[j])
                    if file_name_parts:
                        return ' '.join(file_name_parts)
            
            if possible_file_names:
                return ' '.join(possible_file_names)
            else:
                # Fallback: consider the last word if no proper nouns are found
                if words:
                    return words[-1]
                else:
                    return None

    file_name = extract_file_name(prompt)
    print("file_name: "+file_name)
    
    return {"status":"success","file_name":file_name}

@app.post("/multiple-files-flow/")
async def multiple_files_flow(item: Item):
    # 3 use cases: Update functionality, add new functionality, add new screen
    user_id = item.id
    temp_data = json.loads(item.prompt)
    temp_api_data = json.loads(item.data)
    
    api_data = temp_api_data["project_code"]
    additional_requirements = temp_api_data["added_requirements"]
    figma_api_data = temp_api_data["figma_data"]
    extra_data = temp_api_data["extra_data"]
    
    prompt = temp_data["prompt"]
    user_role = temp_data["user_role"]
    image_url = temp_data["image_url"]
    assistant_id = temp_data["assistant_id"]
    thread_id = temp_data["thread_id"]

    print("prompt: ", prompt)
    print("user_role: ", user_role)
    print("image_url: ", image_url)
    print("assistant_id: ", assistant_id)
    print("thread_id: ", thread_id)
    print("additional_requirements: ",additional_requirements)
    print("api_data: ",api_data)
    print("figma_api_data: ",figma_api_data)    
    
    dir_path = create_user_directory(user_id)
    print(dir_path)
            
    if not assistant_id:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Generate a description for a {user_role} developer."}
            ]
        )
        role_description = completion.choices[0].message.content
        print(role_description)

        
        my_assistant = client.beta.assistants.create(
            instructions=f"{role_description}\nGenerate {user_role} code based on user prompt.\nRemember:\n1. Generate output by studying the uploaded files\n2. Uploaded  Readme.txt contain the proper files and folder structure\n3. Link the output to existing project files\n4. Generate code with proper file and folder name",
            name=f"{user_role} Code Assist",
            tools=[{"type": "file_search"}],
            model="gpt-4o",
        )
        assistant_id = my_assistant.id
        print("assistant_id: ", assistant_id)

    def remove_extension(filename: str) -> str:
        return filename.rsplit('.', 1)[0]

    def write_code_to_file(filename: str, code: str) -> None:
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(code)

    def upload_file_to_vector_store(file_name, vector_id):
        with open(file_name, "rb") as file:
            try:
                uploaded_file = client.files.create(file=file, purpose="assistants")
                file_id = uploaded_file.id
                uploaded_file_ids.append(file_id)
                vector_store_file = client.beta.vector_stores.files.create(vector_store_id=vector_id, file_id=file_id)
                print(file_name + " was successfully stored")
                print(vector_store_file)
            except:
                print("Not able to store file ",file_name)

    def get_file_name(file_path):
        return os.path.basename(file_path.replace('\\', '/'))

    # Function to filter JSON objects
    def filter_json_objects(data,keyword):
        filtered_data = [obj for obj in data if keyword in obj['file_path'] or 'Readme' in obj['file_path']]
        return filtered_data
    
    isValid = False
    try:
        api_data_json_1 = json.loads(api_data)
        #condition to reduce the length of data
        api_data_json = []
        if "flutter" in user_role.lower():
            api_data_json =  filter_json_objects(api_data_json_1,"lib")
        elif "react native" in user_role.lower():
            api_data_json =  filter_json_objects(api_data_json_1,"src")
        else:
            api_data_json = api_data_json_1  
        
        isValid = True
        print("It's a valid JSON")

        for files in api_data_json:
            file_name = get_file_name(files["file_path"])
            files["file_path"] = file_name
        print(api_data_json)
        
    except ValueError as e:
        print("Invalid JSON: ", e)

    if isValid:
        multiple_file_name = []
        uploaded_file_ids = []
        for item in api_data_json:
            code = item['content']
            filename = dir_path+'/'+remove_extension(item['file_path']) + '.txt'
            multiple_file_name.append(filename)
            write_code_to_file(filename, code)
            print(f"Code written to {filename}")
        
        store_name = "Uploaded files to Store"
        vector_store = client.beta.vector_stores.create(name=store_name)
        vector_id = vector_store.id

        for files in multiple_file_name:
            upload_file_to_vector_store(files, vector_id)
            
        upload_file_to_vector_store("Common_Functionality.json",vector_id)
        # uploaded_file_ids.append(file_id)
        print("Common Functionality file was successfully uploaded")
        
        #creating and uploading the additional functionality file to the
        if additional_requirements!="":
            try:
                write_code_to_file(dir_path+"/More requirements.txt",additional_requirements)
                upload_file_to_vector_store(dir_path+"/More requirements.txt",vector_id)
                multiple_file_name.append(dir_path+"/More requirements.txt")
                print("More requirements file was successully uploaded")
            except:
                print("Error uploading more requirements file")
            
        assistant = client.beta.assistants.update(
            assistant_id=assistant_id,
            tool_resources={"file_search": {"vector_store_ids": [vector_id]}},
        )
        print("updated assistant: ", assistant)

    else:
        print("No files uploaded")
        if additional_requirements != "":
            try:
                write_code_to_file(dir_path + "/More requirements.txt",additional_requirements)
                upload_file_to_vector_store(dir_path + "/More requirements.txt",vector_id)
                # multiple_file_name.append("More requirements.txt")
                print("More requirements file was successully uploaded")
                assistant = client.beta.assistants.update(
                    assistant_id=assistant_id,
                    tool_resources={"file_search": {"vector_store_ids": [vector_id]}},
                )
                print("updated assistant: ", assistant)
            except:
                print("Error uploading more requirements file")        

    if not thread_id:
        empty_thread = client.beta.threads.create()
        thread_id = empty_thread.id
        print(empty_thread)
    else:
        print("Thread is already created: ", thread_id)

    def upload_image_file_to_openai(filepath):
        with open(filepath, "rb") as file:
            uploaded_file = client.files.create(file=file, purpose="vision")
            return uploaded_file.id
    ###############################################################
    if figma_api_data!="":
        try:
            ## figma_json = json.loads(figma_api_data)
            # api_converted_data = get_analyzed_api_data(figma_api_data,image_url)
            # with open("figma_api_data.txt", 'w') as file:
            #     json.dump(api_converted_data, file)
            
            with open(dir_path+"/figma_api_data.json", 'w') as file:
                json.dump(json.loads(figma_api_data), file)
                print("API JSON data was successfully written to file")

            upload_file_to_vector_store(dir_path+"/figma_api_data.json",vector_id)
            print("Figma API data file was successully uploaded")
            assistant = client.beta.assistants.update(
                assistant_id=assistant_id,
                tool_resources={"file_search": {"vector_store_ids": [vector_id]}},
            )
            print("updated assistant: ", assistant)
                
        except:
            print("Error while uploading figma_api_data file to assistant api")
    ###############################################################
    image_id = ""
    if image_url:        
        #downloading the image
        response = requests.get(image_url)
        
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            image.save(dir_path+"/output.png", format="PNG")
            print("Image saved as output.png")
        else:
            print("Failed to retrieve the image")

        image_id = upload_image_file_to_openai(dir_path+"/output.png")
        print("Image successfully uploaded to OpenAI: ", image_id)
        
        if figma_api_data=="":
            print("Not using figma data")
            payload_1 = [{"type": "text", "text": f"{prompt} \n Don't provide any unwanted explanation or commentary, give me only exact code with file and folder name."},{"type": "image_url","image_url": {"url": image_url}}]
        else:
            print("Using figma data")
            figma_info = "absoluteBoundingBox: Describes the absolute position and size of the element in the frame, here position is given in the form of x and y coordinates with respect to the screen, so place the components at proper positions"
            if user_role == "Flutter Developer":
                print("Flutter Developer mode output")
                payload_1 = [{"type": "text", "text": f"{prompt}. Figma API data for the UI is uploaded in the figma_api_data.json file you can refer there to understand ui in more detail. \n Some information about figma data is: \n  Note: The colors in figma api data is in the form of RGBA fromat so make so add accurate colors in code \n{figma_info} \n The exisiting code follows MVVM architecture and proper State Management so generate ouput according to it. Make separate files for reusable components, classes, and assets.Also maintain Colors and Strings as a reusable component. \n Don't provide any unwanted explanation or commentary, give me only exact code with file and folder name. Integrate the generated response with the existing uploaded code"}, {"type": "image_url","image_url": {"url": image_url}}]
            else:
                print("Normal output mode")
                payload_1 = [{"type": "text", "text": f"{prompt}. Figma API data for the UI is uploaded in the figma_api_data.json file you can refer there to understand ui in more detail. \n Some information about figma data is: \n  Note: The colors in figma api data is in the form of RGBA fromat so make so add accurate colors in code \n{figma_info} \n Don't provide any unwanted explanation or commentary, give me only exact code with file and folder name. Integrate the generated response with the existing uploaded code"}, {"type": "image_url","image_url": {"url": image_url}}]
        ###############################################################
        
        response_1 = get_response(thread_id, assistant_id, payload_1)
        print(response_1)
        if response_1 == "Failed":
            print("Here image uploading via url is failed, trying a backup route.")
            if figma_api_data=="":
                print("Not using figma data")
                payload_1 = [{"type": "text", "text": f"{prompt} \n Don't provide any unwanted explanation or commentary, give me only exact code with file and folder name."},{"type": "image_file","image_file": {"file_id": image_id}}]
            else:
                print("Using figma data")
                figma_info = "absoluteBoundingBox: Describes the absolute position and size of the element in the frame, here position is given in the form of x and y coordinates with respect to the screen, so place the components at proper positions"
                payload_1 = [{"type": "text", "text": f"{prompt}. Figma API data for the UI is uploaded in the figma_api_data.json file you can refer there to understand ui in more detail. \n Some information about figma data is: \n  Note: The colors in figma api data is in the form of RGBA fromat so make so add accurate colors in code \n{figma_info} \n Don't provide any unwanted explanation or commentary, give me only exact code with file and folder name."}, {"type": "image_file","image_file": {"file_id": image_id}}]
            
            response_1 = get_response(thread_id, assistant_id, payload_1)
            print(response_1)
            
        payload_2 = ""
        
        if additional_requirements =="":
            payload_2 = [{"type": "text", "text": f"Generate logic code for every interactable element and modify the code. The output must be fully functional. Generate logic for the code by yourself don't expect from user. I had uploaded some common functionality steps in the Common_Functionality.json file you can refer from there to create functionality logic for component of UI. No need to add any additional functionality into the code, generate functionality for elements that are already present in the ui.\nAlso see the uploaded ui image and correct the position of any component which is wrong\n Don't provide any unwanted explanation, give me only exact code with comments. Ensure to follow file and folder name."},{"type": "image_url","image_url": {"url": image_url}}]
        else:
            payload_2 = [{"type": "text", "text": f"Generate logic code for every interactable element and modify the code. The output must be fully functional. Generate logic for the code by yourself don't expect from user. I had uploaded some common functionality steps in the Common_Functionality.json and also some specific requirements in More requirements.txt file you can refer from there to create functionality logic for component of UI. No need to add any additional functionality into the code, generate functionality for elements that are already present in the ui.\nAlso see the uploaded ui image and correct the position of any component which is wrong\n Don't provide any unwanted explanation, give me only exact code with comments. Ensure to follow file and folder name."},{"type": "image_url","image_url": {"url": image_url}}]

        
        response_2 = get_response(thread_id, assistant_id, payload_2)
        print(response_2)
        if response_2 == "Failed":
            print("Here image uploading via url is failed, trying a backup route.")
            if additional_requirements =="":
                payload_2 = [{"type": "text", "text": f"Generate logic code for every interactable element and modify the code. The output must be fully functional. Generate logic for the code by yourself don't expect from user. I had uploaded some common functionality steps in the Common_Functionality.json file you can refer from there to create functionality logic for component of UI. No need to add any additional functionality into the code, generate functionality for elements that are already present in the ui.\nAlso see the uploaded ui image and correct the position of any component which is wrong\n Don't provide any unwanted explanation, give me only exact code with comments. Ensure to follow file and folder name."},{"type": "image_file","image_file": {"file_id": image_id}}]
            else:
                payload_2 = [{"type": "text", "text": f"Generate logic code for every interactable element and modify the code. The output must be fully functional. Generate logic for the code by yourself don't expect from user. I had uploaded some common functionality steps in the Common_Functionality.json and also some specific requirements in More requirements.txt file you can refer from there to create functionality logic for component of UI. No need to add any additional functionality into the code, generate functionality for elements that are already present in the ui.\nAlso see the uploaded ui image and correct the position of any component which is wrong\n Don't provide any unwanted explanation, give me only exact code with comments. Ensure to follow file and folder name."},{"type": "image_file","image_file": {"file_id": image_id}}]            
            
            response_2 = get_response(thread_id, assistant_id, payload_2)
            print(response_2)
            
        response = response_2
        
    else:
        if additional_requirements == "":
            payload = f"{prompt} \n Don't provide any unwanted explanation or commentary, give me only exact code with file and folder name."
        else:
            payload = prompt + " .Refer to some requirements in uploaded More requirements.txt file \n Don't provide any unwanted explanation or commentary, give me only exact code with file and folder name."            
        
        response = get_response(thread_id, assistant_id, payload)
        print(response)

    def delete_local_file(file_path):
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
            else:
                print(f"File not found: {file_path}")
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")
            
    delete_local_file(dir_path+"/figma_api_data.json")
    delete_local_file(dir_path+"/output.png")
    
    #deleting the image file
    if image_id:
        delete_image_file = client.files.delete(image_id)
        print(delete_image_file)
    else:
        print("No need to delete the images")
        
    helping_data = {"figma_data":figma_api_data,"user_requirements_data":additional_requirements,"apps_data":extra_data,"project_code":json.dumps(api_data_json, indent=4, ensure_ascii=False)}
    
    delete_folder_recursive(dir_path)
    
    if response != "Failed":    
        return {"status": "success", "assistant_id": assistant_id, "thread_id": thread_id, "response": response,"data":json.dumps(helping_data, indent=4, ensure_ascii=False)}
    else:
        return {"status":"failed","data":json.dumps(helping_data, indent=4, ensure_ascii=False)}

@app.post("/analyze-files/")
async def analyze_files(item: Item):
    api_data=item.data
    print(api_data)

    api_data_json = json.loads(api_data)
    print(api_data_json)

    #Prompt2
    my_assistant = client.beta.assistants.create(
        instructions="You are an AI assistant who responds with JSON only, without any explaining/describing text. You specialize in indexing and summarizing source code files for improved understanding and quick navigation. For the provided file content, generate a detailed structured summary and logic overview in JSON format as shown below:\n\njson\n{\n  \"file_path\": \"<file_path>\",\n  \"overview\": \"Detailed overview of the file.\",\n  \"classes\": [\n    {\n      \"name\": \"ClassName\",\n      \"description\": \"Detailed description of the class.\"\n    }\n  ],\n  \"functions\": [\n    {\n      \"name\": \"FunctionName\",\n      \"description\": \"Detailed description of the function and its logic.\",\n      \"parameters\": [\"param1\", \"param2\"],\n      \"returns\": \"Description of the return value.\"\n    }\n  ],\n  \"variables\": [\n    {\n      \"name\": \"VariableName\",\n      \"description\": \"Brief description of the variable.\"\n    }\n  ],\n  \"logic_overview\": [\n    {\n      \"logic\": \"Description of the logic(how the method works).\",\n      \"location\": \"Where the logic is used in the file.\"\n    }\n  ]\n}\nAdditionally, generate second json object to capture the user\'s coding style and preferences for future reference. This JSON should include information about component reuse, preferred libraries, coding patterns, and any other relevant aspects of the user\'s coding style:\n\njson\n{\n  \"user_style\": {\n    \"component_reuse\": \"Description of how the user reuses components or methods.\",\n    \"preferred_libraries\": [\"Library1\", \"Library2\"],\n    \"coding_patterns\": \"Description of the user\'s common coding patterns or techniques.\",\n    \"excluded_libraries\": [\"Library3\", \"Library4\"],\n    \"additional_notes\": \"Any other relevant information about the user\'s coding style.\"\n  }\n}\nFile Content:\n<file content>\n\nProvide the detailed structured summary and logic overview along with user\'s style in the JSON format as shown above.\nIMPORTANT: Provide only 2 json object, one for file overview and other for user style.\nALSO  FIND PROJECT NAME AND RETURN WITH RESPONSE IN TOP",
        name="AI Assistant",
        tools=[{"type": "file_search"}],
        model="gpt-4o",
    )
    print(my_assistant)
    
    assistant_id  = my_assistant.id
    print(assistant_id)

    def get_file_name(file_path):
            # Normalize the path to handle backslashes
            normalized_path = file_path.replace('\\', '/')
            return os.path.basename(normalized_path)

    def write_code_to_file(filename: str, code: str) -> None:
            with open(filename, 'w') as file:
                file.write(code)

    def remove_extension(filename: str) -> str:
            # Split the filename into name and extension
            name = filename.rsplit('.', 1)[0]
            return name

    uploaded_file_ids=[]

    def upload_file_to_vector_store(file_name,vector_id):
        file = client.files.create(
            file=open(file_name, "rb"),
            purpose="assistants"
        )
        file_id = file.id
        uploaded_file_ids.append(file_id)

        vector_store_file = client.beta.vector_stores.files.create(
            vector_store_id=vector_id,
            file_id=file_id
        )
        print(file_name + " was successfully stored")
        print(vector_store_file)

    multiple_file_name = []
    for item in api_data_json:
        code = item['content']
        filename = remove_extension(get_file_name(item['file_path'])) + '.txt'
        print(filename)
        multiple_file_name.append(filename)
        write_code_to_file(filename, code)
        print(f"Code written to {filename}")
        
    print(multiple_file_name)

    store_name = "Uploaded files to Store"
    vector_store = client.beta.vector_stores.create(
        name=store_name
    )

    print(vector_store)
    vector_id = vector_store.id

    for files in multiple_file_name:
        upload_file_to_vector_store(files,vector_id)

    print(str(len(multiple_file_name))+" was successfully uploaded to vector store "+store_name)

    assistant = client.beta.assistants.update(
        assistant_id=assistant_id,
        tool_resources={"file_search": {"vector_store_ids": [vector_id]}},
    )
    print("updated assistant: ",assistant)

    def delete_files_by_name(directory, filenames):
        for filename in filenames:
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
                else:
                    print(f"File not found: {file_path}")
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")

    directory = './'
    delete_files_by_name(directory, multiple_file_name)

    empty_thread = client.beta.threads.create()
    print(empty_thread)
    thread_id = empty_thread.id

    payload = "Index the uploaded source files"
    response_2 = get_response(thread_id,assistant_id,payload)
    print(response_2)
    
    if response_2 != "Failed":        
        json_strings = response_2.split('```json\n')[1:]
        
        json_list = []

        for json_str in json_strings:
            json_str = json_str.strip().rstrip('```')
            json_data = json.loads(json_str)
            json_list.append(json_data)
        
        final_json_str = json.dumps(json_list,indent = 2)

        print(final_json_str)

    #deleting the uploaded image file
    def delete_openai_files(file_id):
        deleted_image_file = client.files.delete(file_id)
        return deleted_image_file

    for ids in uploaded_file_ids:
        msg=delete_openai_files(ids)
        print(msg)

    deleted_vector_store = client.beta.vector_stores.delete(
        vector_store_id=vector_id
        )
    print(deleted_vector_store)

    response = client.beta.assistants.delete(assistant_id)
    print(response)
    
    if response_2 != "Failed":
        return {"status":"success","response":final_json_str}
    
    return {"status":"failed"}

def get_analyzed_api_data(api_data,image_url):
    print("image_url: ",image_url)
    print("api_data: ",api_data)
    
    if api_data != "":
        try:
            with open("api_data_file.json", 'w') as file:
                json.dump(json.loads(api_data), file)
                print("API JSON data was successfully written to file")
        except:
            print("Some error occured while uploadng the data")
    
    my_assistant = client.beta.assistants.create(
        instructions="Given a UI image and a corresponding Figma API response, analyze the UI elements in the image based on the details provided in the API response. For each UI element, provide the following details for each frame. Identify each frame with the text included in them: \n{\nPositioning and Sizing\nText Elements\nConstraints\nFills and Strokes\nCorner Radius and Effects\nFont Details\nText Content\nspacing\nstyles\n}\nthese details need to be fetched out from figma response and organize them in readable format",
        name="API Code Analyzer",
        tools=[{"type": "file_search"}],
        model="gpt-4o",
    )

    assistant_id = my_assistant.id
    print("assistant_id: ", assistant_id)

    store_name = "Uploaded files to Store"
    vector_store = client.beta.vector_stores.create(name=store_name)
    vector_id = vector_store.id
    print("vector_id: ",vector_id)
    
    def upload_file_to_vector_store(file_name,vector_id):
        file = client.files.create(
            file=open(file_name, "rb"),
            purpose="assistants"
        )
        file_id = file.id
        
        vector_store_file = client.beta.vector_stores.files.create(
            vector_store_id=vector_id,
            file_id=file_id
        )
        print(file_name + " was successfully stored")
        print(vector_store_file)
        return file_id
    
    file_id = upload_file_to_vector_store("api_data_file.json",vector_id)

    assistant = client.beta.assistants.update(
        assistant_id=assistant_id,
        tool_resources={"file_search": {"vector_store_ids": [vector_id]}},
    )
    print("updated assistant: ", assistant)

    empty_thread = client.beta.threads.create()
    thread_id = empty_thread.id
    print(empty_thread)

    payload = [{"type": "text", "text": "Anaylze the figma data"},{"type": "image_url","image_url": {"url": image_url}}]
    response_final = get_response(thread_id, assistant_id, payload)
    print(response_final)
    
    file_path = os.path.join("./","api_data_file.json")
    try:
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Deleted file: {file_path}")
        else:
            print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error deleting file {file_path}: {e}")

    print(file_id)

    try:
        deleted_vector_store_file = client.beta.vector_stores.files.delete(
            vector_store_id=vector_id,
            file_id=file_id
        )
        print(deleted_vector_store_file)
    except:
        print("Error while deleting the fle")

    try:
        deleted_vector_store = client.beta.vector_stores.delete(
            vector_store_id=vector_id
        )
        print(deleted_vector_store)
    except:
        print("Error while deleting the store")

    try:
        response = client.beta.assistants.delete(assistant_id)
        print(response)
    except:
        print("Error while deleting the assistant")
    
    return response_final
    
@app.post("/analyze-api-data/")
async def analyze_api_data(item: Item):
    image_url = item.prompt
    api_data = item.data
    
    response = get_analyzed_api_data(api_data,image_url)
    
    print(response)
    
    return {"status":"success","response":response}

@app.post("/compare-output-code/")
async def compare_output_code(item: Item):
    code = item.data
    figma_data = item.prompt
    
    my_assistant = client.beta.assistants.create(
        instructions="You are a comparision assistant whose task is to compare the data file with the output file and identify the hallucinated and missing elements, colors, font sizes, font styles, locations, positions, text and image placeholders that are present in the data file but not used in the code. Provide only these details without extra explanation, so the user can fix them manually.\nStart response by saying \"Missing Elements -\"",
        name=f"Comparision assistant",
        tools=[{"type": "file_search"}],
        model="gpt-4o",
    )
    print(my_assistant)

    assistant_id = my_assistant.id
    print("assistant_id: ",assistant_id)

    def write_code_to_file(filename: str, code: str) -> None:
        try:
            with open(filename, 'w') as file:
                file.write(code)
        except:
            print("Error while writing code to file")

    write_code_to_file("figma_data.txt",figma_data)
    write_code_to_file("code.txt",code)

    def upload_file_to_vector_store(file_name,vector_id):
        file = client.files.create(
            file=open(file_name, "rb"),
            purpose="assistants"
        )
        file_id = file.id

        vector_store_file = client.beta.vector_stores.files.create(
            vector_store_id=vector_id,
            file_id=file_id
        )
        print(file_name + " was successfully stored")
        print(vector_store_file)
        return file_id

    store_name = "Uploaded files to Store"
    vector_store = client.beta.vector_stores.create(
        name=store_name
    )

    print(vector_store)
    vector_id = vector_store.id

    figma_id = upload_file_to_vector_store("figma_data.txt",vector_id)
    code_id = upload_file_to_vector_store("code.txt",vector_id)
    print(figma_id)
    print(code_id)

    assistant = client.beta.assistants.update(
        assistant_id=assistant_id,
        tool_resources={"file_search": {"vector_store_ids": [vector_id]}},
    )
    print("updated assistant: ",assistant)

    def delete_files_by_name(directory, filenames):
        for filename in filenames:
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
                else:
                    print(f"File not found: {file_path}")
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")

    delete_files_by_name('./',["figma_data.txt","code.txt"])

    empty_thread = client.beta.threads.create()
    print(empty_thread)
    thread_id = empty_thread.id

    payload = "Compare both the uploaded files and predict the missing elements"
    final_response = get_response(thread_id,assistant_id,payload)
    print(final_response)

    def delete_vector_store_files(file_id):
        deleted_vector_store_file = client.beta.vector_stores.files.delete(
            vector_store_id=vector_id,
            file_id=file_id
        )
        print(deleted_vector_store_file)

    delete_vector_store_files(code_id)
    delete_vector_store_files(figma_id)

    deleted_vector_store = client.beta.vector_stores.delete(
        vector_store_id=vector_id
    )
    print(deleted_vector_store)

    dlt_thread = client.beta.threads.delete(thread_id)
    print(dlt_thread)

    dlt_assistant = client.beta.assistants.delete(assistant_id)
    print(dlt_assistant)
    
    return {"status":"success","response":final_response}

@app.post("/get-styles-data/")
async def get_styles_data(item: Item):
    api_data = json.loads(item.data)
    
    if api_data:
        try:
            with open("api_data_file.json", 'w') as file:
                json.dump(api_data, file)
                print("API JSON data was successfully written to file")
        except Exception as e:
            print(f"Error occurred while uploading the data: {e}")

    my_assistant = client.beta.assistants.create(
        instructions='''
        You are an expert software engineer and code reviewer. Your task is to analyze the following code to understand the user's unique coding style. Identify the user's coding conventions, commenting style, code structure, specific patterns, error handling, indentation style, use of libraries or frameworks, and fetch the project name if available. Provide a detailed analysis in JSON format.

        Your analysis should cover the following aspects:

        * Project Name:
          Identify and describe the project name if available within the code or comments. If not available find out project name from the file path, anyways find out root directory name.

        * Naming Conventions:
          Describe the naming conventions used for variables, functions, classes, etc.

        * Commenting Style:
          Describe how comments are used in the code (e.g., inline comments, block comments, docstrings).

        * Code Structure:
          Describe the overall structure of the code (e.g., use of classes, functions, modules).

        * Specific Patterns:
          Identify any recurring patterns or idioms unique to the user's coding style (e.g., frequent use of helper functions, specific ways of handling errors).

        * Error Handling:
          Describe the approach to error handling in the code (e.g., use of try/except blocks, custom error messages).

        * Indentation Style:
          Describe the indentation style used in the code (e.g., spaces vs. tabs, number of spaces per indentation level).

        * Use of Libraries/Frameworks:
          Identify any libraries or frameworks used in the code and describe how they are utilized.

        * Overall Coding Habits:
          Provide any additional insights into the user's coding habits and style.

        Provide the analysis in the following JSON format:
        NOTE: do not include any text rather than JSON as this may break my code
        ```json
        [
        {
          "project_name": "name of the project",
        },
        {
          "naming_conventions": "description of naming conventions",
          "commenting_style": "description of commenting style",
          "code_structure": "description of code structure",
          "specific_patterns": "description of specific patterns",
          "error_handling": "description of error handling",
          "indentation_style": "description of indentation style",
          "libraries_frameworks": "description of libraries or frameworks used",
          "overall_coding_habits": "additional insights into the user's coding habits and style"
        }
        ]
        ```
        ''',
        name="User Code Analyzer",
        tools=[{"type": "file_search"}],
        model="gpt-4o",
    )

    assistant_id = my_assistant.id
    print(f"Assistant ID: {assistant_id}")

    store_name = "Uploaded files to Store"
    vector_store = client.beta.vector_stores.create(name=store_name)
    vector_id = vector_store.id
    print(f"Vector ID: {vector_id}")

    def upload_file_to_vector_store(file_name, vector_id):
        try:
            file = client.files.create(
                file=open(file_name, "rb"),
                purpose="assistants"
            )
            file_id = file.id

            vector_store_file = client.beta.vector_stores.files.create(
                vector_store_id=vector_id,
                file_id=file_id
            )
            print(f"{file_name} was successfully stored")
            print(f"Vector store file: {vector_store_file}")
            return file_id
        except Exception as e:
            print(f"Error uploading file {file_name} to vector store: {e}")
            return None

    file_id = upload_file_to_vector_store("api_data_file.json", vector_id)

    assistant = client.beta.assistants.update(
        assistant_id=assistant_id,
        tool_resources={"file_search": {"vector_store_ids": [vector_id]}},
    )
    print(f"Updated assistant: {assistant}")

    empty_thread = client.beta.threads.create()
    thread_id = empty_thread.id
    print(f"Thread ID: {thread_id}")

    payload = [{"type": "text", "text": "Analyze the user codes and give proper analysis"}]
    response_final = get_response(thread_id, assistant_id, payload)
    print(f"Response: {response_final}")

    file_path = os.path.join("./", "api_data_file.json")
    try:
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Deleted file: {file_path}")
        else:
            print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error deleting file {file_path}: {e}")

    print(f"File ID: {file_id}")

    try:
        deleted_vector_store_file = client.beta.vector_stores.files.delete(
            vector_store_id=vector_id,
            file_id=file_id
        )
        print(f"Deleted vector store file: {deleted_vector_store_file}")
    except Exception as e:
        print(f"Error while deleting the file: {e}")

    try:
        deleted_vector_store = client.beta.vector_stores.delete(
            vector_store_id=vector_id
        )
        print(f"Deleted vector store: {deleted_vector_store}")
    except Exception as e:
        print(f"Error while deleting the store: {e}")

    try:
        response = client.beta.assistants.delete(assistant_id)
        print(f"Deleted assistant: {response}")
    except Exception as e:
        print(f"Error while deleting the assistant: {e}")

    if response_final == "Failed":
        return {"status":"failed"}
    else:
        response_text = response_final
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]    
        return {"status":"success","response":response_text}

async def stream_response():
    
    openai_api_key = os.getenv('OPENAI_API_KEY')
    print('Open ai api key: '+openai_api_key)
    client = OpenAI(api_key=openai_api_key)
    
    stream = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Who is narendra modi"}],
        stream=True,
    )
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            data = chunk.choices[0].delta.content
            print(data)
            yield data

@app.post("/generate-stream-resp/")
async def generate_stream_resp(item: Item):
    return StreamingResponse(stream_response())