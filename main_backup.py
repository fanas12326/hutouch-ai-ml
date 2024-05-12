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
                
                time.sleep(25) # wait for 2 seconds before checking again
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
            elif get_updated_response == "[]":
                time.sleep(65)

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
            
            return {"id": user_id,"user_role": user_role,"task":task_updated,"instruction":final_prompt,"success": True}
        
        else:
            print('No futher execution')
            return {"error":"Unable to find tasks"}
        
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