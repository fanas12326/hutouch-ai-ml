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
from PIL import Image
from io import BytesIO
import spacy

load_dotenv()

app = FastAPI()

class Item(BaseModel):
    id: int
    prompt: str
    data: str

@app.post("/create-a-assistant/")
async def create_a_assistant(item: Item):
    
    temp_data = item.prompt
    data_json_obj = json.loads(temp_data)
    print(data_json_obj)
    
    assitant_name = data_json_obj["name"]
    model_name = data_json_obj["model"]
    instructions = item.data
    print("assitant_name: "+assitant_name)
    print("model_name: "+model_name)
    print("instructions: "+instructions)
    
    openai_api_key = os.getenv('OPENAI_API_KEY')
    # print('Open ai api key: '+openai_api_key)
    client = OpenAI(api_key=openai_api_key)
    
    #Prompt2
    my_assistant = client.beta.assistants.create(
        instructions=instructions,
        name=assitant_name,
        tools=[{"type": "file_search"}],
        model=model_name,
    )
    print(my_assistant)

    assistant_id = my_assistant.id
    print("assistant_id: "+assistant_id)
    
    return {"status": "success", "assistant_id": assistant_id} 

@app.post("/start-a-new-chat/")
async def start_a_new_chat(item: Item):
    
    assistant_id = item.prompt
    
    openai_api_key = os.getenv('OPENAI_API_KEY')
    print('Open ai api key: '+openai_api_key)
    client = OpenAI(api_key=openai_api_key)
    
    thread = client.beta.threads.create()
    thread_id = thread.id
    print(thread_id)
    
    return {"status": "success", "thread_id": thread_id}

@app.post("/send-message/")
async def send_message(item: Item):
    temp_data = item.data
    data_json_obj = json.loads(temp_data)
    print(data_json_obj)
    
    thread_id = data_json_obj["thread_id"]
    assistant_id = data_json_obj["assistant_id"]
    prompt = item.prompt

    openai_api_key = os.getenv('OPENAI_API_KEY')
    print('Open ai api key: '+openai_api_key)
    client = OpenAI(api_key=openai_api_key)
    
    def get_response(threadID,payload):
        message = client.beta.threads.messages.create(
            thread_id = threadID,
            role = "user",
            content = payload
        )
        print(message)
        #run the assistant
        run = client.beta.threads.runs.create(
            thread_id = threadID,
            assistant_id = assistant_id,
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
                        print(f'\n{role}: {response}')

        else:
            print("Something went wrong")
            response = 'Failed'

        return response
    
    response = get_response(thread_id,prompt)
    
    print(response)
    
    if response=="Failed":
        return {"status":"failed"}
    else:
        return {"status":"success","response":response}
 
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
    user_prompt = item.prompt
    user_data = item.data
    
    json_obj_2 = json.loads(user_data)
    json_obj_2
    
    # Serializing json
    json_object = json.dumps(json_obj_2, indent=4)
    
    # Writing to sample.json
    with open("sample.json", "w") as outfile:
        outfile.write(json_object)
        
    openai_api_key = os.getenv('OPENAI_API_KEY')
    print('Open ai api key: '+openai_api_key)
    client = OpenAI(api_key=openai_api_key)
    
    def upload_file_to_assistant(filePath1):
        # Create a vector store caled "Financial Statements"
        vector_store = client.beta.vector_stores.create(name="Uploaded Files")

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
    
    vector_id = upload_file_to_assistant("sample.json")
    
    prompt = f"Fetch Tasks for Today with the Following Criteria:\nPrimary Criteria:\nPriority is urgent and due date is today.\nSecondary Criteria if Primary is Not Met:\nIf there is no due date, check if the priority is urgent.\nIf there is no priority, check if the due date is today.\nIf there are no tasks due today, check for tasks due tomorrow or nearby dates with urgent priority.\nIf there is no priority, check for tasks with due dates in the near future.\nIf there is no due date and no priority, check for tasks whose content (name or description) contains words related to urgency like \"urgent,\" \"EOD,\" \"ASAP.\"\nAdditional Requirements:\nProvide a single external link related to the task content.\nInclude links to open the task, and details such as description, name, due date, and priority level.\nInclude links where available.\nVerify if tasks are completed by checking for user replies; exclude if replied to.\nPresentation Style:\nList tasks sequentially with serial numbers.\nInclude the app name as a parameter within each task's details.\nEnsure each task entry is concise and includes all necessary information without additional explanations.\nMake task details a bit detailed.\nmodify description parameter in such a way that it should tell the senders name as well like this sender is asking for that etc in the apps where there is name in the data , do not modify description for quire\nDo not seperate tasks by app names , instead add app name paramter in it and give bullet points to tasks"
    print(prompt)
    
    assistant = client.beta.assistants.update(
        assistant_id="asst_V16Ar6bvvgdREnsQCZPkibrO",
        tool_resources={"file_search": {"vector_store_ids": [vector_id]}},
        model="gpt-3.5-turbo-0125"
    )
    
    def get_response():
        count = 0
        thread = client.beta.threads.create()
        message = client.beta.threads.messages.create(
            thread_id = thread.id,
            role = "user",
            content = prompt
        )
        #run the assistant
        run = client.beta.threads.runs.create(
            thread_id = thread.id,
            assistant_id = 'asst_V16Ar6bvvgdREnsQCZPkibrO',
        )
        # Waits for the run to be completed
        while True:
            run_status = client.beta.threads.runs.retrieve(thread_id = thread.id, run_id = run.id)
            if run_status.status == "completed":
                break
            elif run_status.status == "failed":
                break
            time.sleep(3) # wait for 2 seconds before checking 
            
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
            return response

        else:
            return "Failed"

    get_updated_response = get_response()
    print(get_updated_response)
    if get_updated_response == "Failed":
        return {"status":"failed"}
    
    response = {"status":"success","tasks":get_updated_response}
    return response

@app.post("/figma-custom-ui/")
async def figma_custom_ui(item: Item):
    api_data = item.data
    
    temp_data = item.prompt
    data_json_obj = json.loads(temp_data)
    print(data_json_obj)
    
    image_url = data_json_obj["image_url"]
    user_role = data_json_obj["user_role"]
    
    print(image_url)
    api_json_data = json.loads(api_data)
    print(api_json_data)
    
    openai_api_key = os.getenv('OPENAI_API_KEY')
    print('Open ai api key: '+openai_api_key)
    client = OpenAI(api_key=openai_api_key)

    #Prompt 1
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
    max_tokens=1000,
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
    
    # uploading ui image to open ai
    def upload_document_file_to_openai(filepath):
        uploaded_file = client.files.create(
            file=open(filepath, "rb"),
            purpose="assistants"
        )
        return uploaded_file.id

    file_id = upload_image_file_to_openai("output.png")
    print(file_id)

    image_file_id = file_id
    
    # Specify the filename
    filename = 'figma_data_file.json'

    # Writing JSON data to a file
    with open(filename, 'w') as file:
        json.dump(api_json_data, file)

    print(f"JSON data has been written to {filename}")

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

    vector_id = upload_file_to_vector_store("figma_data_file.json")

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
    
    def get_response(threadID,payload):
        message = client.beta.threads.messages.create(
            thread_id = threadID,
            role = "user",
            content = payload
        )
        print(message)
        #run the assistant
        run = client.beta.threads.runs.create(
            thread_id = thread.id,
            assistant_id = assistant_id,
        )
        print(run)
        # Waits for the run to be completed
        while True:
            run_status = client.beta.threads.runs.retrieve(thread_id = thread.id, run_id = run.id)
            if run_status.status == "completed":
                break
            elif run_status.status == "failed":
                print("Run failed: ",run_status.last_error)
                break

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

        return response

    payload = [{"type": "text", "text": f"Generate a {user_role} code for the figma UI based on UI image, figma styling data and  Description of the UI: \" {response_1} \"Make separate files for reusable components, classes and asssets"},{"type": "image_file","image_file": {"file_id": file_id}}]
    response_2 = get_response(thread_id,payload)
    print(response_2)

    #deleting the uploaded image file
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

    payload = "Generate logic code for every interactable element and modify the code. The output must be fully functional. Generate logic for the code by yourself don't expect from user. I had uploaded some common functionality steps in the file you can refer from there to create functionality logic for component of UI. No need to add any additional functionality into the code, generate functionality for elements that are already present in the ui."
    response_3 = get_response(thread_id,payload)
    print(response_3)

    deleted_image_file = delete_openai_files(image_file_id)
    print(deleted_image_file)
    
    deleted_document_file = delete_openai_files(file_id)
    print(deleted_document_file)

    #deleting the uploaded vector, so that i can create a new one
    deleted_vector_store = client.beta.vector_stores.delete(
    vector_store_id=vector_id
    )
    print(deleted_vector_store)

    response = client.beta.assistants.delete(assistant_id)
    print(response)
    
    if response_3=="Failed":
        return {"status":"failed"}
    else:
        return {"status":"success","response":response_3}
    
@app.post("/new-functionalities/")
async def new_functionalities(item: Item):
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
    
    def get_response(threadID,payload):
        message = client.beta.threads.messages.create(
            thread_id = threadID,
            role = "user",
            content = payload
        )
        print(message)
        #run the assistant
        run = client.beta.threads.runs.create(
            thread_id = thread.id,
            assistant_id = assistant_id,
        )
        print(run)
        # Waits for the run to be completed
        while True:
            run_status = client.beta.threads.runs.retrieve(thread_id = thread.id, run_id = run.id)
            if run_status.status == "completed":
                break
            elif run_status.status == "failed":
                print("Run failed: ",run_status.last_error)
                break

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

        return response

    payload = [{"type": "text", "text": f"Please modify the code inside the uploaded file to align with the design and layout specifications shown in the provided UI image. Ensure the following:\n\n1. Adapt the visual elements, colors, and layout as per the UI image.\n2. Verify that all interactive elements (buttons, forms, etc.) work correctly according to the new design.\n\nBelow is the functional description of the UI image {response_1}"},{"type": "image_file","image_file": {"file_id": file_id}}]
    response_2 = get_response(thread_id,payload)
    
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