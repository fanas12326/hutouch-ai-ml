import logging
import os
from fastapi_utils.tasks import repeat_every
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
from token_count import TokenCount
from openai import OpenAI
from collections import Counter
import gdown  # Import gdown to use it for downloading files from Google Drive
from dotenv import load_dotenv
import psutil  # Import psutil for resource monitoring
from PIL import Image
from io import BytesIO
import spacy
import asyncio
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator
import shutil

load_dotenv()

# Create log directory if not exists
if not os.path.exists('log'):
    os.makedirs('log')

# Configure logging
logging.basicConfig(
    filename='log/app.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = FastAPI()

class Item(BaseModel):
    id: int
    prompt: str
    data: str

openai_api_key = os.getenv('OPENAI_API_KEY')
logging.info('Open ai api key: %s', openai_api_key)
client = OpenAI(api_key=openai_api_key)

def create_user_directory(user_id):
    try:
        path = f"./users_temp_data/user_{str(user_id)}"
        os.makedirs(path)
        logging.info(f"Folder {path} created successfully")
        return path
    except FileExistsError:
        logging.info(f"Folder '{path}' already exists.")
        path = f"./users_temp_data/user_{str(user_id)}"
        return path
    except Exception as e:
        logging.error(f"An error occurred while creating directory: {e}")
        return ""

def delete_folder_recursive(path):
    try:
        shutil.rmtree(path)
        logging.info(f"Folder '{path}' and all its contents deleted successfully.")
    except FileNotFoundError:
        logging.info(f"Folder '{path}' does not exist.")
    except Exception as e:
        logging.error(f"An error occurred while deleting directory: {e}")

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
        logging.info("Data successfully retrieved from API")
        return {"status": "success", "data": data}
    else:
        logging.error(f"Failed to retrieve data: {response.status_code} - {response.text}")
        return {"status": "failed"}

def store_project_data_locally(user_id, dir_path):
    project_data = get_project_data(user_id)

    if project_data["status"] == "success":
        logging.info("Data successfully retrieved")
        try:
            with open(f"{dir_path}/project_style_data.json", 'w') as file:
                data = json.loads(project_data["data"]["data"]["user_nature"])
                json.dump(data, file)
                logging.info("Project style data successfully stored locally")
                return {"status": "success"}
        except Exception as e:
            logging.error(f"Error while creating the file: {e}")
            return {"status": "failed"}
    else:
        logging.error("Failed to retrieve project data")
        return {"status": "failed"}

def store_error(id, func_name, error):
    url = 'http://35.85.112.192/api/ai-store-error'
    headers = {
        'Accept': 'application/json',
        'X-API-KEY': 'JGIp4AWFmI',
        'Content-Type': 'application/json'
    }
    body = {
        "user_id": id,
        "data": [
            {
                "func_name": func_name,
                "error": error
            }
        ]
    }
    response = requests.post(url, headers=headers, json=body)
    logging.error("Error stored for user_id %s in function %s: %s", id, func_name, error)
    return response

def get_response(threadID, assistantID, payload):
    message = client.beta.threads.messages.create(
        thread_id=threadID,
        role="user",
        content=payload
    )
    logging.info("Message created: %s", message)
    
    run = client.beta.threads.runs.create(
        thread_id=threadID,
        assistant_id=assistantID
    )
    logging.info("Run initiated: %s", run)
    
    while True:
        run_status = client.beta.threads.runs.retrieve(thread_id=threadID, run_id=run.id)
        if run_status.status == "completed":
            break
        elif run_status.status == "failed":
            logging.error("Run failed: %s", run_status.last_error)
            break
            
    if run_status.status == "completed":
        messages = client.beta.threads.messages.list(thread_id=threadID)
        logging.info("Number of messages: %d", len(messages.data))
        
        for message in reversed(messages.data):
            role = message.role
            for content in message.content:
                if content.type == 'text':
                    response = content.text.value
    else:
        logging.error("Run did not complete successfully.")
        response = 'Failed'
        
    return response

# Middleware for logging request, response times, status, and resource usage
@app.middleware("http")
async def log_requests(request: Request, call_next):
    process = psutil.Process(os.getpid())
    cpu_usage_before = process.cpu_percent(interval=None)
    memory_info_before = process.memory_info()

    start_time = time.time()
    try:
        response = await call_next(request)
        status_code = response.status_code
        success = "success"
    except Exception as exc:
        status_code = 500
        success = "error"
        logging.error(f"Unhandled exception: {str(exc)}", exc_info=True)
        raise HTTPException(status_code=status_code, detail=str(exc))
    finally:
        process_time = time.time() - start_time
        cpu_usage_after = process.cpu_percent(interval=None)
        memory_info_after = process.memory_info()

        cpu_usage = cpu_usage_after - cpu_usage_before
        memory_usage = (memory_info_after.rss - memory_info_before.rss) / (1024 * 1024)

        logging.info(
            f"Request: {request.method} {request.url} completed in {process_time:.4f} seconds "
            f"Status: {status_code} Result: {success} "
            f"CPU Usage: {cpu_usage:.2f}% Memory Usage: {memory_usage:.2f} MB"
        )
    return response

# Resource monitoring and usage report
def log_resource_usage():
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()
    logging.info("CPU Usage: %s%%", cpu_usage)
    logging.info("Memory Usage: %s%% used, %s free", memory_info.percent, memory_info.available)

@app.on_event("startup")
@repeat_every(seconds=600)  # Adjust the interval as needed
def monitor_resources():
    log_resource_usage()

@app.get("/")
async def read_root(request: Request):
    logging.info(f"Request received: {request.method} {request.url}")
    response = {"message": "HuTouch Server Live!"}
    logging.info(f"Response: {response}")
    return response
    
@app.post("/random-prompts/")
async def random_prompts(item: Item):
    data = json.loads(item.data)
    logging.info("Received data: %s", data)
    
    user_id = item.id
    prompt = item.prompt
    thread_id = data["thread_id"]
    assistant_id = data["assistant_id"]
    logging.info("prompt: %s", prompt)
    logging.info("thread_id: %s", thread_id)
    logging.info("assistant_id: %s", assistant_id)
    
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
        logging.info("Assistant created: %s", my_assistant)
        assistant_id = my_assistant.id

        thread = client.beta.threads.create()
        thread_id = thread.id
        logging.info("Thread created: %s", thread_id)

        curr_payload = prompt
        response = get_response(thread_id, assistant_id, curr_payload)
        logging.info("Response: %s", response)

    else:
        curr_payload = prompt
        response = get_response(thread_id, assistant_id, curr_payload)
        logging.info("Response: %s", response)
    
    if response == "Failed":
        store_error(user_id, "/random-prompts/", "assistant api failed to generate response")    
        return {"status": "failed"}
    else:
        return {"status": "success", "response": response, "thread_id": thread_id, "assistant_id": assistant_id}
 
@app.post("/upload-file/")
async def upload_file(item: Item):
    
    file_id = item.prompt
    assistant_id = item.data
    
    logging.info('Open ai api key: %s', openai_api_key)
    client = OpenAI(api_key=openai_api_key)
    
    vector_store = client.beta.vector_stores.create(
        name="Uploaded Files"
    )
    logging.info("Vector store created: %s", vector_store)
    
    vector_id = vector_store.id
    logging.info("vector_id: %s", vector_id)
    
    vector_store_file = client.beta.vector_stores.files.create(
        vector_store_id=vector_id,
        file_id=file_id
    )
    logging.info("Vector store file created: %s", vector_store_file)
    
    def update_assistant(vectorId):
        assistant = client.beta.assistants.update(
            assistant_id=assistant_id,
            tool_resources={"file_search": {"vector_store_ids": [vectorId]}},
        )   
        logging.info("Assistant updated with vector ID: %s", vectorId)
        return assistant

    updated_assistant = update_assistant(vector_id)
    logging.info("updated_assistant: %s", updated_assistant)
    
    vector_store_files = client.beta.vector_stores.files.list(
        vector_store_id=vector_id
    )
    
    return {"status": "success", "response": updated_assistant, "vector_id": vector_id}

@app.post("/delete-uploaded-file/")
async def delete_uploaded_file(item: Item):
    try:
        openai_api_key = os.getenv('OPENAI_API_KEY')
        logging.info('Open ai API key: %s', openai_api_key)
        client = OpenAI(api_key=openai_api_key)
        
        vector_id = item.prompt
        vector_store_files = client.beta.vector_stores.files.list(
            vector_store_id=vector_id
        )
        logging.info("Vector store files: %s", vector_store_files)

        total_files = len(vector_store_files.data)
        logging.info("Total files: %d", total_files)
        
        return {"status": "success", "total_files": total_files}
    except Exception as e:
        logging.error("Exception in delete_uploaded_file: %s", str(e))
        return {"status": "failed", "error": str(e)}

@app.post("/calculate-task/")
async def calculate_task(item: Item):
    try:
        user_id = item.id
        logging.info("Calculate task for user_id: %d", user_id)
        
        url = f"http://35.85.112.192/api/ai_profiles?user_id={user_id}"
        headers = {
            'Accept': 'application/json',
            'X-API-KEY': 'JGIp4AWFmI',
            'Content-Type': 'application/json',
        }

        response = requests.post(url, headers=headers)
        logging.info("Response from ai_profiles: %s", response.text)

        user_profile_data = json.loads(response.text)
        actual_user_apps = json.loads(user_profile_data['apps'])
        logging.info("Actual user apps: %s", actual_user_apps)
        
        for i in range(len(actual_user_apps)):
            if actual_user_apps[i] == 'vscode' or actual_user_apps[i] == 'vs code':
                actual_user_apps[i] = 'Visual Studio Code'
        cleaned_list = [item.split('.')[0] for item in actual_user_apps]
        
        df = pd.read_csv('formed_data (20).csv')
        df['parent_id'].fillna(0, inplace=True)

        G = nx.DiGraph()
        for index, row in df.iterrows():
            G.add_node(row['id'], label=row['data'])
        for index, row in df.iterrows():
            if row['parent_id'] != 0:
                G.add_edge(row['parent_id'], row['id'])
        
        def get_all_paths_for_app_as_json(graph, app):
            app_lower = app.lower()
            app_nodes = [node for node, data in graph.nodes(data=True) if app_lower in data.get('label', '').lower()]
            all_paths = []

            for app_node in app_nodes:
                path = {}
                current_node = app_node
                path_nodes = []

                while current_node != 0:
                    path_nodes.append(current_node)
                    predecessors = list(graph.predecessors(current_node))
                    current_node = predecessors[0] if predecessors else 0

                labels = ["type", "developer", "task", "steps", "apps"]
                for node in reversed(path_nodes):
                    if labels:
                        label = labels.pop(0)
                        path[label] = graph.nodes[node]['label']
                        if label == "apps":
                            path["parameters"] = [graph.nodes[child]['label'] for child in graph.successors(node)]

                all_paths.append(path)
            return all_paths

        all_app_paths = []
        for app in cleaned_list:
            app_paths = get_all_paths_for_app_as_json(G, app)
            if app_paths:
                all_app_paths.extend(app_paths)
            else:
                logging.info("No path found for app: %s", app)

        if all_app_paths:
            for path in all_app_paths:
                logging.info("Path: %s", path)
        else:
            logging.info("No data available for any apps.")

        filtered_data = [record for record in all_app_paths if 'apps' in record]
        all_app_paths = filtered_data
        json_formatted_str = json.dumps(all_app_paths, indent=4)
        logging.info("JSON formatted string of all app paths: %s", json_formatted_str)

        url = f"http://35.85.112.192/api/ai_fetchapi_data?userID={user_id}"
        response = requests.post(url, headers=headers)
        logging.info("Response from ai_fetchapi_data: %s", response.text)

        json_user_data = json.loads(response.text)
        df = pd.DataFrame(json_user_data['data'])[['appName', 'data']].rename(columns={'appName': 'App name ', 'data': 'BLOB data'})
        df.to_excel('apps_data.xlsx', index=False)
        logging.info("Excel file created successfully.")

        df_user = pd.read_excel('apps_data.xlsx')
        df_local = pd.read_csv('dev data (2).csv')

        df_user['App name '] = df_user['App name '].fillna('')
        df_user['BLOB data'] = df_user['BLOB data'].fillna('')
        df_user['all_data'] = df_user['App name '] + ' ' + df_user['BLOB data']
        df_user['all_data'] = df_user['all_data'].apply(lambda x: ' '.join(x.split()))
        user_merged_data = " ".join(df_user['all_data'].tolist())

        def clean_text(text):
            words = text.lower()
            tokens = word_tokenize(words)
            english_stopwords = stopwords.words('english')
            filtered_words = [word for word in tokens if word not in english_stopwords]
            cleaned_words = [re.sub(r'[^A-Za-z]', '', word) for word in filtered_words if word.isalnum()]
            cleaned_words = list(filter(None, cleaned_words))
            return cleaned_words

        cleaned_user_data = clean_text(user_merged_data)
        logging.info("Cleaned user data: %s", cleaned_user_data)

        def find_best_match(df, cleaned_user_data):
            max_count = 0
            best_match = None

            for index, row in df.iterrows():
                tags = row['Tags'].lower().split(',')
                matches = [tag for tag in tags if tag in cleaned_user_data]
                for item in matches:
                    logging.info("%s: %s", row['Developer'], item)
                count = len(matches)

                if count > max_count:
                    max_count = count
                    best_match = row['Developer']

            return best_match

        user_role = find_best_match(df_local, cleaned_user_data)
        logging.info("Role of the developer: %s", user_role)

        def select_path(paths, developer_tag):
            filtered_paths = [path for path in paths if path['developer'] == developer_tag]
            if len(filtered_paths) == 1:
                return json.dumps(filtered_paths[0])
            elif len(filtered_paths) > 1:
                logging.info("Multiple paths found for the developer: %s", developer_tag)
                for i, path in enumerate(filtered_paths):
                    logging.info("%d: %s", i + 1, path)
                return json.dumps(filtered_paths)
            else:
                return json.dumps("No paths found for the specified developer.")

        if len(json.loads(json_formatted_str)) > 1:
            user_input_developer = user_role
            dev_json_data = select_path(json.loads(json_formatted_str), user_input_developer)
            logging.info("Selected path: %s", dev_json_data)
        elif len(json.loads(json_formatted_str)) < 1:
            logging.info("No path found for specified app")
        else:
            logging.info("No need, only one task is there")

        def remove_parameters(json_list):
            new_list = []
            for item in json_list:
                if isinstance(item, dict):
                    new_dict = {key: value for key, value in item.items() if key != 'parameters'}
                    new_list.append(new_dict)
                else:
                    new_list.append(item)
            return new_list

        updated_data = remove_parameters(json.loads(dev_json_data))
        logging.info("Updated data: %s", updated_data)

        tc = TokenCount(model_name="gpt-3.5-turbo")

        def excel_to_json_array(excel_file):
            df = pd.read_excel(excel_file)
            json_data = df.to_json(orient='records')
            return json_data

        json_data = excel_to_json_array('apps_data.xlsx')
        logging.info("JSON array string: %s", json_data)

        json_object = json.loads(json_data)

        def get_tokens_count(text):
            tokens = tc.num_tokens_from_string(text)
            logging.info("Tokens in the string: %d", tokens)
            return tokens

        def reduce_token_simp(json_data, max_tokens, reduced_by):
            data_1 = copy.deepcopy(json_data)
            json_string_1 = json.dumps(data_1)
            total_tokens = get_tokens_count(json_string_1)
            logging.info("Initial token count: %d", total_tokens)

            visited = [False] * len(data_1)
            count = 0

            while total_tokens > max_tokens:
                for index, item in enumerate(data_1):
                    if total_tokens <= max_tokens:
                        break

                    if len(item["BLOB data"]) >= reduced_by:
                        item["BLOB data"] = item["BLOB data"][:-reduced_by]
                        json_string_1 = json.dumps(data_1)
                        total_tokens = get_tokens_count(json_string_1)
                    else:
                        if not visited[index]:
                            visited[index] = True
                            count += 1

                if count >= len(data_1):
                    logging.info("All data has been minimized or cannot be reduced further.")
                    break

            return data_1

        logging.info("Initial token count: %d", get_tokens_count(json.dumps(json_object, indent=2)))
        reduced_data = reduce_token_simp(json_object, 30000, 200)
        reduced_str = json.dumps(reduced_data, indent=2)
        logging.info("Reduced token count: %d", get_tokens_count(reduced_str))
        logging.info("Reduced data: %s", reduced_str)

        with open('user_data_file.json', 'w') as file:
            json.dump(reduced_data, file)
        logging.info("JSON data written to user_data_file.json")

        with open('public_data_file.json', 'w') as file:
            json.dump(updated_data, file)
        logging.info("JSON data written to public_data_file.json")

        def upload_file_to_assistant(filePath1, filePath2):
            vector_store = client.beta.vector_stores.create(name="Uploaded Files")
            file_paths = [filePath1, filePath2]
            file_streams = [open(path, "rb") for path in file_paths]

            file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
                vector_store_id=vector_store.id, files=file_streams
            )

            logging.info("File batch status: %s", file_batch.status)
            logging.info("File batch counts: %s", file_batch.file_counts)
            logging.info("Vector store id: %s", vector_store.id)
            return vector_store.id

        vector_id = upload_file_to_assistant("user_data_file.json", "public_data_file.json")
        assistant = client.beta.assistants.update(
            assistant_id="asst_gzeLQXC6s9aGZzrakZbuAho2",
            tool_resources={"file_search": {"vector_store_ids": [vector_id]}},
            model="gpt-3.5-turbo-0125"
        )

        def extract_json_using_brackets(text):
            start_index = text.find('[')
            end_index = text.rfind(']') + 1
            json_data = text[start_index:end_index]
            json_object = json.loads(json_data)
            return json_object

        content = f"Compare the files and generate output. Output must be a valid json array"
        logging.info("Content: %s", content)

        def get_response():
            thread = client.beta.threads.create()
            message = client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=content
            )
            run = client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id='asst_gzeLQXC6s9aGZzrakZbuAho2',
            )
            while True:
                run_status = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
                if run_status.status == "completed":
                    break
                elif run_status.status == "failed":
                    logging.error("Run failed: %s", run_status.last_error)
                    break
                time.sleep(2)
            if run_status.status == "completed":
                messages = client.beta.threads.messages.list(thread_id=thread.id)
                logging.info("Number of messages: %d", len(messages.data))
                for message in reversed(messages.data):
                    role = message.role
                    for content in message.content:
                        if content.type == 'text':
                            response = content.text.value
                            logging.info("%s: %s", role, response)
            else:
                logging.error("Something went wrong")
                response = 'Failed'

            if response != 'Failed':
                json_output = extract_json_using_brackets(response)
                final_response = json.dumps(json_output, indent=4)
                logging.info("Final response: %s", final_response)
                return final_response
            else:
                return "Failed"

        flag = True
        while flag:
            get_updated_response = get_response()
            logging.info("Updated response: %s", get_updated_response)
            if get_updated_response != "[]":
                flag = False

        if get_updated_response != "Failed":
            def format_task_steps(json_input):
                json_data = json.loads(json_input)
                task_steps_map = {}

                for item in json_data:
                    task = item['task']
                    step = item['steps']
                    if task in task_steps_map:
                        task_steps_map[task].append(step)
                    else:
                        task_steps_map[task] = [step]

                output_string = ""
                for task, steps in task_steps_map.items():
                    output_string += f"{task} : [{', '.join(steps)}]\n"

                return output_string.strip()

            task_updated = format_task_steps(get_updated_response)
            logging.info("Task updated: %s", task_updated)

            data = json.loads(get_updated_response)
            tasks = {}
            for item in data:
                task = item['task']
                step = item['steps']
                if task in tasks:
                    tasks[task].append(step)
                else:
                    tasks[task] = [step]

            all_tasks = ''
            for task, steps in tasks.items():
                steps_formatted = ", ".join(steps[:-1]) + ", and " + steps[-1]
                all_tasks += f"The task '{task}' involves the following steps: {steps_formatted}.\n"

            logging.info("All tasks: %s", all_tasks)

            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"I wanted to generate instructions for training my AI assistant which is designed for {user_role} developer. Sample example for React Native developer: You are ReactDev an assistant with the knowledge of React Native, Javascript and Redux. You are expert in developing codes and building logic, also you are expert in fixing faulty codes and provide assistance to the user. So, Generate a short template for the prompt don't provide any commentary or explanation for the "}
                ]
            )

            half_prompt = completion.choices[0].message.content
            logging.info("Half prompt: %s", half_prompt)

            final_prompt = half_prompt + '\n' + "You generally perform tasks like\n" + all_tasks
            logging.info("Final prompt: %s", final_prompt)

            dev_resp = json.loads(dev_json_data)
            updated_resp = json.loads(get_updated_response)

            def match_found(dev_obj, updated_response):
                for response_obj in updated_response:
                    if all(dev_obj[key] == response_obj.get(key) for key in response_obj.keys()):
                        return True
                return False

            matched_objects = [obj for obj in dev_resp if match_found(obj, updated_resp)]
            logging.info("Matched objects: %s", matched_objects)

            url = "http://35.85.112.192/api/ai_task_api"
            headers = {
                'Accept': 'application/json',
                'X-API-KEY': 'JGIp4AWFmI',
                'Content-Type': 'application/json',
            }

            data = {
                'user_id': user_id,
                'rolename': user_role,
                'task': task_updated,
                'instruction': final_prompt
            }
            logging.info("Data to be sent: %s", json.dumps(data))

            response = requests.post(url, json=data, headers=headers)
            logging.info("Response from ai_task_api: %s", response.text)

            return {"id": user_id, "user_role": user_role, "task": task_updated, "instruction": final_prompt, "status": "success"}

        else:
            logging.info("No further execution")
            return {"status": "failed"}
    except Exception as e:
        logging.error("Exception in calculate_task: %s", str(e))
        return {"status": "failed", "error": str(e)}



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
            logging.error("Invalid JSON data for tasks. Skipping task extraction.")
    
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
        filtered_keywords = {kw for kw in initial_keywords if kw not in top_apps.values}
        return top_apps, filtered_keywords
    
    try:
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

        logging.info("Identify task output: %s", output)
        return {"success": True, "output": output}
    except Exception as e:
        logging.error("Exception in identify_task: %s", str(e))
        return {"success": False, "error": str(e)}

@app.post("/task-priority/")
async def task_priority(item: Item):
    try:
        user_id = item.id
        temp_data = json.loads(item.prompt)
        user_prompt = temp_data["prompt"]
        assistant_id = temp_data["assistant_id"]
        mode = temp_data["mode"]
        thread_id = temp_data["thread_id"]
        user_data = item.data
        
        dir_path = create_user_directory(user_id)
        logging.info("Directory path: %s", dir_path)
        
        # Uploading figma api data to vector store
        def upload_file_to_vector_store(filePath1):
            # Create a vector store called "Document Files"
            vector_store = client.beta.vector_stores.create(name="Document Files")

            # Ready the files for upload to OpenAI
            file_paths = [filePath1]
            file_streams = [open(path, "rb") for path in file_paths]

            # Use the upload and poll SDK helper to upload the files, add them to the vector store,
            # and poll the status of the file batch for completion.
            file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
                vector_store_id=vector_store.id, files=file_streams
            )

            # Log the status and the file counts of the batch to see the result of this operation.
            logging.info("File batch status: %s", file_batch.status)
            logging.info("File batch counts: %s", file_batch.file_counts)
            logging.info("Vector store ID: %s", vector_store.id)

            return vector_store.id

        def update_assistant(vectorId, assistantID):
            assistant = client.beta.assistants.update(
                assistant_id=assistantID,
                tool_resources={"file_search": {"vector_store_ids": [vectorId]}},
            )
            return assistant

        if not assistant_id:
            my_assistant = client.beta.assistants.create(
                instructions="You are Task Priority Assist, an AI assistant that will help the user to find the priority task.",
                name="Task Priority Assist",
                tools=[{"type": "file_search"}],
                model="gpt-4o",
            )
            logging.info("New assistant created: %s", my_assistant)
            assistant_id = my_assistant.id
        else:
            logging.info("Assistant is already created")

        file_path = os.path.join(dir_path, "sample.json")
        if user_data:
            user_json_obj = json.loads(user_data)
            logging.info("User data: %s", user_json_obj)
            
            with open(file_path, 'w') as json_file:
                json.dump(user_json_obj, json_file, indent=4)
            
            vector_id = upload_file_to_vector_store(file_path)
            updated_assistant = update_assistant(vector_id, assistant_id)
            logging.info("New files added: %s", updated_assistant)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    logging.info("Deleted file: %s", file_path)
                else:
                    logging.info("File not found: %s", file_path)
            except Exception as e:
                logging.error("Error deleting file %s: %s", file_path, e)
        else:
            logging.info("No need to add files to assistant")

        if not thread_id:
            thread = client.beta.threads.create()
            thread_id = thread.id
            logging.info("New thread created: %s", thread_id)
        else:
            logging.info("No need to create a thread")
        
        response = get_response(thread_id, assistant_id, user_prompt)
        logging.info("Initial response: %s", response)

        def update_assistant_instruction(assistant_id, instructions):
            assistant = client.beta.assistants.update(
                assistant_id=assistant_id,
                instructions=instructions
            )
            return assistant

        if user_prompt:
            if mode == "meet":
                inst = ("You are Task Priority Assist, an AI assistant that will fetch the meetings for the asked date. "
                        "Format the response as follows:\n\nTitle: [Title of the meeting]\nDescription: [Description of the meeting (if provided, otherwise exclude this line)]"
                        "\nTime: [Date and time of the meeting]\nRequirements for the meeting: [Details about what is required or the purpose of the meeting]"
                        "\nLink: [Link to the meeting (if provided, otherwise exclude this line)]")
                updated_assistant = update_assistant_instruction(assistant_id, inst)
                logging.info("Updated assistant: %s", updated_assistant)
            else:
                inst = "You are Task Priority Assist, an AI assistant that will help the user to find the priority task."
                updated_assistant = update_assistant_instruction(assistant_id, inst)
                logging.info("Updated assistant: %s", updated_assistant)

            if mode == "meet":
                response = get_response(thread_id, assistant_id, user_prompt)
            else:
                prompt_1 = ("Given the task data, structure each task with the following details one below the other:\n"
                            "Title, Description : Summarize the description in short, Source : App name, Due date, Link to open task : Add the URL from the data.")
                response_1 = get_response(thread_id, assistant_id, prompt_1)
                logging.info("Response 1: %s", response_1)
                
                prompt_2 = ("Only show tasks that have not been marked as completed, dev complete, test complete etc or the messages which contain tasks and do not have any replies indicating the completion of tasks.")
                response_2 = get_response(thread_id, assistant_id, prompt_2)
                logging.info("Response 2: %s", response_2)
                
                prompt_3 = (f"User's Prompt - {user_prompt}\n Filter the tasks to display only those that have the due date specified in the user's prompt and priority is urgent or high, along with any overdue tasks."
                            "\n If no due date is found for any task, check the priority level. If no priority is found, then check the due date."
                            "\n If two or fewer tasks are found, include tasks from the next day or the next week or the tasks which do not have any due date or priority mentioned.")
                response_3 = get_response(thread_id, assistant_id, prompt_3)
                logging.info("Response 3: %s", response_3)
                
                response = response_3

            delete_folder_recursive(dir_path)
                
            helping_data = {"prompt": user_prompt, "mode": mode, "api_data": json.dumps(user_data, indent=4, ensure_ascii=False)}

            if response == "Failed":
                store_error(user_id, "/task-priority/", "assistant API failed to generate response")
                return {"status": "failed", "data": json.dumps(helping_data, indent=4, ensure_ascii=False)}
            return {"status": "success", "assistant_id": assistant_id, "thread_id": thread_id, "response": response, "data": json.dumps(helping_data, indent=4, ensure_ascii=False)}
        else:
            store_error(user_id, "/task-priority/", "Prompt is not entered")
            return {"status": "failed", "exception": "Prompt is not entered"}
    except Exception as e:
        logging.error("Exception in task_priority: %s", str(e))
        return {"status": "failed", "error": str(e)}

@app.post("/figma-custom-ui/")
async def figma_custom_ui(item: Item):
    user_id = item.id
    temp_api_data = json.loads(item.data)
    
    api_data = temp_api_data["figma_data"]
    added_requirements = temp_api_data["added_requirements"]
    temp_data = item.prompt
    data_json_obj = json.loads(temp_data)
    
    image_url = data_json_obj["image_url"]
    user_role = data_json_obj["user_role"]
    
    logging.info(f"Image URL: {image_url}")
    logging.info(f"User Role: {user_role}")
    
    api_json_data = json.loads(api_data)
    logging.info(api_json_data)
    
    dir_path = create_user_directory(user_id)
    logging.info(f"Directory Path: {dir_path}")
    
    openai_api_key = os.getenv('OPENAI_API_KEY')
    logging.info(f'OpenAI API Key: {openai_api_key}')
    client = OpenAI(api_key=openai_api_key)

    def write_code_to_file(filename: str, code: str) -> None:
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(code)
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Generate functional requirements of the UI having each component functionality explanation"},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        max_tokens=500,
    )

    response_1 = response.choices[0].message.content
    logging.info(response_1)

    my_assistant = client.beta.assistants.create(
        instructions="You are a helpful assistant",
        name="Good Assistant",
        tools=[{"type": "file_search"}],
        model="gpt-4o",
    )
    logging.info(my_assistant)

    def upload_document_file_to_openai(filepath):
        uploaded_file = client.files.create(
            file=open(filepath, "rb"),
            purpose="assistants"
        )
        return uploaded_file.id

    if api_data != "":
        try:
            with open(f"{dir_path}/figma_data_file.json", 'w') as file:
                json.dump(json.loads(api_data), file)
                logging.info("API JSON data was successfully written to file")
        except Exception as e:
            logging.error(f"Some error occurred while uploading the data: {e}")

    def upload_file_to_vector_store(filePath1, filePath2=""):
        vector_store = client.beta.vector_stores.create(name="Document Files")

        file_paths = [filePath1] if filePath2 == "" else [filePath1, filePath2]
        file_streams = [open(path, "rb") for path in file_paths]

        file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vector_store.id, files=file_streams
        )

        logging.info(file_batch.status)
        logging.info(file_batch.file_counts)
        logging.info(vector_store.id)

        return vector_store.id
    
    if added_requirements != "":
        write_code_to_file(f"{dir_path}/More requirements.txt", added_requirements)
        logging.info("More requirements were successfully made")
        vector_id = upload_file_to_vector_store(f"{dir_path}/figma_data_file.json", f"{dir_path}/More requirements.txt")
    else:
        vector_id = upload_file_to_vector_store(f"{dir_path}/figma_data_file.json")

    assistant_id = my_assistant.id
    logging.info(f"Assistant ID: {assistant_id}")
    logging.info(f"Vector ID: {vector_id}")
    
    thread = client.beta.threads.create()
    thread_id = thread.id
    logging.info(f"Thread ID: {thread_id}")

    status_project_code = store_project_data_locally(user_id, dir_path)
    
    if status_project_code["status"] == "success":
        logging.info("Uploading the styles file to vector store")
        created_file = client.files.create(
            file=open(f"{dir_path}/project_style_data.json", "rb"),
            purpose="assistants"
        )
        logging.info(f"Created file: {created_file}")
        
        styles_id = created_file.id
        
        vector_store_file = client.beta.vector_stores.files.create(
            vector_store_id=vector_id,
            file_id=styles_id
        )
        logging.info(vector_store_file)
        logging.info("File was successfully uploaded")
        
    def update_assistant(vector_id):
        assistant = client.beta.assistants.update(
            assistant_id=assistant_id,
            tool_resources={"file_search": {"vector_store_ids": [vector_id]}},
        )   
        return assistant

    updated_assistant = update_assistant(vector_id)
    logging.info(updated_assistant)
    
    figma_info = "absoluteBoundingBox: Describes the absolute position and size of the element in the frame, here position is given in the form of x and y coordinates with respect to the screen, so place the components at proper positions"
    
    if user_role == "Flutter Developer 2":
        logging.info("In flutter dev prompt mode")
        payload = [
            {
                "type": "text",
                "text": f"Generate a {user_role} code with MVC architecture and proper State Management for the figma UI based on UI image, figma styling data (note: figma data is uploaded in figma_data_file.json file) and Description of the UI: \" {response_1} \".\nMake separate files for reusable components, classes, and assets. Also maintain Colors and Strings as a reusable component. \n Some information about figma data is: \n Note: The colors in figma API data are in the form of RGBA format so add accurate colors in code \n{figma_info} \n Ensure the code includes: 1. Proper error handling for each code file and method. 2. Proper commenting so that every non-coder can also understand the code. 3. Provide only the exact code with file and folder names. No need to generate code for Status bar showing battery, time, etc."
            },
            {
                "type": "image_url",
                "image_url": {"url": image_url}
            }
        ]
    else:
        logging.info("Not in Flutter Dev Mode")
        if status_project_code["status"] == "success":
            logging.info("Using the styles data")
            styles_prompt = "User coding styles data is present in project_style_data.json, create response according to it. Here is the short general description of how coding style would be: * architecture_used: Details use of an architectural pattern like MVC or MVVM, with clear separation of concerns for maintainability.\n\n* state_management: Highlights use of state management, evidenced by related classes/functions for managing and updating application state reactively.\n\n* naming_conventions: Describes naming styles. Classes use CamelCase (e.g., NextButton), while variables and functions use camelCase (e.g., selectedLanguage).\n\n* commenting_style: Indicates the approach to comments. Sparse, with occasional inline comments; some files may lack comments.\n\n* code_structure: Explains project organization into directories by concern (e.g., utils, components, modules) and use of classes/functions for different parts of the application.\n\n* specific_patterns: Identifies common patterns, such as using a state management library, constants for strings and paths, and utility classes for common functions.\n\n* error_handling: Describes the approach to handling errors, mainly using utility functions for error messages or exceptions; extensive handling may not be present.\n\n* indentation_style: Specifies indentation style, usually 2 spaces per level, using spaces instead of tabs.\n\n* libraries_frameworks: Lists primary libraries and frameworks used, which vary by tech stack (e.g., React, Angular, Vue for UI; Redux, MobX for state management).\n\n* overall_coding_habits: Describes coding habits, such as organizing code into modules, preferring a declarative style, using a specific state management library, and maintaining consistent structure for components."
            payload = [
                {
                    "type": "text",
                    "text": f"Generate a {user_role} code with architecture and state management given in \"project_style_data.json\" for the figma UI based on UI image, figma styling data (note: figma data is uploaded in figma_data_file.json file) and Description of the UI: \" {response_1} \".\nMake separate files for reusable components, classes, and assets.\n Some information about figma data is: \n Note: The colors in figma API data are in the form of RGBA format so add accurate colors in code \n{figma_info} \n {styles_prompt}\n Ensure the code includes: 1. Proper error handling for each code file and method. 2. Proper commenting so that every non-coder can also understand the code. 3. Provide only the exact code with file and folder names. No need to generate code for Status bar showing battery, time, etc."
                },
                {
                    "type": "image_url",
                    "image_url": {"url": image_url}
                }
            ]
        else:
            logging.info("Not using the styles data")
            payload = [
                {
                    "type": "text",
                    "text": f"Generate a {user_role} code for the figma UI based on UI image, figma styling data (note: figma data is uploaded in figma_data_file.json file) and Description of the UI: \" {response_1} \".\nMake separate files for reusable components, classes, and assets.\n Some information about figma data is: \n Note: The colors in figma API data are in the form of RGBA format so add accurate colors in code \n{figma_info} \n Ensure the code includes: 1. Proper error handling for each code file and method. 2. Proper commenting so that every non-coder can also understand the code. 3. Provide only the exact code with file and folder names. No need to generate code for Status bar showing battery, time, etc."
                },
                {
                    "type": "image_url",
                    "image_url": {"url": image_url}
                }
            ]

    response_2 = get_response(thread_id, assistant_id, payload)
    logging.info(response_2)

    def delete_openai_files(file_id):
        deleted_image_file = client.files.delete(file_id)
        return deleted_image_file
        
    vector_store_files = client.beta.vector_stores.files.list(
        vector_store_id=vector_id
    )
    logging.info(vector_store_files)

    file_id = vector_store_files.data[0].id
    logging.info(f"File ID: {file_id}")

    deleted_vector_store_file = client.beta.vector_stores.files.delete(
        vector_store_id=vector_id,
        file_id=file_id
    )
    logging.info(deleted_vector_store_file)

    file_id = upload_document_file_to_openai("Common_Functionality.json")
    logging.info(f"Common Functionality File ID: {file_id}")

    vector_store_file = client.beta.vector_stores.files.create(
        vector_store_id=vector_id,
        file_id=file_id
    )
    logging.info(vector_store_file)
    
    logging.info(f"Thread ID: {thread_id}")

    if added_requirements == "":
        payload = [
            {
                "type": "text",
                "text": f"Generate logic code for every interactable element and modify the code. The output must be fully functional. Generate logic for the code by yourself, don't expect it from the user. I had uploaded some common functionality steps in the file you can refer to create functionality logic for components of UI. No need to add any additional functionality into the code, generate functionality for elements that are already present in the UI. Also see the uploaded UI image and correct the position of any component which is wrong. \n Ensure the code includes: 1. Proper error handling for each code file and method. 2. Proper commenting so that every non-coder can also understand the code. 3. Provide only the exact code with file and folder names. No need to generate code for Status bar showing battery, time, etc."
            },
            {
                "type": "image_url",
                "image_url": {"url": image_url}
            }
        ]
        logging.info("Generating prompt 3 normally")
    else:
        payload = [
            {
                "type": "text",
                "text": f"Generate logic code for every interactable element and modify the code. The output must be fully functional. Generate logic for the code by yourself, don't expect it from the user. I had uploaded some common functionality steps in the Common_Functionality.json and also some specific requirements in More requirements.txt file you can refer to create functionality logic for components of UI. No need to add any additional functionality into the code, generate functionality for elements that are already present in the UI. Also see the uploaded UI image and correct the position of any component which is wrong. \n Ensure the code includes: 1. Proper error handling for each code file and method. 2. Proper commenting so that every non-coder can also understand the code. 3. Provide only the exact code with file and folder names. No need to generate code for Status bar showing battery, time, etc."
            },
            {
                "type": "image_url",
                "image_url": {"url": image_url}
            }
        ]
        logging.info("Generating prompt 3 for UI image")
    response_3 = get_response(thread_id, assistant_id, payload)
    logging.info(response_3)

    def remove_locally_stored_files(file_path):
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                logging.info(f"Deleted file: {file_path}")
            else:
                logging.info(f"File not found: {file_path}")
        except Exception as e:
            logging.error(f"Error deleting file {file_path}: {e}")
        
    remove_locally_stored_files(f"{dir_path}/More requirements.txt")
    remove_locally_stored_files(f"{dir_path}/figma_data_file.json")
        
    deleted_document_file = delete_openai_files(file_id)
    logging.info(deleted_document_file)
    
    if styles_id:
        deleted_document_file = client.files.delete(styles_id)
        logging.info(deleted_document_file)
    
    deleted_vector_store = client.beta.vector_stores.delete(vector_store_id=vector_id)
    logging.info(deleted_vector_store)

    response = client.beta.assistants.delete(assistant_id)
    logging.info(response)
    
    helping_data = {"figma_data": api_data, "user_requirements_data": added_requirements}
    
    delete_folder_recursive(dir_path)
    
    if response_3 == "Failed":
        store_error(user_id, "/figma-custom-ui/", "assistant API failed to generate response")
        return {"status": "failed", "data": json.dumps(helping_data, indent=4, ensure_ascii=False)}
    else:
        return {"status": "success", "response": response_3, "data": json.dumps(helping_data, indent=4, ensure_ascii=False)}

@app.post("/new-functionalities/")
async def new_functionalities(item: Item):
    try:
        user_id = item.id
        code_data = item.data
        
        temp_data = item.prompt
        data_json_obj = json.loads(temp_data)
        logging.info("Data JSON object: %s", data_json_obj)
        
        image_url = data_json_obj["image_url"]
        user_role = data_json_obj["user_role"]
        
        logging.info("Image URL: %s", image_url)
        
        client = OpenAI(api_key="sk-proj-csMstUJ72UbVhyIBeE5ET3BlbkFJyTfXZ9fEZ4eC5N14i4X8")

        # Prompt 1
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Generate functional requirements of the UI having each component functionality explanation with proper positioning"},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            max_tokens=1000,
        )

        response_1 = response.choices[0].message.content
        logging.info("Response 1: %s", response_1)

        # Download the image
        response = requests.get(image_url)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            image.save("output.png", format="PNG")
            logging.info("Image saved as output.png")
        else:
            logging.error("Failed to retrieve the image")
            return {"status": "failed", "error": "Failed to retrieve the image"}

        def upload_image_file_to_openai(filepath):
            uploaded_file = client.files.create(
                file=open(filepath, "rb"),
                purpose="vision"
            )
            return uploaded_file.id
        
        file_id = upload_image_file_to_openai("output.png")
        logging.info("Image file ID: %s", file_id)

        image_file_id = file_id

        file_path = "code.txt"
        with open(file_path, "w") as file:
            file.write(code_data)
        logging.info("File created successfully at %s", file_path)
        
        def upload_file_to_vector_store(filePath1):
            vector_store = client.beta.vector_stores.create(name="Document Files")
            file_paths = [filePath1]
            file_streams = [open(path, "rb") for path in file_paths]
            file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
                vector_store_id=vector_store.id, files=file_streams
            )
            logging.info("File batch status: %s", file_batch.status)
            logging.info("File batch counts: %s", file_batch.file_counts)
            logging.info("Vector store ID: %s", vector_store.id)
            return vector_store.id

        vector_id = upload_file_to_vector_store("code.txt")

        my_assistant = client.beta.assistants.create(
            instructions="You are a helpful assistant",
            name="Good Assistant",
            tools=[{"type": "file_search"}],
            model="gpt-4o",
        )
        logging.info("New assistant created: %s", my_assistant)

        assistant_id = my_assistant.id
        logging.info("Assistant ID: %s", assistant_id)
        logging.info("Vector ID: %s", vector_id)
        
        thread = client.beta.threads.create()
        thread_id = thread.id
        logging.info("Thread ID: %s", thread_id)

        def update_assistant(vectorId):
            assistant = client.beta.assistants.update(
                assistant_id=assistant_id,
                tool_resources={"file_search": {"vector_store_ids": [vectorId]}},
            )
            return assistant

        updated_assistant = update_assistant(vector_id)
        logging.info("Updated assistant: %s", updated_assistant)
        
        payload = [
            {
                "type": "text", 
                "text": f"Please modify the code inside the uploaded file to align with the design and layout specifications shown in the provided UI image. Ensure the following:\n\n1. Adapt the visual elements, colors, and layout as per the UI image.\n2. Verify that all interactive elements (buttons, forms, etc.) work correctly according to the new design.\n\nBelow is the functional description of the UI image {response_1}"
            },
            {
                "type": "image_file",
                "image_file": {"file_id": file_id}
            }
        ]
        response_2 = get_response(thread_id, assistant_id, payload)
        logging.info("Response 2: %s", response_2)
        
        def delete_openai_files(file_id):
            deleted_image_file = client.files.delete(file_id)
            return deleted_image_file

        deleted_image_file = delete_openai_files(image_file_id)
        logging.info("Deleted image file: %s", deleted_image_file)

        deleted_vector_store = client.beta.vector_stores.delete(vector_store_id=vector_id)
        logging.info("Deleted vector store: %s", deleted_vector_store)

        response = client.beta.assistants.delete(assistant_id)
        logging.info("Deleted assistant: %s", response)
        
        if response_2 == "Failed":
            store_error(user_id, "/new-functionalities/", "assistant API failed to generate response")
            return {"status": "failed"}
        else:
            return {"status": "success", "response": response_2}
    except Exception as e:
        logging.error("Exception in new_functionalities: %s", str(e))
        store_error(item.id, "/new-functionalities/", str(e))
        return {"status": "failed", "error": str(e)}

@app.post("/detect-file-name/")
async def detect_file_name(item: Item):
    try:
        prompt = item.prompt
        logging.info("Detect file name prompt: %s", prompt)
        
        nlp = spacy.load('en_core_web_sm')
        
        def extract_file_name(statement):
            file_name_pattern = r'\b\w+\.\w+\b'
            file_name_match = re.search(file_name_pattern, statement)
            
            if file_name_match:
                return file_name_match.group()
            else:
                doc = nlp(statement)
                possible_file_names = []

                for token in doc:
                    if token.pos_ == 'PROPN':
                        possible_file_names.append(token.text)
                
                keywords = ["file name", "document", "file"]
                words = statement.split()
                for i, word in enumerate(words):
                    if any(keyword in word.lower() for keyword in keywords):
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
                    if words:
                        return words[-1]
                    else:
                        return None

        file_name = extract_file_name(prompt)
        logging.info("Extracted file name: %s", file_name)
        
        return {"status": "success", "file_name": file_name}
    except Exception as e:
        logging.error("Exception in detect_file_name: %s", str(e))
        return {"status": "failed", "error": str(e)}

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

    logging.info(f"Prompt: {prompt}")
    logging.info(f"User Role: {user_role}")
    logging.info(f"Image URL: {image_url}")
    logging.info(f"Assistant ID: {assistant_id}")
    logging.info(f"Thread ID: {thread_id}")
    logging.info(f"Additional Requirements: {additional_requirements}")
    logging.info(f"API Data: {api_data}")
    logging.info(f"Figma API Data: {figma_api_data}")
    
    dir_path = create_user_directory(user_id)
    logging.info(f"Directory Path: {dir_path}")
            
    if not assistant_id:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Generate a description for a {user_role} developer."}
            ]
        )
        role_description = completion.choices[0].message.content
        logging.info(role_description)

        my_assistant = client.beta.assistants.create(
            instructions=f"{role_description}\nGenerate {user_role} code based on user prompt.\nRemember:\n1. Generate output by studying the uploaded files\n2. Uploaded Readme.txt contains the proper files and folder structure\n3. Link the output to existing project files\n4. Generate code with proper file and folder name",
            name=f"{user_role} Code Assist",
            tools=[{"type": "file_search"}],
            model="gpt-4o",
        )
        assistant_id = my_assistant.id
        logging.info(f"Assistant ID: {assistant_id}")

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
                logging.info(f"{file_name} was successfully stored")
                logging.info(vector_store_file)
            except Exception as e:
                logging.error(f"Not able to store file {file_name}: {e}")

    def get_file_name(file_path):
        return os.path.basename(file_path.replace('\\', '/'))

    def filter_json_objects(data, keyword):
        filtered_data = [obj for obj in data if keyword in obj['file_path'] or 'Readme' in obj['file_path']]
        return filtered_data
    
    is_valid = False
    try:
        api_data_json_1 = json.loads(api_data)
        api_data_json = []
        if "flutter" in user_role.lower():
            api_data_json = filter_json_objects(api_data_json_1, "lib")
        elif "react native" in user_role.lower():
            api_data_json = filter_json_objects(api_data_json_1, "src")
        else:
            api_data_json = api_data_json_1
        
        is_valid = True
        logging.info("It's a valid JSON")

        for files in api_data_json:
            file_name = get_file_name(files["file_path"])
            files["file_path"] = file_name
        logging.info(api_data_json)
        
    except ValueError as e:
        logging.error(f"Invalid JSON: {e}")

    if is_valid:
        multiple_file_name = []
        uploaded_file_ids = []
        for item in api_data_json:
            code = item['content']
            filename = f"{dir_path}/{remove_extension(item['file_path'])}.txt"
            multiple_file_name.append(filename)
            write_code_to_file(filename, code)
            logging.info(f"Code written to {filename}")
        
        store_name = "Uploaded files to Store"
        vector_store = client.beta.vector_stores.create(name=store_name)
        vector_id = vector_store.id

        for files in multiple_file_name:
            upload_file_to_vector_store(files, vector_id)
            
        upload_file_to_vector_store("Common_Functionality.json", vector_id)
        logging.info("Common Functionality file was successfully uploaded")
        
        status_project_code = store_project_data_locally(user_id, dir_path)
        if status_project_code["status"] == "success":
            logging.info("Uploading the styles file to vector store")
            upload_file_to_vector_store(f"{dir_path}/project_style_data.json", vector_id)
        
        if additional_requirements != "":
            try:
                write_code_to_file(f"{dir_path}/More requirements.txt", additional_requirements)
                upload_file_to_vector_store(f"{dir_path}/More requirements.txt", vector_id)
                multiple_file_name.append(f"{dir_path}/More requirements.txt")
                logging.info("More requirements file was successfully uploaded")
            except Exception as e:
                logging.error(f"Error uploading more requirements file: {e}")
            
        assistant = client.beta.assistants.update(
            assistant_id=assistant_id,
            tool_resources={"file_search": {"vector_store_ids": [vector_id]}},
        )
        logging.info(f"Updated assistant: {assistant}")

    else:
        logging.info("No files uploaded")
        if additional_requirements != "":
            try:
                write_code_to_file(f"{dir_path}/More requirements.txt", additional_requirements)
                upload_file_to_vector_store(f"{dir_path}/More requirements.txt", vector_id)
                logging.info("More requirements file was successfully uploaded")
                assistant = client.beta.assistants.update(
                    assistant_id=assistant_id,
                    tool_resources={"file_search": {"vector_store_ids": [vector_id]}},
                )
                logging.info(f"Updated assistant: {assistant}")
            except Exception as e:
                logging.error(f"Error uploading more requirements file: {e}")

    if not thread_id:
        empty_thread = client.beta.threads.create()
        thread_id = empty_thread.id
        logging.info(empty_thread)
    else:
        logging.info(f"Thread is already created: {thread_id}")

    def upload_image_file_to_openai(filepath):
        with open(filepath, "rb") as file:
            uploaded_file = client.files.create(file=file, purpose="vision")
            return uploaded_file.id

    if figma_api_data != "":
        try:
            with open(f"{dir_path}/figma_api_data.json", 'w') as file:
                json.dump(json.loads(figma_api_data), file)
                logging.info("API JSON data was successfully written to file")

            upload_file_to_vector_store(f"{dir_path}/figma_api_data.json", vector_id)
            logging.info("Figma API data file was successfully uploaded")
            assistant = client.beta.assistants.update(
                assistant_id=assistant_id,
                tool_resources={"file_search": {"vector_store_ids": [vector_id]}},
            )
            logging.info(f"Updated assistant: {assistant}")
                
        except Exception as e:
            logging.error(f"Error while uploading figma_api_data file to assistant API: {e}")

    image_id = ""
    if image_url:
        response = requests.get(image_url)
        
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            image.save(f"{dir_path}/output.png", format="PNG")
            logging.info("Image saved as output.png")
        else:
            logging.info("Failed to retrieve the image")

        image_id = upload_image_file_to_openai(f"{dir_path}/output.png")
        logging.info(f"Image successfully uploaded to OpenAI: {image_id}")
        
        if figma_api_data == "":
            logging.info("Not using figma data")
            payload_1 = [
                {"type": "text", "text": f"{prompt} \n Don't provide any unwanted explanation or commentary, give me only exact code with file and folder name."},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        else:
            logging.info("Using figma data")
            figma_info = "absoluteBoundingBox: Describes the absolute position and size of the element in the frame, here position is given in the form of x and y coordinates with respect to the screen, so place the components at proper positions"
            if user_role == "Flutter Developer 2":
                logging.info("Flutter Developer mode output")
                payload_1 = [
                    {"type": "text", "text": f"{prompt}. Figma API data for the UI is uploaded in the figma_api_data.json file you can refer there to understand UI in more detail. \n Some information about figma data is: \n Note: The colors in figma API data are in the form of RGBA format so add accurate colors in code \n{figma_info} \n The existing code follows MVC architecture and proper State Management so generate output according to it. Make separate files for reusable components, classes, and assets. Also maintain Colors and Strings as a reusable component. \n Don't provide any unwanted explanation or commentary, give me only exact code with file and folder name. Integrate the generated response with the existing uploaded code"},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            else:
                logging.info("Normal output mode")
                if status_project_code["status"] == "success":
                    logging.info("Using the styles data")
                    styles_prompt = "User coding styles data is present in project_style_data.json, create response according to it."
                    payload_1 = [
                        {"type": "text", "text": f"{prompt} & integrate it with existing uploaded project code. The ouput should follow the archtitecture style and state management mentioned in the \"figma_api_data.json\" file. Figma API data for the UI is uploaded in the figma_api_data.json file you can refer there to understand UI in more detail. \n Some information about figma data is: \n Note: The colors in figma API data are in the form of RGBA format so add accurate colors in code \n{figma_info} \n {styles_prompt} \n Don't provide any unwanted explanation or commentary, give me only exact code with file and folder name. Integrate the generated response with the existing uploaded code"},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]    
                else:
                    payload_1 = [
                        {"type": "text", "text": f"{prompt}. Figma API data for the UI is uploaded in the figma_api_data.json file you can refer there to understand UI in more detail. \n Some information about figma data is: \n Note: The colors in figma API data are in the form of RGBA format so add accurate colors in code \n{figma_info} \n Don't provide any unwanted explanation or commentary, give me only exact code with file and folder name. Integrate the generated response with the existing uploaded code"},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]

        response_1 = get_response(thread_id, assistant_id, payload_1)
        logging.info(response_1)
        if response_1 == "Failed":
            logging.info("Here image uploading via url is failed, trying a backup route.")
            if figma_api_data == "":
                logging.info("Not using figma data")
                payload_1 = [
                    {"type": "text", "text": f"{prompt} \n Don't provide any unwanted explanation or commentary, give me only exact code with file and folder name."},
                    {"type": "image_file", "image_file": {"file_id": image_id}}
                ]
            else:
                logging.info("Using figma data")
                figma_info = "absoluteBoundingBox: Describes the absolute position and size of the element in the frame, here position is given in the form of x and y coordinates with respect to the screen, so place the components at proper positions"
                payload_1 = [
                    {"type": "text", "text": f"{prompt}. Figma API data for the UI is uploaded in the figma_api_data.json file you can refer there to understand UI in more detail. \n Some information about figma data is: \n Note: The colors in figma API data are in the form of RGBA format so add accurate colors in code \n{figma_info} \n Don't provide any unwanted explanation or commentary, give me only exact code with file and folder name."},
                    {"type": "image_file", "image_file": {"file_id": image_id}}
                ]
            
            response_1 = get_response(thread_id, assistant_id, payload_1)
            logging.info(response_1)
            
        payload_2 = ""
        
        if additional_requirements == "":
            payload_2 = [
                {"type": "text", "text": f"Generate logic code for every interactable element and modify the code. The output must be fully functional. Generate logic for the code by yourself, don't expect it from user. I had uploaded some common functionality steps in the Common_Functionality.json file you can refer to create functionality logic for component of UI. No need to add any additional functionality into the code, generate functionality for elements that are already present in the UI.\nAlso see the uploaded UI image and correct the position of any component which is wrong\n Don't provide any unwanted explanation, give me only exact code with comments. Ensure to follow file and folder name."},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        else:
            payload_2 = [
                {"type": "text", "text": f"Generate logic code for every interactable element and modify the code. The output must be fully functional. Generate logic for the code by yourself, don't expect it from user. I had uploaded some common functionality steps in the Common_Functionality.json and also some specific requirements in More requirements.txt file you can refer to create functionality logic for component of UI. No need to add any additional functionality into the code, generate functionality for elements that are already present in the UI.\nAlso see the uploaded UI image and correct the position of any component which is wrong\n Don't provide any unwanted explanation, give me only exact code with comments. Ensure to follow file and folder name."},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]

        response_2 = get_response(thread_id, assistant_id, payload_2)
        logging.info(response_2)
        if response_2 == "Failed":
            logging.info("Here image uploading via url is failed, trying a backup route.")
            if additional_requirements == "":
                payload_2 = [
                    {"type": "text", "text": f"Generate logic code for every interactable element and modify the code. The output must be fully functional. Generate logic for the code by yourself, don't expect it from user. I had uploaded some common functionality steps in the Common_Functionality.json file you can refer to create functionality logic for component of UI. No need to add any additional functionality into the code, generate functionality for elements that are already present in the UI.\nAlso see the uploaded UI image and correct the position of any component which is wrong\n Don't provide any unwanted explanation, give me only exact code with comments. Ensure to follow file and folder name."},
                    {"type": "image_file", "image_file": {"file_id": image_id}}
                ]
            else:
                payload_2 = [
                    {"type": "text", "text": f"Generate logic code for every interactable element and modify the code. The output must be fully functional. Generate logic for the code by yourself, don't expect it from user. I had uploaded some common functionality steps in the Common_Functionality.json and also some specific requirements in More requirements.txt file you can refer to create functionality logic for component of UI. No need to add any additional functionality into the code, generate functionality for elements that are already present in the UI.\nAlso see the uploaded UI image and correct the position of any component which is wrong\n Don't provide any unwanted explanation, give me only exact code with comments. Ensure to follow file and folder name."},
                    {"type": "image_file", "image_file": {"file_id": image_id}}
                ]            
            
            response_2 = get_response(thread_id, assistant_id, payload_2)
            logging.info(response_2)
            
        response = response_2
        
    else:
        if additional_requirements == "":
            payload = f"{prompt} \n Don't provide any unwanted explanation or commentary, give me only exact code with file and folder name."
        else:
            payload = f"{prompt}. Refer to some requirements in uploaded More requirements.txt file \n Don't provide any unwanted explanation or commentary, give me only exact code with file and folder name."            
        
        response = get_response(thread_id, assistant_id, payload)
        logging.info(response)

    def delete_local_file(file_path):
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                logging.info(f"Deleted file: {file_path}")
            else:
                logging.info(f"File not found: {file_path}")
        except Exception as e:
            logging.error(f"Error deleting file {file_path}: {e}")
            
    delete_local_file(f"{dir_path}/figma_api_data.json")
    delete_local_file(f"{dir_path}/output.png")
    
    if image_id:
        delete_image_file = client.files.delete(image_id)
        logging.info(delete_image_file)
    else:
        logging.info("No need to delete the images")
        
    helping_data = {
        "figma_data": figma_api_data,
        "user_requirements_data": additional_requirements,
        "apps_data": extra_data,
        "project_code": json.dumps(api_data_json, indent=4, ensure_ascii=False)
    }
    
    delete_folder_recursive(dir_path)
    
    if response != "Failed":    
        return {
            "status": "success",
            "assistant_id": assistant_id,
            "thread_id": thread_id,
            "response": response,
            "data": json.dumps(helping_data, indent=4, ensure_ascii=False)
        }
    else:
        return {
            "status": "failed",
            "data": json.dumps(helping_data, indent=4, ensure_ascii=False)
        }

@app.post("/analyze-files/")
async def analyze_files(item: Item):
    try:
        api_data = item.data
        logging.info("API data: %s", api_data)

        api_data_json = json.loads(api_data)
        logging.info("API data JSON: %s", api_data_json)

        my_assistant = client.beta.assistants.create(
            instructions="You are an AI assistant who responds with JSON only, without any explaining/describing text. You specialize in indexing and summarizing source code files for improved understanding and quick navigation. For the provided file content, generate a detailed structured summary and logic overview in JSON format as shown below:\n\njson\n{\n  \"file_path\": \"<file_path>\",\n  \"overview\": \"Detailed overview of the file.\",\n  \"classes\": [\n    {\n      \"name\": \"ClassName\",\n      \"description\": \"Detailed description of the class.\"\n    }\n  ],\n  \"functions\": [\n    {\n      \"name\": \"FunctionName\",\n      \"description\": \"Detailed description of the function and its logic.\",\n      \"parameters\": [\"param1\", \"param2\"],\n      \"returns\": \"Description of the return value.\"\n    }\n  ],\n  \"variables\": [\n    {\n      \"name\": \"VariableName\",\n      \"description\": \"Brief description of the variable.\"\n    }\n  ],\n  \"logic_overview\": [\n    {\n      \"logic\": \"Description of the logic(how the method works).\",\n      \"location\": \"Where the logic is used in the file.\"\n    }\n  ]\n}\nAdditionally, generate second json object to capture the user's coding style and preferences for future reference. This JSON should include information about component reuse, preferred libraries, coding patterns, and any other relevant aspects of the user's coding style:\n\njson\n{\n  \"user_style\": {\n    \"component_reuse\": \"Description of how the user reuses components or methods.\",\n    \"preferred_libraries\": [\"Library1\", \"Library2\"],\n    \"coding_patterns\": \"Description of the user's common coding patterns or techniques.\",\n    \"excluded_libraries\": [\"Library3\", \"Library4\"],\n    \"additional_notes\": \"Any other relevant information about the user's coding style.\"\n  }\n}\nFile Content:\n<file content>\n\nProvide the detailed structured summary and logic overview along with user's style in the JSON format as shown above.\nIMPORTANT: Provide only 2 json object, one for file overview and other for user style.\nALSO  FIND PROJECT NAME AND RETURN WITH RESPONSE IN TOP",
            name="AI Assistant",
            tools=[{"type": "file_search"}],
            model="gpt-4o",
        )
        logging.info("Assistant created: %s", my_assistant)

        assistant_id = my_assistant.id
        logging.info("Assistant ID: %s", assistant_id)

        def get_file_name(file_path):
            normalized_path = file_path.replace('\\', '/')
            return os.path.basename(normalized_path)

        def write_code_to_file(filename: str, code: str) -> None:
            with open(filename, 'w') as file:
                file.write(code)

        def remove_extension(filename: str) -> str:
            return filename.rsplit('.', 1)[0]

        uploaded_file_ids = []

        def upload_file_to_vector_store(file_name, vector_id):
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
            logging.info("%s was successfully stored", file_name)
            logging.info("Vector store file: %s", vector_store_file)

        multiple_file_name = []
        for item in api_data_json:
            code = item['content']
            filename = remove_extension(get_file_name(item['file_path'])) + '.txt'
            logging.info("Filename: %s", filename)
            multiple_file_name.append(filename)
            write_code_to_file(filename, code)
            logging.info("Code written to %s", filename)

        logging.info("Multiple file names: %s", multiple_file_name)

        store_name = "Uploaded files to Store"
        vector_store = client.beta.vector_stores.create(name=store_name)
        logging.info("Vector store created: %s", vector_store)

        vector_id = vector_store.id

        for files in multiple_file_name:
            upload_file_to_vector_store(files, vector_id)

        logging.info("%d files successfully uploaded to vector store %s", len(multiple_file_name), store_name)

        assistant = client.beta.assistants.update(
            assistant_id=assistant_id,
            tool_resources={"file_search": {"vector_store_ids": [vector_id]}},
        )
        logging.info("Updated assistant: %s", assistant)

        def delete_files_by_name(directory, filenames):
            for filename in filenames:
                file_path = os.path.join(directory, filename)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        logging.info("Deleted file: %s", file_path)
                    else:
                        logging.warning("File not found: %s", file_path)
                except Exception as e:
                    logging.error("Error deleting file %s: %s", file_path, e)

        directory = './'
        delete_files_by_name(directory, multiple_file_name)

        empty_thread = client.beta.threads.create()
        logging.info("Empty thread created: %s", empty_thread)

        thread_id = empty_thread.id

        payload = "Index the uploaded source files"
        response_2 = get_response(thread_id, assistant_id, payload)
        logging.info("Response 2: %s", response_2)
        
        if response_2 != "Failed":        
            json_strings = response_2.split('```json\n')[1:]
            json_list = []

            for json_str in json_strings:
                json_str = json_str.strip().rstrip('```')
                json_data = json.loads(json_str)
                json_list.append(json_data)
            
            final_json_str = json.dumps(json_list, indent=2)
            logging.info("Final JSON string: %s", final_json_str)

        def delete_openai_files(file_id):
            deleted_image_file = client.files.delete(file_id)
            return deleted_image_file

        for ids in uploaded_file_ids:
            msg = delete_openai_files(ids)
            logging.info("Deleted file ID: %s", msg)

        deleted_vector_store = client.beta.vector_stores.delete(vector_store_id=vector_id)
        logging.info("Deleted vector store: %s", deleted_vector_store)

        response = client.beta.assistants.delete(assistant_id)
        logging.info("Deleted assistant: %s", response)
        
        if response_2 != "Failed":
            return {"status": "success", "response": final_json_str}
        
        return {"status": "failed"}
    except Exception as e:
        logging.error("Exception in analyze_files: %s", str(e))
        return {"status": "failed", "error": str(e)}

@app.post("/compare-output-code/")
async def compare_output_code(item: Item):
    try:
        code = item.data
        figma_data = item.prompt

        my_assistant = client.beta.assistants.create(
            instructions="You are a comparison assistant whose task is to compare the data file with the output file and identify the hallucinated and missing elements, colors, font sizes, font styles, locations, positions, text and image placeholders that are present in the data file but not used in the code. Provide only these details without extra explanation, so the user can fix them manually.\nStart response by saying \"Missing Elements -\"",
            name=f"Comparison assistant",
            tools=[{"type": "file_search"}],
            model="gpt-4o",
        )
        logging.info("Created assistant: %s", my_assistant)

        assistant_id = my_assistant.id
        logging.info("Assistant ID: %s", assistant_id)

        def write_code_to_file(filename: str, code: str) -> None:
            try:
                with open(filename, 'w') as file:
                    file.write(code)
            except Exception as e:
                logging.error("Error while writing code to file %s: %s", filename, e)
        
        write_code_to_file("figma_data.txt", figma_data)
        write_code_to_file("code.txt", code)

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
                logging.info("%s was successfully stored", file_name)
                logging.info("Vector store file: %s", vector_store_file)
                return file_id
            except Exception as e:
                logging.error("Error uploading file to vector store: %s", e)

        store_name = "Uploaded files to Store"
        vector_store = client.beta.vector_stores.create(name=store_name)
        logging.info("Created vector store: %s", vector_store)

        vector_id = vector_store.id
        figma_id = upload_file_to_vector_store("figma_data.txt", vector_id)
        code_id = upload_file_to_vector_store("code.txt", vector_id)
        logging.info("Figma ID: %s, Code ID: %s", figma_id, code_id)

        assistant = client.beta.assistants.update(
            assistant_id=assistant_id,
            tool_resources={"file_search": {"vector_store_ids": [vector_id]}},
        )
        logging.info("Updated assistant: %s", assistant)

        def delete_files_by_name(directory, filenames):
            for filename in filenames:
                file_path = os.path.join(directory, filename)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        logging.info("Deleted file: %s", file_path)
                    else:
                        logging.warning("File not found: %s", file_path)
                except Exception as e:
                    logging.error("Error deleting file %s: %s", file_path, e)

        delete_files_by_name('./', ["figma_data.txt", "code.txt"])

        empty_thread = client.beta.threads.create()
        logging.info("Created thread: %s", empty_thread)
        thread_id = empty_thread.id

        payload = "Compare both the uploaded files and predict the missing elements"
        final_response = get_response(thread_id, assistant_id, payload)
        logging.info("Final response: %s", final_response)

        def delete_vector_store_files(file_id):
            try:
                deleted_vector_store_file = client.beta.vector_stores.files.delete(
                    vector_store_id=vector_id,
                    file_id=file_id
                )
                logging.info("Deleted vector store file: %s", deleted_vector_store_file)
            except Exception as e:
                logging.error("Error deleting vector store file %s: %s", file_id, e)

        delete_vector_store_files(code_id)
        delete_vector_store_files(figma_id)

        deleted_vector_store = client.beta.vector_stores.delete(vector_store_id=vector_id)
        logging.info("Deleted vector store: %s", deleted_vector_store)

        dlt_thread = client.beta.threads.delete(thread_id)
        logging.info("Deleted thread: %s", dlt_thread)

        dlt_assistant = client.beta.assistants.delete(assistant_id)
        logging.info("Deleted assistant: %s", dlt_assistant)
        
        return {"status": "success", "response": final_response}
    except Exception as e:
        logging.error("Exception in compare_output_code: %s", str(e))
        return {"status": "failed", "error": str(e)}

def store_user_styles(user_id, project_name, overview, user_nature):
    url = 'http://35.85.112.192/api/store-project-data'
    
    # Define the headers
    headers = {
        'Accept': 'application/json',
        'X-API-KEY': 'JGIp4AWFmI',
        'Content-Type': 'application/json'
    }

    # Define the body
    body = {
        "user_id": user_id,
        "project_name": project_name,
        "overview": overview,
        "user_nature": user_nature
    }

    try:
        # Make the POST request
        response = requests.post(url, headers=headers, json=body)
        response.raise_for_status()
        logging.info("Data successfully stored")
        return response
    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP error occurred: {http_err}")
    except Exception as err:
        logging.error(f"An error occurred: {err}")

    return None

@app.post("/get-styles-data/")
async def get_styles_data(item: Item):
    user_id = item.id
    api_data = json.loads(item.data)
    
    dir_path = create_user_directory(user_id)
    logging.info(f"Directory path: {dir_path}")
    
    if api_data:
        try:
            with open(os.path.join(dir_path, "api_data_file.json"), 'w') as file:
                json.dump(api_data, file)
                logging.info("API JSON data was successfully written to file")
        except Exception as e:
            logging.error(f"Error occurred while uploading the data: {e}")

    my_assistant = client.beta.assistants.create(
        instructions='''
        You are an expert software engineer and code reviewer. Your task is to analyze the following code to understand the user's unique coding style. Identify the user's coding conventions, commenting style, code structure, specific patterns, error handling, indentation style, use of libraries or frameworks, and fetch the project name if available. Provide a detailed analysis in JSON format.

        Your analysis should cover the following aspects:
        * Project Name:
          Identify and describe the project name if available within the code or comments. If not available, find out the project name from the file path, or find out the root directory name.
          
        * Architecture used:
          Identify the architecture according to which the project is made (e.g., MVC or MVVM or any other).
          
        * State Management:
          Identify which state management approach is used in the project (e.g., for flutter - Provider or Riverpod or Redux or any other).

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
        NOTE: do not include any text other than JSON as this may break my code
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
          "architecture_used": "description of architecture used",
          "state_management": "description of state management used",
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
    logging.info(f"Assistant ID: {assistant_id}")

    store_name = "Uploaded files to Store"
    vector_store = client.beta.vector_stores.create(name=store_name)
    vector_id = vector_store.id
    logging.info(f"Vector ID: {vector_id}")

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
            logging.info(f"{file_name} was successfully stored")
            logging.info(f"Vector store file: {vector_store_file}")
            return file_id
        except Exception as e:
            logging.error(f"Error uploading file {file_name} to vector store: {e}")
            return None

    file_id = upload_file_to_vector_store(os.path.join(dir_path, "api_data_file.json"), vector_id)

    assistant = client.beta.assistants.update(
        assistant_id=assistant_id,
        tool_resources={"file_search": {"vector_store_ids": [vector_id]}},
    )
    logging.info(f"Updated assistant: {assistant}")

    empty_thread = client.beta.threads.create()
    thread_id = empty_thread.id
    logging.info(f"Thread ID: {thread_id}")

    payload = [{"type": "text", "text": "Analyze the user codes and give proper analysis"}]
    response_final = get_response(thread_id, assistant_id, payload)
    logging.info(f"Response: {response_final}")

    logging.info(f"File ID: {file_id}")

    try:
        deleted_vector_store_file = client.beta.vector_stores.files.delete(
            vector_store_id=vector_id,
            file_id=file_id
        )
        logging.info(f"Deleted vector store file: {deleted_vector_store_file}")
    except Exception as e:
        logging.error(f"Error while deleting the file: {e}")

    try:
        deleted_vector_store = client.beta.vector_stores.delete(
            vector_store_id=vector_id
        )
        logging.info(f"Deleted vector store: {deleted_vector_store}")
    except Exception as e:
        logging.error(f"Error while deleting the store: {e}")

    try:
        response = client.beta.assistants.delete(assistant_id)
        logging.info(f"Deleted assistant: {response}")
    except Exception as e:
        logging.error(f"Error while deleting the assistant: {e}")

    delete_folder_recursive(dir_path)

    if response_final == "Failed":
        return {"status": "failed"}
    else:
        response_text = response_final
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]

        try:
            json_obj = json.loads(response_text)
            project_name = json_obj[0]["project_name"]
            user_nature = json_obj[1]
            stored_response = store_user_styles(user_id, project_name, "", user_nature)
            logging.info(f"Stored response: {stored_response}")
            return {"status": "success", "response": response_text}
        except Exception as e:
            logging.error(f"Error while fetching the data: {e}")
            return {"status": "failed"}

async def stream_response():
    try:
        openai_api_key = os.getenv('OPENAI_API_KEY')
        logging.info('Open ai API key: %s', openai_api_key)
        client = OpenAI(api_key=openai_api_key)
        
        stream = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Who is Narendra Modi"}],
            stream=True,
        )
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                data = chunk.choices[0].delta.content
                logging.info("Stream data: %s", data)
                yield data
    except Exception as e:
        logging.error("Exception in stream_response: %s", str(e))

@app.post("/generate-stream-resp/")
async def generate_stream_resp(item: Item):
    try:
        return StreamingResponse(stream_response())
    except Exception as e:
        logging.error("Exception in generate_stream_resp: %s", str(e))
        return {"status": "failed", "error": str(e)}