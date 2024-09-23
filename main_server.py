import logging
import os
from logging.handlers import TimedRotatingFileHandler
import glob
import time
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
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator
import shutil

load_dotenv()

"""
# Create log directory if not exists
if not os.path.exists('log'):
    os.makedirs('log')

# Configure logging
logging.basicConfig(
    filename='log/app.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
"""

# Create log directory if it doesn't exist
log_dir = "log"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Configure logging with UTF-8 encoding
log_handler = TimedRotatingFileHandler(
    filename=os.path.join(log_dir, "app.log"),
    when="midnight",
    interval=1,
    backupCount=5,
    encoding="utf-8",
)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
log_handler.setFormatter(formatter)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(log_handler)


# Function to delete log files older than 5 days
def delete_old_logs(log_dir, days=5):
    cutoff = time.time() - (days * 86400)  # 86400 seconds in a day
    for log_file in glob.glob(os.path.join(log_dir, "app.log.*")):
        if os.path.getmtime(log_file) < cutoff:
            os.remove(log_file)


# Call the function to delete old log files
delete_old_logs(log_dir, days=5)

# Create the FastAPI app
app = FastAPI()


class Item(BaseModel):
    id: int
    prompt: str
    data: str


openai_api_key = os.getenv("OPENAI_API_KEY")
# logging.info("Open ai api key: %s", openai_api_key)
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
    api_url = f"http://35.85.112.192/api/get-project-data/{user_id}"

    # Define the headers
    headers = {
        "Accept": "application/json",
        "X-API-KEY": "JGIp4AWFmI",
        "Content-Type": "application/json",
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
        logging.error(
            f"Failed to retrieve data: {response.status_code} - {response.text}"
        )
        return {"status": "failed"}


def store_project_data_locally(user_id, dir_path):
    project_data = get_project_data(user_id)

    if project_data["status"] == "success":
        logging.info("Data is successfully retrieved")
        logging.info(project_data["data"]["data"]["user_nature"])
        logging.info(type(project_data["data"]["data"]["user_nature"]))

        return {
            "status": "success",
            "data": project_data["data"]["data"]["user_nature"],
        }

    else:
        logging.info("Failed to retrieve data")
        return {"status": "failed"}


def store_error(id, func_name, error):
    url = "http://35.85.112.192/api/ai-store-error"
    headers = {
        "Accept": "application/json",
        "X-API-KEY": "JGIp4AWFmI",
        "Content-Type": "application/json",
    }
    body = {"user_id": id, "data": [{"func_name": func_name, "error": error}]}
    response = requests.post(url, headers=headers, json=body)
    logging.error(
        "Error stored for user_id %s in function %s: %s", id, func_name, error
    )
    return response


def get_response(threadID, assistantID, payload):
    message = client.beta.threads.messages.create(
        thread_id=threadID, role="user", content=payload
    )
    logging.info("Message created: %s", message)

    run = client.beta.threads.runs.create(thread_id=threadID, assistant_id=assistantID)
    logging.info("Run initiated: %s", run)

    while True:
        run_status = client.beta.threads.runs.retrieve(
            thread_id=threadID, run_id=run.id
        )
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
                if content.type == "text":
                    response = content.text.value
    else:
        logging.error("Run did not complete successfully.")
        response = "Failed"

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
    logging.info(
        "Memory Usage: %s%% used, %s free", memory_info.percent, memory_info.available
    )


@app.on_event("startup")
@repeat_every(seconds=600)  # Adjust the interval as needed
def monitor_resources():
    log_resource_usage()


@app.get("/")
async def read_root(request: Request):
    logging.info(f"Request received: {request.method} {request.url}")
    response = {"message": "HuTouch Server Live! \nv3.1"}
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
        store_error(
            user_id, "/random-prompts/", "assistant api failed to generate response"
        )
        return {"status": "failed"}
    else:
        return {
            "status": "success",
            "response": response,
            "thread_id": thread_id,
            "assistant_id": assistant_id,
        }


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

            with open(file_path, "w") as json_file:
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
                assistant_id=assistant_id, instructions=instructions
            )
            return assistant

        if user_prompt:
            if mode == "meet":
                inst = (
                    "You are Task Priority Assist, an AI assistant that will fetch the meetings for the asked date. "
                    "Format the response as follows:\n\nTitle: [Title of the meeting]\nDescription: [Description of the meeting (if provided, otherwise exclude this line)]"
                    "\nTime: [Date and time of the meeting]\nRequirements for the meeting: [Details about what is required or the purpose of the meeting]"
                    "\nLink: [Link to the meeting (if provided, otherwise exclude this line)]"
                )
                updated_assistant = update_assistant_instruction(assistant_id, inst)
                logging.info("Updated assistant: %s", updated_assistant)
            else:
                inst = "You are Task Priority Assist, an AI assistant that will help the user to find the priority task."
                updated_assistant = update_assistant_instruction(assistant_id, inst)
                logging.info("Updated assistant: %s", updated_assistant)

            if mode == "meet":
                response = get_response(thread_id, assistant_id, user_prompt)
            else:
                prompt_1 = (
                    "Given the task data, structure each task with the following details one below the other:\n"
                    "Title, Description : Summarize the description in short, Source : App name, Due date, Link to open task : Add the URL from the data."
                )
                response_1 = get_response(thread_id, assistant_id, prompt_1)
                logging.info("Response 1: %s", response_1)

                prompt_2 = "Only show tasks that have not been marked as completed, dev complete, test complete etc or the messages which contain tasks and do not have any replies indicating the completion of tasks."
                response_2 = get_response(thread_id, assistant_id, prompt_2)
                logging.info("Response 2: %s", response_2)

                prompt_3 = (
                    f"User's Prompt - {user_prompt}\n Filter the tasks to display only those that have the due date specified in the user's prompt and priority is urgent or high, along with any overdue tasks."
                    "\n If no due date is found for any task, check the priority level. If no priority is found, then check the due date."
                    "\n If two or fewer tasks are found, include tasks from the next day or the next week or the tasks which do not have any due date or priority mentioned."
                )
                response_3 = get_response(thread_id, assistant_id, prompt_3)
                logging.info("Response 3: %s", response_3)

                response = response_3

            delete_folder_recursive(dir_path)

            helping_data = {
                "prompt": user_prompt,
                "mode": mode,
                "api_data": json.dumps(user_data, indent=4, ensure_ascii=False),
            }

            if response == "Failed":
                store_error(
                    user_id,
                    "/task-priority/",
                    "assistant API failed to generate response",
                )
                return {
                    "status": "failed",
                    "data": json.dumps(helping_data, indent=4, ensure_ascii=False),
                }
            return {
                "status": "success",
                "assistant_id": assistant_id,
                "thread_id": thread_id,
                "response": response,
                "data": json.dumps(helping_data, indent=4, ensure_ascii=False),
            }
        else:
            store_error(user_id, "/task-priority/", "Prompt is not entered")
            return {"status": "failed", "exception": "Prompt is not entered"}
    except Exception as e:
        logging.error("Exception in task_priority: %s", str(e))
        return {"status": "failed", "error": str(e)}


@app.post("/figma-custom-ui/")
async def figma_custom_ui(item: Item):
    print("Figma Custom UI - Started for user_id: ", item.id)
    user_id = item.id
    logging.info(f"user_id: {user_id}")
    temp_api_data = json.loads(item.data)

    api_data = temp_api_data["figma_data"]
    added_requirements = temp_api_data["added_requirements"]
    logging.info(f"added_requirements: {added_requirements}")

    temp_data = item.prompt
    data_json_obj = json.loads(temp_data)
    is_styles_used = "no"
    image_url = data_json_obj["image_url"]
    user_role = data_json_obj["user_role"]
    assets_used = data_json_obj["assets_used"]
    logging.info(f"image_url: {image_url}")
    logging.info(f"user_role: {user_role}")
    logging.info(f"assets_used: {assets_used}")

    #checking if the esential params is available

    #if figma data is null
    if api_data == "null" or api_data == "":
        #if image url is also null
        if image_url == "null" or image_url == "":
            store_error(user_id,"/figma-custom-ui/","Unable to fetch Figma Data and UI image")
            return {"status":"failed","response":"Screen not found. Please use Upload button to provide the Figma where the screen exists"}
        #if image url is not null
        else:
            store_error(user_id,"/figma-custom-ui/","Unable to fetch Figma Data")
            return {"status":"failed","response":"We've encountered a glitch, please try again. If you see this error again, please add comments"}
    # if figma data is not null
    else:
        #if image url is null
        if image_url == "null" or image_url == "":
            store_error(user_id,"/figma-custom-ui/","Unable to fetch UI image")
            return {"status":"failed","response":"We've encountered a glitch, please try again. If you see this error again, please add comments"}
        #if image url is not null
        else:
            logging.info("Both image url and figma data is received")


    api_json_data = json.loads(api_data)
    logging.info(f"api_json_data: {api_json_data}")
    

    dir_path = create_user_directory(user_id)
    logging.info(dir_path)

    def upload_file_to_vector_store(file_name, vector_id):
        with open(file_name, "rb") as file:
            try:
                uploaded_file = client.files.create(file=file, purpose="assistants")
                file_id = uploaded_file.id
                vector_store_file = client.beta.vector_stores.files.create(
                    vector_store_id=vector_id, file_id=file_id
                )
                logging.info(f"{file_name} was successfully stored")
                logging.info(vector_store_file)
            except:
                logging.error(f"Not able to store file {file_name}")

    def write_code_to_file(filename: str, code: str) -> None:
        with open(filename, "w", encoding="utf-8") as file:
            file.write(code)

    # Prompt 1
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Examine the uploaded UI image and perform the following tasks:\n\nIdentify Components: List each visible component (e.g., buttons, text fields, icons, overall background colour/gradient, background color for card/elements, etc ) along with its position on the screen using relative positioning.\n\nDescribe Functionality: Explain the visiblity of each component(eg. product card with curved border and 3d image overlaying card ), purpose and function of each component (e.g. what happens when the user interacts with it).\n\nDetect Repeatation: Identify components that are repeated or have same functions.\n\nImportant Notes:\n\nCover All Elements: Include every visible component in the UI; do not omit any item, no matter how small like shadows & gradients also.\nDetail Repetative Elements: Highlight repeated structures clearly.\nEnsure that every component is accounted for with precise descriptions.\nDo not provide extra explanation or summary.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                        },
                    },
                ],
            }
        ],
    )

    response_1 = response.choices[0].message.content
    logging.info("response for prompt1: \n")
    logging.info(response_1)

    # Prompt 2
    my_assistant = client.beta.assistants.create(
        instructions="You are an expert coder. \n  ##REMEMBER: \n      1. For the next 9-10 prompts, treat all interactions as part of a single task related to creating UI or coding from scratch. Do not lose context; keep track of all inputs and responses to ensure continuity in the design and coding process.\n        2. Build the UI or code sequentially based on user instructions. If a new prompt introduces changes or additions, integrate them without losing the overall structure and consistency of the previous work.\n        3. When generating UI, ensure that the design matches the description provided by the user up to 90%, with all specified components present. If the description evolves, adjust the code accordingly without losing context.\n",
        name="Figma Assistant",
        tools=[{"type": "file_search"}],
        model="gpt-4o",
    )
    logging.info(my_assistant)

    def upload_document_file_to_openai(filepath):
        uploaded_file = client.files.create(
            file=open(filepath, "rb"), purpose="assistants"
        )
        return uploaded_file.id

    def write_code_to_file(filename: str, code: str) -> None:
        with open(filename, "w", encoding="utf-8") as file:
            file.write(code)

    if api_data != "":
        api_converted_data = get_analyzed_api_data(api_data, image_url, dir_path)
        filename = dir_path + "/figma_data_file.txt"
        write_code_to_file(filename, api_converted_data)

    # Creating the vector store
    vector_store = client.beta.vector_stores.create(name="Uploaded Document files")
    logging.info(vector_store)
    vector_id = vector_store.id
    logging.info(f"vector_id: {vector_id}")

    # Uploading the figma data assistant
    upload_file_to_vector_store(dir_path + "/figma_data_file.txt", vector_id)

    assistant_id = my_assistant.id
    logging.info(f"assistant_id: {assistant_id}")
    logging.info(f"vector_id: {vector_id}")

    if assets_used != "null" and assets_used != "":
        try:
            write_code_to_file(dir_path + "/assets_file.txt", assets_used)
            assets_file_id = upload_document_file_to_openai(
                dir_path + "/assets_file.txt"
            )
            vector_store_file = client.beta.vector_stores.files.create(
                vector_store_id=vector_id, file_id=assets_file_id
            )
            logging.info(vector_store_file)
        except:
            logging.error("Some error occurred while using the file")
            assets_used = ""

    thread = client.beta.threads.create()
    thread_id = thread.id
    logging.info(thread_id)

    def update_assistant(vectorId):
        assistant = client.beta.assistants.update(
            assistant_id=assistant_id,
            tool_resources={"file_search": {"vector_store_ids": [vectorId]}},
        )
        return assistant

    updated_assistant = update_assistant(vector_id)
    logging.info(updated_assistant)
    
# -----------Generating First level of code----------------
    figma_info = "absoluteBoundingBox: Describes the absolute position and size of the element in the frame, here position is given in the form of x and y coordinates with respect to the screen, so place the components at proper positions, for size, use the approximate sie in percentage according to screen size"
    assets_info = 'The assets which can be used while generating the code is mentioned in uploaded file "assets info.txt" use whichever assets used as image or icon required to generate the code.'

    if user_role == "Flutter Developer 2":
        logging.info("In flutter dev prompt mode")
        payload = [
            {
                "type": "text",
                "text": f'Generate a {user_role} code with MVC architecture and proper State Management for the figma UI based on UI image, figma styling data(note: figma data is uploaded in figma_data_file.json file) and  Description of the UI: " {response_1} ".\nMake separate files for reusable components, classes, and assets. Also maintain Colors and Strings as a reusable component. \n Some information about figma data is: \n  Note: The colors in figma API data is in the form of RGBA format so add accurate colors in code \n{figma_info} \n Ensure the code includes: 1. Proper error handling for each code file and method. 2. Proper commenting so that every non-coder can also understand the code. 3. Provide only the exact code with file and folder names. No need to generate code for Status bar showing battery, time, etc.',
            },
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
    elif user_role == "Web Developer (HTML & CSS & JavaScript only)":
        payload = [
            {
                "type": "text",
                "text": f"Generate a code for given Web UI based on uploaded UI image, description of the UI and figma styling data (note: figma data is uploaded in figma_data_file.json file). Description of the UI: {response_1}. \nThe logic should be self-contained, ensuring that the code is fully functional without requiring additional user input. Refer to the common functionality steps provided in the file to create logic for each component. Avoid adding new functionalities; focus on creating exact same elements present in the UI.\n\nPlease ensure the code includes:\n\nProper error handling for each function to manage exceptions gracefully.\nDetailed comments in both the HTML and JavaScript files.\nAccurate file and folder names for the HTML, CSS, and JavaScript files involved, reflecting the exact structure needed.\nProvide the updated project structure.",
            },
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
    else:
        logging.info("Generating Prompt 2")

        if assets_used != "null" and assets_used != "":
            logging.info("Using assets to generate code")
            payload = [
                {
                    "type": "text",
                    "text": f'First breakdown step by step how you would implement this and then Generate a {user_role} code for the figma UI based on UI image, figma styling data(note: figma data is uploaded in figma_data_file.txt file) and  Description of the UI: " {response_1} ".\n Some information about figma data is: \n  Note: The colors in figma api data is in the form of RGBA format so make so add accurate colors in code \n{figma_info} \n No need to generate code for Status bar showing battery, time, etc.\nMake components reusable instead of hardcoding repetitive components to avoid duplicate code. \n{assets_info}\nGenerated code must contain some explanatory comments.\nThe generated output must not contain any unwanted explanation or discussion of the output.\n Also Make sure the generated code doesnot include deprecated snippets or imports also the generated code is without any error.',
                },
                {"type": "image_url", "image_url": {"url": image_url}},
            ]
        else:
            logging.info("Not using assets")
            payload = [
                {
                    "type": "text",
                    "text": f'First breakdown step by step how you would implement this and then Generate a {user_role} code for the figma UI based on UI image, figma styling data(note: figma data is uploaded in figma_data_file.txt file) and  Description of the UI: " {response_1} ".\n Some information about figma data is: \n  Note: The colors in figma api data is in the form of RGBA format so make so add accurate colors in code \n{figma_info} \n No need to generate code for Status bar showing battery, time, etc.\nMake components reusable instead of hardcoding repetitive components to avoid duplicate code.\nGenerated code must contain some explanatory comments.\nThe generated output must not contain any unwanted explanation of the output.\n Also Make sure the generated code doesnot include deprecated snippets or imports also the generated code is without any error.',
                },
                {"type": "image_url", "image_url": {"url": image_url}},
            ]

    response_2 = get_response(thread_id, assistant_id, payload)
    logging.info("response for prompt2: \n")
    logging.info(response_2)
    
# -----------ENDED Generating First level of code----------------
    
    
    
# ----------HARDCODED ADAPTIBILITY ----------------
    # payload = [
    #     {
    #         "type": "text",
    #         "text": f"The generated code might not contain proper screen adaptability. The code must be such that it should adapt seamlessly to different screen sizes and orientations. The component should resize text, images, and interactive elements appropriately based on the screen size. Refer uploaded ui image to study ui. Generate entire code with comments and without any explanation",
    #     },
    #     {"type": "image_url", "image_url": {"url": image_url}},
    # ]
    # response_interim = get_response(thread_id, assistant_id, payload)
    # logging.info(response_interim)
# ----------ENDING HARDCODED ADAPTIBILITY ----------------


    # #deleting the uploaded image file
    def delete_openai_files(file_id):
        deleted_file = client.files.delete(file_id)
        return deleted_file

    def remove_file_from_vector_store(fileID, vectorID):
        deleted_vector_store_file = client.beta.vector_stores.files.delete(
            vector_store_id=vectorID, file_id=fileID
        )
        logging.info(deleted_vector_store_file)

    def retrieve_current_files_and_remove(vectorID):
        try:
            vector_store_files = client.beta.vector_stores.files.list(
                vector_store_id=vectorID
            )
            logging.info(vector_store_files)
            file_obj = vector_store_files.data
            for files in file_obj:
                remove_file_from_vector_store(files.id, vector_id)
        except:
            logging.error("Some error occurred while deleting the files")

    retrieve_current_files_and_remove(vector_id)

    logging.info("Correcting the code according to the ui image")
    payload = [
        {
            "type": "text",
            "text": f"The generated UI is not accurate and is not matching with the UI image can you please enhance the code such that it would be exactly matching to the ui. See whichever elements is missing or not properly adjusted in the code or the spacing between the ui components is improper or the position of the element is improper and modify the code accordingly, do not loose original comments & properties of code like adaptiblity and other. Make sure to give complete code without extra explaination about generated code.",
        },
        {"type": "image_url", "image_url": {"url": image_url}},
    ]

    response_3 = get_response(thread_id, assistant_id, payload)
    logging.info(response_3)
# -----------ENDED Correcting code a/c ui image----------------

# -----------adding coding styles to the code----------------

    status_project_code = store_project_data_locally(user_id, dir_path)

    if status_project_code["status"] == "success":
        data_content = status_project_code["data"]
        payload = f"Restructure the generated code with the exact architecture, state management, code structure, adaptive and responsive design, app constants, and error handling as specified in the coding styles below. Ensure that the code is separated into the appropriate folders and files, with clear folder and file names & path displayed. Maintain all existing UI components and functionalities while applying the specified styles. Thoroughly verify that no part of the original code, especially UI elements, interactions, or functionality, is lost during the restructuring process. Use detailed checks to ensure all elements are correctly styled and integrated as per the guidelines.\n\nIf any ambiguity arises in implementing styles without affecting the code's functionality, maintain the original code logic, and add comments highlighting potential adjustments needed to fully align with the coding standards.\n\nCoding Styles are as follows:\n{data_content}\n"

        logging.info(payload)

    if status_project_code["status"] == "success":
        is_styles_used = "yes"
        logging.info("Using the styles data")
        data_content = status_project_code["data"]
        payload = f"Restructure the generated code with the exact architecture, state management, code structure, adaptive and responsive design, app constants, and error handling as specified in the coding styles below. Ensure that the code is separated into the appropriate folders and files, with clear folder and file names & path displayed. Maintain all existing UI components and functionalities while applying the specified styles. Thoroughly verify that no part of the original code, especially UI elements, interactions, or functionality, is lost during the restructuring process. Use detailed checks to ensure all elements are correctly styled and integrated as per the guidelines.\n\nIf any ambiguity arises in implementing styles without affecting the code's functionality, maintain the original code logic, and add comments highlighting potential adjustments needed to fully align with the coding standards.\n\nCoding Styles are as follows:\n{data_content}\n"

        logging.info(payload)
        response_4 = get_response(thread_id, assistant_id, payload)
        response_final = response_4
        logging.info("generated response_4")
        logging.info(response_final)
    
# -----------ENDING adding coding styles to the code----------------


# -----------adding functionalities to the code----------------
    if added_requirements != "":
        write_code_to_file(dir_path + "/More requirements.txt", added_requirements)
        logging.info("More requirements was successfully made")
        upload_file_to_vector_store(dir_path + "/More requirements.txt", vector_id)
    else:
        logging.info("Additional requirements don't exist")

    file_id = upload_document_file_to_openai("Common_Functionality.json")
    logging.info(file_id)

    vector_store_file = client.beta.vector_stores.files.create(
        vector_store_id=vector_id, file_id=file_id
    )
    logging.info(vector_store_file)

    logging.info(thread_id)

    if added_requirements == "":
        payload = 'The current generated code needs functionalities added to it. Refer to "Common_Functionality.json" for available functionality descriptions. This file contains common functionalities with the following details:\n- *Functionality Name*: The name of the functionality.\n- *Description*: Steps to implement the functionality.\n- *Type*: Specifies the type of UI element the functionality is linked to.\n\n### Instructions:\n1. *Strictly Maintain the Existing UI Structure*: Do not modify, alter, or disturb the existing UI elements, layout, or structure in any way. Only focus on adding the required functionalities.\n   \n2. *Match Functionalities to Existing UI Elements*: Implement only those functionalities that directly correspond to UI elements already present in the code. Avoid creating new UI elements (e.g., text boxes, buttons) that are not already in the design, even if the functionality suggests it.\n\n3. *Avoid Unnecessary Additions*: If a functionality in "Common_Functionality.json" does not naturally fit the existing UI elements, do not implement it. For instance, if the code has a "Sign up with Email" button but no email text box, do not add the text box. Only apply functionalities that match the current UI elements without altering or adding new UI components.\n\n4. *Do Not Assume Functionalities*: Only implement functionalities that are clearly defined and directly applicable to existing elements. Do not assume or create new features that are not explicitly needed by the UI.\n\n5. *Seamless Integration*: Ensure that added functionalities integrate smoothly with the existing code without breaking or changing the design or structure of the UI.\n\n6. *No Extra Commentary or Unnecessary Code Changes*: Focus solely on implementing the relevant functionalities. Avoid adding any explanations or changes that are not directly related to the functional aspects.\n\n### Output Expectations:\n- *Functionality Alignment*: Ensure all functionalities align with the UI elements present in the code and do not introduce new elements.\n- *Maintain UI Integrity*: The structure and design of the existing UI code must remain completely unchanged.\n- *Precision and Relevance*: Only add what is necessary and relevant. Do not add functionalities that require UI modifications unless those elements already exist in the current code.'
        logging.info("Generating prompt 3 normally")
    else:
        payload = 'The current generated code needs functionalities added to it. Refer to "More requirements.txt" for available functionality descriptions. This file contains functionalities description, analyze it and strictly add every mentioned functionality from the file.\n\n### Instructions:\n1. *Strictly Maintain the Existing UI Structure*: Do not modify, alter, or disturb the existing UI elements, layout, or structure in any way. Only focus on adding the mentioned functionalities.\n\n2. *Do Not Assume Functionalities*: Only implement functionalities that are clearly defined and directly applicable to existing elements. Do not assume or create new features that are not explicitly needed by the UI.\n\n3. *Seamless Integration*: Ensure that added functionalities integrate smoothly with the existing code without breaking or changing the design or structure of the UI.\n\n4. *No Extra Commentary or Unnecessary Code Changes*: Focus solely on implementing the relevant functionalities. Avoid adding any explanations or changes that are not directly related to the functional aspects.\n\n### Output Expectations:\n- *Functionality Alignment*: Ensure all functionalities align with the UI elements present in the code and do not introduce new elements.\n- *Maintain UI Integrity*: Each functionality should be added and the structure and design of the existing UI code must remain completely unchanged.'
        logging.info("Generating prompt 3 with More requirements")

    response_5 = get_response(thread_id, assistant_id, payload)
    logging.info(response_5)

    response_final = response_5

    retrieve_current_files_and_remove(vector_id)
    
# -----------ENDING adding functionalities to the code----------------

# ------------Validating itself----------------
    if status_project_code["status"] == "success":
        payload = [{"type": "text", "text": f"{response_final}\n First breakdown step by step how and why you created this code also Can you check the code line by line and ensure that the code is properly structured and adhered to coding standards given above as well as functionalities in generated code. Ensure that the code contains proper explainatory commented and is according to the coding standards. Here is coding standards to check properly in json string: \n {data_content} \n Additionally, make sure that the code doesnot contain deprecated or error implementation."}]
        response_interim1 = get_response(thread_id,assistant_id,payload)
        logging.info(response_interim1)
    else:
        payload = [{"type": "text", "text": f"{response_final}\n First breakdown step by step how and why you created this code also Can you check the code line by line and make sure that the code is properly structured and adhered to General coding standards as well as functionalities in generated code. Ensure that the code contains proper explainatory commented and is according to the best practice coding standards. \n Additionally, make sure that the code doesnot contain deprecated or error implementation."}]
        response_interim1 = get_response(thread_id,assistant_id,payload)
        logging.info(response_interim1)
# ------------ENDING Validating itself----------------
    
# -----------Correcting code a/c ui image----------------

    logging.info("Correcting the code according to the ui image")
    payload = [{"type": "text", "text": f"Further Enhance the generated code, as the generated code might overlookedd certain UI elements, can you please check and fix the code such that it should be about 95% match according to the ui (keeping the adaptablity and coding styles unchanged.).Do not modify, alter, or loose the existing UI elements, layout, or structure in any way. See whichever elements is missing in the code or the position of the element is improper and modify the code accordingly, (Image url attached). Provide project structure with name for each file and provide complete formated code for the ui. \n for the generated project structure, please provide terminal command to create structure in ide. give two seperate commands for windows and macos, command should be one liner without any comments and discussion. it should be such that, code and project structure is setup directly by running the command."},{"type": "image_url","image_url": {"url": image_url}}]

    response_6 = get_response(thread_id, assistant_id, payload)
    if response_6!="Failed":
        if is_styles_used == "no":
            alert_statement = "Seems like personalization process was not done. The generated code will be based on generic coding standards. If you would like a personalized code, please go to home page, click personalization button and retry the prompt\n\n"
            response_6 = alert_statement + response_6 

    logging.info(response_6)

    try:
        deleted_document_file = delete_openai_files(file_id)
        logging.info(deleted_document_file)
    except:
        logging.error("Unable to delete specified file")

    deleted_vector_store = client.beta.vector_stores.delete(vector_store_id=vector_id)
    logging.info(deleted_vector_store)

    response = client.beta.assistants.delete(assistant_id)
    logging.info(response)

# -----------ENDING Correcting code a/c ui image----------------

    helping_data = {
        "figma_data": api_converted_data,
        "user_requirements_data": added_requirements,
        "image_url": image_url,
        "assets_used": assets_used,
    }
    logging.info(helping_data)
    delete_folder_recursive(dir_path)

    if response_6 == "Failed":
        store_error(
            user_id, "/figma-custom-ui/", "assistant api failed to generate response"
        )
        print("Figma Custom UI - Failed")
        response_payload = {
            "status": "failed",
            "data": json.dumps(helping_data, indent=4, ensure_ascii=False),
        }
        return response_payload
    else:
        print("Figma Custom UI - Success")
        response_payload = {
            "status": "success",
            "response": response_6,
            "data": json.dumps(helping_data, indent=4, ensure_ascii=False),
        }
        print(response_payload)
        return response_payload


@app.post("/multiple-files-flow/")
async def multiple_files_flow(item: Item):
    # 3 use cases: Update functionality, add new functionality, add new screen
    print("Multiple Files Flow - Started")
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
    is_screen_used = temp_data["is_screen"]
    assets_used = temp_data["assets_used"]
    assistant_id = temp_data["assistant_id"]
    thread_id = temp_data["thread_id"]

    is_styles_used = "no"

    logging.info(f"prompt: {prompt}")
    logging.info(f"user_role: {user_role}")
    logging.info(f"is_screen_used: {is_screen_used}")
    logging.info(f"image_url: {image_url}")
    logging.info(f"assistant_id: {assistant_id}")
    logging.info(f"thread_id: {thread_id}")
    logging.info(f"assets_used: {assets_used}")
    logging.info(f"additional_requirements: {additional_requirements}")
    logging.info(f"api_data: {api_data}")
    logging.info(f"figma_api_data: {figma_api_data}")

    #checking if the esential params is available

    if is_screen_used == "yes":
        #if figma data is null
        if figma_api_data == "null" or figma_api_data == "":
            #if image url is also null
            if image_url == "null" or image_url == "":
                store_error(user_id,"/figma-custom-ui/","Unable to fetch Figma Data and UI image")
                return {"status":"failed","response":"Screen not found. Please use Upload button to provide the Figma where the screen exists"}
            #if image url is not null
            else:
                store_error(user_id,"/figma-custom-ui/","Unable to fetch Figma Data")
                return {"status":"failed","response":"We've encountered a glitch, please try again. If you see this error again, please add comments"}
        # if figma data is not null
        else:
            #if image url is null
            if image_url == "null" or image_url == "":
                store_error(user_id,"/figma-custom-ui/","Unable to fetch UI image")
                return {"status":"failed","response":"We've encountered a glitch, please try again. If you see this error again, please add comments"}
            #if image url is not null
            else:
                logging.info("Both image url and figma data is received")

        #if project code is present or not
        if api_data == "null" or api_data == "":
            store_error(user_id,"/figma-custom-ui/","Unable to fetch UI image")
            return {"status":"failed","response":"Alert: Unable to fetch project code\n1. Kindly check if your VS code is opened\n2. Check if the HuTouch AI extension is installed on VS code\n3. There should not be more than one project opened in VS code at once\n4. Check if you have  a stable internet connection"}
        else:
            logging.info("Project code received")
    else:
        #if project code is present or not
        if api_data == "null" or api_data == "":
            store_error(user_id,"/figma-custom-ui/","Unable to fetch UI image")
            return {"status":"failed","response":"Alert: Unable to fetch project code\n1. Kindly check if your VS code is opened\n2. Check if the HuTouch AI extension is installed on VS code\n3. There should not be more than one project opened in VS code at once\n4. Check if you have  a stable internet connection"}
        else:
            logging.info("Project code received")

    dir_path = create_user_directory(user_id)
    logging.info(dir_path)

    def remove_extension(filename: str) -> str:
        return filename.rsplit(".", 1)[0]

    def write_code_to_file(filename: str, code: str) -> None:
        with open(filename, "w", encoding="utf-8") as file:
            file.write(code)

    def upload_document_file_to_openai(filepath):
        uploaded_file = client.files.create(
            file=open(filepath, "rb"), purpose="assistants"
        )
        return uploaded_file.id

    def upload_file_to_vector_store(file_name, vector_id):
        with open(file_name, "rb") as file:
            try:
                uploaded_file = client.files.create(file=file, purpose="assistants")
                file_id = uploaded_file.id
                uploaded_file_ids.append(file_id)
                vector_store_file = client.beta.vector_stores.files.create(
                    vector_store_id=vector_id, file_id=file_id
                )
                logging.info(file_name + " was successfully stored")
                logging.info(vector_store_file)
            except:
                logging.error(f"Not able to store file {file_name}")

    def get_file_name(file_path):
        return os.path.basename(file_path.replace("\\", "/"))

    # Function to filter JSON objects
    def filter_json_objects(data, keyword):
        filtered_data = [
            obj
            for obj in data
            if keyword in obj["file_path"] or "Readme" in obj["file_path"]
        ]
        return filtered_data

    def update_assistant(vectorId):
        assistant = client.beta.assistants.update(
            assistant_id=assistant_id,
            tool_resources={"file_search": {"vector_store_ids": [vectorId]}},
        )
        return assistant

    # #deleting the uploaded image file
    def delete_openai_files(file_id):
        deleted_file = client.files.delete(file_id)
        return deleted_file

    def remove_file_from_vector_store(fileID, vectorID):
        deleted_vector_store_file = client.beta.vector_stores.files.delete(
            vector_store_id=vectorID, file_id=fileID
        )
        logging.info(deleted_vector_store_file)

    def retrieve_current_files_and_remove(vectorID):
        try:
            vector_store_files = client.beta.vector_stores.files.list(
                vector_store_id=vectorID
            )
            logging.info(vector_store_files)
            file_obj = vector_store_files.data
            for files in file_obj:
                remove_file_from_vector_store(files.id, vector_id)
        except:
            logging.error("Some error ocurred while deleting the files")

    if is_screen_used == "yes":
        # add new screen

        # Prompt 1
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Examine the uploaded UI image and perform the following tasks:\n\nIdentify Components: List each visible component (e.g., buttons, text fields, icons, background colour/gradient) along with its exact position on the screen using relative positioning.\n\nDescribe Functionality: Explain the purpose and function of each component (e.g., what happens when the user interacts with it).\n\nDetect Repeatation: Identify components that are repeated or have same functions.\n\nImportant Notes:\n\nCover All Elements: Include every visible component in the UI; do not omit any item, no matter how small.\nDetail Repetative Elements: Highlight repeated structures clearly.\nEnsure that every component is accounted for with precise descriptions.\nDo not provide extra explanation or summary.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                            },
                        },
                    ],
                }
            ],
            # max_tokens=1000,
        )

        response_1 = response.choices[0].message.content
        logging.info(response_1)

        multiple_file_name = []
        uploaded_file_ids = []

        # Prompt2
        my_assistant = client.beta.assistants.create(
            instructions="You are an Code assistant. \n   1. For the next 9-10 prompts, treat all interactions as part of a single task related to creating UI or coding from scratch. Do not lose context; keep track of all inputs and responses to ensure continuity in the design and coding process.\n        2. Build the UI or code sequentially based on user instructions. If a new prompt introduces changes or additions, integrate them without losing the overall structure and consistency of the previous work.\n        3. When generating UI, ensure that the design matches the description provided by the user up to 90%, with all specified components present. If the description evolves, adjust the code accordingly without losing context.\n",
            name="Good Assistant",
            tools=[{"type": "file_search"}],
            model="gpt-4o",
        )
        logging.info(my_assistant)

        if figma_api_data != "":

            # api_converted_data = "Based on the Figma API response data and the provided UI image, here is an organized breakdown of the components shown in the UI screen:\n\n---\n\n### A. Header Section\n\n1. **Top Navigation Bar:**\n   - **Positioning and Sizing:** \n     - Position: `x: -676, y: -2272`\n     - Size: `width: 375, height: 54`\n   - **Shapes:** Rectangle\n   - **Fill Color:** Solid, rgba(1, 1, 1, 1)\n   - **Effects (Shadows):** \n     - Type: Drop Shadow\n     - Color: rgba(60, 60, 67, 0.29)\n     - Offset: `x: 0, y: 0.33`\n   - **Constraints:** Vertical: SCALE, Horizontal: SCALE\n\n2. **Instagram Logo (Center):**\n   - **Positioning and Sizing:**\n     - Position: `x: -326, y: -2466.33`\n     - Size: `width: 88, height: 24`\n   - **Shapes:** Combination of paths and vectors (Boolean operations)\n   - **Fill Color:** Various gradients and solid fills for different parts\n\n3. **Messenger Icon (Top Right):**\n   - **Positioning and Sizing:**\n     - Position: `x: -626, y: -2486`\n     - Size: `width: 24, height: 24`\n   - **Shapes:** Vector\n   - **Fill Color:** Solid, rgba(23, 122, 240, 1)\n   - **Constraints:** Vertical: SCALE, Horizontal: SCALE\n\n4. **Add Post Icon (Top Left):**\n   - **Positioning and Sizing:**\n     - Position: `x: -576, y: -2486`\n     - Size: `width: 24, height: 24`\n   - **Shapes:** Vector\n   - **Fill Color:** Solid, rgba(0, 0, 0, 1)\n   - **Constraints:** Vertical: SCALE, Horizontal: SCALE\n\n---\n\n### B. Story Section\n\n1. **Story Thumbnails:**\n   - **Positioning and Sizing:** Varies for each thumbnail (e.g., the first thumb is at `x: -676, y: -2244`, size: `width: 76, height: 76`)\n   - **Shapes:** Oval (Vector)\n   - **Fill Type:** Image\n   - **Stroke:** Color rgba(0, 0, 0, 0.1), Width: 0.5\n   - **Constraints:** Vertical: SCALE, Horizontal: SCALE\n\n2. **Story Thumbnails (More Icon):**\n   - **Shapes:** Boolean operation of several small circles\n   - **Fill Color:** Solid, rgba(216, 216, 216, 1)\n   - **Constraints:** Vertical: SCALE, Horizontal: SCALE\n\n---\n\n### C. Main Feed Section\n\n1. **Post Image:**\n   - **Positioning and Sizing:**\n     - Position: `x: -676, y: -2216`\n     - Size: `width: 375, height: 375`\n   - **Shapes:** Rectangle\n   - **Fill Type:** Image\n\n2. **Post Header:**\n   - **Positioning and Sizing:**\n     - Position: `x: -676, y: -2272`\n     - Size: `width: 375, height: 54`\n   - **Background Color:** Solid, rgba(1, 1, 1, 1)\n   - **Shadows:** \n     - Color: rgba(60, 60, 67, 0.3)\n     - Offset: `x: 0, y: 0.33`\n\n3. **User Profile Picture (Top Left of Post Header):**\n   - **Positioning and Sizing:**\n     - Positioned within the header at: `x: -666, y: -2261`\n     - Size: `width: 32, height: 32`\n   - **Shapes:** Oval (Vector)\n   - **Fill Type:** Image\n   - **Stroke:** Color rgba(0, 0, 0, 0.1), Width: 0.5\n\n4. **Username and Location:**\n   - **Positioning and Sizing:**\n     - Positioned within the header with username at `x: -624, y: -2242`\n     - Location below username\n   - **Text Details:**\n     - Font Size: Username (17), Location (14)\n     - Font Weight: Normal/Bold for username\n     - Font Family: San Francisco\n     - Text Color for Username: rgba(38, 38, 38, 1)\n     - Text Color for Location: rgba(38, 38, 38, 1)\n\n5. **More Icon (Top Right of Post Header):**\n   - **Positioning and Sizing:**\n     - Positioned within the header at: `x: -330, y: -2246.5`\n     - Size: `width: 14, height: 3`\n   - **Shapes:** Boolean operation\n   - **Fill Color:** Solid, rgba(216, 216, 216, 1)\n\n---\n\n### D. Interaction Buttons (Under Post Image)\n\n1. **Like Button:**\n   - **Positioning and Sizing:** Left-most icon (e.g., at `x: -626, y: -1829`)\n   - **Shapes:** Vector\n   - **Fill Color:** Solid, rgba(38, 38, 38, 1)\n   - **Constraints:** Vertical: SCALE, Horizontal: SCALE\n\n2. **Comment Button:**\n   - **Positioning and Sizing:** Next to Like button (e.g., at `x: -601, y: -1829`)\n   - **Shapes:** Vector\n   - **Fill Color:** Solid, rgba(38, 38, 38, 1)\n   - **Constraints:** Vertical: SCALE, Horizontal: SCALE\n\n3. **Share Button:**\n   - **Positioning and Sizing:** Positioned to the right of the Comment button\n   - **Shapes:** Vector\n   - **Fill Color:** Solid, rgba(38, 38, 38, 1)\n   - **Constraints:** Vertical: SCALE, Horizontal: SCALE\n\n4. **Save Button:**\n   - **Positioning and Sizing:** Positioned to the right-most of the interaction buttons\n   - **Shapes:** Vector\n   - **Fill Color:** Solid, rgba(38, 38, 38, 1)\n   - **Constraints:** Vertical: SCALE, Horizontal: SCALE\n\n---\n\n### E. Post Description and Comments\n\n1. **Text Elements (like description, comments):**\n   - **Positioning and Sizing:** Below the interaction buttons section\n   - **Font Details:** \n     - Font Size: Varies (e.g., 14 for comment text)\n     - Font Weight: Normal\n     - Font Family: San Francisco\n     - Text Color: rgba(38, 38, 38, 1)\n   - **Constraints:** Vertical: TOP, Horizontal: LEFT\n\n---\n\nThis comprehensive breakdown ensures coverage of all visible components in the provided UI image, based on the Figma API response data. The data provided includes details on positioning, sizing, colors, shapes, and other relevant properties."

            api_converted_data = get_analyzed_api_data(
                figma_api_data, image_url, dir_path
            )
            # Specify the filename
            filename = dir_path + "/figma_data_file.txt"

            write_code_to_file(filename, api_converted_data)

        # creating the vector store
        vector_store = client.beta.vector_stores.create(name="Uploaded Document files")
        logging.info(vector_store)
        vector_id = vector_store.id
        logging.info(f"vector_id: {vector_id}")

        # uploading the figma data assistant
        upload_file_to_vector_store(dir_path + "/figma_data_file.txt", vector_id)
        assistant_id = my_assistant.id
        # logging.info("file_id: ",file_id)
        logging.info(f"assistant_id: {assistant_id}")
        logging.info(f"vector_id: {vector_id}")

        if assets_used:
            try:
                write_code_to_file(dir_path + "/assets_file.txt", assets_used)
                assets_file_id = upload_document_file_to_openai(
                    dir_path + "/assets_file.txt"
                )
                upload_file_to_vector_store(dir_path + "/assets_file.txt", vector_id)
            except:
                logging.error("Some error occured while using the file")
                assets_used = ""
            else:
                logging.info("Assets are empty")

        thread = client.beta.threads.create()
        thread_id = thread.id
        logging.info(thread_id)

        updated_assistant = update_assistant(vector_id)
        logging.info(updated_assistant)

        figma_info = "absoluteBoundingBox: Describes the absolute position and size of the element in the frame, here position is given in the form  of x and y coordinated with respect to the screen, so place the components at proper positions"
        assets_info = 'The assets which can be used while generating the code is mentioned in uploaded file "assets info.txt" use whichever assets used as image or icon required to generate the code.'
        if user_role == "Flutter Developer 2":
            logging.info("In flutter dev prompt mode")
            payload = [
                {
                    "type": "text",
                    "text": f'Generate a {user_role} code with MVC architecture and proper State Management for the figma UI based on UI image, figma styling data(note: figma data is uploaded in figma_data_file.json file) and  Description of the UI: " {response_1} ".\nMake separate files for reusable components, classes, and assets. Also maintain Colors and Strings as a reusable component. \n Some information about figma data is: \n  Note: The colors in figma API data is in the form of RGBA format so add accurate colors in code \n{figma_info} \n Ensure the code includes: 1. Proper error handling for each code file and method. 2. Proper commenting so that every non-coder can also understand the code. 3. Provide only the exact code with file and folder names. No need to generate code for Status bar showing battery, time, etc.',
                },
                {"type": "image_url", "image_url": {"url": image_url}},
            ]
        elif user_role == "Web Developer (HTML & CSS & JavaScript only)":
            payload = [
                {
                    "type": "text",
                    "text": f"Generate a code for given Web UI based on uploaded UI image, description of the UI and figma styling data (note: figma data is uploaded in figma_data_file.json file). Description of the UI: {response_1}. \nThe logic should be self-contained, ensuring that the code is fully functional without requiring additional user input. Refer to the common functionality steps provided in the file to create logic for each component. Avoid adding new functionalities; focus on creating exact same elements present in the UI.\n\nPlease ensure the code includes:\n\nProper error handling for each function to manage exceptions gracefully.\nDetailed comments in both the HTML and JavaScript files.\nAccurate file and folder names for the HTML, CSS, and JavaScript files involved, reflecting the exact structure needed.\nProvide the updated project structure.",
                },
                {"type": "image_url", "image_url": {"url": image_url}},
            ]
        else:
            logging.info("Generating Prompt 2")

            if assets_used != "null" and assets_used != "":
                logging.info("Using assets to generate code")
                payload = [
                    {
                        "type": "text",
                        "text": f'First breakdown step by step how you would implement this and then Generate a {user_role} code for the figma UI based on UI image, figma styling data(note: figma data is uploaded in figma_data_file.txt file) and  Description of the UI: " {response_1} ".\n Some information about figma data is: \n  Note: The colors in figma api data is in the form of RGBA fromat so make so add accurate colors in code \n{figma_info} \n No need to generate code for Status bar showing battery, time, etc.\nMake components reusable instead of hardcoding repetative components to avoid duplicate code. {assets_info}\nGenerated code must contain some explanatory comments.\nThe generated output must not contain any unwanted commentary or explanation of the output. \n Also Make sure the generated code doesnot include deprecated snippets or imports also the generated code is without any error.',
                    },
                    {"type": "image_url", "image_url": {"url": image_url}},
                ]
            else:
                logging.info("Not using assets")
                payload = [
                    {
                        "type": "text",
                        "text": f'First breakdown step by step how you would implement this and then Generate a {user_role} code for the figma UI based on UI image, figma styling data(note: figma data is uploaded in figma_data_file.txt file) and  Description of the UI: " {response_1} ".\n Some information about figma data is: \n  Note: The colors in figma api data is in the form of RGBA fromat so make so add accurate colors in code \n{figma_info} \n No need to generate code for Status bar showing battery, time, etc.\nMake components reusable instead of hardcoding repetative components to avoid duplicate code\nGenerated code must contain some explanatory comments.\nThe generated output must not contain any unwanted commentary or explanation of the output. \n Also Make sure the generated code doesnot include deprecated snippets or imports also the generated code is without any error.',
                    },
                    {"type": "image_url", "image_url": {"url": image_url}},
                ]

        response_2 = get_response(thread_id, assistant_id, payload)
        logging.info(response_2)

        # payload = [
        #     {
        #         "type": "text",
        #         "text": f"The generated code doesn't contain proper screen adaptability. The code must be such that it should adapt seamlessly to different screen sizes and orientations. The component should resize text, images, and interactive elements appropriately based on the screen size. Refer uploaded ui image to study ui. Generate entire code without any commentory and explanation",
        #     },
        #     {"type": "image_url", "image_url": {"url": image_url}},
        # ]
        # response_interim = get_response(thread_id, assistant_id, payload)
        # logging.info(response_interim)

        retrieve_current_files_and_remove(vector_id)

        logging.info("Correcting the code accoring to the ui image")
        payload = [
            {
                "type": "text",
                "text": f"The generated UI is not proper and is not matching with the UI image can you please fix the code such that it would be exactly according to the ui. See whichever elements is missing in the code or the spacing between the ui components is improper or the position of the element is improper and modify the code accordingly",
            },
            {"type": "image_url", "image_url": {"url": image_url}},
        ]

        response_3 = get_response(thread_id, assistant_id, payload)
        logging.info(response_3)
        
                # storing the user styles data files
        status_project_code = store_project_data_locally(user_id, dir_path)

        styles_id = ""
        if status_project_code["status"] == "success":
            logging.info("Uploading the styles file to vector store")

            try:
                with open(dir_path + "/project_style_data.json", "w") as file:
                    data = json.loads(status_project_code["data"])
                    json.dump(data, file)
            except:
                logging.error("error while creating the file")

            created_file = client.files.create(
                file=open(dir_path + "/project_style_data.json", "rb"),
                purpose="assistants",
            )
            logging.info(f"created_file: {created_file}")

            styles_id = created_file.id

            vector_store_file = client.beta.vector_stores.files.create(
                vector_store_id=vector_id, file_id=styles_id
            )
            logging.info(vector_store_file)
            logging.info("File was successfully uploaded")

        if status_project_code["status"] == "success":
            is_styles_used = "yes"
            logging.info("Using the styles data")
            data_content = status_project_code["data"]
            payload = f"Restructure the generated code with the exact architecture, state management, code structure, adaptive and responsive design, app constants, and error handling as specified in the coding styles below. Ensure that the code is separated into the appropriate folders and files, with clear folder and file names displayed. Maintain all existing UI components and functionalities while applying the specified styles. Thoroughly verify that no part of the original code, especially UI elements, interactions, or functionality, is lost during the restructuring process. Use detailed checks to ensure all elements are correctly styled and integrated as per the guidelines.\n\nIf any ambiguity arises in implementing styles without affecting the codes functionality, maintain the original code logic, and add comments highlighting potential adjustments needed to fully align with the coding standards.\n\nCoding Styles are as follows:\n{data_content}\n"

            
            response_5 = get_response(thread_id, assistant_id, payload)
            logging.info(response_5)


        if additional_requirements != "":
            write_code_to_file(
                dir_path + "/More requirements.txt", additional_requirements
            )
            logging.info("More requirements was successfully made")
            upload_file_to_vector_store(dir_path + "/More requirements.txt", vector_id)
        else:
            logging.info("Additional requirements doesn't exist")

        upload_file_to_vector_store("Common_Functionality.json", vector_id)

        # prompt 3

        logging.info(thread_id)

        if additional_requirements == "":
            payload = 'The current generated code needs functionalities added to it. Refer to "Common_Functionality.json" for available functionality descriptions. This file contains common functionalities with the following details:\n- *Functionality Name*: The name of the functionality.\n- *Description*: Steps to implement the functionality.\n- *Type*: Specifies the type of UI element the functionality is linked to.\n\n### Instructions:\n1. *Strictly Maintain the Existing UI Structure*: Do not modify, alter, or disturb the existing UI elements, layout, or structure in any way. Only focus on adding the required functionalities.\n   \n2. *Match Functionalities to Existing UI Elements*: Implement only those functionalities that directly correspond to UI elements already present in the code. Avoid creating new UI elements (e.g., text boxes, buttons) that are not already in the design, even if the functionality suggests it.\n\n3. *Avoid Unnecessary Additions*: If a functionality in "Common_Functionality.json" does not naturally fit the existing UI elements, do not implement it. For instance, if the code has a "Sign up with Email" button but no email text box, do not add the text box. Only apply functionalities that match the current UI elements without altering or adding new UI components.\n\n4. *Do Not Assume Functionalities*: Only implement functionalities that are clearly defined and directly applicable to existing elements. Do not assume or create new features that are not explicitly needed by the UI.\n\n5. *Seamless Integration*: Ensure that added functionalities integrate smoothly with the existing code without breaking or changing the design or structure of the UI.\n\n6. *No Extra Commentary or Unnecessary Code Changes*: Focus solely on implementing the relevant functionalities. Avoid adding any explanations or changes that are not directly related to the functional aspects.\n\n### Output Expectations:\n- *Functionality Alignment*: Ensure all functionalities align with the UI elements present in the code and do not introduce new elements.\n- *Maintain UI Integrity*: The structure and design of the existing UI code must remain completely unchanged.\n- *Precision and Relevance*: Only add what is necessary and relevant. Do not add functionalities that require UI modifications unless those elements already exist in the current code.'
            logging.info("Generating prompt 3 normally")
        else:
            payload = 'The current generated code needs functionalities added to it. Refer to "More requirements.txt" for available functionality descriptions. This file contains functionalities description, analyze it and strictly add every mentioned functionality from the file.\n\n### Instructions:\n1. *Strictly Maintain the Existing UI Structure*: Do not modify, alter, or disturb the existing UI elements, layout, or structure in any way. Only focus on adding the mentioned functionalities.\n\n2. *Do Not Assume Functionalities*: Only implement functionalities that are clearly defined and directly applicable to existing elements. Do not assume or create new features that are not explicitly needed by the UI.\n\n3. *Seamless Integration*: Ensure that added functionalities integrate smoothly with the existing code without breaking or changing the design or structure of the UI.\n\n4. *No Extra Commentary or Unnecessary Code Changes*: Focus solely on implementing the relevant functionalities. Avoid adding any explanations or changes that are not directly related to the functional aspects.\n\n### Output Expectations:\n- *Functionality Alignment*: Ensure all functionalities align with the UI elements present in the code and do not introduce new elements.\n- *Maintain UI Integrity*: Each functionality should be added and the structure and design of the existing UI code must remain completely unchanged.'
            logging.info("Generating prompt 3 for with More requirements")

        response_4 = get_response(thread_id, assistant_id, payload)
        logging.info(response_4)

        response_final = response_4
        
        if status_project_code["status"] == "success":
            payload = [{"type": "text", "text": f"{response_final}\n First breakdown step by step how and why you created this code also Can you check the code line by line and ensure that the code is properly structured and adhered to coding standards given above as well as functionalities in generated code. Ensure that the code contains proper explainatory commented and is according to the coding standards. Here is coding standards to check properly in json string: \n {data_content} \n Additionally, make sure that the code doesnot contain deprecated or error implementation."}]
            response_interim1 = get_response(thread_id,assistant_id,payload)
            logging.info(response_interim1)
        else:
            payload = [{"type": "text", "text": f"{response_final}\n First breakdown step by step how and why you created this code also Can you check the code line by line and make sure that the code is properly structured and adhered to General coding standards as well as functionalities in generated code. Ensure that the code contains proper explainatory commented and is according to the best practice coding standards. \n Additionally, make sure that the code doesnot contain deprecated or error implementation."}]
            response_interim1 = get_response(thread_id,assistant_id,payload)
            logging.info(response_interim1)


        retrieve_current_files_and_remove(vector_id)


        logging.info("Correcting the code accoring to the ui image")
        payload = [
            {
                "type": "text",
                "text": f"Enhance the generated code, as the generated code might miss certain UI elements, can you please fix the code such that it would be about 95% match according to the ui (keeping the adaptablity and coding styles unchanged.).Do not modify, alter, or loose the existing UI elements, layout, or structure in any way. See whichever elements is missing in the code or the spacing between the ui components is improper or the position of the element is improper and modify the code accordingly, Image url attached. Provide project structure with name for each file and try to give complete code for ui",
            },
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
        response_corr = get_response(thread_id, assistant_id, payload)
        logging.info(response_corr)

        isValid = False
        try:
            api_data_json_1 = json.loads(api_data)

            api_data_json = api_data_json_1

            isValid = True
            logging.info("It's a valid JSON")

            for files in api_data_json:
                file_name = get_file_name(files["file_path"])
                files["file_path"] = file_name
            logging.info(api_data_json)

        except ValueError as e:
            logging.error(f"Invalid JSON: {e}")

        if isValid:
            for item in api_data_json:
                code = item["content"]
                onlyName = remove_extension(item["file_path"])
                filename = dir_path + "/" + onlyName + ".txt"
                multiple_file_name.append(filename)
                write_code_to_file(filename, code)
                logging.info(f"Code written to {filename}")

                if onlyName == "Readme":
                    read_me_content = code

            for files in multiple_file_name:
                upload_file_to_vector_store(files, vector_id)

            assistant = client.beta.assistants.update(
                assistant_id=assistant_id,
                tool_resources={"file_search": {"vector_store_ids": [vector_id]}},
            )
            logging.info(f"updated assistant: {assistant}")

        else:
            logging.error("No files uploaded")

        payload = f"Generated response signifies a new screen along with its components and state management that need to be added into the existing project. Note: Don't replace or remove any existing screen or component. If the project follows a particular statemanagement then add the newly generate states into the existing state. The project files are uploaded, analyze it and check if there are any components which are similar in new screen and if there is then 're-use' the code and don't do the repeatative work. Generate accurate response and give full code. This is the existing project structure:\n{read_me_content}.\n please digest this information and once you understand existing code then Give me production ready code which is formatted, with their projected structure and file/folder name for each genereated code."
        response_6 = get_response(thread_id, assistant_id, payload)

        
        if response_6!="Failed":
            if is_styles_used == "no":
                alert_statement = "Seems like personalization process was not done. The generated code will be based on generic coding standards. If you would like a personalized code, please go to home page, click personalization button and retry the prompt\n\n"
                response_6 = alert_statement + response_6    

        logging.info(response_6)

        final_response = response_6
        
        figma_and_assets = {"figma_analysed_data":api_converted_data,"assets_used":assets_used}

    else:
        # Add new functionality & update functionality

        txt_content = f"You are given two inputs: a UI image and a message. Your task is to identify if any elements mentioned in the message are present in the UI image. Specifically, look for any components or words from the message within the UI. Compare adjacent word combinations from the message with elements visible in the UI. If you find any matches, output them as an array of strings representing the words from the message that match elements in the UI. If no matches are found, output an empty array.\n\nInput Examples:\n\nMessage: \"In main_page.dart, add a search bar and modify the code accordingly.\"\n\nOutput: [\"search bar\"] if the corresponding UI element is found.\n\nMessage: \"Update the button styles in home_page.dart.\"\n\nOutput: [] if no corresponding elements are found.\n\nOutput Requirements:\n\nIf an element is found, output in the format: [\"Element 1\", \"Element 2\"].\nIf no elements are found, output an empty array: [].\n\nInput:\n\nMessage: {prompt}\nUI Image: [Uploaded image]\n\nEnsure the response strictly follows the format specified, with no additional commentary or explanation."
        list_of_elements_found = []

        # Prompt 1
        if image_url !="" and api_data !="":
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": txt_content},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url,
                                },
                            },
                        ],
                    }
                ],
                # max_tokens=1000,
            )

            response_elems = response.choices[0].message.content
            logging.info(response_elems)

            list_of_elements_found = json.loads(response_elems)
            logging.info(list_of_elements_found)

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
            instructions=f"{role_description}\nGenerate {user_role} code based on user prompt.\nRemember:\n1. Generate output by studying the uploaded files\n2. Uploaded Readme.txt contain the proper files and folder structure, use the given directory structure to create files in it.\n3. Link the output to existing project files\n4. Generate code with proper file and folder name",
            name=f"{user_role} Code Assist",
            tools=[{"type": "file_search"}],
            model="gpt-4o",
        )
        assistant_id = my_assistant.id
        logging.info(f"assistant_id: {assistant_id}")

        isValid = False
        try:
            api_data_json_1 = json.loads(api_data)
            api_data_json = api_data_json_1
            isValid = True
            logging.info("It's a valid JSON")

            for files in api_data_json:
                file_name = get_file_name(files["file_path"])
                files["file_path"] = file_name
            logging.info(api_data_json)

        except ValueError as e:
            logging.info(f"Invalid JSON: {e}")

        vector_id = ""  # changes done

        if isValid:
            multiple_file_name = []
            uploaded_file_ids = []
            for item in api_data_json:
                code = item['content']
                filename = dir_path + '/' + remove_extension(item['file_path']) + '.txt'
                multiple_file_name.append(filename)
                write_code_to_file(filename, code)
                logging.info(f"Code written to {filename}")

            store_name = "Uploaded files to Store"
            vector_store = client.beta.vector_stores.create(name=store_name)
            vector_id = vector_store.id

            for files in multiple_file_name:
                upload_file_to_vector_store(files, vector_id)

            if additional_requirements != "":
                try:
                    write_code_to_file(dir_path + "/More requirements.txt", additional_requirements)
                    upload_file_to_vector_store(dir_path + "/More requirements.txt", vector_id)
                    multiple_file_name.append(dir_path + "/More requirements.txt")
                    logging.info("More requirements file was successfully uploaded")
                except:
                    logging.info("Error uploading more requirements file")

            if assets_used != "null" and assets_used != "":
                try:
                    write_code_to_file(dir_path + "/assets_file.txt", assets_used)
                    assets_file_id = upload_document_file_to_openai(dir_path + "/assets_file.txt")
                    vector_store_file = client.beta.vector_stores.files.create(
                        vector_store_id=vector_id,
                        file_id=assets_file_id
                    )
                    logging.info(vector_store_file)
                except:
                    logging.info("Some error occurred while using the file")
                    assets_used = ""

            updated_assistant = client.beta.assistants.update(
                assistant_id=assistant_id,
                tool_resources={"file_search": {"vector_store_ids": [vector_id]}},
            )
            logging.info(f"updated assistant: {updated_assistant}")

            empty_thread = client.beta.threads.create()
            logging.info(empty_thread)
            thread_id = empty_thread.id

            if len(list_of_elements_found) == 0:
                logging.info("No element found so not using figma data and coding styles")
                if additional_requirements == "":
                    payload = f"{prompt} \n. The code is uploaded refer it. Don't provide any unwanted explanation or commentary, give me only exact code with file and folder name."
                else:
                    payload = prompt + " . Make the necessary changes as specified in the \"More requirements.txt\" file uploaded. Provide only the exact code with the specified file and folder names, without any additional explanations or commentary."

                response = get_response(thread_id, assistant_id, payload)
                logging.info(response)

                final_response = response

                figma_and_assets = {"figma_analysed_data": "", "assets_used": assets_used}

            else:
                logging.info("Element found so need to use figma data and coding styles")
                figma_data = get_exact_api_data(figma_api_data, response_elems, image_url, dir_path)

                if additional_requirements == "":
                    logging.info("Not using additional requirements")
                    payload = [{"type": "text", "text": f"{prompt} . \nThe code is uploaded refer it. \nAnalyze the ui image and figma data and make modifications according to it \nThe figma data that would be required to create the ui element is given below \n{figma_data}\nThe assets used in the project is also uploaded as \"assets_file.txt\" if required make use of assets from it to make modifications\nDon\'t provide any unwanted explanation or commentary, give me only exact code with file and folder name."}, {"type": "image_url", "image_url": {"url": image_url}}]
                else:
                    logging.info("using additional requirements")
                    payload = [{"type": "text", "text": prompt + f" . Make the necessary changes as specified in the \"More requirements.txt\" file uploaded. \nAnalyze the ui image and figma data and make modifications according to it \nThe figma data that would be required to create the ui element is given below \n{figma_data}\nThe assets used in the project is also uploaded as \"assets_file.txt\" if required make use of assets from it to make modifications\nProvide only the exact code with the specified file and folder names, without any additional explanations or commentary."}, {"type": "image_url", "image_url": {"url": image_url}}]

                figma_and_assets = {"figma_analysed_data": figma_data, "assets_used": assets_used}
            
                response = get_response(thread_id, assistant_id, payload)
                logging.info(response)

                payload = [{"type": "text", "text": f"The generated code for the component {response_elems} doesn't match with the one present in the ui, can you make it similar to the one present in ui image. Generate the entire code for {response_elems}"},{"type": "image_url","image_url": {"url": image_url}}]
                response = get_response(thread_id, assistant_id, payload)
                logging.info(response)
                
                final_response = response
                
                figma_and_assets = {"figma_analysed_data":figma_data,"assets_used":assets_used}
        else:
            logging.info("Didn't get data from extension")
            final_response = "Error while fetching data from extension"

    if uploaded_file_ids:
        logging.info("deleting openai files")
        for fileID in uploaded_file_ids:
            dlt_file = client.files.delete(fileID)
            logging.info(f"Deleted Files: {dlt_file}")

    deleted_vector_store = client.beta.vector_stores.delete(
        vector_store_id=vector_id
    )
    logging.info(f"Deleted vector store: {deleted_vector_store}")

    dlt_assistant = client.beta.assistants.delete(assistant_id)
    logging.info(f"Deleted Assistant: {dlt_assistant}")

    def delete_local_file(file_path):
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                logging.info(f"Deleted file: {file_path}")
            else:
                logging.info(f"File not found: {file_path}")
        except Exception as e:
            logging.info(f"Error deleting file {file_path}: {e}")

    delete_local_file(dir_path + "/figma_api_data.json")
    delete_local_file(dir_path + "/output.png")

    figma_and_assets = {"figma_analysed_data": figma_and_assets, "assets_used": assets_used}

    helping_data = {
        "figma_data": json.dumps(figma_and_assets, indent=4, ensure_ascii=False),
        "user_requirements_data": additional_requirements,
        "apps_data": extra_data,
        "project_code": json.dumps(api_data_json, indent=4, ensure_ascii=False),
    }

    delete_folder_recursive(dir_path)

    if final_response != "Failed":
        return {"status": "success", "assistant_id": assistant_id, "thread_id": thread_id, "response": final_response, "data": json.dumps(helping_data, indent=4, ensure_ascii=False)}
    else:
        return {"status": "failed", "data": json.dumps(helping_data, indent=4, ensure_ascii=False)}



def get_exact_api_data(api_data, list_of_elem, image_url, dir_path):
    logging.info(f"List of elements: {list_of_elem}")
    # logging.info(f"api_data: {api_data}")

    if api_data != "":
        try:
            with open(dir_path + "/api_data_file.json", 'w') as file:
                json.dump(json.loads(api_data), file)
                logging.info("API JSON data was successfully written to file")
        except:
            logging.info("Some error occurred while uploading the data")

            def write_code_to_file(filename: str, code: str) -> None:
                with open(filename, 'w', encoding='utf-8') as file:
                    file.write(code)

            write_code_to_file(dir_path + "/api_data_file.json", api_data)

    my_assistant = client.beta.assistants.create(
        instructions=f"Given a list of elements and a corresponding Figma API response, for each UI element present in list, provide the following details. Identify frame with the text included in them: \n(\nPositioning and Sizing\nShapes\nText Elements\nConstraints i.e spacing between the components\nImages used(which component is used as an image [usually type of Vector is considered as image])\nColors(Note: if gradient used mentioned that also)\nBorders represent as strokes in figma (like border color and width) and Shadows\nCorner Radius and Effects\nPadding\nFont Details(fontFamily,fontSize, fontWeight,color in rgba format[present in fills])\nspacing\nstyles\n)\nthese details need to be fetched out from figma response and organize them in readable format.\nGenerate output for the element present in the below list. No need to generate it for every ui element\n\nInput List: {list_of_elem}\n\nOutput example\ne.g A. Header Section\n\t1. Icon1:\n\t   Positioning:.....\n\t   Shapes.....\t\n\t   ..........",
        name="API Code Analyzer",
        tools=[{"type": "file_search"}],
        model="gpt-4o",
    )

    assistant_id = my_assistant.id
    logging.info(f"assistant_id: {assistant_id}")

    store_name = "Uploaded files to Store"
    vector_store = client.beta.vector_stores.create(name=store_name)
    vector_id = vector_store.id
    logging.info(f"vector_id: {vector_id}")

    def upload_file_to_vector_store(file_name, vector_id):
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
        logging.info(vector_store_file)
        return file_id

    file_id = upload_file_to_vector_store(dir_path + "/api_data_file.json", vector_id)

    assistant = client.beta.assistants.update(
        assistant_id=assistant_id,
        tool_resources={"file_search": {"vector_store_ids": [vector_id]}},
    )
    logging.info(f"updated assistant: {assistant}")

    empty_thread = client.beta.threads.create()
    thread_id = empty_thread.id
    logging.info(empty_thread)

    payload = [{"type": "text", "text": "Analyze the figma data and generated response must cover every mentioned points"}, {"type": "image_url", "image_url": {"url": image_url}}]
    response_final = get_response(thread_id, assistant_id, payload)
    logging.info(response_final)

    file_path = dir_path + "/api_data_file.json"
    try:
        if os.path.isfile(file_path):
            os.remove(file_path)
            logging.info(f"Deleted file: {file_path}")
        else:
            logging.info(f"File not found: {file_path}")
    except Exception as e:
        logging.info(f"Error deleting file {file_path}: {e}")

    logging.info(file_id)

    try:
        deleted_vector_store_file = client.beta.vector_stores.files.delete(
            vector_store_id=vector_id,
            file_id=file_id
        )
        logging.info(deleted_vector_store_file)
    except:
        logging.info("Error while deleting the file")

    try:
        deleted_vector_store = client.beta.vector_stores.delete(
            vector_store_id=vector_id
        )
        logging.info(deleted_vector_store)
    except:
        logging.info("Error while deleting the store")

    try:
        response = client.beta.assistants.delete(assistant_id)
        logging.info(response)
    except:
        logging.info("Error while deleting the assistant")

    return response_final

def get_analyzed_api_data(api_data, image_url, dir_path):
    logging.info(f"image_url: {image_url}")
    logging.info(f"api_data: {api_data}")

    if api_data != "":
        try:
            with open(dir_path + "/api_data_file.json", "w") as file:
                json.dump(json.loads(api_data), file)
                logging.info("API JSON data was successfully written to file")
        except:
            logging.error("Some error occurred while uploading the data")

            def write_code_to_file(filename: str, code: str) -> None:
                with open(filename, "w", encoding="utf-8") as file:
                    file.write(code)

            write_code_to_file(dir_path + "/api_data_file.json", api_data)

    my_assistant = client.beta.assistants.create(
        instructions='Given a UI image and a corresponding Figma API response, analyze the UI elements in the image based on the details provided in the API response. For each UI element, provide the following details for each UI component. Identify each frame with the text included in them. If some Component are similar no need to create separate data for them, create a combined data for them: \n{\nPositioning and Sizing\nShapes\nText Elements\nConstraints i.e spacing between the components\nImages used(which component is used as an image [usually type of Vector is considered as image])\nColors(Note: if gradient used mentioned that also)\nBorders represent as strokes in figma (like border color and width) and Shadows\nCorner Radius and Effects\nPadding\nFont Details(fontFamily,fontSize, fontWeight,color in rgba format[present in fills])\nspacing\nstyles\n}\nthese details need to be fetched out from figma response and organize them in readable format.\nCover each and every component present in the ui screen don\'t miss out any \nNote: Suppose some component doesn\'t have a parameter value or the value is "null" or "none" so don\'t include that parameter in that component description\ne.g A. Header Section\n\t1. Icon1:\n\t   Positioning:.....\n\t   Shapes.....\t\n\t   ..........',
        name="API Code Analyzer",
        tools=[{"type": "file_search"}],
        model="gpt-4o",
    )

    assistant_id = my_assistant.id
    logging.info(f"assistant_id: {assistant_id}")

    store_name = "Uploaded files to Store"
    vector_store = client.beta.vector_stores.create(name=store_name)
    vector_id = vector_store.id
    logging.info(f"vector_id: {vector_id}")

    def upload_file_to_vector_store(file_name, vector_id):
        file = client.files.create(file=open(file_name, "rb"), purpose="assistants")
        file_id = file.id

        vector_store_file = client.beta.vector_stores.files.create(
            vector_store_id=vector_id, file_id=file_id
        )
        logging.info(f"{file_name} was successfully stored")
        logging.info(vector_store_file)
        return file_id

    file_id = upload_file_to_vector_store(dir_path + "/api_data_file.json", vector_id)

    assistant = client.beta.assistants.update(
        assistant_id=assistant_id,
        tool_resources={"file_search": {"vector_store_ids": [vector_id]}},
    )
    logging.info(f"updated assistant: {assistant}")

    empty_thread = client.beta.threads.create()
    thread_id = empty_thread.id
    logging.info(empty_thread)

    payload = [
        {
            "type": "text",
            "text": "Analyze the figma data and generated response must cover every mentioned points",
        },
        {"type": "image_url", "image_url": {"url": image_url}},
    ]
    response_final = get_response(thread_id, assistant_id, payload)
    logging.info(response_final)

    file_path = dir_path + "/api_data_file.json"
    try:
        if os.path.isfile(file_path):
            os.remove(file_path)
            logging.info(f"Deleted file: {file_path}")
        else:
            logging.warning(f"File not found: {file_path}")
    except Exception as e:
        logging.error(f"Error deleting file {file_path}: {e}")

    logging.info(file_id)

    try:
        deleted_vector_store_file = client.beta.vector_stores.files.delete(
            vector_store_id=vector_id, file_id=file_id
        )
        logging.info(deleted_vector_store_file)
    except:
        logging.error("Error while deleting the file")

    try:
        deleted_vector_store = client.beta.vector_stores.delete(
            vector_store_id=vector_id
        )
        logging.info(deleted_vector_store)
    except:
        logging.error("Error while deleting the store")

    try:
        response = client.beta.assistants.delete(assistant_id)
        logging.info(response)
    except:
        logging.error("Error while deleting the assistant")

    return response_final


@app.post("/analyze-api-data/")
async def analyze_api_data(item: Item):
    user_id = item.id
    image_url = item.prompt
    api_data = item.data

    dir_path = create_user_directory(user_id)
    logging.info("Directory path: %s", dir_path)

    response = get_analyzed_api_data(api_data, image_url, dir_path)

    logging.info("Response: %s", response)

    return {"status": "success", "response": response}


def store_user_styles(user_id, project_name, overview, user_nature):
    url = "http://35.85.112.192/api/store-project-data"

    # Define the headers
    headers = {
        "Accept": "application/json",
        "X-API-KEY": "JGIp4AWFmI",
        "Content-Type": "application/json",
    }

    # Define the body
    body = {
        "user_id": user_id,
        "project_name": project_name,
        "overview": overview,
        "user_nature": user_nature,
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
    logging.info(dir_path)

    def remove_extension(filename: str) -> str:
        return filename.rsplit(".", 1)[0]

    def write_code_to_file(filename: str, code: str) -> None:
        with open(filename, "w", encoding="utf-8") as file:
            file.write(code)

    def get_file_name(file_path):
        return os.path.basename(file_path.replace("\\", "/"))

    multiple_file_name = []
    uploaded_file_ids = []
    if api_data:
        for files in api_data:
            file_name = get_file_name(files["file_path"])
            files["file_path"] = file_name
        logging.info(api_data)

        for item in api_data:
            code = item["content"]
            filename = dir_path + "/" + remove_extension(item["file_path"]) + ".txt"
            multiple_file_name.append(filename)
            write_code_to_file(filename, code)
            logging.info(f"Code written to {filename}")

    my_assistant = client.beta.assistants.create(
        instructions="""
        You are an expert software engineer and code reviewer. Your task is to analyze the following code to understand the user's unique coding style. Identify the user's coding conventions, commenting style, code structure, organization ,specific patterns, error handling, indentation style, use of libraries or frameworks, and fetch the project name if available. Provide a detailed analysis in JSON format.

        Your analysis should cover the following aspects:
        * Project Name:
        Identify and describe the project name if available within the code or comments. If not available, find out the project name from the file path, or find out the root directory name.

        * Architecture used:
        Identify the architecture by determine the architectural pattern used in the project, such as MVC, MVVM, MVP, or other architectures. Look for specific structuring of code, the separation of concerns, and naming conventions that indicate the architecture type, like controllers, models, views, view models, or presenters and finally explain how is the architecture used by used in code.

        * State Management (if applicable):
        Identify which state management approach is used in the project (e.g., for flutter - Provider or Riverpod or Getx or any other) and explain what technique of state management with particular library user prefers to work and how user manages state.

        * Code Structure & Organization:
        Describe the detailed overall structure of the code organization, explain how directories and files are stored also what are levels of directories. Use of object-oriented principles, Patterns in class design (e.g., use of classes, functions, modules), such as inheritance and composition , Preferred length and complexity of functions and methods.
        Describe in details how are files organised in the project.
        (for eg. :The project follows a modular approach with a structured hierarchy within the lib directory. For each screen, there is a dedicated folder, and within each screen's folder, there are subfolders for different MVVM/MVC components such as models, view models, utilities, views (presentation), etc. Example Structure: lib/modules/\nhome/\nmodel/\nview_model/\nutils/\nview/\nhome_screen.dart, lib/profile/\nmodel/\nview_model/\nutils/\nview/\nprofile_screen.dart. Platform-specific directories such as android/, ios/, linux/, macos/, and web/ are maintained outside the lib directory for platform-dependent code. This structure ensures that each screen in the app is organized within its folder, making the project modular and scalable. Example Structure: `lib/modules/\<screen_name>/<other folders for screen name>`, ) 

        * App constants:
        Identify the use of constants for managing application-wide elements such as strings, paths, colors, styles, and themes. Look for examples like:
            •	Strings: Constant string definitions (const val APP_NAME = "MyApp" in Kotlin, static let appName = "MyApp" in Swift).
            •	Paths: Centralized path management (public static final String IMAGE_PATH = "/assets/images/" in Java).
            •	Colors: Defined color constants (export const PRIMARY_COLOR = '#FF5733'; in JavaScript).
            •	Styles: Consistent style definitions (const TextStyle headingStyle = TextStyle(fontSize: 20); in Flutter).
            •	Themes: Theme management (ThemeData(primaryColor: Colors.blue) in Flutter, @mixin theme-variables { $primary-color: #123456; } in SCSS).”

        * Adaptive and Responsive design:
        Examine the code to determine the methods used for creating adaptive and responsive designs across various UI frameworks. Look for techniques such as media queries (e.g., CSS @media rules), flexible layouts (e.g., Flexbox, Grid in web, StackPanel in WPF), adaptive component sizing (e.g., useWindowDimensions() in React Native, UIScreen.main.bounds in Swift), and responsive design patterns like auto-layouts or constraints in iOS, and ConstraintLayout in Android.

        * Error Handling:
        Describe the approach to error handling in the code (e.g., use of try/except blocks, custom error messages). Identify how the user handles exceptions, errors, and edge cases in the code. Look for error messages, logging, and error handling strategies.
        
        * Use of Libraries/Frameworks:
        Identify any libraries or frameworks used in the code and describe how they are utilized, Patterns for injecting dependencies (if applicable). Look for common libraries or frameworks used in the project, such as network libraries, image loading libraries, state management libraries, etc. Explain how these libraries are integrated into the project and how they are used to enhance the functionality of the application.

        * Overall Coding Habits:
        Provide any additional insights into the user's coding habits and style. refer to the consistent behaviors and practices that a programmer or developer follows while writing, maintaining, and reviewing code, analyse reusable component in this field, inform does user reuses components like search bar, adPlaceholder etc.
        [eg.  (for flutter) User uses network images only and SVG for icons, Image and elements size adapt dynamically using media query technique. No image size is hardcoded.]    

        Provide the analysis in the following JSON format and use description given above:
        NOTE: do not include any text other than JSON as this may break my code
        json
        [
        {
        "project_name": "name of the project"
        },
        {
        "architecture_used": "description of architecture used in project",
        "state_management": "description of state management used",
        "code_structure": "description of code structure and c ode organization",
        "app_constants":"descriptions for app constants",
        "adaptive_responsive_design":"description of adaptive and responsive design",
        "error_handling": "description of error handling",
        "libraries_frameworks": "description of libraries or frameworks used",
        "overall_coding_habits": "other coding habits of user"
        }
        ]

        Analyse these details and provide a detailed analysis in above format only.
        Dont provide any unwanted explanation or commentory, give only the json object.
        End goal of this task is to understand the user and user's coding style so that other AI can code exactly like users preference.
        """,
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
        with open(file_name, "rb") as file:
            try:
                uploaded_file = client.files.create(file=file, purpose="assistants")
                file_id = uploaded_file.id
                uploaded_file_ids.append(file_id)
                vector_store_file = client.beta.vector_stores.files.create(
                    vector_store_id=vector_id, file_id=file_id
                )
                logging.info(file_name + " was successfully stored")
                logging.info(vector_store_file)
            except:
                logging.info(f"Not able to store file {file_name}")

    for files in multiple_file_name:
        upload_file_to_vector_store(files, vector_id)

    assistant = client.beta.assistants.update(
        assistant_id=assistant_id,
        tool_resources={"file_search": {"vector_store_ids": [vector_id]}},
    )
    logging.info(f"Updated assistant: {assistant}")

    empty_thread = client.beta.threads.create()
    thread_id = empty_thread.id
    logging.info(f"Thread ID: {thread_id}")

    payload = [
        {
            "type": "text",
            "text": "Analyze all the uploaded files and give proper analysis",
        }
    ]
    response_final = get_response(thread_id, assistant_id, payload)
    logging.info(f"Response: {response_final}")

    if uploaded_file_ids:
        logging.info("Deleting OpenAI files")
        for fileID in uploaded_file_ids:
            dlt_file = client.files.delete(fileID)
            logging.info(f"Deleted Files: {dlt_file}")

    try:
        deleted_vector_store = client.beta.vector_stores.delete(
            vector_store_id=vector_id
        )
        logging.info(f"Deleted vector store: {deleted_vector_store}")
    except Exception as e:
        logging.info(f"Error while deleting the store: {e}")

    try:
        response = client.beta.assistants.delete(assistant_id)
        logging.info(f"Deleted assistant: {response}")
    except Exception as e:
        logging.info(f"Error while deleting the assistant: {e}")

    if response_final == "Failed":
        return {"status": "failed"}
    else:
        response_text = response_final
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]

        delete_folder_recursive(dir_path)

        try:
            json_obj = json.loads(response_text)
            project_name = json_obj[0]["project_name"]
            user_nature = json_obj[1]
            # stored_response = store_user_styles(user_id, project_name, "", user_nature)
            # logging.info(f"Stored response: {stored_response}")
            return {
                "status": "success",
                "project_name": project_name,
                "response": json.dumps(user_nature, indent=4),
            }
        except:
            logging.info("Error while fetching the data")
            return {"status": "failed"}

@app.post("/structure-code/")
async def structure_code(item: Item):
    ai_response = item.data
    logging.info("Structuring the code")
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
        "role": "system",
        "content": [
            {
            "type": "text",
            "text": "You are an assistant that specializes in converting structured plain text input into a JSON object format. Your goal is to ensure that the input is accurately transformed into a JSON object that can be used for further processing.\n\n- Identify file paths and their associated content blocks.\n- The input will be structured in a way where each file starts with a line formatted as `### <file_path>`.\n- The content of each file is enclosed within triple backticks and specifies the language (e.g., ` ```dart `).\n- The output JSON should have a \"project_name\" field and a \"files\" array.\n- Each file object in the \"files\" array should contain:\n  - \"path\": the relative path of the file.\n  - \"content\": the complete content of the file as a string, preserving line breaks.\n- Ensure proper formatting, escaping special characters where necessary, and validate the JSON to ensure it conforms to the expected schema.\n\nUse the following schema as a guide:\n```json\n{\n  \"type\": \"object\",\n  \"properties\": {\n    \"project_name\": {\n      \"type\": \"string\"\n    },\n    \"files\": {\n      \"type\": \"array\",\n      \"items\": {\n        \"type\": \"object\",\n        \"properties\": {\n          \"path\": {\n            \"type\": \"string\"\n          },\n          \"content\": {\n            \"type\": \"string\"\n          }\n        },\n        \"required\": [\"path\", \"content\"]\n      }\n    }\n  },\n  \"required\": [\"project_name\", \"files\"]\n}\n\nmake sure the conversion process is accurate and error free and no code to lose during conversion."
            }
        ]
        },
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": f"Convert the following structured text into a JSON object. The text includes file paths and their respective content blocks. The output should include a project name and a list of files with their paths and contents as shown in the schema.\n\nInput:\n{ai_response}"
            }
        ]
        },
    ],
    temperature=0.6,
    max_tokens=15339,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    response_format={
        "type": "json_object"
    }
    )
    
    response_code = response.choices[0].message.content
    logging.info(response_code)

    return {"status":"success","response":response_code}