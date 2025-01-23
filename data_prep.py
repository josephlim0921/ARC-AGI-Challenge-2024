from data_augmentation.main import *
from data_augmentation.dsl import *
from data_augmentation.generators import *
from data_augmentation.utils import *
from data_augmentation.verifiers import *
import os
import json
import pandas as pd
import re
import numpy as np
import random
import logging
import shutil


def load_tasks_from_file(task_set):
    with open(task_set['challenges'], "r") as tasks:
        challenges = json.load(tasks)
    with open(task_set['solutions'], "r") as tasks:
        solutions = json.load(tasks)
    return challenges, solutions

def json_task_to_string(challenge_tasks: dict, task_id: str, test_input_index: int) -> str:
    json_task = challenge_tasks[task_id]

    final_output = "Training Examples\n"
    train_tasks = json_task['train']
    test_task = json_task['test']

    for i, task in enumerate(train_tasks):
        final_output += f"Example {i + 1}: Input\n[{', '.join(map(str, task['input']))}]\n\n"
        final_output += f"Example {i + 1}: Output\n[{', '.join(map(str, task['output']))}]\n\n"

    final_output += "Test\n["
    final_output += ', '.join(map(str, test_task[test_input_index]['input']))
    final_output += "]"

    final_output = final_output.replace("],", "],\n")

    return final_output

def format_array(array):
    """
    Format a 2D array to match the style: all rows aligned, brackets without indentation.
    
    :param array: 2D list to format.
    :return: String representation of the array.
    """
    formatted_rows = [f"[{', '.join(map(str, row))}]" for row in array]
    return "[" + ",\n ".join(formatted_rows) + "]"


if __name__ == "__main__":

    #---------------Setting up the challenges/solutions for train and validation, and creating empty train_df and empty val_df------------#
    
    base_path = "arc-prize-2024/"
    task_sets = {
        'training': {
            'challenges': f'{base_path}arc-agi_training_challenges.json',
            'solutions': f'{base_path}arc-agi_training_solutions.json',
        },
        'evaluation': {
            'challenges': f'{base_path}arc-agi_evaluation_challenges.json',
            'solutions': f'{base_path}arc-agi_evaluation_solutions.json',
        }
    }

    # Load data
    train_challenges, train_solutions = load_tasks_from_file(task_set=task_sets['training'])
    val_challenges, val_solutions = load_tasks_from_file(task_set=task_sets["evaluation"])

    train_df = pd.DataFrame(columns=["id", "input", "output"])
    val_df = pd.DataFrame(columns=["id", "input", "output"])

    #------------Creating the train_dataset.csv and val_dataset.csv------------------#

    for count, (id, _) in enumerate(train_challenges.items()):
        for test_id in range(len(train_solutions[id])):
            input = json_task_to_string(train_challenges, id, test_id)
            output = train_solutions[id][test_id]
            output = str(output).replace("],", "],\n")
            train_df.loc[len(train_df)] = [id, input, output]

    for id, _ in val_challenges.items():
        for test_id in range(len(val_solutions[id])):
            input = json_task_to_string(val_challenges, id, test_id)
            output = val_solutions[id][test_id]
            output = str(output).replace("],", "],\n")
            val_df.loc[len(val_df)] = [id, input, output]

    train_df.to_csv("data/train_dataset.csv", index=False)  # Save without the index
    train_df.to_csv("data/train_dataset_augmented.csv", index=False) # Creating a dataset in which the augmented samples will go in
    val_df.to_csv("data/val_dataset.csv", index=False)


    #------------------Generating Augmented Dataset-------------------#
    """
    num_examples = 100
    diff_lb=0.2
    diff_up=0.8
    path = 'augmented_data_v2'

    if os.path.exists(path):
        shutil.rmtree(path)

    # Generating Dataset
    generate_dataset(path=path, n_examples=num_examples, diff_lb=diff_lb, diff_ub=diff_up)
    """
    
    #------------------Combining Augmented Dataset with Original Training Dataset-------------------#



    df = pd.read_csv("data/train_dataset_augmented.csv")
    samples_per_id = 20
    id_augmented_dict = {key:False for key in df['id'].unique()} # Keeping track of which keys have been augmented with data

    for i, row in df.iterrows():

        if id_augmented_dict[row['id']] == False:

            # Find all unique "Example X" pattern
            example_matches = re.findall(r"Example \d+:", row['input'])
            unique_examples = set(example_matches)

            # Count unique examples
            example_count = len(unique_examples)

            # Obtaining the task id
            temp_id = row['id']

            # Opening json file in the augmented_data folder that corresponds to the temporary id
            with open(f'augmented_data_v2/tasks/{temp_id}.json', 'r') as temp_id_file:
                temp_id_data = json.load(temp_id_file) # data here is a list of dictionaries

            for i in range(samples_per_id):
                augmented_train_samples = random.sample(temp_id_data, example_count)
                augmented_test_sample = random.sample(temp_id_data, 1)

                formatted_data = "Training Examples\n"

                # Add training examples to the string
                for j, one_train_pair in enumerate(augmented_train_samples, start=1):
                    formatted_data += f"Example {j}: Input\n"
                    formatted_data += format_array(one_train_pair['input']) + "\n\n"
                    formatted_data += f"Example {j}: Output\n"
                    formatted_data += format_array(one_train_pair['output']) + "\n\n"

                # Add the test example to the string
                formatted_data += "Test\n"
                formatted_data += format_array(augmented_test_sample[0]['input']) + "\n"

                new_row = {'id': temp_id, 'input':formatted_data, 'output': format_array(augmented_test_sample[0]['output'])}
                df.loc[len(df)] = new_row

            id_augmented_dict[row['id']] = True

    df.to_csv("data/train_dataset_augmented.csv", index=False)
