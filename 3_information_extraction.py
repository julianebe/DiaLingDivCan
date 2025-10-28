import statistics
import os
import openai
import json
import preprocessing
import pandas as pd
import numpy as np
import copy
import re



def open_json(filename:str) -> list[dict]:
    '''
    open a jsonfile with right encoding
    input: filename (string) .json
    output: list of dictionary objects representing the json format
    '''
    with open(filename, "r", encoding="utf-8") as file:
        f = json.load(file)
    return f

def csv_to_json(filename:str) -> list[dict]:
    '''
    takes a filename(without .csv) and transform it into a .json file with the same filename
    returns the json file as a list of dictionary objects representing the json instances/ rows of the csv
    input: filename (string)
    output: data in json format (list of dict)
    '''
    df = preprocessing.read_csv(f"{filename}.csv")
    preprocessing.save_json(df, f"{filename}.json")
    json_df = open_json(f"{filename}.json")
    return json_df

def create_json(content:list[dict], filename:str):
    '''
    saves a list of dictionary objects representing the json format as an actual json file
    with right parameters
    input: content (list of dict), filename with .json ending
    '''
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(content, file, ensure_ascii=False, indent=4)

def openai_client() -> openai.OpenAI:
    '''
    get an openAI client from api
    prerequisite: there has to be the environment variable "OPENAI_API_KEY"
    output: client
    '''
    openai.api_key = os.environ["OPENAI_API_KEY"]
    client = openai.OpenAI()
    return client

### set client as global variable for other functions
client = openai_client()

### prompt for information extraction for older editions, should also work for newer ones
prompt_old = """
Your task is to extract the information and structure it in the following way.

You should extract any information on: user, L1 user, L2 user, semi-speaker, monolinguals
The information should be specified if it's information on only canada or not (all)
regarding the specification on canada, it should be mentioned explicitely in the text or the source points to it
Furthermore the year should be added, the year is within the source in brackets
If no year is provided for the information add the year of the edition (ed_year) to the structure
You should also extract information all languages L1 and ethnic population
For this information there is no specification canada or not needed, but information on the year
Any further information should also be saved with the year of the edition
No information can be lost, but not all possible features are in one text.
Provide a json format.

###
like this the following information structure is wanted:
user_all_year/ user_canada_year/ user_all_ed_year/ user_canada_ed_year
L1_
L2_
semi_
monoling_
all_languages_L1_year/ all_languages_L1_ed_year
ethnic_population_
extra_info_ed_year

###
year should be changed into the real year
the edition year is {edition}
no known speaker is equivalent to 0 speaker
_canada_ if "in Canada" mentioned for the information or the source is census or source is FPCC, source is Ministere de la Sante el des Services Sociaux or Govt. report or MSSS, or the source contains the words Canada or Quebec 
a range is to be written in a list, with the min, max
the first number, if not stated precisely otherwise, is the number for user all or canada.

###
Example 1 with output: {example1}  {output1}
Example 2 with output: {example2}  {output2}
Example 3 with output: {example3}  {output3}

###
extract the information for the following texts, 
if a text is an empty string, the output for the text is an empty json object
{user_text}

"""

def format_prompt(edition:int, example1:str, output1:str, example2:str, output2:str, example3:str, output3:str, prompt:str = prompt_old) -> str:
    '''
    format the prompt (based on prompt_old, if not specified differently, but needs the same input variables)
    input: year of edition (int), 3 examples and their corresponding outputs for the few-shot prompt approach, prompt optional
    output: the formatted prompt
    '''
    formatted_prompt = prompt.format(
                    edition=edition, 
                    example1=example1,
                    output1=output1,
                    example2=example2,
                    output2=output2,
                    example3=example3,
                    output3=output3)
    return formatted_prompt

def create_messages(prompt:str) -> list[dict]:
    '''
    creates the messages, for the information extraction task
    input: prompt (string)
    output: messages (listof dict)
    '''
    messages = [
        {"role": "system", "content": "You are an expert in information extraction."},
        {"role": "user", "content": prompt},
        ]
    return messages

def completion_output(messages:list[dict], client:openai.OpenAI = client, model: str = "gpt-4o") -> list[dict]:
    '''
    get openAI chat completions and structure output to json format
    input: messages (list of dict; see also messages funtion), and optional client and model
    output: json structured list of dictionaries
    '''
    response = client.chat.completions.create(model=model,
                                          messages=messages,
                                          temperature=0,
                                          frequency_penalty=0,
                                          presence_penalty=0,
                                          stop=None,
                                          )
    response_content = response.choices[0].message.content
    response_content = response_content.replace("```json", "").replace("```", "").strip()
    response_content = response_content.replace("```", "").replace("```", "").strip()
    output = json.loads(response_content)
    print(output)
    return output

def info_ex_solo( jsonfile:list[dict], edition:int, example1:str, output1:str, example2:str, output2:str, example3:str, output3:str, prompt:str = prompt_old, client:openai.OpenAI = client) -> list[dict]:
    '''
    takes the data, on which the information extraction should be performed. 
    It will be performed separately for each json instance in the data.
    input: jsonfile, 
        year of edition (int), 3 examples and their corresponding outputs for the few-shot prompt approach, prompt optional
        client optional
    output: list of dictionary objects representing the json format
    '''
    prompt = format_prompt(edition=edition, 
                    example1=example1,
                    output1=output1,
                    example2=example2,
                    output2=output2,
                    example3=example3,
                    output3=output3, 
                    prompt=prompt_old)
    for record in jsonfile:
        if "population" in record:
            user_text = record["population"]
            if user_text != "":
                formatted_prompt = prompt.format(user_text=user_text)
                messages = create_messages(formatted_prompt)
                output = completion_output(messages, client)
                record["extracted_info"] = output
    return jsonfile

### prompt for splitting the details text structured information and examples with output, relevant for edition 1996
prompt_splitting_1996 = """
    Your task is to split the text and structure it in the following way.

    You should extract the following information by splitting the text: population, details_info, classification, dialects, comments and status
    population contains information about speakers are there or how many people are connected
    details_info contains information about a space or area
    classification is a listing of language families
    dialects is a list marked by the beginning dialects and written in capitals
    comments contains all other information
    status contains information if it's extinct or nearly extinct

    Any further information should also be saved in comments
    No information can be lost, but not all possible features are always in one text.
    Provide a json format.

    ###
    The output should have json format with the following keys and the extracted information as value:
    population
    details_info
    classification
    dialects
    comments
    status

    ###
    Example 1 with output: {example1}  {output1}
    Example 2 with output: {example2}  {output2}
    Example 3 with output: {example3}  {output3}

    ###
    extract the information for the following text, 
    if a text is an empty string, the output for the text is an empty json object
    {text}

    """
example1_split = "20 speakers (1991 M. Krauss) out of 1,800 population including USA (1982 SIL). Total population probably evenly divided between the two dialects. Quebec on St. Lawrence River between Montreal and Quebec City. Algic, Algonquian, Eastern. Dialects: WESTERN ABNAKI (ABENAKI, ST. FRANCIS, ABENAQUI), PENOBSCOT (EASTERN ABNAKI). Dictionary. Grammar. Bible portions 1844. Nearly extinct."
output1_split = {
    "details_info": "Quebec on St. Lawrence River between Montreal and Quebec City.",
    "dialects": "WESTERN ABNAKI (ABENAKI, ST. FRANCIS, ABENAQUI), PENOBSCOT (EASTERN ABNAKI).",
    "population": "20 speakers (1991 M. Krauss) out of 1,800 population including USA (1982 SIL). Total population probably evenly divided between the two dialects.",
    "status": "Nearly extinct.",
    "classification": "Algic, Algonquian, Eastern.",
    "comments": "Dictionary. Grammar. Bible portions 1844."
    }

example2_split = "English speaking areas of Canada. Deaf sign language. Reported to have similarities to British Sign Language. Strong influence from American Sign Language. Structurally and grammatically distinct from French Canadian Sign Language (FCSL). Has grammatical characteristics independent of English. A few adults know both CSL and FCSL. Recognized by the government as a real language. Sign language interpreters required for deaf people in court. Used for deaf college students sometimes, important public functions, job training and social service programs. There is sign language instruction for parents of deaf children, and many classes for hearing people. There is a committee on national sign language. There is an organization for sign language teachers. Some research on the language. There is a manual alphabet. Dictionary, videos, film. Some signed interpretation on TV. Survey needed."
output2_split = {
    "details_info": "English speaking areas of Canada.",
    "classification": "Deaf sign language.",
    "comments": "Reported to have similarities to British Sign Language. Strong influence from American Sign Language. Structurally and grammatically distinct from French Canadian Sign Language (FCSL). Has grammatical characteristics independent of English. A few adults know both CSL and FCSL. Recognized by the government as a real language. Sign language interpreters required for deaf people in court. Used for deaf college students sometimes, important public functions, job training and social service programs. There is sign language instruction for parents of deaf children, and many classes for hearing people. There is a committee on national sign language. There is an organization for sign language teachers. Some research on the language. There is a manual alphabet. Dictionary, videos, film. Some signed interpretation on TV. Survey needed."
    }

example3_split = "40 speakers (1991 M. Dale Kinkade), out of 750 population (1977 SIL). Telegraph Creek, northwest British Columbia and other scattered locations. Na-Dene, Nuclear Na-Dene, Athapaskan-Eyak, Athapaskan, Tahltan-Kaska. Closely related to Kaska. Tahltan is seldom used. Only elderly speakers left (1991). Bilingual in English. Nearly extinct."
output3_split = {
        "details_info": "Telegraph Creek, northwest British Columbia and other scattered locations.",
        "population": "40 speakers (1991 M. Dale Kinkade), out of 750 population (1977 SIL).",
        "status": "Nearly extinct.",
        "classification": "Na-Dene, Nuclear Na-Dene, Athapaskan-Eyak, Athapaskan, Tahltan-Kaska.",
        "comments": "Closely related to Kaska. Tahltan is seldom used. Only elderly speakers left (1991). Bilingual in English."
    }

def split_1996( jsonfile:list[dict], client:openai.OpenAI = client) -> list[dict]:
    '''
    takes the data, on which the information splitting should be performed. 
    It will be performed separately for each json instance in the data.
    relevant for edition 1996
    input: jsonfile, client optional
    output: list of dictionary objects representing the json format
    '''
    prompt = prompt_splitting_1996.format(
                        example1=example1_split,
                        output1=output1_split,
                        example2=example2_split,
                        output2=output2_split,
                        example3=example3_split,
                        output3=output3_split)
    for record in jsonfile:
        print(record)
        if "details" in record:
            text = record["details"]
            print(text)
            if text != "":
                formatted_prompt = prompt.format(text=text)
                messages = create_messages(formatted_prompt)
                output = completion_output(messages, client=client)
                record["split_details"] = output
    return jsonfile

def info_ex_1996_solo(jsonfile:list[dict], example1:str, output1:str, example2:str, output2:str, example3:str, output3:str, client:openai.OpenAI = client) -> list[dict]:
    '''
    takes the data, on which the information extraction should be performed, only edition 1996) 
    It will be performed separately for each json instance in the data.
    Function based on splitting data, from beforehand
    input: jsonfile (edition 1996), 
        3 examples and their corresponding outputs for the few-shot prompt approach, prompt optional
        client optional
    output: list of dictionary objects representing the json format
    '''
    prompt = format_prompt(edition=1996, 
                    example1=example1,
                    output1=output1,
                    example2=example2,
                    output2=output2,
                    example3=example3,
                    output3=output3)
    for record in jsonfile:
        print(record)
        if "split_details" in record and "population" in record["split_details"]:
            user_text = record["split_details"]["population"]
            print(user_text)
            if user_text != "":
                formatted_prompt = prompt.format(user_text=user_text)
                messages = create_messages(formatted_prompt)
                output = completion_output(messages, client)
                record["extracted_info"] = output
    return jsonfile

def prompt_testdata_to_json(test_data_raw:str, input_example:str, output_example:str) -> str:
    '''
    creates prompt for transforming the manually annotated information into json format
    input: semi-structured testdata (string), example (string) and output (string) (does only need the right information, not structure)
    output: prompt (string)
    '''
    prompt = f"""" I have an input, transform it into json \
    each instance has data from a csv and manual extracted information
    I want create json instances with information of the iso-code and the manual extracted information
    the keys of the json instances are in lowercase, except L1 and L2.
    The key for the iso-code is "iso-code"

    ###
    Example:
    Input: 
    {input_example}

    Output should have the following information:
    {output_example}\

    ###
    Create a json file (list of json objects) for the following input, the input is separated by multiple newlines
    return just the json

    {test_data_raw}
    """
    return prompt

def testdata_to_json(test_data_raw:str, input_example:str, output_example:str, client:openai.OpenAI = client) -> list[dict]:
    '''
    application of transforming the structuring of the manually annotated test data
    input: semi-structured testdata (string), example (string) and output (string) (does only need the right information, not structure),
        client optional
    output: list of dictionary objects representing the json format
    '''
    prompt = prompt_testdata_to_json(test_data_raw, input_example, output_example)
    messages=[{"role": "user", "content": prompt}]
    testset = completion_output(messages=messages, client=client, model="gpt-4o-mini")
    return testset

def correct_testset(testset:list[dict], edition:int) -> list[dict]:
    '''
    correct the keys of the structured manually testset
    input: structured testset (list of dict), year of edition (int)
    output: list of dictionary objects representing the json format
    '''
    for instance in testset:
        keys_to_rename = {key: key.replace("l1", "L1").replace("l2", "L2") for key in instance if key.startswith(("l1", "l2"))}
        for old_key, new_key in keys_to_rename.items():
            instance[new_key] = instance.pop(old_key)
        if f"all_languages_l1_ed_{edition}" in instance:
            instance[f"all_languages_L1_ed_{edition}"] = instance.pop(f"all_languages_l1_ed_{edition}")
    return testset

def prepare_testset(testset_raw:str, input_ex:str, output_ex:str, edition:int, client:openai.OpenAI = client) -> list[dict]:
    '''
    prepare the manually annotated testset and save it (combined application)
    input: semi-structured testdata (string), example (string) and output (string) (does only need the right information, not structure),
        year of edition (int)
        client optional
    output: list of dictionary objects representing the json format    
    '''
    t_prompt = prompt_testdata_to_json(testset_raw, input_ex, output_ex)
    testset = testdata_to_json(client, t_prompt)
    testset = correct_testset(testset, edition)
    create_json(testset, f"testset_{edition}.json")
    return 

def validation_split(testset:list[dict], output:list[dict]) -> dict:
    '''
    validate the splitting output of edition 1996 per instance (accuracy)
    input: prepared testdata (list of dict), output information extraction (list of dict)
    output: scores (dict; accuracy of tested instances)
    '''
    reference_dict = {item["iso-code"]: item for item in output}
    scores = {}
    for test_instance in testset:
        iso_code = test_instance["iso-code"]
        if len(test_instance) == 1:
            scores[iso_code] = None
            print(f"{iso_code}: No additional information provided. Score set to None.")
            continue
        if iso_code in reference_dict:
            ref_instance = reference_dict[iso_code]
            extracted_info = ref_instance.get("split_details", {})
            total_keys = len(test_instance) - 1
            matching_keys = sum(
                1 for key in test_instance if key != "iso_code" and test_instance[key] == extracted_info.get(key)
            )
            score = (matching_keys / total_keys) * 100 if total_keys > 0 else 0
            scores[iso_code] = score
            print(f"{iso_code}: {score:.2f}% match")
        else:
            print(f"Warning: {iso_code} not found in reference dataset.")
            scores[iso_code] = None
    print("\nFinal Matching Scores:", scores)
    return scores

def validation(testset:list[dict], output:list[dict], year:int = None) -> dict:
    '''
    validate the information extraction output per instance (accuracy)
    input: prepared testdata (list of dict), output information extraction (list of dict)
        year (int) is optional, if keyname was already transformed
    output: scores (dict; accuracy of tested instances)
    '''
    reference_dict = {item["iso-code"]: item for item in output}
    scores = {}
    for test_instance in testset:
        iso_code = test_instance["iso-code"]
        if len(test_instance) == 1:
            scores[iso_code] = None
            print(f"{iso_code}: No additional information provided. Score set to None.")
            continue
        if iso_code in reference_dict:
            ref_instance = reference_dict[iso_code]
            if year is None:
                keyname = "extracted_info"
            elif year is not None:
                keyname = f"extracted_info_ed_{year}" 
            extracted_info = ref_instance.get(keyname, {})
            total_keys = len(test_instance) - 1
            matching_keys = sum(
                1 for key in test_instance if key != "iso_code" and test_instance[key] == extracted_info.get(key)
            )
            score = (matching_keys / total_keys) * 100 if total_keys > 0 else 0
            scores[iso_code] = score
            print(f"{iso_code}: {score:.2f}% match")
        else:
            print(f"Warning: {iso_code} not found in reference dataset.")
            scores[iso_code] = None 
    print("\nFinal Matching Scores:", scores)
    return scores

def calculate_accuracy(scores:dict) -> tuple[float, float]:
    '''
    calculates the average accuracy of the tested instances, and the median
    input: scores (dict)
    print calculation
    output: average accuracy per edition, median
    '''
    values = [value for value in scores.values() if value is not None]
    total_sum = sum(values)
    average = total_sum / len(values)
    print(f"Average: {average}")
    median_value = statistics.median(values)
    print(f"Median: {median_value}")
    return average, median_value

def test_outcome(year:int, output:list[dict]) -> dict:
    '''
    combine functions for testing output
    input: year of edition (int), output information extraction (list of dict)
    output: scores (dict); print calculation
    '''
    testset = open_json(f"testset_{year}.json")
    scores = validation(testset, output)
    calculate_accuracy(scores)
    return scores

def drop_empty_values(jsonfile:list[dict]) -> list[dict]:
    '''
    recursive function to remove empty values in json file
    input: jsonfile (list of dictionary objects representing the json format)
    output: same jsonfile (list of dictionary objects representing the json format)
    '''
    if isinstance(jsonfile, dict):
        return {k: drop_empty_values(v) for k, v in jsonfile.items() if v not in [None, "", [], {}]}
    elif isinstance(jsonfile, list):
        return [drop_empty_values(item) for item in jsonfile if item not in [None, "", [], {}]]
    else:
        return jsonfile

def clean_json(year:int) -> list[dict]:
    '''
    clean the json file of the output of the information extraction
    input: year of edition (int)
    output: clean jsonfile (list of dictionary objects representing the json format)
    '''
    jsonfile = open_json(f"data_info_ex/output_{year}_solo_1.json")
    clean_json = drop_empty_values(jsonfile)
    create_json(clean_json, f"clean_can_{year}.json")
    return clean_json

def rename_key(json_data:list[dict], old_key:str, new_key:str) -> list[dict]:
    '''
    it changes the json_data and returns it
    input: json_data (list of dictionary objects representing the json format), old key (string), new key (string)
    output: modified json_data
    '''
    for record in json_data:
        if old_key in record:
            record[new_key] = record.pop(old_key)
    return json_data

def remove_key(json_data:list[dict], key_to_remove:str) -> list[dict]:
    '''
    Removes a specified key from each dictionary in a list (json file).
    input: json data (list[dict]; The list of dictionaries to modify), key_to_remove (str; The key to remove from each dictionary)
    output: updated json data (list[dict])
    '''
    for entry in json_data:
        entry.pop(key_to_remove, None)
    return json_data

def update_json_with_csv_old(json_data:list[dict], year:int) -> list[dict]:
    '''
    Updates a JSON file with missing fields from a CSV file.
    Loads through year the csv of the raw scraped data
    Relevant for editions 1996 and 2000
    saves as json file and returns it
    Input: json_data (list[dict]), year ()int
    output: updated json data (list[dict])
    '''
    csv_data = preprocessing.read_csv_correct_encoding(f"data_scraping_final/ethnologue_{year}_languages_canada.csv")
    csv_data["abbreviation"] = csv_data["abbreviation"].str.lower()
    merge_key = "abbreviation"
    csv_dict = csv_data.set_index(merge_key).to_dict(orient="index")
    existing_keys = set()
    for entry in json_data:
        existing_keys.update(entry.keys())
    for entry in json_data:
        abbr = entry.get(merge_key, "").strip().lower()
        if abbr in csv_dict:
            for key, value in csv_dict[abbr].items():
                if key not in existing_keys and pd.notna(value) and str(value).strip() != "":
                    entry[key] = value
    create_json(json_data, f"clean_can_{year}_merge.json")
    print(f"JSON file updated successfully! Saved as clean_can_{year}_merge.json")
    return json_data

def update_json_with_csv_iso(json_data:list[dict], year:int) -> list[dict]:
    '''
    Updates a JSON file with missing fields from a CSV file.
    Loads through year the csv of the raw scraped data
    Relevant for editions 2005 and newer (with iso-code)
    saves as json file and returns it
    input: json_data (list[dict]), year ()int
    output: updated json data (list[dict])
    '''
    csv_data = preprocessing.read_csv_correct_encoding(f"data_scraping_final/ethnologue_{year}_languages_canada.csv")
    csv_key = "abbreviation"
    json_key = "iso-code"
    csv_dict = csv_data.set_index(csv_key).to_dict(orient="index")
    print(csv_dict)
    for entry in json_data:
        iso_code = entry.get(json_key, "").strip().lower()
        if iso_code in csv_dict:
            for key, value in csv_dict[iso_code].items():
                if key not in entry and pd.notna(value) and str(value).strip() != "":
                    entry[key] = value
    create_json(json_data, f"clean_can_{year}_merge.json")
    print(f"JSON file updated successfully! Saved as clean_can_{year}_merge.json")
    return json_data

def add_suffix_to_keys(json_data:list[dict], year:int) -> list[dict]:
    '''
    Adds a suffix to each key in a list of dictionaries, except for 'iso_code' 
    and keys that end in a number.
    suffix is composed with year of edition
    input: json data (list of dictionaries), year of edition (int) 
    output: updated JSON data with modified keys (list of dict)
    '''
    updated_data = []
    suffix = f"_ed_{year}"
    for entry in json_data:
        new_entry = {
            f"{key}{suffix}" if key != "iso-code" and not re.search(r"\d$", key) else key: value
            for key, value in entry.items()
        }
        updated_data.append(new_entry)
    return updated_data

def merge_json_by_iso_code(json_file_list:list[list[dict]], output_path:str) -> list[dict]:
    '''
    Merges multiple JSON data lists by 'iso-code', ensuring no information is lost.
    input: json_file_list (list[list[dict]]), output path (str; to save the merged file)
    output: merged json file (list[dict])
    '''
    merged_data = {}
    for data in json_file_list:
        for entry in data:
            iso_code = entry.get("iso-code")
            if not iso_code:
                continue
            if iso_code not in merged_data:
                merged_data[iso_code] = entry
            else:
                for key, value in entry.items():
                    if key not in merged_data[iso_code]:
                        merged_data[iso_code][key] = value
                    elif merged_data[iso_code][key] != value:
                        existing_value = merged_data[iso_code][key]
                        if not isinstance(existing_value, list):
                            merged_data[iso_code][key] = [existing_value]
                        if value not in merged_data[iso_code][key]:
                            merged_data[iso_code][key].append(value)
    merged_list = list(merged_data.values())
    create_json(merged_list, output_path)
    print(f"Merged JSON saved as {output_path}")
    return merged_list

def get_validation_desc(big_json:list[dict], scores_filename:str, validation_desc_filename:str) -> pd.DataFrame:
    '''
    get overview of the evalutation of the dataset (validation description)
    input: json file (list of dict), scores_filename for saving (string), validation_desc_filename for saving (string)
    output: data frame
    '''
    years = [1996, 2000, 2005, 2009, 2013, 2015, 2016, 2017, 2018, 2019, 2024]
    testsets = []
    number_processed_languages = []
    scores_list = []
    for year in years:
        if year != 1996:
            testset = open_json(f"data_info_ex/testset_{year}.json")
        elif year == 1996:
            testset = open_json("data_info_ex/testset_1996_small_info_ex.json")
        testsets.append(testset)
        data = preprocessing.read_csv(f"data_preprocessing/canada_{year}.csv")
        number_processed_languages.append(len(data))
        scores = validation(testset, big_json, year)
        scores_list.append(scores)
    scores_df = pd.DataFrame(scores_list).T
    scores_df.columns = years
    scores_df.sort_index(axis=0, inplace=True)
    scores_df = scores_df.rename_axis("iso-code")
    scores_df.to_csv(scores_filename) 
    number_annotated = [len(x) for x in testsets]
    percentages = [(a / b) * 100 for a, b in zip(number_annotated, number_processed_languages)]
    average = []
    median = []
    for s in scores_list:
        a, m = calculate_accuracy(s)
        average.append(a)
        median.append(m)
    se = [2*(np.sqrt(((p/100) * (1 - (p/100))) / n)) for p,n in zip(average, number_annotated)]
    print(scores_df)
    valid_scores = scores_df.count().to_list()
    validation_desc = pd.DataFrame({"edition":years, 
                                "number_languages": number_processed_languages, 
                                "number_annotated_languages": number_annotated, 
                                "percentage_validated_languages": percentages,
                                "average_accuracy": average,
                                "median_accuracy": median,
                                "number_valid_annotated_languages": valid_scores,
                                "standard_error_proportion": se
                                })
    preprocessing.save_csv(validation_desc, validation_desc_filename)
    return validation_desc

def get_all_keys(json_file: list[dict]) -> set:
    '''
    get a list of all keys of the main entry
    input: json file (list of dict)
    output: set of all keys
    '''
    all_keys = set()
    for entry in json_file:
        all_keys.update(entry.keys())
    all_keys = sorted(all_keys)
    return all_keys

def get_nested_keys(json_file: list[dict], prefix="extracted_info_ed_") -> set:
    '''
    get a list of all keys that were identified during the information extraction task
    possible changes
    input: json file (list of dict), optional prefex
    output: set of extracted keys
    '''
    nested_keys = set()
    for entry in json_file:
        for key, value in entry.items():
            if key.startswith(prefix) and isinstance(value, dict):
                nested_keys.update(value.keys())
    return nested_keys

def is_valid_key(key: str) -> bool:
    ''''
    Check if a given dictionary key matches the allowed structure.
    input: key of dict
    output: boolean
    '''
    pattern = re.compile(
        r"^(user|L1|L2|semi|monoling)_(all|canada)_(\d{4}|ed_\d{4})$|"
        r"^(all_languages_L1|ethnic_population)_(\d{4}|ed_\d{4})$|"
        r"^extra_info_ed_\d{4}$"
    )
    return bool(pattern.match(key))

def get_wrong_keys_with_values(json_file: list[dict], prefix="extracted_info_ed_") -> list[list[dict]]:
    '''
    get a list of all keys that where wrongly created during the information extraction task
    and get them in a dict
    input: json file (list of dict), optional prefex
    output: set of extracted keys
    '''
    language_wrong_keys = {}
    for entry in json_file:
        iso_code = entry['iso-code']
        wrong_keys = []
        for key, value in entry.items():
            if key.startswith(prefix) and isinstance(value, dict):
                for k, v in value.items():
                    if not is_valid_key(k):
                        wrong_keys.append({k:v})
                if wrong_keys:
                    language_wrong_keys[iso_code] = wrong_keys
    return language_wrong_keys

def merge_dict_lists(input_dict: dict) -> list:
    '''
    Merges all lists in a dictionary into a single list. (to see afterwards how many keys are wrong)
    input: input_dict (dict): A dictionary where values are lists.
    output: list: A merged list containing all elements from the dictionary's lists.
    '''
    merged_list = []
    for value in input_dict.values():
        if isinstance(value, list): 
            merged_list.extend(value)
        print(len(merged_list))
    return merged_list

def create_json_with_gold_standard(big_json: list[dict]) -> list[dict]:
    '''
    creates a json file, that replaces the tested data with the manual annotated data to increase the quality
    input: json file (list of dict)
    output: json file (list of dict)
    '''
    new_big_json = copy.deepcopy(big_json)
    years = [1996, 2000, 2005, 2009, 2013, 2015, 2016, 2017, 2018, 2019, 2024]
    testsets = []
    for year in years:
        if year != 1996:
            testset = open_json(f"data_info_ex/testset_{year}.json")
        elif year == 1996:
            testset = open_json("data_info_ex/testset_1996_small_info_ex.json")
        testsets.append(testset)
        reference_dict = {item["iso-code"]: item for item in new_big_json}
        keyname = f"extracted_info_ed_{year}"
        for test_instance in testset:
            iso_code = test_instance["iso-code"]
            if len(test_instance) == 1:
                if iso_code in reference_dict and keyname in reference_dict[iso_code]:
                        reference_dict[iso_code].pop(keyname, None)                    
                print(f"{iso_code}: No additional information provided {year}.")
                continue
            if iso_code in reference_dict:
                ref_instance = reference_dict[iso_code]
                keyname = f"extracted_info_ed_{year}"
                new_info = {k: v for k, v in test_instance.items() if k != "iso-code"}
                ref_instance[keyname] = new_info
            else:
                print(f"Warning: {iso_code} not found in reference dataset {year}.")
    return new_big_json