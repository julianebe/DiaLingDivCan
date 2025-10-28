import information_extraction
import preprocessing
import pandas as pd



def add_official_names(merged_data:list[dict], output_path:str) -> list[dict]:
    '''
    Adds 'official_name' to a JSON file (big merged file) using a CSV lookup table (languages_canada.csv)
    input: merged data (list of dict), output path (str; to save the updated file)
    output: updated json file (list[dict])
    '''
    official_names_df = preprocessing.read_csv("languages_canada.csv")
    official_names_dict = dict(zip(official_names_df["iso-code"], official_names_df["official_name"]))
    print(official_names_dict)
    for entry in merged_data:
        iso_code = entry.get("iso-code")
        if iso_code in official_names_dict:
            entry["official_name"] = official_names_dict[iso_code]
        else:
            print(f"no official name found for {iso_code}")
    information_extraction.create_json(merged_data, output_path)
    print(f"Updated JSON with official names saved as {output_path}")
    return merged_data

def get_names(json_file:list[dict]) -> list[dict]:
    '''
    Adds a key (names) to each JSON entry (language) containing a set of all values from different editions
    extracted from keys that start with name_
    input: json_file (list[dict])
    output: updated json file list[dict]
    '''
    for entry in json_file:
        name_list = []
        for key, value in entry.items():
            if key.startswith(f"name_") and isinstance(value, str):
                formatted_name = value.title() if value.isupper() else value
                name_list.append(formatted_name)
        entry[f"names"] = list(set(name_list))
    return json_file

def transform_entries_to_set(json_file: list[dict], entry_name:str) -> list[dict]:
    '''
    add a key for each language that combines in a set all values of different editions extracted from keys starting alike
    relevant for alternate_names and autonyms
    input: json_file (list[dict]), entry name (str; how the key starts)
    output: updated json file list[dict]
    '''
    for entry in json_file:
        entry_list = set()
        for key, value in entry.items():
            if key.startswith(f"{entry_name}") and isinstance(value, str):
                formatted_name = value.title() if value.isupper() else value
                names = formatted_name.split(", ")
                normalized_names = {name.strip().rstrip(".") for name in names}
                entry_list.update(normalized_names)
        if entry_list:
            entry[f"{entry_name}s"] = list(entry_list)
            print(entry_list)
    return json_file

def extract_split_details(json_file: list[dict]) -> list[dict]:
    '''
    extract the information inside of split_details (relevant for edition 1996) into main entry
    dealing with key name changes
    input: json_file (list[dict])
    output: updated json file list[dict]
    '''
    for entry in json_file:
        for key in list(entry.keys()):
            if key.startswith("split_details_ed_"):
                suffix = key.replace("split_details_", "")
                if isinstance(entry[key], dict):
                    for sub_key, sub_value in entry[key].items():
                        new_key = f"{sub_key}_{suffix}"
                        entry[new_key] = sub_value
                del entry[key]
    return json_file

def mark_macrolanguages(data: list[dict]):
    '''
    add new element macrolanguage if is macrolanguage
    input: data in json format (list of dict) and changes the data directly
    '''
    for instance in data:
        if "includes_ed_2024" in instance:
            instance["macrolanguage"] = True

def overview_languages() -> pd.DataFrame:
    '''
    create an overview dataframe for the existence of languages in each edition
    reads automatically the json files in question for all years
    make sure the file path to the data is correct
    output: pandas DataFrame representing iso-code and year in boolean
    '''
    years = [1996, 2000, 2005, 2009, 2013, 2015, 2016, 2017, 2018, 2019, 2024]
    language_dict = {}
    for year in years:
        data = information_extraction.open_json(f"data_clean_year/new_clean_{year}.json")
        iso_codes = [item["iso-code"] for item in data]
        for code in iso_codes:
            if code not in language_dict:
                language_dict[code] = {}
            language_dict[code][year] = True
    df = pd.DataFrame.from_dict(language_dict, orient='index')
    df = df.fillna(False).astype(bool)
    df.index.name = "iso-code"
    df = df.sort_index()
    return df

def subset_status_json(data: list[dict]) -> list[dict]:
    '''
    get a json that contains the iso-codes and the status, if it's macrolanguage then indicates that
    input: data in json format (list of dict)
    output: subset in json format (list of dict)
    '''
    subset = []
    for instance in data:
        new_instance = {}
        if "iso-code" in instance:
            new_instance["iso-code"] = instance["iso-code"]
        if "macrolanguage" in instance:
            new_instance["macrolanguage"] = instance["macrolanguage"]
        for key, value in instance.items():
            if key.startswith("status_ed"):
                new_instance[key] = value.split('.')[0].strip()
        subset.append(new_instance)
    return subset

def cluster_status(status_str: str, year: int) -> str:
    '''
    get the clustered status of of a year (extinct, endangered, living, unestablished, unknown)
    if language not in edition: returns none
    input: string containing the status, year of edition (int)
    output: string of clustered status
    '''
    if not status_str:
        return "unknown"
    status_str = status_str.strip()
    if isinstance(status_str, str):
        if year == 1996:
            if status_str.startswith("Extinct"):
                return "extinct"
            if status_str.startswith("Nearly"):
                return "endangered"
            return "living"
        elif year in [2000, 2005, 2009]:
            lower = status_str.lower()
            if "living" in lower or "deaf" in lower:
                return "living"
            if status_str == "Nearly extinct":
                return "endangered"
            if status_str.startswith("Extinct"):
                return "extinct"
            return "unknown"
        elif year >= 2013:
            code = status_str.split()[0]
            code = code.strip("*")
            if code in ["1", "2", "3", "4", "5", "6a"]:
                return "living"
            elif code in ["6b", "7", "8a", "8b"]:
                return "endangered"
            elif code in ["9", "10"]:
                return "extinct"
            elif code in ["Unestablished"]:
                return "unestablished"
            return "unknown"
    return "unknown"

def status_table_overview(status_raw: list[dict], overview_lang: pd.DataFrame) -> pd.DataFrame:
    '''
    get a dataframe that is an overview of the status of a language for the editions
    input: subset of the status (status_raw in json format list of dict) and overview dataframe (with overview_languages)
    output: dataframe with status for each language per year
    '''
    index_list = overview_lang.index.tolist()
    years = overview_lang.columns.to_list()
    df = pd.DataFrame(index=index_list, columns=years)
    status_lookup = {entry["iso-code"]: entry for entry in status_raw}
    for iso, lang in overview_lang.iterrows():
        status_data = status_lookup.get(iso, {})
        is_macro = status_data.get("macrolanguage", False)
        for year in years:
            present = lang.get(year, False)
            if not present:
                df.loc[iso, year] = "none"
            elif is_macro:
                df.loc[iso, year] = "macrolanguage"
            else:
                key = f"status_ed_{year}"
                raw_status = status_data.get(key, None)
                if raw_status:
                    clustered = cluster_status(raw_status, year)
                    df.loc[iso, year] = clustered
                else:
                    df.loc[iso, year] = "unknown"
    df.index.name = "iso-code"
    df.to_csv("status_overview_canada.csv")
    return df

def egids_status(status_str: str, year: int) -> str:
    '''
    preparation of egids values
    input: status (string) and year (int)
    output: normalized egids values
    '''
    if not status_str:
        return "unknown"
    status_str = status_str.strip()
    if isinstance(status_str, str):
        code = status_str.split()[0]
        code = code.strip("*")
        if code in ["1", "2", "3", "4", "5", "6a", "6b", "7", "8a", "8b", "9", "10"]:
            return code
        elif code in ["Unestablished"]:
            return "unestablished"
        return "unknown"
    
def status_table_egids(status_raw: list[dict], overview_lang: pd.DataFrame) -> pd.DataFrame:
    '''
    get a dataframe that is an overview of the status of a language for the editions
    input: subset of the status (status_raw in json format list of dict) and overview dataframe (with overview_languages)
    output: dataframe with egids score for each language per year
    '''
    index_list = overview_lang.index.tolist()
    years = [2013, 2015, 2016, 2017, 2018, 2019, 2024]
    df = pd.DataFrame(index=index_list, columns=years)
    status_lookup = {entry["iso-code"]: entry for entry in status_raw}
    for iso, lang in overview_lang.iterrows():
        status_data = status_lookup.get(iso, {})
        is_macro = status_data.get("macrolanguage", False)
        for year in years:
            present = lang.get(year, False)
            if not present:
                df.loc[iso, year] = "none"
            elif is_macro:
                df.loc[iso, year] = "macrolanguage"
            else:
                key = f"status_ed_{year}"
                raw_status = status_data.get(key, None)
                if raw_status:
                    egids = egids_status(raw_status, year)
                    df.loc[iso, year] = egids
                else:
                    df.loc[iso, year] = "unknown"
    df.index.name = "iso-code"
    return df    

def subset_json(data: list[dict]) -> list[dict[str, dict]]:
    '''
    creates a subset of data in json format. 
    Keeps only the iso-code and all extracted information found in the information_extraction task
    input: data in json format (list of dict)
    output : subset as list of iso codes (dict) and extracted infos (dict)
    '''
    subset = []
    for instance in data:
        new_instance = {}
        if "iso-code" in instance:
            iso = instance.get("iso-code")
            for key, value in instance.items():
                if key.startswith("extracted_info_ed") and isinstance(value, dict):
                    for entry, val in value.items():
                        if entry not in new_instance:
                            new_instance[entry] = [val] 
                        else:
                            if val not in new_instance[entry]:
                                new_instance[entry].append(val)
        subset.append({iso: new_instance})
    return subset

def normalize_isocode_data(json_list: list[dict[str, dict]]) -> list[dict]:
    '''
    creates a flattend json file (list of dict) by turning the keys of the subset (iso code) into a feature
    input: subset as list of iso codes (dict) and extracted infos (dict)
    output: normalized data in json format (list of dict)
    '''
    normalized = []
    for entry in json_list:
        for iso, data in entry.items():
            flattened = {"iso-code": iso}
            for key, value in data.items():
                if isinstance(value, list) and len(value) == 1:
                    flattened[key] = value[0]   
                else:
                    flattened[key] = value 
            normalized.append(flattened)
    return normalized

def extract_contradictive_items(data:list[dict])->list[dict]:
    '''
    get a subset of all contradictive information in pioulation data (for manual corrections)
    input: data in json format (list of dict)
    output: list of dict of dict (iso code with contradictive information)
    '''
    contradictions = []
    for entry in data:
        multi_items = {}
        for key, value in entry.items():
            if isinstance(value, list):
                multi_items[key] = value
        if multi_items:
            iso = entry.get("iso-code")
            contradictions.append({iso:multi_items})
    return contradictions

def enrich_json_immigrant_language_user(ex_info:list[dict], gold_big:list[dict]): 
    '''
    enrich the corrected extracted population information with information of immigrant languages
    input: corrected extracted information in json format (list of dict) and json file containing all information

    '''
    enrichment_lookup = {entry["iso-code"]: entry for entry in gold_big if "iso-code" in entry}
    for instance in ex_info:
        iso = instance.get("iso-code")
        if not iso or iso not in enrichment_lookup:
            continue
        enrich_info = enrichment_lookup[iso]
        for key, value in enrich_info.items():
            if key.startswith("user_ed_"):
                instance[key] = value
            if key == "user_1998":
                instance[key] = value

def reorder_columns(df:pd.DataFrame) -> pd.DataFrame:
    '''
    reorder the columns of dataframe containing the population information
    input: dataframe
    output dataframe
    '''
    columns = list(df.columns)
    first = ['iso-code'] if 'iso-code' in columns else []
    remaining = [col for col in columns if col != 'iso-code']
    extra_cols = sorted([col for col in remaining if col.startswith('extra_')])
    other_cols = sorted([col for col in remaining if not col.startswith('extra_')])
    new_order = first + other_cols + extra_cols
    return df[new_order]

def check_datatypes(data_json:list[dict]) -> list[dict[str, dict]]:
    '''
    check datatypes of extracted population data, to make sure the data is numerical for the analysis
    input: data in json format (list of dict)
    output: list of dict of dict (iso code with data with wrong datatype)
    '''
    wrong_datatypes = []
    for entry in data_json:
        wrong_items = {}
        for key, value in entry.items():
            if key == "iso-code" or key.startswith("extra_"):
                if type(value) is not str:
                    wrong_items[key] = value
            else:
                if not isinstance(value, (int, float)):
                    wrong_items[key] = value, type(value)
        if wrong_items:
            iso = entry.get("iso-code")
            wrong_datatypes.append({iso:wrong_items})
    return wrong_datatypes

def convert_numeric_fields(data:list[dict]) -> list[dict]:
    '''
    convert strings with numbers into numerical value (int)
    input: data in json format (list of dict)
    output: data in json format (list of dict)
    '''
    for entry in data:
        for key, value in entry.items():
            if key == "iso-code" or key.startswith("extra_"):
                continue
            if isinstance(value, str):
                try:
                    num = float(value)
                    entry[key] = int(num) if num.is_integer() else num
                except ValueError:
                    pass
    return data

def create_ex_info_csv(ex_info_json:list[dict]) -> pd.DataFrame:
    '''
    transform the data from json format into a data frame
    input: data in json format (list of dict)
    output: dataframe
    '''
    df_exinfo = pd.DataFrame(ex_info_json)
    df_exinfo = reorder_columns(df_exinfo)
    df_exinfo = df_exinfo.sort_values(by='iso-code')
    return df_exinfo

def get_all_keys(json_data:list[dict]) -> set:
    '''
    get a list (set) of all keys (here all columnnames with population data)
    input: data in json format (list of dict)
    output: set
    '''
    all_keys = set()
    for entry in json_data:
        all_keys.update(entry.keys())
    all_keys = sorted(all_keys)
    return all_keys

def subset_speaker(ex_info_json:list[dict]) -> list[dict]:
    '''
    get a subset with relevant speaker information (user and L1)
    input: data in json format (list of dict)
    output: data in json format (list of dict)
    '''
    subset = []
    for entry in ex_info_json:
        new_entry = {}
        if "iso-code" in entry:
            new_entry["iso-code"] = entry["iso-code"]
        for key, value in entry.items():
            if key.startswith("L1") or key.startswith("user"):
                new_entry[key] = value
        subset.append(new_entry)
    return subset