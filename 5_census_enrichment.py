import json
import information_extraction
import preprocessing
import dataset_creation
import pandas as pd
import re
import numpy as np
from typing import Optional, Tuple



def create_census_languages_df(census:pd.Dataframe, year:int) -> pd.DataFrame:
    '''
    prepare census data for enrichment
    make sure tables where manually prepared beforehand (delete columns, first line setting, names of columns)
    manual correction needed afterwards
    input: census dataframe, year of census (int)
    output: prepared census data in dataframe
    '''
    if year == 2016 or year == 2021:
        census_sub = census[census['Topic'].isin(['Mother tongue', 'Knowledge of languages'])]
        census_sub['Counts'] = pd.to_numeric(census_sub['Counts'], errors='coerce')
        mother_tongue = census_sub[census_sub["Topic"]=="Mother tongue"]
        mother_tongue = mother_tongue.drop(columns=(["Topic"]))
        mother_tongue = mother_tongue.rename(columns={"Characteristics": "language","Counts" : "l1_counts"})
        knowledge_lang = census_sub[census_sub["Topic"]!="Mother tongue"]
        knowledge_lang = knowledge_lang.drop(columns=(["Topic"]))
        knowledge_lang = knowledge_lang.rename(columns={"Characteristics": "language","Counts" : "knowledge_counts"})
        mother_tongue.columns = mother_tongue.columns.str.strip()
        knowledge_lang.columns = knowledge_lang.columns.str.strip()
        mother_tongue['language'] = mother_tongue['language'].str.strip()
        knowledge_lang['language'] = knowledge_lang['language'].str.strip()
        overview = pd.merge(mother_tongue, knowledge_lang, on= "language", how= "inner")
        overview['iso-code'] = overview['language'].apply(preprocessing.get_iso_code)
        filtered_df = overview[~overview['language'].str.contains('languages', case=False, na=False)]
        filtered_df = filtered_df[~filtered_df['language'].str.contains('n.o.s.', case=False, na=False)]
        filtered_df = filtered_df[~filtered_df['language'].str.contains('n.i.e.', case=False, na=False)]
        return filtered_df
    elif year == 2011:
        census = census[census['Topic']=="Detailed mother tongue"]
        census['Total'] = pd.to_numeric(census['Total'], errors='coerce')
        census = census.drop(columns=(["Topic"]))
        census = census.rename(columns={"Characteristics": "language","Total" : "l1_counts"})
        census = census[['language', 'l1_counts']]
        census.columns = census.columns.str.strip()
        census['language'] = census['language'].str.strip()
        census['language'] = census['language'].str.strip("�")
        census['language'] = census['language'].str.strip()
        census['iso-code'] = census['language'].apply(preprocessing.get_iso_code)
        census = census[~census['language'].str.contains('languages', case=False, na=False)]
        census = census[~census['language'].str.contains('n.i.e.', case=False, na=False)]
        census['language'] = census['language'].str.strip()
        return census
    elif year == 2006:
        census['Total'] = pd.to_numeric(census['Total'], errors='coerce')
        census = census.rename(columns={"Detailed mother tongue": "language","Total" : "l1_counts"})
        census = census[['language', 'l1_counts']]
        census['iso-code'] = census['language'].apply(preprocessing.get_iso_code)
        census = census[~census['language'].str.contains('languages', case=False, na=False)]
        census = census[~census['language'].str.contains('n.i.e.', case=False, na=False)]
        return census
    elif year == 2001:
        census['Total'] = pd.to_numeric(census['Total'], errors='coerce')
        census = census.rename(columns={"Detailed Mother Tongue": "language","Total" : "l1_counts"})
        census.columns = census.columns.str.strip()
        census['language'] = census['language'].str.strip()
        census['iso-code'] = census['language'].apply(preprocessing.get_iso_code)
        census = census[~census['language'].str.contains('languages', case=False, na=False)]
        return census

def census_2011_postprocessing(census11:pd.DataFrame) -> pd.DataFrame:
    '''
    after a first manual correction further processing is needed (nan was divided in two lines, and is treated here; 
    has to be extra, because of automatic problems in iso-code findings)
    input: dataframe of census 2011
    output: dataframe of census 2011 after correction
    '''
    df_grouped = census11.groupby("iso-code", as_index=False).agg({
            "language": lambda x: ", ".join(x.unique()),
            "l1_counts": "sum",
        })
    return df_grouped

def create_census_1996_languages_df(mother_tongue96:pd.DataFrame, knowledge_lang96:pd.DataFrame, knowledge_offi96:pd.DataFrame) -> Tuple[pd.DataFrame,pd.DataFrame]:
    '''
    prepare census data from 1996 for enrichment
    input: 3 dataframes including the information of the census 1996
    output: 2 dataframes with information on mothertongue and on knowledge of languages
    '''
    mother_tongue96['Total'] = pd.to_numeric(mother_tongue96['Total'], errors='coerce')
    mother_tongue96 = mother_tongue96.rename(columns={"Mother Tongue": "language","Total" : "l1_counts"})
    mother_tongue96.columns = mother_tongue96.columns.str.strip()
    mother_tongue96['language'] = mother_tongue96['language'].str.strip()
    mother_tongue96['iso-code'] = mother_tongue96['language'].apply(preprocessing.get_iso_code)
    mother_tongue96 = mother_tongue96[~mother_tongue96['language'].str.contains('languages', case=False, na=False)]
    knowledge_lang96['Total - Age groups'] = pd.to_numeric(knowledge_lang96['Total - Age groups'], errors='coerce')
    knowledge_lang96 = knowledge_lang96.rename(columns={"Knowledge of Non-official Languages (73)": "language","Total - Age groups" : "knowledge_counts"})
    knowledge_lang96.columns = knowledge_lang96.columns.str.strip()
    knowledge_lang96['language'] = knowledge_lang96['language'].str.strip()
    knowledge_lang96['iso-code'] = knowledge_lang96['language'].apply(preprocessing.get_iso_code)
    knowledge_lang96 = knowledge_lang96[~knowledge_lang96['language'].str.contains('languages', case=False, na=False)]
    knowledge_offi96['Total'] = pd.to_numeric(knowledge_offi96['Total'], errors='coerce')
    knowledge_offi96 = knowledge_offi96.rename(columns={"Knowledge of Official Languages": "language","Total" : "knowledge_counts"})
    knowledge_offi96.columns = knowledge_offi96.columns.str.strip()
    knowledge_offi96['language'] = knowledge_offi96['language'].str.strip()
    eng_o = knowledge_offi96.loc[knowledge_offi96['language'] == "English only", "knowledge_counts"].values[0]
    fra_o = knowledge_offi96.loc[knowledge_offi96['language'] == "French only", "knowledge_counts"].values[0]
    both = knowledge_offi96.loc[knowledge_offi96['language'] == "English and French", "knowledge_counts"].values[0]
    knowledge_offi96.loc[len(knowledge_offi96)] = ["English", (eng_o + both)]
    knowledge_offi96.loc[len(knowledge_offi96)] = ["French", (fra_o + both)]
    knowledge_offi96 = knowledge_offi96[knowledge_offi96['language'].isin(["English", "French"])]
    knowledge_offi96['iso-code'] = knowledge_offi96['language'].apply(preprocessing.get_iso_code)
    knowledge_1996 = pd.concat([knowledge_offi96, knowledge_lang96], ignore_index=True)
    return mother_tongue96, knowledge_1996

def find_shared_data(json:list[dict], census:pd.DataFrame, year:int) -> Tuple[list, list, list, dict[str, dict]]:
    '''
    find shared data in Ethnologue and the Census, or exclusive data of one data source
    input: Ethnologue data in json format (list of dict), census data (dataframe), year of the census data
    output: the shared iso codes (list), iso codes present only in all editions of Ethnologue (list), 
        iso codes present only in the census of the specific year (list) 
        and iso codes of shared codes with if available data from Ethnologue for the specific year of the census
    '''
    json_iso_codes = {entry.get('iso-code') for entry in json if 'iso-code' in entry}
    csv_iso_codes = set(census['iso-code'].dropna().astype(str).str.strip())
    in_both = json_iso_codes & csv_iso_codes
    only_in_json = json_iso_codes - csv_iso_codes
    only_in_csv = csv_iso_codes - json_iso_codes
    result = {}
    for entry in json:
        iso = entry.get("iso-code")
        if iso in in_both:
            result[iso] = {k: v for k, v in entry.items() if k.endswith(f"_{year}")}
    return in_both, only_in_json, only_in_csv, result

def enrich_with_census(census_df:pd.DataFrame, json:list[dict], year:int) -> list[dict]:
    '''
    enrich the ethnologue data with the prepared census data (user and L1)
    input: prepared census data as a dataframe, ethnologue data in json format (list of dict), year of census (int)
    output: enriched data in json format (list of dict)
    '''
    json_dict = {entry['iso-code']: entry for entry in json}
    for _, row in census_df.iterrows():
        iso_code = row['iso-code']
        knowledge_count = row.get('knowledge_counts', '')
        l1_count = row.get('l1_counts', '')
        name_census = row.get('language', '')
        if iso_code in json_dict:
            entry = json_dict[iso_code]
            entry[f'user_canada_{year}'] = knowledge_count
            entry[f'L1_canada_{year}'] = l1_count
        else:
            json_dict[iso_code] = {
                'iso-code': iso_code,
                f'user_canada_{year}': knowledge_count,
                f'L1_canada_{year}': l1_count,
                f'name_census_{year}': name_census
            }
    enriched_json = list(json_dict.values())
    return enriched_json

def enrich_with_census_l1(census_df:pd.DataFrame, json:list[dict], year:int) -> list[dict]:
    '''
    enrich the ethnologue data with the prepared census data (L1)
    input: prepared census data as a dataframe, ethnologue data in json format (list of dict), year of census (int)
    output: enriched data in json format (list of dict)
    '''
    json_dict = {entry['iso-code']: entry for entry in json}
    for _, row in census_df.iterrows():
        iso_code = row['iso-code']
        l1_count = row.get('l1_counts', '')
        name_census = row.get('language', '')
        if iso_code in json_dict:
            entry = json_dict[iso_code]
            entry[f'L1_canada_{year}'] = l1_count 
        else:
            json_dict[iso_code] = {
                'iso-code': iso_code,
                f'L1_canada_{year}': l1_count,
                f'name_census_{year}': name_census
            }
    enriched_json = list(json_dict.values())
    return enriched_json

def enrich_with_census_k(census_df:pd.DataFrame, json:list[dict], year:int) -> list[dict]:
    '''
    enrich the ethnologue data with the prepared census data (user)
    input: prepared census data as a dataframe, ethnologue data in json format (list of dict), year of census (int)
    output: enriched data in json format (list of dict)
    '''
    json_dict = {entry['iso-code']: entry for entry in json}
    for _, row in census_df.iterrows():
        iso_code = row['iso-code']
        k_count = row.get('knowledge_counts', '')
        name_census = row.get('language', '')
        if iso_code in json_dict:
            entry = json_dict[iso_code]
            entry[f'user_canada_{year}'] = k_count
        else:
            json_dict[iso_code] = {
                'iso-code': iso_code,
                f'user_canada_{year}': k_count,
                f'name_census_{year}': name_census
            }
    enriched_json = list(json_dict.values())
    return enriched_json

def save_enriched_data(enriched_json:list[dict], version:int):
    '''
    save a new version of the enriched data in json and as csv
    input: new data to save in json format (list of dict), number of version (int)
    '''
    information_extraction.create_json(enriched_json, f"extracted_info_canada_{version}_census.json")
    df_exinfo = dataset_creation.create_ex_info_csv(enriched_json)
    preprocessing.save_csv(df_exinfo, f"extracted_info_canada_{version}_census.csv")

def show_whole_dataframe():
    '''
    function to set the display mode to the whole dataframe
    '''
    pd.set_option('display.max_rows', None)

def reset_display_dataframe():
    '''
    function to reset the display mode to see the head
    '''
    pd.reset_option('display.max_rows')

def get_ethnologue_census_subsets(enriched_json:list[dict]) -> Tuple[list[dict], list[dict]]:
    '''
    find languages that were and were not in Ethnologue before the enrichment for manual correction that no language is added with a wrong iso code
    input: enriched data in json format (list of dict)
    output: two subsets in json format 
    '''
    has_name_census = []
    no_name_census = []

    for entry in enriched_json:
        if any(key.startswith("name_census") for key in entry):
            has_name_census.append(entry)
        else:
            no_name_census.append(entry)
    return no_name_census, has_name_census

def update_ethnologe_subset_census(ethnologue:list[dict], census:list[dict]) -> list[dict]:
    '''
    after manual correction of census subset update the ethnologue subset with remaining census data
    input: ethnologue data in json format (list of dict), corrected census data in json format
    output: enriched and updated data json format
    '''
    base_dict = {entry['iso-code']: entry for entry in ethnologue}
    for update_entry in census:
        iso = update_entry.get('iso-code')
        if iso in base_dict:
            base_dict[iso].update(update_entry)
            keys_to_remove = [key for key in base_dict[iso] if key.startswith("name_census")]
            for key in keys_to_remove:
                del base_dict[iso][key]           
        else:
            base_dict[iso] = update_entry
    enriched_data = list(base_dict.values())
    return enriched_data

def code_characteristics(df_enriched:pd.DataFrame):
    '''
    create a table with all languages (iso codes) in Canada with their scope and language types
    input: dataframe of enriched data
    saved as a csv
    infos:
        scope: I = individuel, M = macrolanguage, (C = collective)
        language type: L = living, E = extinct
        mostly iso 630-3, but also tutchone, aig, and creoles
        manuel correction: 
        Creoles scope = Collective (like on sil website: code tables) but language type set to living (not genetic) because of more relevance for the project
        aig, not sure of the iso code, but set to aig, because of researchers decision (I,L)
        tutchone, because of 2 languages M (even though no iso code available for the macrolanguage) und L for living
    '''
    iso639 = pd.read_csv("iso-639-3.tab", sep='\t', keep_default_na=False)
    iso_df = iso639.rename(columns={
        "Id": "iso-code",
        "Scope": "scope",
        "Language_Type": "language_type"
    })
    languages = df_enriched[["iso-code"]]
    languages = languages.merge(iso_df[["iso-code", "scope", "language_type"]], on="iso-code", how="left")
    languages = languages.sort_values(by="iso-code")
    preprocessing.save_csv(languages, "codes_characteristics.csv")

