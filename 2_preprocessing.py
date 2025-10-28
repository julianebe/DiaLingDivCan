import pandas as pd
import re
import pycountry
from langcodes import find
import unicodedata
from typing import Optional



def read_csv(path:str) -> pd.DataFrame:
    '''
    read csv from path (no NAN)
    input: path of saved csv (string)
    output: data frame of table
    '''
    return pd.read_csv(path, keep_default_na=False)

def read_csv_correct_encoding(path:str) -> pd.DataFrame:
    '''
    read csv from path with latin1 and transform it to utf-8
    input: path of saved csv (string)
    output: data frame of table
    '''
    df = pd.read_csv(path, encoding='latin1', keep_default_na=False)
    df = df.applymap(lambda x: x.encode('latin1').decode('utf-8') if isinstance(x, str) else x)
    return df

def get_dataframe_fraction(df:pd.DataFrame, year:int) -> pd.DataFrame:
    '''
    get relevant dataframe fraction to examine the speaker number more
    input: dataframe (pandas DataFrame), year (int) of edition
    output: dataframe (pandas DataFrame)
    '''
    if year == 1996:
        df = df['Language Name', 'Abbreviation', 'Details'].copy()
        df.rename(columns={'Language Name':'name', 'Abbreviation':'abbreviation', 'Details':'details'})
        return df
    elif year == 2000 or 2005 or 2009:
        df = df['name', 'status', 'abbreviation', 'details'].copy()
        if year == 2000:
            return df        
    elif year == 2013 or 2015<= year <= 2018:
        df = df['name', 'status', 'abbreviation', 'details'].copy()
    elif year == 2019:
        df = df['name', 'abbreviation', 'population'].copy()
    elif year == 2024:
        df = df['name', 'abbreviation', 'users'].copy()
        df.rename(columns={'users':'population'})
    df.rename(columns={'abbreviation':'iso-code'})
    return df        

def save_csv(df:pd.DataFrame, filename:str):
    '''
    save dataframe as csv 
    input: dataframe (pandas DataFrame), target filename .csv (string)
    '''
    df.to_csv(filename, index=False)

def save_json(df:pd.DataFrame, filename:str):
    '''
    save dataframe as json file
    input: dataframe (pandas DataFrame), target filename .json (string)
    '''
    df.to_json(filename, orient='records', indent=4, force_ascii=False)

def split_details_old(text:str) -> pd.Series:
    '''
    split details, for edition 1996
    input: text of the details column (string)
    output: pandas Series with details information (details_info and population) 
    '''
    if "\xa0" in text: 
        parts = text.split("\xa0", 1)
        if 'macrolanguage' in text:
            return pd.Series([parts[0].strip(), parts[1].strip()])
        return pd.Series([parts[1].strip(), parts[0].strip()])
    else:
        if any(char.isdigit() for char in text):
            return pd.Series([None, text])
        else:
            return pd.Series([text, None])
        
def split_details_2000(text:str) -> pd.Series:
    '''
    split details, for edition 2000
    input: text of the details column (string)
    output: pandas Series with details information (details_info and population) 
    '''
    if "\xa0\xa0" in text: 
        parts = text.split("\xa0\xa0", 1)
        return pd.Series([parts[1].strip(), parts[0].strip()])
    else:
        if any(char.isdigit() for char in text):
            return pd.Series([None, text])
        else:
            return pd.Series([text, None])

def split_details(text:str) -> pd.Series:
    '''
    split details, relevant for edition 2005 to 2018
    input: text of the details column (string)
    output: pandas Series with details information (details_info and population) 
    '''
    if "  " in text: 
        parts = text.split("  ", 1) 
        return pd.Series([parts[0], parts[1]])
    else:
        if any(char.isdigit() for char in text):
            return pd.Series([None, text])
        else:
            return pd.Series([text, None])
        
def apply_split_details(df:pd.DataFrame, year:int) -> pd.DataFrame:
    '''
    apply different split details for different editions
    by adding the columns details_info, containing infos on the area
    and population, containing infos on speaker
    input: dataframe (pandas DataFrame), year (int) of edition in question 
    output: same dataframe with added columns
    '''
    if year == 1996:
        df[['details_info', 'population']] = df['details'].apply(split_details_old)
    elif year == 2000:
        df[['details_info', 'population']] = df['details'].apply(split_details_2000)
    else:
        df[['details_info', 'population']] = df['details'].apply(split_details)
    return df

def extract_first_number(text) -> Optional[int]:
    '''
    find the first number of the population cell for a language (mostly L1 or user)
    for overview (edition based)
    input: text of the population column (string)
    output: a number (int), if present
    '''
    regex_pattern = r"(^|\:|\(|\s|\.\s*)(\b(?!\d{4}\b)\d{1,3}(?:,\d{3})*)"
    if text is None or pd.isna(text):
        return None
    elif "No known L1 speakers" in text:
        return 0
    elif "Extinct." in text:
        return 0
    elif "No speakers out of" in text:
        return 0
    elif "became extinct" in text:
        return 0
    match = re.search(regex_pattern, text)
    if match:
        return int(match.group(2).replace(',', ''))
    return None

def apply_extract_first_number(df:pd.DataFrame, year:int) -> pd.DataFrame:
    '''
    extracts the first_number, relevant for speaker, mostly users, oder L1, sometimes ethnic population
    input: dataframe (pandas DataFrame), year (int) of edition in question 
    output: same dataframe with added column
    '''
    if year <= 2009:
        df['first_number'] = df['details'].apply(extract_first_number).astype('Int64')
        if df['status']:
            df.loc[df['status'] == 'Extinct languages', 'first_number'] = 0
    elif year >= 2013:
        df['first_number'] = df['population'].apply(extract_first_number).astype('Int64')
    return df

def get_il_from_sum_csv(path:str) -> str:
    '''
    get immigrant languages from summary file csv
    input: filepath of summary.csv
    output: string of in question
    '''
    sum = pd.read_csv(path)
    il = sum.loc[sum['label'] == 'Immigrant Languages', 'information'].values[0]
    return il

def get_il_from_sum_txt(path:str) -> str:
    '''
    get immigrant languages from summary file txt
    input: filepath of summary.txt
    output: string of in question
    '''
    with open(path, 'r') as file:
        sum = file.read()
    sum = sum.replace("\n", " ")
    match = re.search(r'Immigrant languages\:.*?\.', sum)
    x= ""
    if match:
        x = match.group().replace("Immigrant languages: ", "")
        x = x.rstrip(".") + ", "
    match = re.search(r'Also includes.*?\.', sum)
    il = match.group()
    il = il.replace("Also includes ", "")
    il = re.sub(r',[^,]*$', '', il)
    il = x  + il
    return il

def get_immigrant_languages(il:str, year:int) -> pd.DataFrame:
    '''
    transform immigrant language string into dataframe depending on year (int) of edition
    input: immigrant language (string), year (int) of edition
    output: dataframe (pandas DataFrame) of immigrant languages (name and user)
    '''
    if year <= 2000:
        pattern = r"([\w\s\-éÉ'’]+)\s(\d{1,3}(?:,\d{3})*)(?:\s\((\d{4})\))?"
        matches = re.findall(pattern, il)
        rows = []
        for match in matches:
            name, user, y = match
            user = int(user.replace(",", "")) if user else None
            rows.append({
                "name": name.strip(),
                f"user_ed_{year}": user if not year else None,
                f"user_{y}": user if y else None,
            })
        df_il = pd.DataFrame(rows)
    elif year >= 2005:
        pattern = r"\(([\d,]+)\)"
        cleaned_data = re.sub(pattern, lambda x: f"({x.group(1).replace(',', '')})", il)
        im_lang = cleaned_data.split(", ")
        pattern_summary = r"([\w\s\-éÉ]+?)\s\((\d+)\)?"
        parsed_data = []
        for entry in im_lang:
            match = re.match(pattern_summary, entry)
            if match:
                language = match.group(1).strip()
                number = int(match.group(2))
            else:
                language = entry.strip()
                number = None
            parsed_data.append((language, number))
        df_il= pd.DataFrame(parsed_data, columns=["name", f"user_ed_{year}"])
    return df_il

def merge_edition(df:pd.DataFrame, df_il:pd.DataFrame, year:int) -> pd.DataFrame:
    '''
    merge to get complete edition (languages and immigrant languages)
    get speaker information (first_number resp. user)
    input: 2 dataframes (pandas DataFrame), year (int) of edition
    output: 1 combined dataframe (pandas DataFrame)
    '''
    df_merged = pd.merge(df, df_il, on=['name', 'iso-code'], how='outer')
    user_columns = [col for col in df_merged.columns if col.startswith("user")]
    df_merged[f'ed_{year}'] = df_merged["first_number"].fillna(df_merged[user_columns].bfill(axis=1).iloc[:, 0])
    return df_merged

def normalize_name(name:str) -> str:
    '''
    for the normalizaiton of the Language names (here aposthroph)
    input: name (string)
    output: name (string)
    '''
    if isinstance(name, str):
        return unicodedata.normalize("NFKC", name).replace("’", "'") 
    return name

def format_text(name):
    '''
    for the normalizaiton of the Language names (here comma handling and capitalizaiton)
    input: name (string)
    output: name (string)
    '''
    if isinstance(name, str):
        return ', '.join([word.capitalize() for word in name.lower().split(', ')])
    return name

def get_iso_code(language: str) -> Optional[str]:
    '''
    get iso-code by taking language name. 
    apply like df[iso-code] = df['name'].apply(get_iso_code)
    manual validation and correction needed
    input: iso code of the language (string)
    output: language name (string) if findable
    '''
    try:
        return find(language).to_alpha3()
    except LookupError:
        return None

def get_name_iso_table(dfs:list[pd.DataFrame]) -> pd.DataFrame:
    '''
    get language table with iso codes and names
    for further work and overview
    manual validation and correction needed afterwards
    input: a list of dataframes to be merged (all editions)
    output: a dataframe that contains the iso-codes, the names (given in the editions)
    '''
    df_merged = pd.concat(dfs)
    df_languages = ['name', 'iso-code']
    df_languages = df_merged.groupby('iso-code')['name'].apply(lambda x: list(set(x))).reset_index()
    return df_languages

def get_missing_iso_code(df_languages:pd.DataFrame) -> pd.DataFrame:
    '''
    get rows of dataframe where iso-code is missing, 
    helpful to revise language table
    input: dataframe of iso code - names table 
    output: subset of row with missing data
    '''
    missing_iso_rows = df_languages[df_languages['iso-code'].isna() | (df_languages['iso-code'] == '')]
    return missing_iso_rows

def get_spreadsheet_editions(dfs:list[pd.DataFrame]) -> pd.DataFrame:
    '''
    get first_number/user as dataframe from all editions and name + iso-code
    input: list of dataframes
    output: dataframe (pandas DataFrame)
    '''
    merged_df = pd.concat(dfs, ignore_index=True)
    cols_to_merge = ['name', 'iso-code'] + [col for col in merged_df.columns if col.startswith('ed_')]
    final_df = merged_df[cols_to_merge].groupby(['name', 'iso-code'], dropna=False).first().reset_index()
    final_df.loc[:, final_df.columns.str.startswith('ed_')] = final_df.loc[:, final_df.columns.str.startswith('ed_')].apply(pd.to_numeric, errors='coerce').astype('Int64')
    df_sorted_iso = final_df.sort_values(by='iso-code', ascending=True)
    df_editions = df_sorted_iso.groupby('name').agg(lambda x: list(set(x.dropna())) if x.nunique() > 1 else x.iloc[0]).reset_index() 
    return df_editions

def get_official_name(iso_code:str) -> Optional[str]:
    '''
    get official name using iso-code 
    manual validation and correction needed afterwards
    apply like df_languages['official_name'] = df_languages['iso-code'].apply(get_official_name)
    input: iso code (string)
    output: string of the official name found for the iso code
    '''
    try:
        language = pycountry.languages.get(alpha_3=iso_code)
        return language.name if language else None
    except:
        return None

def get_missing_official_name(df_languages:pd.DataFrame) -> pd.DataFrame:
    '''
    get rows of dataframe where official name is missing, 
    helpful to revise language table (manual correction)
    input: dataframe of iso code - names - official name table 
    output: subset of row with missing official name
    '''
    missing_ofn_rows = df_languages[df_languages['official_name'].isna() | (df_languages['official_name'] == '')]
    return missing_ofn_rows

def replace_newline(text:str) -> str:
    '''
    clean text by removing newlines
    input: text (string) of population column
    output: cleaned text (string) of the populaiton column
    '''
    if isinstance(text, str): 
        return text.replace("\n", " ")
    return text

def remove_tabs(text: str) -> str:
    '''
    clean text by removing all tab characters (\t) from a given string.
    input: text (string) of population column
    output: cleaned text (string) of the populaiton column
    '''
    if isinstance(text, str):
        return text.replace("\t", "")
    return text

def deal_with_macrolanguage (text: str) -> str:
    '''
    correct macrolanguages
    relevant for edition 2009
    input: text (string) of population column
    output: cleaned text (string) of the populaiton column
    '''
    if isinstance(text):
        text = text.replace("\n", " ")
        text = text.replace("\t", "")
        text = text.replace("More information.", "")
        text = text.replace("Amacrolanguage.", "A macrolanguage.")
    return text
