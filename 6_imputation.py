import json
import information_extraction
import preprocessing
import pandas as pd
import numpy as np
import dataset_creation



def subset_json(data:list[dict], key_name:str) -> list[dict]:
    '''
    create a subset that includes just specific data (L1 or user)
    input: data in json format (list of dict), key_name ('L1' or 'user')
    output: subset in json format (list of dict)
    '''
    subset = []
    for entry in data:
        new_entry = {}
        if "iso-code" in entry:
            new_entry["iso-code"] = entry["iso-code"]
        for key, value in entry.items():
            if key.startswith(key_name): 
                new_entry[key] = value
        subset.append(new_entry)
    return subset

def remove_key_prefix(data:list[dict], prefix:str) -> list[dict]:
    '''
    Removes the given prefix from keys in each dictionary in the list. Keeps other keys unchanged.
    input: data in json format (list of dict), prefix (str)
    output: cleaned data in json format (list of dict)
    '''
    cleaned_data = []
    for entry in data:
        new_entry = {}
        for key, value in entry.items():
            if key.startswith(prefix):
                new_key = key[len(prefix):]
                new_entry[new_key] = value
            else:
                new_entry[key] = value
        cleaned_data.append(new_entry)
    return cleaned_data

def clean_ed_keys(data:list[dict]) -> list[dict]:
    '''
    handle data without specific year: If a key starts with the prefix (e.g., 'ed_'),
    extract the suffix (e.g., the year). If the suffix already exists as a key, drop the prefixed key. 
    If not, rename the prefixed key to the suffix.
    input: data in json format (list of dict)
    output: json (list of dict)
    '''
    cleaned_data = []
    prefix="ed_"
    for entry in data:
        new_entry = {}
        for key, value in entry.items():
            if key.startswith(prefix):
                year = key[len(prefix):]
                if year not in entry:
                    new_entry[year] = value
            else:
                new_entry[key] = value
        cleaned_data.append(new_entry)
    return cleaned_data

def subset_clean(data:list[dict], prefix:str) -> list[dict]:
    '''
    get a clean subset
    input: data in json format (list of dict), key_name ('L1' or 'user')
    output: subset in json format (list of dict)    
    '''
    subset = subset_json(data, prefix)
    subset = remove_key_prefix(subset, prefix=f"{prefix}_")
    subset = clean_ed_keys(subset)
    return subset

def enrich_zero(base_data:list[dict], enrichment_data:list[dict])-> list[dict]:
    '''
    enrich the data set of canada (base_data) with zero user in general (enrichment_data)
    if there are no user in all countries, there are also none in Canada
    input: base_data in json format (list of dict), enrichment_data in json format (list of dict)
    output: enriched data in json format (list of dict)
    '''
    base_index = {entry['iso-code']: entry for entry in base_data}
    for enrich_entry in enrichment_data:
        iso = enrich_entry.get('iso-code')
        base_entry = base_index.get(iso)

        if base_entry:
            for key, value in enrich_entry.items():
                if key != 'iso-code' and value == 0:
                    base_entry[key] = value
    enriched_data = list(base_index.values())
    return enriched_data

def filter_instances_with_data(data:list[dict], base_data:list[dict]) -> list[dict]:
    '''
    enrich canada data with not specific canada data and return data that has to be checked manually
    input: data (list of dict): data to enrich the basisdata, json file with base data (canada specific)
    output: data in json format (list of dict) that contains data, to be checked manually for enrichment
    '''
    filtered = []
    f = [entry for entry in data if len(entry) > 1]
    for entry in f:
        other_values = [v for k, v in entry.items() if k != 'iso-code']
        if other_values and any(v != 0 for v in other_values):
            filtered.append(entry)
    base_index = {entry["iso-code"]: entry for entry in base_data}
    check_data = []
    for enrich_entry in filtered:
        iso = enrich_entry.get('iso-code')
        base_entry = base_index.get(iso)
        new_entry = {"iso-code": iso}
        for key, value in enrich_entry.items():
            if key != 'iso-code':
                if key in base_entry:
                        continue
                if int(key) >= 2021 and "2021" in base_entry:
                        continue
                new_entry[key] = value
        if len(new_entry) > 1:
            check_data.append(new_entry)
    return check_data

def user_all_subset_clean(data:list[dict])-> list[dict]:
    '''
    create a subset for user_all
    input: data in json format (list of dict)
    output data in json format (list in dict)
    '''
    user = subset_json(data, "user_")
    user_all = [
            {k: v for k, v in entry.items() if not k.startswith("user_canada")}
            for entry in user
        ]
    user_all = dataset_creation.rename_key(user_all, "user_1998", "user_all_ed_1998")
    user_all = remove_key_prefix(user_all, prefix="user_all_")
    user_all = remove_key_prefix(user_all, prefix="user_")
    user_all = clean_ed_keys(user_all)
    return user_all

def get_canadian_languages()-> set:
    '''
    create a set that contains all languages that are classified as Canadian by Ethnologue
    output: set
    '''
    langcodes = pd.read_csv("LanguageCodes.tab", sep='\t', keep_default_na=False)
    canadian_langs_df = langcodes[langcodes["CountryID"] == "CA"]
    canadian_langs_df = canadian_langs_df[["LangID"]]
    canadian_langs = set(canadian_langs_df["LangID"])
    return canadian_langs

def enrich_canadian_langs(base_data:list[dict], enrichment_data:list[dict])-> list[dict]:
    '''
    updates base_data (canada specific data) with enrichment_data (not canada specific) for languages native to Canada.
    input: base_data in json format (list of dict), enrichment_data in json format(list of dict)
    output: enriched data in json format (list of dict)
    '''
    canadian_langs = get_canadian_languages()
    base_index = {entry["iso-code"]: entry for entry in base_data}
    for enrich_entry in enrichment_data:
        iso = enrich_entry.get("iso-code")
        if not iso:
            continue
        if iso in canadian_langs:
            if iso in base_index:
                base_index[iso].update(enrich_entry)
            else:
                base_index[iso] = enrich_entry
    return [
        entry for entry in enrichment_data
        if entry.get("iso-code") not in canadian_langs
    ]

def enrich_zero_user(user:list[dict], L1:list[dict])-> list[dict]:
    '''
    enrich L1 data with zero user: if there are no user in general, there are also no L1 speaker
    input: user data in json format (list of dict), L1 data in json format (list of dict)
    output: L1 data enriched in json format (list of dict)
    '''
    for col in user.columns:
        if (user[col] == 0).any():
            if col not in L1.columns:
                L1[col] = pd.NA
            L1.loc[user[col] == 0, col] = 0
    L1 = dataset_creation.reorder_columns(L1)
    return L1

def normalize_year_columns(df:pd.DataFrame) -> pd.DataFrame:
    '''
    ensures all year columns are present (add missing years and empty columns), even if missing in some rows
    input: dataframe
    output dataframe 
    '''
    df = df.set_index("iso-code")
    all_years = [int(col) for col in df.columns if str(col).isdigit()]
    full_year_range = list(map(str, range(min(all_years), max(all_years) + 1)))
    return df.reindex(columns=full_year_range)

def linear_interpolation_imputation(df:pd.DataFrame)-> pd.DataFrame:
    '''
    impute missing data with linear interpolation method
    input: dataframe
    output: dataframe
    '''
    df_numeric = df.apply(pd.to_numeric, errors='coerce')
    result = pd.DataFrame(index=df_numeric.index, columns=df_numeric.columns, dtype=float)
    for idx, row in df_numeric.iterrows():
        valid = row.dropna()
        if valid.empty:
            result.loc[idx] = row
            continue
        first_idx = valid.index[0]
        last_idx = valid.index[-1]
        mask = (row.index >= first_idx) & (row.index <= last_idx)
        interp_slice = row[mask].interpolate(method='linear', limit_direction='both')
        filled_row = pd.Series(index=row.index, dtype=float)
        filled_row.loc[mask] = interp_slice
        result.loc[idx] = filled_row
    return result

def moving_average_imputation(df:pd.DataFrame, window:int=3)-> pd.DataFrame:
    '''
    impute missing data with moving average method
    input: dataframe
    output: dataframe
    '''
    df_ma = df.copy()
    df_ma = df_ma.apply(pd.to_numeric, errors='coerce')
    for idx, row in df_ma.iterrows():
        values = row.to_numpy(dtype='float64')
        mask = ~np.isnan(values)
        if np.count_nonzero(mask) < 2:
            continue
        first_valid = np.argmax(mask)
        last_valid = len(mask) - 1 - np.argmax(mask[::-1])
        new_values = values.copy()
        slice_idx = slice(first_valid, last_valid + 1)
        segment = row.iloc[slice_idx].copy()
        segment_filled = segment.fillna(method='ffill').fillna(method='bfill')
        smoothed = segment_filled.rolling(window=window, min_periods=1, center=True).mean()
        segment_imputed = segment.where(segment.notna(), smoothed)
        new_values[first_valid:last_valid+1] = segment_imputed.values
        df_ma.loc[idx] = new_values
    return df_ma

def number_imputed_data_points(df_before:pd.DataFrame, df_after:pd.DataFrame):
    number_languages = df_after.replace("", np.nan).set_index("iso-code").dropna(how="all").shape[0]
    datapoints_before = df_before.replace("", np.nan).set_index("iso-code").notna().sum().sum()
    datapoints_after = df_after.replace("", np.nan).set_index("iso-code").notna().sum().sum()
    return datapoints_before, datapoints_after, number_languages


