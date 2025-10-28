import json
import information_extraction
import preprocessing
import dataset_creation
import imputation
import pandas as pd
import re
import numpy as np
import plotly.express as px
import squarify
import matplotlib.pyplot as plt
import seaborn as sns
import jinja2
from typing import Tuple



def get_pop_dict() -> dict:
    '''
    get the population estimates of Canada
    output: dictionnary of the population estimates
    '''
    population = preprocessing.read_csv("population_estimates_canada.csv")
    population = population[["REF_DATE", "VALUE"]]
    pop_est = population.rename(columns={"REF_DATE":"year", "VALUE": "population"})
    pop_est_dict = dict(zip(pop_est['year'], pop_est['population']))
    pd.Series(pop_est_dict).plot(title="Population estimates Canada")
    return pop_est_dict

def get_canadian_official_languages() -> dict:
    '''
    creates a dictionary of the official languages of Canada
    output: dict of official languages
    '''
    official_languages = {'eng', 'fra'}
    return official_languages

def get_subset_since_1971(df:pd.DataFrame) -> pd.DataFrame:
    '''
    get a subset since 1971
    input: pandas Dataframe of imputed speaker numbers
    output: subset of the pandas Dataframe
    '''
    df = df.set_index("iso-code")
    years = [col for col in df.columns if col.isdigit() and int(col) >= 1971]
    return df[years].reset_index()

def get_language_groups(df:pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''
    creates subsets of a dataframe with imputed data for the language groups, just individual languages
    input: dataframe of imputed data
    output: 5 subsets of individual languages for the language groups
    '''
    canadian_langs = imputation.get_canadian_indigenous_languages()
    official_languages = get_canadian_official_languages()
    subset = get_subset_since_1971(df)
    characteristics = preprocessing.read_csv("codes_characteristics.csv")
    df_individual = characteristics[characteristics['scope'] == 'I']
    all_I = subset[subset['iso-code'].isin(df_individual['iso-code'])].set_index("iso-code").apply(pd.to_numeric, errors='coerce')
    indi_I = all_I[all_I.index.isin(canadian_langs)].apply(pd.to_numeric, errors='coerce')
    offi_I = all_I[all_I.index.isin(official_languages)].apply(pd.to_numeric, errors='coerce')
    immi_I = all_I[~all_I.index.isin(set(indi_I.index)| set(offi_I.index))]
    loca_I = all_I[all_I.index.isin(set(indi_I.index)| set(offi_I.index))]
    return all_I, indi_I, offi_I, immi_I, loca_I

def melt_group(df:pd.DataFrame, group_name:str) ->pd.DataFrame:
    '''
    get a long table of iso code, year, speaker number, language group
    input: dataframe of imputed data (language group), name of the group
    output: dataframe long table of a language group
    '''
    df_long = df.reset_index().melt(id_vars="iso-code", var_name="Year", value_name="Speakers")
    df_long.rename(columns={"index": "Language"}, inplace=True)
    df_long["Group"] = group_name
    return df_long

def prepare_df_long(df:pd.DataFrame)->pd.DataFrame:
    '''
    creates a long table out of the big dataframe
    input: dataframe of imputed data (all)
    return: dataframe of long table of all data combined
    '''
    all_I, indi_I, offi_I, immi_I, loca_I = get_language_groups(df)
    df_indi_long = melt_group(indi_I, "Indigenous languages")
    df_offi_long = melt_group(offi_I, "Official languages")
    df_immi_long = melt_group(immi_I, "Immigrant languages")
    df_all = pd.concat([df_indi_long, df_offi_long, df_immi_long], ignore_index=True)
    return df_all

def create_color_palette()-> dict:
    '''
    creates the color palette for visualizations (without measurement specifics: column name is language gorup)
    output: dict of colors for the different language groups (columns)
    '''
    palette = {
        "Population": "tab:pink",
        "Indigenous languages": "tab:blue",
        "Immigrant languages": "tab:orange",
        "Official languages": "tab:green",
        "All languages": "tab:red",  
        "Local languages": "tab:purple"
    }
    return palette

def create_color_palette_prefix(prefix:str) ->dict:
    '''
    creates the color palette for visualizations (Measurement specific)
    input: prefix (string) which identifies the measurement (depending on column name)
    output: dict of colors for the different language groups (columns)
    '''
    palette = {
        f"{prefix} for Population": "tab:pink",
        f"{prefix} Indigenous languages": "tab:blue",
        f"{prefix} Immigrant languages": "tab:orange",
        f"{prefix} Official languages": "tab:green",
        f"{prefix} for all Data": "tab:red",  
        f"{prefix} Local languages": "tab:purple" 
    }
    return palette

def language_description(df:pd.DataFrame, name_df:str)-> Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    '''
    creates a tables with information on the language groups population sizes (min, max and mean)
    input: dataframe with imputed data, name of the dataframe
    output: tuple od three dataframes (min, max, mean)
    '''
    all_I, indi_I, offi_I, immi_I, loca_I = get_language_groups(df)
    min = {'all_I': all_I.min(), 
        'indi_I': indi_I.min(),
        'offi_I': offi_I.min(),
        'immi_I': immi_I.min(),
        'loca_I': loca_I.min()}
    df_min = pd.concat(min, axis=1)
    df_min.to_csv(f"min_I_{name_df}.csv")
    max = {'all_I': all_I.max(), 
        'indi_I': indi_I.max(),
        'offi_I': offi_I.max(),
        'immi_I': immi_I.max(),
        'loca_I': loca_I.max()}
    df_max = pd.concat(max, axis=1)
    df_max.to_csv(f"max_I_{name_df}.csv")
    mean = {'All languages': all_I.mean(), 
        'Indigenous languages': indi_I.mean(),
        'Official languages': offi_I.mean(),
        'Immigrant languages': immi_I.mean(),
        'Local languages': loca_I.mean()}
    df_mean = pd.concat(mean, axis=1)
    df_mean.to_csv(f"mean_I_{name_df}.csv")
    return df_min, df_max, df_mean

def language_populations(df_all:pd.DataFrame, df_mean:pd.DataFrame, title:str):
    '''
    for the plot language populations with mean of the group
    input: df_all (long Data frame created with prepare_df_long), 
            df_mean (Data frame created with language_description),
            title (string of the title)
    '''
    plt.figure(figsize=(14, 6))
    palette = create_color_palette()
    df_mean_long = df_mean.loc["1996":"2021"].reset_index().melt(id_vars="index", 
                                                    var_name="Group", 
                                                    value_name="Speakers")
    df_mean_long.rename(columns={"index": "Year"}, inplace=True)
    sns.stripplot(
        x="Year",
        y="Speakers",
        hue="Group",
        data=df_all,
        dodge=True,
        size=6,
        alpha=0.35,
        palette=palette,
        legend=False 
    )
    sns.lineplot(
        x="Year",
        y="Speakers",
        hue="Group",
        data=df_mean_long,
        palette=palette,
        linewidth=2,
    )
    plt.title(title)
    plt.ylabel("Number of Speakers (symlog)")
    plt.xlabel("Year")
    plt.yscale("symlog", linthresh=1) #alternative "log", but symlog also includes languages with zero speakers
    plt.ylim(0, df_all["Speakers"].max() * 2) 
    plt.xticks(rotation=45)
    plt.legend(title="Group")
    plt.show()

def transform_to_latex(df:pd.DataFrame):
    '''
    transforms a table (dataframe) into a table in latex format; uses library jinja2
    input: dataframe
    '''
    print(df.to_latex())

def compare_column(speaker_L1_plus_lin:pd.DataFrame,
                    speaker_L1_plus_mav:pd.DataFrame,
                    speaker_L1_enriched_lin:pd.DataFrame,
                    speaker_L1_enriched_mav:pd.DataFrame,
                    speaker_user_enriched_lin:pd.DataFrame,
                    speaker_user_enriched_mav:pd.DataFrame, 
                    column_name:str) -> pd.DataFrame:
    '''
    For the comparison of the same column for different imputation methods
    input: the different dataframes containing the speaker numbers for the categories per year (see compute_speaker_detailed),
            the name of the column that should be compared
    output: dataframe that contains the speaker numbers for the languages group (column) for each Dataframe
    '''
    comparison_df = pd.DataFrame({
        "L1_plus_lin": speaker_L1_plus_lin[column_name],
        "L1_plus_mav": speaker_L1_plus_mav[column_name],
        "L1_enriched_lin": speaker_L1_enriched_lin[column_name],
        "L1_enriched_mav": speaker_L1_enriched_mav[column_name],
        "user_enriched_lin": speaker_user_enriched_lin[column_name],
        "user_enriched_mav": speaker_user_enriched_mav[column_name],
    })
    names = {'speaker_all': 'All Languages', 
       'speaker_indi': 'Indigenous Languages',
       'speaker_offi': 'Official Languages',
       'speaker_immi': 'Immigrant Languages',
       'speaker_loca' : 'Local Languages'}
    title_name = names.get(column_name, column_name)
    comparison_df.to_csv(f"results/{column_name}.csv")
    comparison_df.loc["1996":"2021"].plot(title=f"Imputation Methods Comparison {title_name}", ylabel="Speaker", xlabel="Year")
    return comparison_df

def richness_dataset(L1_plus_lin:pd.DataFrame,
                     L1_plus_mav:pd.DataFrame,
                     L1_enriched_lin:pd.DataFrame,
                     L1_enriched_mav:pd.DataFrame,
                     user_enriched_lin:pd.DataFrame,
                     user_enriched_mav:pd.DataFrame,
                     individual:bool=True)-> pd.DataFrame:
    '''
    calculates the linguistic richness (based on the different imputed dataframes) and plots it
    input: 6 dataframes with the imputed data, boolean if richness should be calculated on individual languages only (true by default)
    output: dataframe of the linguistic richness
    '''
    characteristics = preprocessing.read_csv("codes_characteristics.csv")
    df_individual = characteristics[characteristics['scope'] == 'I']
    if individual == False:
        counts = {'L1 plus lin': L1_plus_lin.set_index("iso-code").replace('', np.nan).count(), 
       'L1 plus mav': L1_plus_mav.set_index("iso-code").replace('', np.nan).count(),
       'L1 enriched lin': L1_enriched_lin.set_index("iso-code").replace('', np.nan).count(),
       'L1 enriched mav': L1_enriched_mav.set_index("iso-code").replace('', np.nan).count(),
       'user enriched lin': user_enriched_lin.set_index("iso-code").replace('', np.nan).count(),
       'user enriched mav': user_enriched_mav.set_index("iso-code").replace('', np.nan).count()}
    elif individual == True:
        counts = {'L1 plus lin': L1_plus_lin[L1_plus_lin['iso-code'].isin(df_individual['iso-code'])].set_index("iso-code").replace('', np.nan).count(), 
        'L1 plus mav': L1_plus_mav[L1_plus_mav['iso-code'].isin(df_individual['iso-code'])].set_index("iso-code").replace('', np.nan).count(),
        'L1 enriched lin': L1_enriched_lin[L1_enriched_lin['iso-code'].isin(df_individual['iso-code'])].set_index("iso-code").replace('', np.nan).count(),
        'L1 enriched mav': L1_enriched_mav[L1_enriched_mav['iso-code'].isin(df_individual['iso-code'])].set_index("iso-code").replace('', np.nan).count(),
        'user enriched lin': user_enriched_lin[user_enriched_lin['iso-code'].isin(df_individual['iso-code'])].set_index("iso-code").replace('', np.nan).count(),
        'user enriched mav': user_enriched_mav[user_enriched_mav['iso-code'].isin(df_individual['iso-code'])].set_index("iso-code").replace('', np.nan).count()}
    df_counts = pd.concat(counts, axis=1)
    df_counts.loc["1971":"2024"].plot()
    df_counts.loc["1996":"2021"].plot()
    return df_counts

def richness_language_counts(df:pd.DataFrame, name_df:str) ->pd.DataFrame:
    '''
    calculates the linguistic richness (based on language groups) and plots it
    input: dataframe with the imputed data, name of the dataframe
    output: dataframe of the linguistic Richness
    '''
    palette = create_color_palette()
    all_I, indi_I, offi_I, immi_I, loca_I = get_language_groups(df)
    counts = {
        'All languages': all_I.mask(all_I <= 0).count(), 
        'Indigenous languages': indi_I.mask(indi_I <= 0).count(),
        'Official languages': offi_I.mask(offi_I <= 0).count(),
        'Immigrant languages': immi_I.mask(immi_I <= 0).count(),
        'Local languages': loca_I.mask(loca_I <= 0).count()}
    df_counts = pd.concat(counts, axis=1)
    df_counts.to_csv(f"results/richness_I_L_{name_df}.csv")
    df_counts.loc["1996":"2021"].plot(title="Linguistic Richness", ylabel="Number of Languages", xlabel= "Year",color=[palette[col] for col in df_counts.columns])
    stacked_plot = df_counts.loc["1996":"2021"].drop(columns=["All languages", "Local languages"])
    desired_order = ["Official languages", "Indigenous languages", "Immigrant languages"]
    stacked_plot = stacked_plot[desired_order]
    stacked_plot.plot(kind="bar", stacked=True, title="Linguistic Richness", ylabel="Number of Languages", xlabel= "Year", color=[palette[col] for col in stacked_plot.columns])
    df_counts.loc["1996":"2021"]
    return df_counts

def plot_canadian_population():
    '''
    plots the population estimates for Canada
    '''
    pop_est_dict = get_pop_dict()
    pd.Series(pop_est_dict).plot(title="Population estimates Canada")

def compute_speaker_detailed (df:pd.DataFrame)->pd.DataFrame:
    '''
    computes the speaker numbers (abundance) for each language group (also in comparison to population estimates)
    input: dataframe of imputed speaker numbers
    output: data frame with speaker numbers per language group
    '''
    all_I, indi_I, offi_I, immi_I, loca_I = get_language_groups(df)
    palette = create_color_palette()
    pop_est_dict = get_pop_dict()
    speaker = {'Population': pop_est_dict,
        'All languages': all_I.sum(), 
        'Indigenous languages': indi_I.sum(),
        'Official languages': offi_I.sum(),
        'Immigrant languages': immi_I.sum(),
        'Local languages' : loca_I.sum()}
    df_speaker = pd.concat(speaker, axis=1)
    df_speaker.plot(title="Absolute Abundance", ylabel="Number of Speakers", xlabel= "Year",color=[palette[col] for col in df_speaker.columns])
    stacked_plot = df_speaker.loc["1996":"2021"].drop(columns=["All languages", "Local languages", "Population"])
    desired_order = ["Official languages", "Indigenous languages", "Immigrant languages"]
    stacked_plot = stacked_plot[desired_order]
    stacked_plot.plot(kind="bar", stacked=True, title="Absolute Abundance", ylabel="Number of Speaker", xlabel= "Year", color=[palette[col] for col in stacked_plot.columns])
    return df_speaker

def calculate_language_user_fractions(df:pd.DataFrame)->pd.DataFrame:
    '''
    calculates the language user fration relative to the population estimates
    input: dataframe (subset) of speaker numbers
    output: dataframe
    '''
    df.columns = df.columns.astype(str)
    pop_est_dict = get_pop_dict()
    df_frac = df.copy()
    pop_dict = {str(k): v for k, v in pop_est_dict.items()}
    for year in df_frac.columns:
        if year in pop_est_dict:
            df_frac[year] = pd.to_numeric(df_frac[year], errors='coerce')/ pop_est_dict[year]
    return df_frac

def compute_speaker_fraction_detailed (df:pd.DataFrame, name_df:str)->pd.DataFrame:
    '''
    computes the linguistic evenness (relative abundance of speaker in language groups) relative to population estimates
    input: dataframe (big) of speaker numbers name of the data set
    output: dataframe of relative abundance of speakers
    '''
    palette = create_color_palette()
    canadian_langs = imputation.get_canadian_indigenous_languages()
    official_languages = get_canadian_official_languages()
    subset = get_subset_since_1971(df)
    characteristics = preprocessing.read_csv("codes_characteristics.csv")
    df_individual = characteristics[characteristics['scope'] == 'I']
    pop_est_dict = get_pop_dict()
    fraction = calculate_language_user_fractions(subset, pop_est_dict)
    all_I = fraction[fraction['iso-code'].isin(df_individual['iso-code'])].set_index("iso-code").apply(pd.to_numeric, errors='coerce')
    indi_I = all_I[all_I.index.isin(canadian_langs)].apply(pd.to_numeric, errors='coerce')
    offi_I = all_I[all_I.index.isin(official_languages)].apply(pd.to_numeric, errors='coerce')
    immi_I = all_I[~all_I.index.isin(set(indi_I.index)| set(offi_I.index))]
    loca_I = all_I[all_I.index.isin(set(indi_I.index)| set(offi_I.index))]
    speaker = {'All languages': all_I.sum(), 
       'Indigenous languages': indi_I.sum(),
       'Official languages': offi_I.sum(),
       'Immigrant languages': immi_I.sum(),
       'Local languages' : loca_I.sum()}
    df_speaker_fraction = pd.concat(speaker, axis=1)
    df_speaker_fraction.to_csv(f"results/speaker_fraction_detailed_{name_df}.csv")
    stacked_plot = df_speaker_fraction.loc["1996":"2021"].drop(columns=["All languages", "Local languages"])
    desired_order = ["Official languages", "Indigenous languages", "Immigrant languages"]
    stacked_plot = stacked_plot[desired_order]
    stacked_plot.plot(kind="bar", stacked=True, title="Linguistic Evenness", ylabel="Relative Abundance", xlabel= "Year", color=[palette[col] for col in stacked_plot.columns])
    return df_speaker_fraction

def treemap_language_population(df:pd.DataFrame, year:str, min_pop_label:int):
    '''
    creates a treemap for the languages for a specific year
    input: dataframe of imputed data, year (string), min_pop_label (int) from which language size the speaker number has to be added
    '''
    df_long = prepare_df_long(df)
    df_year = df_long[(df_long["Year"] == year) & (df_long["Speakers"].notna()) & (df_long["Speakers"] > 0)] 
    df_year = df_year.sort_values("Speakers", ascending=False)
    palette = sns.color_palette("Set2", df_year["Group"].nunique())
    color_map = dict(zip(df_year["Group"].unique(), palette))
    colors = df_year["Group"].map(color_map)
    top20 = set(df_year.head(20)["iso-code"])
    labels = []
    for iso, speakers in zip(df_year["iso-code"], df_year["Speakers"]):
        if iso in top20:
            labels.append(f"{iso}\n{int(speakers):,}")
        elif speakers >= min_pop_label:
            labels.append(iso)
        else:
            labels.append("") 
    plt.figure(figsize=(16, 10))
    squarify.plot(
        sizes=df_year["Speakers"],
        label=labels, 
        color=colors,
        alpha=0.8,
        ec='white'
    )
    plt.title(f"Treemap of Languages by Number of Speakers ({year})", fontsize=16)
    plt.axis("off")
    plt.show()

def calculate_language_user_fractions_ild(df:pd.DataFrame)->pd.DataFrame:
    '''
    calculates the language user fration relative to the population estimates specific for the Index of Linguistic Diversity (ILD) following Harmon & Loh (2010)
    input: dataframe of speaker numbers
    output: dataframe
    '''
    df.columns = df.columns.astype(str)
    pop_est_dict = get_pop_dict()
    df_frac = df.copy()
    pop_dict = {str(k): v for k, v in pop_dict.items()}
    for year in df_frac.columns:
        if year in pop_dict:
            df_frac[year] = (pd.to_numeric(df_frac[year], errors='coerce')+1)/ pop_dict[year]
    return df_frac

def compute_fraction_ild(df:pd.DataFrame):
    '''
    computes the speaker fractions specific for the Index of Linguistic Diversity (ILD) following Harmon & Loh (2010)
    input: dataframe of speaker numbers name of the data set
    output: dataframe of relative abundance of speakers
    '''
    col_sums = df.sum(axis=0)
    df_plus1 = df + 1
    df_fraction = df_plus1.div(col_sums, axis=1)
    return df_fraction

def log_ratio_yearly_changes_index(df:pd.DataFrame, start_year:int=1971, end_year:int=2024)->pd.DataFrame:
    '''
    Calculates log-transformed year-to-year ratios for each row of a DataFrame.
    input: dataframe with speaker numbers fractions, start_year (int), end_year (int)
    output: dataframe with log-ratio columns (e.g., '1972/1971').
    '''
    years = [str(y) for y in range(start_year, end_year + 1)]
    df_years = df[years].copy()
    log_ratio_data = {}
    for i in range(len(years) - 1):
        y0, y1 = years[i], years[i + 1]
        ratio = df_years[y1] / df_years[y0]
        log_ratio_data[f"{y1}/{y0}"] = np.log10(ratio)
    return pd.DataFrame(log_ratio_data, index=df.index)

def compute_ILD (df:pd.DataFrame, name_df:str, start_year:int, end_year:int)->pd.DataFrame:
    '''
    computes the Index of Linguistic Diversity (ILD) on a specific start year
    input: dataframe of (imputed) speaker data, name of the dataframe, start and end year for the calculation
    output: dataframe with ILD values
    '''
    palette = create_color_palette_prefix("ILD")
    all_I, indi_I, offi_I, immi_I, loca_I = get_language_groups(df)
    pop_dict = get_pop_dict()
    fraction_ild = calculate_language_user_fractions_ild(all_I, pop_dict)
    ratio_ild = log_ratio_yearly_changes_index(fraction_ild, start_year, end_year)
    fraction_all =  compute_fraction_ild(all_I)
    fraction_indi = compute_fraction_ild(indi_I)
    fraction_offi = compute_fraction_ild(offi_I)
    fraction_immi = compute_fraction_ild(immi_I)
    fraction_loca = compute_fraction_ild(loca_I)
    ratio_all = log_ratio_yearly_changes_index(fraction_all, start_year, end_year)
    ratio_indi = log_ratio_yearly_changes_index(fraction_indi, start_year, end_year)
    ratio_offi = log_ratio_yearly_changes_index(fraction_offi, start_year, end_year)
    ratio_immi = log_ratio_yearly_changes_index(fraction_immi, start_year, end_year)
    ratio_loca = log_ratio_yearly_changes_index(fraction_loca, start_year, end_year)
    ild_pop = (10**ratio_ild.mean().T).cumprod()
    ild_all = (10**ratio_all.mean().T).cumprod()
    ild_indi = (10**ratio_indi.mean().T).cumprod()
    ild_offi = (10**ratio_offi.mean().T).cumprod()
    ild_immi = (10**ratio_immi.mean().T).cumprod()
    ild_loca = (10**ratio_loca.mean().T).cumprod()
    ild = {'ILD for Population': ild_pop,
       'ILD for all Data': ild_all, 
       'ILD Indigenous languages': ild_indi,
       'ILD Official languages': ild_offi,
       'ILD Immigrant languages': ild_immi,
       'ILD Local languages': ild_loca}
    df_ild = pd.concat(ild, axis=1)
    baseline = pd.DataFrame({col: 1 for col in df_ild.columns}, index=[start_year])
    df_ild = pd.concat([baseline, df_ild])
    df_ild.to_csv(f"ild_{start_year}_I_{name_df}.csv")
    df_ild.plot(title=f"Index of Linguistic Diversity (base year {start_year})", ylabel="ILD", xlabel= "Year", color=[palette[col] for col in df_ild.columns])
    return df_ild

def compute_fraction(df:pd.DataFrame)->pd.DataFrame:
    '''
    computes the speaker fractions
    input: dataframe of speaker numbers name of the data set
    output: dataframe of relative abundance of speakers
    '''
    col_sums = df.sum(axis=0)
    df_fraction = df.div(col_sums, axis=1)
    return df_fraction

def prepare_fractions(df:pd.DataFrame)-> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''
    prepares fractions for a dataframe for all language groups
    input: dataframe with speaker numbers
    output: a tuple with 6 dataframes containing speaker fractions    
    '''
    all_I, indi_I, offi_I, immi_I, loca_I = get_language_groups(df)
    pop_est_dict = get_pop_dict()
    fraction_pop_all_I = calculate_language_user_fractions(all_I, pop_est_dict)
    fraction_all_I = compute_fraction(all_I)
    fraction_indi_I = compute_fraction(indi_I)        
    fraction_offi_I = compute_fraction(offi_I)
    fraction_immi_I = compute_fraction(immi_I)
    fraction_loca_I = compute_fraction(loca_I)
    return fraction_pop_all_I, fraction_all_I, fraction_indi_I, fraction_offi_I, fraction_immi_I, fraction_loca_I

def compute_LDI(df:pd.DataFrame, name_df:str)->pd.DataFrame:
    '''
    computes the Linguistic Diversity Index (LDI) by Greenberg
    input: dataframe with imputed speaker numbers, name of the underlying data set
    output: dataframe of LDI values per language group
    '''
    palette = create_color_palette_prefix("LDI")
    fraction_pop_all_I, fraction_all_I, fraction_indi_I, fraction_offi_I, fraction_immi_I, fraction_loca_I = prepare_fractions(df)
    ldi_pop = (1 -(fraction_pop_all_I**2).sum())
    ldi_all = (1 -(fraction_all_I**2).sum())
    ldi_indi = (1 -(fraction_indi_I**2).sum())
    ldi_offi = (1 -(fraction_offi_I**2).sum())
    ldi_immi = (1 -(fraction_immi_I**2).sum())
    ldi_loca = (1 -(fraction_loca_I**2).sum())
    ldi = {'LDI for Population': ldi_pop,
       'LDI for all Data': ldi_all, 
       'LDI Indigenous languages': ldi_indi,
       'LDI Official languages': ldi_offi,
       'LDI Immigrant languages': ldi_immi,
       'LDI Local languages': ldi_loca}
    df_ldi = pd.concat(ldi, axis=1)
    df_ldi.to_csv(f"ldi_I_{name_df}.csv")
    df_ldi.loc["1996":"2021"].plot(title="Linguistic Diversity Index", ylabel="LDI", xlabel= "Year", color=[palette[col] for col in df_ldi.columns])
    return df_ldi

def compute_s_entropy(df:pd.DataFrame, name_df:str)->pd.DataFrame:
    '''
    computes the Shannon entropy
    input: dataframe with imputed speaker numbers, name of the underlying data set
    output: dataframe of entropy values per language group
    '''
    palette = create_color_palette_prefix("Entropy")
    fraction_pop_all_I, fraction_all_I, fraction_indi_I, fraction_offi_I, fraction_immi_I, fraction_loca_I = prepare_fractions(df)
    fraction_pop_all_I = fraction_pop_all_I[fraction_pop_all_I>0]
    fraction_all_I = fraction_all_I[fraction_all_I>0]
    fraction_indi_I = fraction_indi_I[fraction_indi_I>0]    
    fraction_offi_I = fraction_offi_I[fraction_offi_I>0]
    fraction_immi_I = fraction_immi_I[fraction_immi_I>0]
    fraction_loca_I = fraction_loca_I[fraction_loca_I>0]
    s_entropy_pop = (-np.sum(fraction_pop_all_I*np.log(fraction_pop_all_I)))
    s_entropy_all = (-np.sum(fraction_all_I*np.log(fraction_all_I)))
    s_entropy_indi = (-np.sum(fraction_indi_I*np.log(fraction_indi_I)))
    s_entropy_offi = (-np.sum(fraction_offi_I*np.log(fraction_offi_I)))
    s_entropy_immi = (-np.sum(fraction_immi_I*np.log(fraction_immi_I)))
    s_entropy_loca = (-np.sum(fraction_loca_I*np.log(fraction_loca_I)))
    s_entropy = {'Entropy for Population': s_entropy_pop,
       'Entropy for all Data': s_entropy_all,
       'Entropy Indigenous languages': s_entropy_indi,
       'Entropy Official languages': s_entropy_offi,
       'Entropy Immigrant languages': s_entropy_immi,
       'Entropy Local languages': s_entropy_loca}
    df_s_entropy = pd.concat(s_entropy, axis=1)
    df_s_entropy.to_csv(f"s_entropy_I_{name_df}.csv")
    df_s_entropy.loc["1996":"2021"].plot(title="Linguistic Diversity (Entropy)", ylabel="Shannon Entropy", xlabel= "Year", color=[palette[col] for col in df_s_entropy.columns])
    return df_s_entropy

def hill_number_entropy(df_s_entropy:pd.DataFrame, df_name:str)->pd.DataFrame:
    ''' 
    compute the hill number of order one based on shannon entropy
    input: dataframe containing the entropy values (see compute_s_entropy), name of underlying dataset
    output: df containing the hill numbers
    '''
    palette = create_color_palette_prefix("Hill number")
    hill = np.exp(df_s_entropy)
    hill.columns = ['Hill number for Population', 
                    'Hill number for all Data', 
                    'Hill number Indigenous languages', 
                    'Hill number Official languages',
                    'Hill number Immigrant languages',
                    'Hill number Local languages']  
    hill.to_csv(f"results/hill_s_entropy_{df_name}_I.csv")
    hill.loc["1996":"2021"].plot(title="Linguistic Diversity (Hill Numbers of order q=1)", ylabel="Numbers equivalient", xlabel= "Year", color=[palette[col] for col in hill.columns])
    return hill

def compute_inverse_simpson(df_ldi:pd.DataFrame, df_name:str)->pd.DataFrame:
    ''' 
    compute the hill number of order two (inverse simpson) based on ldi (greenberg diversity)
    input: dataframe containing the ldi values (see compute_LDI), name of underlying dataset
    output: df containing the inverse simpson
    '''
    palette = create_color_palette_prefix("Inverse Simpson")
    inverse_simpson = 1 / (1 - df_ldi)
    inverse_simpson.columns = ['Inverse Simpson for Population', 
                               'Inverse Simpson for all Data', 
                               'Inverse Simpson Indigenous languages', 
                               'Inverse Simpson Official languages',
                               'Inverse Simpson Immigrant languages',
                               'Inverse Simpson Local languages']  
    inverse_simpson.to_csv(f"results/inverse_simpson_{df_name}_I.csv")
    inverse_simpson.loc["1996":"2021"].plot(title="Linguistic Diversity (Hill Numbers of order q=2)", ylabel="Numbers equivalient", xlabel= "Year", color=[palette[col] for col in inverse_simpson.columns])
    return inverse_simpson

def plot_hill_two_one(hill:pd.DataFrame, inverse_simpson:pd.DataFrame):
    palette_i = create_color_palette_prefix("Inverse Simpson")
    palette_h = create_color_palette_prefix("Hill number")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    inverse_simpson.loc["1996":"2021"].plot(ax=ax1, title="Linguistic Diversity (Hill Numbers of order q=2)",  color=[palette_i[col] for col in inverse_simpson.columns])
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Numbers equivalent")
    hill.loc["1996":"2021"].plot(ax=ax2, title="Linguistic Diversity (Hill Numbers of order q=1)",  color=[palette_h[col] for col in hill.columns])
    ax2.set_xlabel("Year")  
    ax2.set_ylabel("") 
    plt.tight_layout()
    plt.show()

def LOCF_imputation_status(df:pd.DataFrame, start_year:int, csv:bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''
    Imputes missing data in the status table using the last observation carried forward method
    input: dataframe with information on status, start year of the covered time span (1996 or 2013), information if working directly from csv file
    info: df has to be read from csv; if working directly with df from creation set csv = False
    output: tuple of dataframes for each language group containing information on the status 
    '''
    if csv == True: 
        all_years = list(map(str, range(start_year, 2025)))
        status = df.set_index("iso-code").reindex(columns=all_years)
    elif csv == False:
        all_years = list(map(int, range(start_year, 2025)))
        status = df.reindex(columns=all_years)
    characteristics = preprocessing.read_csv("codes_characteristics.csv")
    df_individual = characteristics[characteristics['scope'] == 'I']
    df = status[status.index.isin(df_individual['iso-code'])]
    df = df.replace("none", np.nan)
    mask = df.notna()
    first_valid = mask.idxmax(axis=1)
    last_valid = mask.iloc[:, ::-1].idxmax(axis=1)
    valid_range_mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    for i in df.index:
        start = first_valid[i]
        end = last_valid[i]
        if pd.isna(start) or pd.isna(end):
            continue
        col_range = (df.columns >= start) & (df.columns <= end)
        valid_range_mask.loc[i, col_range] = True
    df_to_fill = df.where(valid_range_mask)
    df_filled = df_to_fill.ffill(axis=1)
    all_I = df.where(~valid_range_mask, df_filled)
    if start_year == 1996:
        cols_to_replace = all_I.loc[:, '1996':'1999']
        all_I.loc[:, '1996':'1999'] = cols_to_replace.replace('unknown', 'living')
    canadian_langs = imputation.get_canadian_indigenous_languages()
    official_languages = get_canadian_official_languages()
    indi_I = all_I[all_I.index.isin(canadian_langs)]
    offi_I = all_I[all_I.index.isin(official_languages)]
    immi_I = all_I[~all_I.index.isin(set(indi_I.index)| set(offi_I.index))]
    loca_I = all_I[all_I.index.isin(set(indi_I.index)| set(offi_I.index))]
    return all_I, indi_I, offi_I, immi_I, loca_I

def get_rank_counts(df_status:pd.DataFrame, group:str, status:bool = True, csv:bool = True)->pd.DataFrame:
    '''
    calculates the counts of the ranks (status/ egids levels)
    input: dataframe containing the status, language group name, information if status (or egids), information if working directly from csv file
    info: df has to be read from csv; if working directly with df from creation set csv = False
    output: dataframe with rank counts
    '''
    if csv == True:
        df_status.replace("", pd.NA, inplace=True)
        df_status = df_status.set_index("iso-code")
    status_counts = df_status.apply(lambda col: col.value_counts(dropna=False)).fillna(0)
    if status == False:
        custom_order = [
            '1', '2', '4', '5', '6a', '6b', '7', '8a', '8b', '9', '10', 
            'unestablished', 'unknown'
        ]
        status_counts.index = pd.CategoricalIndex(status_counts.index, categories=custom_order, ordered=True)
        status_counts = status_counts.sort_index()
    status = "Status" if status == True else "EGIDS"
    status_counts.T.plot(title = f"{status} {group}", kind="bar", stacked=True)
    status_counts.to_csv(f"results/{status}_counts_{group}.csv")
    return status_counts

def plot_status_counts(status_counts:pd.DataFrame, title:str):
    '''
    plots the status counts
    input: dataframe of status counts (big table or language group), title for the visualization
    '''
    palette_status = {
        "living": 'green',
        "endangered": 'orange',
        "extinct": 'red',
        "unestablished": 'blue',
        "unknown": 'purple',
        pd.NA: 'brown'
    }
    status_c = status_counts.set_index('iso-code').apply(lambda col: col.value_counts(dropna=False)).fillna(0).T
    order = [ "living", "endangered", "extinct", "unestablished", "unknown"]
    status_c = status_c[order]
    status_c.plot(kind="bar", stacked=True, color=[palette_status[col] for col in status_c.columns], title=title, ylabel="Number of Languages", xlabel= "Year")

def plot_egids_counts(egids_counts:pd.DataFrame, title:str):
    '''
    plots the egids counts
    input: dataframe of egids counts (big table or language group), title for the visualization
    '''
    palette_egids = {
        "1": 'darkgreen',
        "2": 'lightgreen',
        "4": 'darkblue',
        "5": 'lightblue',
        "6a": 'deeppink',
        "6b": 'darkviolet',
        "7": 'magenta',
        "8a": 'brown',
        "8b": 'darkorange',
        "9": 'tomato',
        "10": 'darkred',
        "unestablished": 'teal',
        "unknown": 'darkgray',
        pd.NA: 'brown'
    }
    status_c = egids_counts.set_index('iso-code').apply(lambda col: col.value_counts(dropna=False)).fillna(0).T
    order = ['1', '2', '4', '5', '6a', '6b', '7', '8a', '8b', '9', '10', 'unestablished', 'unknown']
    status_c = status_c[order]
    status_c.plot(kind="bar", stacked=True, color=[palette_egids[col] for col in status_c.columns], title=title, ylabel="Number of Languages", xlabel= "Year")

def status_weighted(df_status:pd.DataFrame) ->pd.DataFrame:
    '''
    transforms the status levels into weights
    input: dataframe with information on status
    output: dataframe with the weight of the status
    '''
    status_weights  = {
        'living' : 0,
        'unestablished' : 1,
        'unknown' : 1,
        'endangered' : 2,
        'extinct' : 3}
    w_status = df_status.replace(status_weights)
    return w_status

def status_changes_languages(df_status_w:pd.DataFrame) ->pd.DataFrame:
    '''
    creates overview table on status changes per language (postive change means change to a higher rank, means change to less stable)
    input: dataframe of status weights (of language group)
    output: dataframe of a summary on status change
    '''
    df_diff = df_status_w.set_index("iso-code").apply(pd.to_numeric, errors='coerce').diff(axis=1)
    num_changes = df_diff.ne(0) & df_diff.notna()
    num_changes = num_changes.sum(axis=1)
    total_change_size = df_diff.abs().sum(axis=1)
    net_change = df_diff.sum(axis=1)
    num_positive_changes = (df_diff > 0).sum(axis=1)
    num_negative_changes = (df_diff < 0).sum(axis=1)
    df_summary = pd.DataFrame({
        "num_changes": num_changes,
        "total_change_size": total_change_size,
        "net_change": net_change,
        "num_positive_changes": num_positive_changes,
        "num_negative_changes": num_negative_changes
    })
    return df_summary

def status_changes_year(df_status_w:pd.DataFrame) ->pd.DataFrame:
    '''
    creates overview table on status changes per year (postive change means change to a higher rank, means change to less stable)
    input: dataframe of status weights (of language group)
    output: dataframe of a summary on status change
    '''
    df_diff = df_status_w.set_index("iso-code").apply(pd.to_numeric, errors='coerce').diff(axis=1)
    num_changes = df_diff.ne(0) & df_diff.notna()
    num_changes = num_changes.sum(axis=0)
    total_change_size = df_diff.abs().sum(axis=0)
    net_change = df_diff.sum(axis=0)
    num_positive_changes = (df_diff > 0).sum(axis=0)
    num_negative_changes = (df_diff < 0).sum(axis=0)
    df_summary = pd.DataFrame({
        "num_changes": num_changes,
        "total_change_size": total_change_size,
        "net_change": net_change,
        "num_positive_changes": num_positive_changes,
        "num_negative_changes": num_negative_changes
    })
    return df_summary

def overview_rank_change(rank_change:pd.DataFrame, group:str, status:bool = True):
    '''
    creates an overview of rank changes (works with summary per year and per language based on weights per group)
    input: dataframe rank_changes (df_summary of status_changes_year or status_changes_languages), group name and information if status (or egids) for filename
    output: overview table as dataframe
    '''
    if status == True:
        rank = "status"
    elif status == False:
        rank = "egids"
    overview_status = rank_change.sum(axis=0).to_frame().reset_index().rename(columns={'index':f'{rank}', 0:'number'}).set_index(f'{rank}')
    overview_status.to_csv(f'results/overview_{rank}_change_{group}.csv')
    return overview_status

def overview_rank_change_groups(rank:str, group_1:str, group_2:str, group_3:str, group_4:str, group_5:str) ->pd.DataFrame:
    '''
    combines the overview tables on rank changes for the different groups (based on csv files)
    input: rank (string 'status' or 'egids'), 5 language groups names (strings)
    output: an overview table (dataframe) of the rank changes
    '''
    overview_1 = preprocessing.read_csv(f"results/overview_{rank}_change_{group_1}.csv")
    overview_2 = preprocessing.read_csv(f"results/overview_{rank}_change_{group_2}.csv")
    overview_3 = preprocessing.read_csv(f"results/overview_{rank}_change_{group_3}.csv")
    overview_4 = preprocessing.read_csv(f"results/overview_{rank}_change_{group_4}.csv")
    overview_5 = preprocessing.read_csv(f"results/overview_{rank}_change_{group_5}.csv")
    overview_1 = overview_1.rename(columns={"number": f"{group_1}"})
    overview_2 = overview_2.rename(columns={"number": f"{group_2}"})
    overview_3 = overview_3.rename(columns={"number": f"{group_3}"})
    overview_4 = overview_4.rename(columns={"number": f"{group_4}"})
    overview_5 = overview_5.rename(columns={"number": f"{group_5}"})
    overview = overview_1.merge(overview_2, on=f"{rank}", how="outer") \
                 .merge(overview_3, on=f"{rank}", how="outer") \
                 .merge(overview_4, on=f"{rank}", how="outer") \
                 .merge(overview_5, on=f"{rank}", how="outer")
    return overview

def categorize_iso_code(iso:str)->str:
    '''
    finds the language group based on the iso code
    input: iso code as string
    output: language group as string
    '''
    canadian_langs = imputation.get_canadian_indigenous_languages()
    official_languages = get_canadian_official_languages()
    if iso in canadian_langs:
        return 'Indigenous languages'
    elif iso in official_languages:
        return 'Official languages'
    else:
        return 'Immigrant languages'

def create_interactive_plot_status(ranks_all:pd.DataFrame, status:bool=True):
    '''
    creates an interactive parallel sets plot for the status/ egids that shows the flow/ changes over time
    input: dataframe with information on the ranks (weighted and non weighted
    '''
    ranks_all['group'] = ranks_all['iso-code'].apply(categorize_iso_code)
    if status == True:
        dimensions = [str(year) for year in range(1996, 2025)]
        title = 'Language Status Changes Over Time'
        ranks = ["","1.0","2.0","3.0"]
        rank = "status"
    else:
        dimensions = [str(year) for year in range(2013, 2025)]
        title='EGIDS level Changes Over Time'
        rank = "egids"
    fig = px.parallel_categories(
            ranks_all,
            dimensions=dimensions,
            color=ranks_all['group'].astype('category').cat.codes, 
            color_continuous_scale=['red', 'green', 'blue'],
            labels={col: col for col in dimensions},
            title=title
        )
    if status == True:
        fig.update_traces(
            dimensions=[{"categoryorder": "category descending"} for _ in ranks]
        )
    else:
        fig.update_traces(
            dimensions=[
                {
                    "label": year,
                    "values": ranks_all[year],
                    "categoryorder": "array"
                } for year in dimensions
            ],
            selector=dict(type='parcats'),
        )
    fig.update_coloraxes(showscale=False)
    fig.write_html(f"viz/{rank}_changes.html")
    fig.show()

def compute_rli_status(df_status:pd.DataFrame, csv:bool = True) -> pd.Series:
    '''
    computes the red list index (RLI) based on the status
    input: dataframe with information on status, information if working directly from csv file
    output: rli values in dataframe
    '''
    if csv == True:
        df_status.replace("", pd.NA, inplace=True)
        df_status = df_status.set_index("iso-code")
    status_weights  = {
        'living' : 0,
        'unestablished' : 1,
        'unknown' : 1,
        'endangered' : 2,
        'extinct' : 3}
    w_status = df_status.replace(status_weights)
    W_max = max(status_weights.values())
    rli = 1 - w_status.sum(axis=0) / (w_status.notna().sum(axis=0) * W_max)
    return rli

def compute_rli_status_groups(df_status_all_I:pd.DataFrame, csv:bool = True) -> pd.DataFrame:
    '''
    computes the red list index (RLI) based on the status
    input: dataframe with information on status (dataframe of all individual languages), information if working directly from csv file
    output: dataframe rli values per language group
    '''
    palette = create_color_palette_prefix("RLI")
    all_I = df_status_all_I
    canadian_langs = imputation.get_canadian_indigenous_languages()
    official_languages = get_canadian_official_languages()
    indi_I = all_I[all_I.index.isin(canadian_langs)]
    offi_I = all_I[all_I.index.isin(official_languages)]
    immi_I = all_I[~all_I.index.isin(set(indi_I.index)| set(offi_I.index))]
    loca_I = all_I[all_I.index.isin(set(indi_I.index)| set(offi_I.index))]
    outputs = {f'RLI for all Data': compute_rli_status(all_I, csv), 
       f'RLI Indigenous languages': compute_rli_status(indi_I, csv),
       f'RLI Official languages': compute_rli_status(offi_I, csv),
       f'RLI Immigrant languages': compute_rli_status(immi_I, csv),
       f'RLI Local languages': compute_rli_status(loca_I, csv)}
    rli_status = pd.concat(outputs, axis=1)
    rli_status.plot(title="Red List Index (Status)", ylabel="RLI", xlabel= "Year", color=[palette[col] for col in rli_status.columns])
    rli_status.to_csv("results/rli_status.csv")
    return rli_status

def egids_weighted(df_egids:pd.DataFrame)-> pd.DataFrame:
    '''
    transforms the egids levels into weights
    input: dataframe with information on egids
    output: dataframe with the weight of the egids
    '''
    egids_weights  = {
        '0' : 0,
        '1' : 1,
        '2' : 2,
        '3' : 3,
        '4' : 4, 
        '5' : 5,
        '6a' : 6,
        '6b' : 7,
        '7' : 8,
        '8a' : 9,
        '8b' : 10,
        '9' : 11,
        '10' : 12}
    w_egids = df_egids.replace(egids_weights)
    return w_egids

def compute_rli_egids_plus(df_egids:pd.DataFrame, csv:bool = True) -> pd.Series:
    '''
    computes the red list index (RLI) based on the egids (with weights for unestablished and unknown)
    input: dataframe with information on status, information if working directly from csv file
    output: rli values in dataframe
    '''
    if csv == True:
        df_egids.replace("", pd.NA, inplace=True)
        df_egids = df_egids.set_index("iso-code")
    egids_weights  = {
        '0' : 0,
        '1' : 1,
        '2' : 2,
        '3' : 3,
        '4' : 4, 
        '5' : 5,
        '6a' : 6,
        '6b' : 7,
        '7' : 8,
        '8a' : 9,
        '8b' : 10,
        '9' : 11,
        '10' : 12,
        'unestablished': 6,
        'unknown': 6}
    w_egids = df_egids.replace(egids_weights)
    W_max = max(egids_weights.values())
    rli = 1 - w_egids.sum(axis=0) / (w_egids.notna().sum(axis=0) * W_max)
    return rli

def compute_rli_egids_only(df_egids:pd.DataFrame, csv:bool = True)-> pd.Series:
    '''
    computes the red list index (RLI) based on the egids (without weights for unestablished and unknown)
    input: dataframe with information on status, information if working directly from csv file
    output: rli values in dataframe
    '''
    if csv == True:
        df_egids.replace("", pd.NA, inplace=True)
        df_egids = df_egids.set_index("iso-code")
    egids_weights  = {
        '0' : 0,
        '1' : 1,
        '2' : 2,
        '3' : 3,
        '4' : 4, 
        '5' : 5,
        '6a' : 6,
        '6b' : 7,
        '7' : 8,
        '8a' : 9,
        '8b' : 10,
        '9' : 11,
        '10' : 12,
        'unestablished': pd.NA,
        'unknown': pd.NA}
    w_egids = df_egids.replace(egids_weights)
    W_max = 12
    rli = 1 - w_egids.sum(axis=0) / (w_egids.notna().sum(axis=0) * W_max)
    return rli

def compute_rli_egids_groups(df_egids_all_I:pd.DataFrame, csv:bool = True, only:bool=False)->pd.DataFrame:
    '''
    computes the red list index (RLI) based on the egids
    input: dataframe with information on egids (dataframe of all individual languages), information if working directly from csv file, information if unestablished and unknown should be ignored
    output: dataframe rli values per language group
    '''
    all_I = df_egids_all_I
    palette = create_color_palette_prefix("RLI")
    canadian_langs = imputation.get_canadian_indigenous_languages()
    official_languages = get_canadian_official_languages()
    indi_I = all_I[all_I.index.isin(canadian_langs)]
    offi_I = all_I[all_I.index.isin(official_languages)]
    immi_I = all_I[~all_I.index.isin(set(indi_I.index)| set(offi_I.index))]
    loca_I = all_I[all_I.index.isin(set(indi_I.index)| set(offi_I.index))]
    if only == True:
        outputs = {f'RLI for all Data': compute_rli_egids_only(all_I, csv), 
                f'RLI Indigenous languages': compute_rli_egids_only(indi_I, csv),
                f'RLI Official languages': compute_rli_egids_only(offi_I, csv),
                f'RLI Immigrant languages': compute_rli_egids_only(immi_I, csv),
                f'RLI Local languages': compute_rli_egids_only(loca_I, csv)}
        title = "Red List Index (EGIDS only)"
    else:
        outputs = {f'RLI for all Data': compute_rli_egids_plus(all_I, csv), 
                f'RLI Indigenous languages': compute_rli_egids_plus(indi_I, csv),
                f'RLI Official languages': compute_rli_egids_plus(offi_I, csv),
                f'RLI Immigrant languages': compute_rli_egids_plus(immi_I, csv),
                f'RLI Local languages': compute_rli_egids_plus(loca_I, csv)}
        title = "Red List Index (EGIDS)"
    rli_egids = pd.concat(outputs, axis=1)
    rli_egids.plot(title=title, ylabel="RLI", xlabel= "Year", color=[palette[col] for col in rli_egids.columns])
    if only == True:
        rli_egids.drop(columns="RLI Official languages").plot(title=title, ylabel="RLI", xlabel= "Year", color=[palette[col] for col in rli_egids.drop(columns="RLI Official languages").columns])
    return rli_egids
