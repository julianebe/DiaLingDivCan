import pandas as pd
import requests
from bs4 import BeautifulSoup as bs
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from waybackpy import WaybackMachineCDXServerAPI
from typing import Optional, Tuple
import pycountry



def save_summary_csv(df:pd.DataFrame, year:int, country:str):
    '''
    saves a pandas Dataframe as csv, creates automatically the filename for the summary
    input: dataframe (df), year of edition, country name
    '''
    df.to_csv(f"data/ethnologue_{year}_summary_{country}.csv", index=False)
    print("file saved")

def save_summary_txt(general_info:str, year:int, country:str):
    '''
    saves a pandas Dataframe as txt, creates automatically the filename for the summary
    input: string of general information, year of edition, country name
    '''
    with open(f"data/ethnologue_{year}_summary_{country}.txt", "w", encoding="utf-8") as file:
        file.write(general_info)

def save_languages(df_languages:pd.DataFrame, year:int, country:str):
    '''
    saves a pandas Dataframe as csv, creates automatically the filename for the summary
    input: dataframe of languages(df), year of edition, country name
    '''
    df_languages.to_csv(f"data/ethnologue_{year}_languages_{country}.csv", index=False)
    print("file saved")

def get_dynamic_page(url:str) -> bs:
    '''
    get a beautiful soup from a dynamic page, such as ethnologue
    input: url as string
    output: beautifulsoup object
    note: manual login requiered, necessary for recent edition of ethnologue
    '''
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    driver.get(url)
    print("Please log in within the next 30 seconds.")
    time.sleep(30)
    page_source = driver.page_source
    soup = bs(page_source, 'html.parser')
    driver.quit()
    print("page parsed")
    return soup

def get_static_page(url:str) -> bs:
    '''
    get a beautiful soup from a static page, such as the wayback machine
    input: url as string
    output: beautifulsoup object
    '''
    response = requests.get(url)
    if response.status_code == 200:
        html_content = response.text
    else:
        print(f"Failed to retrieve the webpage. Status code: {response.status_code}")
        exit()
    soup = bs(html_content, 'html.parser')
    return soup

def set_country_links(df:pd.DataFrame, year:int, country_name:str, link_abbr:str) -> pd.DataFrame:
    '''
    updates the country_links dataframe
    input: country_links dataframe, year of edition, country_name to updated, link_abbr to be added
    output: dataframe
    '''
    if country_name in df["Country Name"].values:
        df.loc[df["Country Name"] == country_name, f"Link_{year}"] = link_abbr
    else:
        new_row = {"Country Name": country_name, f"Link_{year}": link_abbr}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    return df

def get_links(country_links:bs, year:int, df:pd.DataFrame) -> pd.DataFrame:
    '''
    get links that schould be added, updates the country_links dataframe
    input: set of country_links, year of edition, country_name to updated
    output: dataframe
    note: relevant for editions 2013 to 2024
    '''
    for link in country_links:
        country_name = link.text.strip()
        href = link["href"]
        link_abbr = href.split('/')[-1]
        df = set_country_links(df, year, country_name, link_abbr)
    return df

def get_links_table_structure(table:bs, df:pd.DataFrame, year:int) -> pd.DataFrame:
    '''
    get links that schould be added, updates the country_links dataframe
    input: table (tag object), year of edition, country_name to updated
    output: dataframe
    note: relevant for editions 2000 to 2009
    '''
    columns = table.find_all('td')
    for col in columns:
        links = col.find_all('a')
        for link in links:
            country_name = link.text.strip()  
            href = link.get('href')          
            link_abbr = href.split('=')[-1]
            df = set_country_links(df,year, country_name, link_abbr)
    return df

def get_country_links_wayback(soup:bs, year:int, df:pd.DataFrame) -> pd.DataFrame:
    '''
    get links that schould be added from soup, updates the country_links dataframe
    processes soups from editions from 2013 to 2019
    input: soup (BeautifulSoup object), year of edition, country_name to updated
    output: dataframe
    '''
    if year <=2016:
        spans = soup.find_all('div', class_='span-5')
    elif year >= 2017:
        spans = soup.find_all('div', class_='span-1')
    if spans:
        for span in spans:
            content = span.find('div', class_='view-content')
            country_links = content.find_all("a")
            df = get_links(country_links, year, df)
    return df

def get_country_links_current_ethnologue(year:int, df:pd.DataFrame) -> pd.DataFrame:
    '''
    get links that schould be added from soup of current ethnologue edition, updates the country_links dataframe
    input: year of edition, country_name to updated
    output: dataframe
    '''
    soup_ethnologue_countries = get_dynamic_page('https://www-ethnologue-com.uaccess.univie.ac.at/browse/countries/')
    world = soup_ethnologue_countries.find('div', id='world')
    if world:
        countries_divs = soup_ethnologue_countries.find_all("div", class_="browse__countries")
        for div in countries_divs:
            country_links = div.find_all("a")
            get_links(country_links, year, df)
    return df

def get_first_available_link(row:pd.DataFrame)-> Optional[str]:
    '''
    adds column to dataframe with iso-code in ethnologue data
    input: row of Dataframe
    output: string (sets string in new cell)
    '''
    link_columns = ['Current Link', 'Link_2005', 'Link_2009', 'Link_2013', 'Link_2015',
                'Link_2016', 'Link_2017', 'Link_2018', 'Link_2019']
    for col in link_columns:
        if pd.notna(row[col]):
            return row[col]
    return None

def get_country_name(row:pd.DataFrame) -> Optional[str]:
    '''
    looks for official name of iso-code or if no iso-code takes "Country Name" as name
    input: row of Dataframe
    output: string (sets string in new cell)
    '''
    iso_code = row['link_code']
    if not iso_code or not isinstance(iso_code, str):
        return row['Country Name']
    country = pycountry.countries.get(alpha_2=iso_code.upper())
    if country:
        return country.name
    historic_country = pycountry.historic_countries.get(alpha_2=iso_code.upper())
    if historic_country:
        return historic_country.name
    return row['Country Name']
    
def create_country_links_df() -> pd.DataFrame:
    '''
    combines all functions to get the country links dataframe and csv from 1996 to 2019 and the recent ethnologue edition
    output: dataframe
    note: has to be corrected manually (incl. merging); 
    missing links should be looked up manually and added to 'link_code' if necessary, 
    official name (get_country_name) could looked up after filling in missing codes, for automatisation
    '''
    years =  [1996, 2000, 2005, 2009, 2013, 2015, 2016, 2017, 2018, 2019, 2024] 
    df = pd.DataFrame(columns=["Country Name"])
    for year in years:
        if year == 1996:
            urls_1996 = [f'https://web.archive.org/web/20000815100151/http://www.sil.org/ethnologue/countries/Americas.html', 
                f'https://web.archive.org/web/20000815095859/http://www.sil.org/ethnologue/countries/Africa.html',
                f'https://web.archive.org/web/20000815100032/http://www.sil.org/ethnologue/countries/Europe.html',
                f'https://web.archive.org/web/20000815095957/http://www.sil.org/ethnologue/countries/Asia.html',
                f'https://web.archive.org/web/20000815100213/http://www.sil.org/ethnologue/countries/Pacific.html']
            soups_1996 = []
            for url in urls_1996:
                soup = get_static_page(url)
                soups_1996.append(soup)
            for soup in soups_1996:
                table = soup.find('table', {"cellspacing": "10"})
                links = table.find_all('a')
                for link in links:
                    country_name = link.text.strip()
                    href = link.get('href')
                    link_abbr = href.split('.')[0]
                    df = set_country_links(df, year, country_name, link_abbr)
        elif year == 2000:
            soup_countries_2000 = get_static_page(f'https://web.archive.org/web/20011031111423/http://www.ethnologue.com/country_index.asp')
            table =  soup_countries_2000.find('table', {"cellspacing": "10"})
            if table:
                df = get_links_table_structure(table, df, year)
        elif year == 2005 or year == 2009:
            if year == 2005:
                url = 'https://web.archive.org/web/20080730070500/http://www.ethnologue.com/country_index.asp?place=all'
            elif year == 2009:
                url = 'https://web.archive.org/web/20100421072337/http://www.ethnologue.com/country_index.asp?place=all'
            soup = get_static_page(url)
            table =  soup.find('table', {"cellspacing": "20"})
            if table:
                df = get_links_table_structure(table, df, year)
        elif 2013 <= year <= 2019:
            if year == 2013:
                url = 'https://web.archive.org/web/20130911095018/http://www.ethnologue.com/browse/countries'
            elif year == 2015:
                url = 'https://web.archive.org/web/20150812045043/http://www.ethnologue.com/browse/countries'
            elif year == 2016:
                url = 'https://web.archive.org/web/20160726030829/http://www.ethnologue.com/browse/countries'
            elif year == 2017:                
                url = 'https://web.archive.org/web/20170620052147/https://www.ethnologue.com/browse/countries'
            elif year == 2018:
                url = 'https://web.archive.org/web/20180831004017/https://www.ethnologue.com/browse/countries'
            elif year == 2019:
                url = 'https://web.archive.org/web/20190713181744/https://www.ethnologue.com/browse/countries'  
            soup = get_static_page(url)
            df = get_country_links_wayback(soup, year)
        elif year == 2024:
            df = get_country_links_current_ethnologue(year)
    df['link_code'] = df.apply(get_first_available_link, axis=1)
    df['Name'] = df.apply(get_country_name, axis=1)
    df = df.sort_values(by='Name')
    df.to_csv('countries_links.csv', index=False)
    return df

def get_summary(soup:bs) -> Optional[pd.DataFrame]:
    '''
    takes BeautifulSoup object, extracts the information on the summary and returns it as df
    input: soup 
    output: df of summary
    note: works for recent edition of ethnologue
    '''
    section_summary = soup.find('section', id='summary')
    if section_summary:
        description_list = section_summary.find('dl')
        if description_list:
            dt_elements = [dt.text.strip() for dt in description_list.find_all('dt')]
            dd_elements = [dd.text.strip() for dd in description_list.find_all('dd')]
            if len(dt_elements) != len(dd_elements):
                print("Warning: Mismatched number of <dt> and <dd> elements.")
            summary = pd.DataFrame({'Term (dt)': dt_elements, 'Definition (dd)': dd_elements})
            print(summary)
            return summary
        else:
            print("No summary found on the page.")
    else:
        print("No summary found on the page.")

def get_languages(soup:bs) -> Optional[pd.DataFrame]:
    '''
    takes BeautifulSoup object, extracts the information on the languages and returns it as df
    input: soup 
    output: df of summary
    note: works for recent edition of ethnologue
    '''
    section = soup.find('section', id='languages')
    if section:
        table = section.find('div', class_='description-list')
        if table:
            data = []
            for dt, dd in zip(table.find_all('div', class_='languages__label'), table.find_all('ul', class_='languages__content')):
                row = {}
                row['name'] = dt.get_text(strip=True).replace(dt.find('a').get_text(strip=True), '').strip()
                row['abbreviation'] = dt['id']
                content_text = dd.get_text(separator=' ', strip=True)
                first_i = dd.find('i')
                if first_i:
                    details_end = content_text.find(first_i.get_text(strip=True))
                    details = content_text[:details_end].strip()
                else:
                    details = ""
                row['details'] = details
                for i_tag in dd.find_all('i'):
                    category = i_tag.get_text(strip=True).rstrip(':').lower().replace(' ', '_')
                    value = ""
                    sibling = i_tag.next_sibling
                    while sibling and sibling.name != 'br' and sibling.name != 'i':
                        if hasattr(sibling, 'get_text'):  
                            value += sibling.get_text(strip=True) + " "
                        elif isinstance(sibling, str):  
                            value += sibling.strip() + " "
                        sibling = sibling.next_sibling
                    row[category] = value.strip()
                if details == "A macrolanguage.":
                    includes_start = "Includes: "
                    includes = content_text[content_text.find(includes_start):].strip("Includes: ")
                    row['includes'] = includes
                data.append(row)
            df = pd.DataFrame(data)
            print(df.head(10))
            return df
        else:
            print("No table with languages found on the page.")
    else:
        print("No table with languages found on the page.")

def get_split_content_page(url_country:str, url_languages:str) -> Tuple[bs, bs]:
    '''
    get two beautiful soups for one country (one for the summary, one for the languages) from static pages, 
    such as ethnologue editions 2013 to 2019
    input: two urls as string
    output: two beautifulsoup object
    '''
    headers = {"User-Agent": "Mozilla/5.0"}
    response_country = requests.get(url_country, headers=headers)
    response_languages = requests.get(url_languages, headers=headers)
    if response_country.status_code == 200:
        html_content_country = response_country.text
    else:
        print(f"Failed to retrieve the webpage. Status code: {response_country.status_code}")
        exit()
    if response_languages.status_code == 200:
        html_content_languages = response_languages.text
    else:
        print(f"Failed to retrieve the webpage. Status code: {response_languages.status_code}")
        exit()
    soup_country = bs(html_content_country, 'html.parser')
    soup_languages = bs(html_content_languages, 'html.parser')
    return soup_country, soup_languages

def get_exclusive_summary(soup_country:bs) -> Optional[pd.DataFrame]:
    '''
    get information of summary and store it in a pandas Dataframe
    input: beautifulsoup object
    output: dataframe
    '''
    content_div = soup_country.find('div', class_='view-content')
    table = content_div.find('div', class_='views-row')
    if table:
        labels = [label.text.strip() for label in table.find_all(class_='views-label')] #sometimes em, somtimes span
        content = [info.text.strip() for info in table.find_all('div', class_='field-content')]
        if len(labels) != len(content):
            print("Warning: Mismatched number of <dt> and <dd> elements.")
        else:
            summary = pd.DataFrame({'label': labels, 'information': content})
            print(summary)
            return summary
    else:
        print("No summary found on the page.")

def get_exclusive_languages(soup_languages:bs) -> Optional[pd.DataFrame]:
    '''
    get information of languages and store it in a pandas Dataframe
    input: beautifulsoup object
    output: dataframe
    '''
    content_div = soup_languages.find('div', class_='view-content')
    if content_div:
        data = []
        for row in content_div.find_all('div', class_='views-row'):
            row_data = {}
            title_div = row.find('div', class_='title')
            row_data['name'] = title_div.get_text(strip=True) if title_div else None
            content_div = row.find('div', class_='content')
            if content_div:
                abbr = content_div.find('a')
                if abbr and '[' in abbr.text:
                    row_data['abbreviation'] = abbr.text.strip('[]')
                if abbr:
                    content_text = content_div.get_text(separator=' ', strip=True)
                    details_start = content_text.find(f"[{row_data['abbreviation']}]") + len(f"[{row_data['abbreviation']}]")
                    first_em = content_div.find('em')
                    if first_em:
                        details_end = content_text.find(first_em.get_text(strip=True))
                        row_data['details'] = content_text[details_start:details_end].strip()
                    else:
                        row_data['details'] = content_text[details_start:].strip()
                for em in content_div.find_all('em'):
                    em_title = em.get_text(strip=True).rstrip(':')
                    em_content = ''
                    next_sibling = em.next_sibling
                    while next_sibling and next_sibling.name != 'em' and next_sibling.name != 'p':
                        if hasattr(next_sibling, 'get_text'):  
                            em_content += next_sibling.get_text(strip=True) + ' '
                        elif isinstance(next_sibling, str): 
                            em_content += next_sibling.strip() + ' '
                        next_sibling = next_sibling.next_sibling
                    row_data[em_title.lower().replace(' ', '_')] = em_content.strip()
            data.append(row_data)
        df = pd.DataFrame(data)
        print(df.head(10))
        return df
    else:
        print("No languages found on the page.")

def get_wayback_summary(soup:bs) -> Optional[str]:
    '''
    takes a BeautifulSoup object and returns a string of the blockquote tag: relevant for summary of editions 2000 to 2009
    input: soup
    output: blockquote string
    '''
    if soup.find('blockquote'):
        blockquote = soup.find('blockquote').text.strip()
        print(blockquote)
        return blockquote
    else:
        print("No <blockquote> element with information found.")

def parse_language_table(table:bs, default_status:str) -> list:
    '''
    takes a table (tag object) of languages, organise the information and assigns status
    input: table, and default status
    output: list of rows for a language table
    note: relevant for summary of editions 2000 to 2009
    '''
    data = []
    for row in table.find_all("tr", valign="TOP"):
        name = row.find("td", width="25%").get_text(strip=True)
        row_data = {"name": name, "status": default_status}
        details_cell = row.find("td", width="75%")
        abbreviation = details_cell.find("p").get_text(strip=True).split('[')[1].split(']')[0]
        row_data["abbreviation"] = abbreviation
        content_text = details_cell.get_text(strip=True)
        details_start = content_text.find("]") + 1
        first_i = details_cell.find('i')
        if first_i:
            details_end = content_text.find(first_i.get_text(strip=True))
            row_data['details'] = content_text[details_start:details_end].strip()
        else:
            row_data['details'] = content_text[details_start:].strip() if details_start > 0 else ""
        for i_tag in details_cell.find_all("i"):
            category = i_tag.get_text(strip=True).rstrip(':').lower().replace(' ', '_')
            value = ""
            sibling = i_tag.next_sibling
            while sibling:
                if sibling.name == "i" or sibling.name == "a":  
                    break
                if hasattr(sibling, 'get_text'): 
                    value += sibling.get_text(strip=True) + " "
                elif isinstance(sibling, str):
                    value += sibling.strip() + " "
                sibling = sibling.next_sibling
            row_data[category] = value.strip()
        if details_cell.find("a", href="nearly_extinct.asp"):
            row_data["status"] = "Nearly extinct"
        data.append(row_data)
    return data

def get_wayback_languages(soup:bs) -> Optional[pd.DataFrame]:
    '''
    takes a BeautifulSoup object and return a dataframe of languages, already organised (editions 2000 to 2009)
    input: soup
    output: dataframe
    '''
    if soup.find("h3") and soup.find("table", {"cellspacing": "12"}):
        h3_tags = soup.find_all("h3")
        tables = soup.find_all("table", {"cellspacing": "12"})
        if len(h3_tags) != len(tables):
            print("Warning: Mismatched number of <h3> and <table cellspacing \"12\">  elements.")
        else:
            data = []
            for x in range(len(h3_tags)):
                default_status = h3_tags[x].get_text(strip=True)
                table = tables[x]
                d = parse_language_table(table, default_status)
                data += d
            df = pd.DataFrame(data)
            print(df)
            return df
    elif soup.find("table", {"cellspacing": "12"}):
        default_status = "living"
        table = soup.find("table", {"cellspacing": "12"})
        data = parse_language_table(table, default_status)
        df = pd.DataFrame(data)
        print(df)
        return df
    else:
        print("No language table available")

def get_information_1996(soup:bs) -> Optional[str]:
    '''
    takes a BeautifulSoup object and extracts the general information (summary) of edition 1996
    input: soup
    output string of summary
    '''
    if soup.find('p'):
        general_info = soup.find('p').text.strip()
        print(general_info)
        return general_info 
    else:
        print("No general information available")

def get_languages_1996(soup:bs) -> Optional[pd.DataFrame]:
    '''
    takes a BeautifulSoup object and return a dataframe of languages, already organised of the languages of edition 1996
    input: soup
    output: dataframe
    '''
    if soup.find('p', class_='HIN'):
        language_sections = soup.find_all('p', class_='HIN')
        language_data = []
        for section in language_sections:
            language_name = section.find('a').text.strip() if section.find('a') else "Unknown Language"
            alt_name_tag = section.find('i', class_='NAL')
            alt_names = alt_name_tag.text.strip('()') if alt_name_tag else ""
            abbrev = ""
            if "[" in section.text and "]" in section.text:
                abbrev = section.text.split("[")[1].split("]")[0].strip()
            details_start = section.text.find("]") + 1
            details = section.text[details_start:].strip() if details_start > 0 else ""
            language_data.append({
                "Language Name": language_name,
                "Alternative Names": alt_names,
                "Abbreviation": abbrev,
                "Details": details
            })
        df = pd.DataFrame(language_data)
        print(df)
        return df
    else:
        print("No languages found.") 

def get_link_variable(df_countries:pd.DataFrame, year:int, country:str) -> Optional[str]:
    '''
    lookes up the link variable to build the url for a country of a specific year
    input: dataframe of country links, year and country of question
    output: string with link variable
    '''
    link = df_countries.loc[df_countries['Name'] == country, f'Link_{year}']
    if link.empty:
        print (f'No link for {country} in {year}')
        return None
    else:
        return link.iloc[0]    

def get_url(year:int, country:str) -> Optional[str]:
    '''
    builds the url on the basis of the year and the country
    input: year, country
    output: string of url
    '''
    df_countries = pd.read_csv('countries_links.csv', keep_default_na=False)
    link_variable = get_link_variable(df_countries, year, country)
    if link_variable:
        if year == 1996:
            url_1996 = f'http:/www.sil.org/ethnologue/countries/{link_variable}.html'
            return url_1996
        elif year == 2000:
            url_2000 = f'http:/www.ethnologue.com/show_country.asp?name={link_variable}'
            return url_2000
        elif year == 2005 or year == 2009:
            url_2005_2009 = f'http://www.ethnologue.com/show_country.asp?name={link_variable}'
            return url_2005_2009
        elif year == 2013 or year == 2015 or year == 2016:
            url_2013_2016 = f'http://www.ethnologue.com/country/{link_variable}'
            return url_2013_2016
        elif year >= 2017 and year <= 2019:
            url_2017_2019 = f'https://www.ethnologue.com/country/{link_variable}'
            return url_2017_2019
        elif year == 2024:
            url_ethnologue = f'https://www-ethnologue-com.uaccess.univie.ac.at/country/{link_variable}/'
            return url_ethnologue

def get_wayback_url(url:str, year:int) -> Optional[str]:
    '''
    gets timestamp of time in question for a specific url for available wayback screenshot
    input: url, year
    output: string of wayback url
    '''
    user_agent = "Mozilla/5.0"
    cdx_api = WaybackMachineCDXServerAPI(url, user_agent)
    if year == 1996 or year == 2005 or year == 2013:
        oldest = cdx_api.oldest()
        return oldest.archive_url
    elif year == 2000:
        near_2000 = cdx_api.near(year=2001, month=11)
        return near_2000.archive_url
    elif year == 2009:
        near_2009 = cdx_api.near(year=2010)
        return near_2009.archive_url
    elif year >= 2015 and year <=2019:
        near = cdx_api.near(year=year, month=4)
        return near.archive_url

def scrape_1996(soup:bs) -> Tuple[str, pd.DataFrame]:
    '''
    combines functions to extract the information and the languages of edition 1996
    input: soup
    output: summary (string) and languages in dataframe
    '''
    summary = get_information_1996(soup)
    languages = get_languages_1996(soup)
    return summary, languages

def scrape_2000_2009(soup:bs) -> Tuple[str, pd.DataFrame]:
    '''
    combines functions to extract the information and the languages of editions 2000, 2005 and 2009
    input: soup
    output: summary (string) and languages in dataframe
    '''
    summary = get_wayback_summary(soup)
    languages = get_wayback_languages(soup)
    return summary, languages

def get_current_ethnologue(url:str, year:int, country:str):
    '''
    scrape a current country page of ethnologue
    input: url (string), year (int) of the edition, country (string) name, see valid list
    saves the summary as csv and the languages as csv
    '''
    soup = get_dynamic_page(url)
    summary = get_summary(soup)
    languages = get_languages(soup)
    save_summary_csv(summary, year, country)
    save_languages(languages, year, country)

def get_ethnologue_2013_2019(url_country:str, url_languages:str, year, country:str):
    '''
    combines functions to scrape the information and the languages of editions 2013 to 2019
    input: url_country (for summary information), url_languages (for information on languages), year, country
    saves summary and languages in dataframes
    '''
    soup_country, soup_languages = get_split_content_page(url_country, url_languages)
    summary = get_exclusive_summary(soup_country)
    languages = get_exclusive_languages(soup_languages)
    save_summary_csv(summary, year, country)
    save_languages(languages, year, country)

def scrape_country_year(country:str, year:int):
    '''
    scraping of country site of ethnologue for specific year
    input: country name (string) has to be in csv countries_links and year of asked edition
    output: saves 2 files summary and languages respectivly
    '''
    if year in [1996,2000,2005,2009,2024]: # change of 2024 to 2025 for newest available edition
        url = get_url(year, country)
        if url:
            if year != 2024:
                wayback_url = get_wayback_url(url, year)
                soup = get_static_page(wayback_url)
                if year == 1996:
                    summary, languages = scrape_1996(soup)
                else:
                    summary, languages = scrape_2000_2009(soup)
                save_summary_txt(summary, year, country)
                save_languages(languages, year, country)
            elif year == 2024:
                get_current_ethnologue(url, year, country)
        else:
            print(f"No ethnologue information for {country} or {country} in {year}.")
    elif year == 2013 or (2015 <= year <= 2019):
        url = get_url(year, country)
        if url:
            wayback_url_country = get_wayback_url(url, year)
            url_languages = url + '/languages'
            wayback_url_languages = get_wayback_url(url_languages, year)
            get_ethnologue_2013_2019(wayback_url_country, wayback_url_languages, year, country)
        else:
            print(f"No ethnologue information for {country} or {country} in {year}.")
    else:
        print(f"No ethnologue edition in {year}")