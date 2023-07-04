from selenium.common import NoSuchElementException, TimeoutException
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.by import By
from selenium import webdriver
from selenium.webdriver.support import expected_conditions as EC
import time
import chromedriver_autoinstaller
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from deep_translator import GoogleTranslator
from transformers import pipeline
from transformers import AutoTokenizer
import pandas as pd
import plotly.io as pio
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from collections import Counter
import nltk
import re
import plotly.subplots as sp
from pyabsa import available_checkpoints
from pyabsa import ATEPCCheckpointManager
from itertools import islice
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from textblob import Word


def execute_analysis(my_place_id):
    ### DATA ACQUISITION

    def scrape_reviews(my_place_id):

        google_url = "https://www.google.com/maps/place/?q=place_id:" + my_place_id

        chromedriver_autoinstaller.install()

        driver = webdriver.Chrome()

        driver.set_window_size(1024, 600)
        driver.maximize_window()

        driver.get(google_url)

        names = []
        rating = []
        dates = []
        comments = []

        try:
            # Wait for consent page to load
            accept_all_button = WebDriverWait(driver, 10).until(
                # By css_selector (1 or more classes) is more robust choice because by class_name can only handle 1 class
                EC.presence_of_element_located((By.CSS_SELECTOR, "[jsname='b3VHJd']"))
            )
            # Accept cookies because they are good
            accept_all_button.click()
        except:
            print("An error occurred when trying to accept Google's terms of conditions.")

        # get numbers of comments
        total_reviews = 1
        try:
            # Wait for consent page to load
            reviews_info = WebDriverWait(driver, 10).until(
                # By css_selector (1 or more classes) is more robust choice because by class_name can only handle 1 class
                EC.presence_of_element_located((By.CSS_SELECTOR, ".F7nice"))
            )
            # get all the spans
            span_elements = reviews_info.find_elements(By.TAG_NAME, 'span')
            total_reviews = int(span_elements[8].text[1:-1].replace(",", "").replace("\u202f", ""))
            if total_reviews > 100:
                total_reviews = 100
        except:
            print("Couldn't get number of reviews.")

        # get to reviews
        try:
            reviews_button = WebDriverWait(driver, 10).until(
                # EC.presence_of_element_located((By.CSS_SELECTOR, "[data-tab-index='1']"))
                # EC.presence_of_element_located((By.CSS_SELECTOR, "[aria-label*='Review']"))
                EC.presence_of_element_located((By.CSS_SELECTOR, "[jslog='145620;track:click;']"))
            )
            reviews_button.click()
        except:
            print("An error occurred when trying to get to reviews.")

        # scroll until the end
        try:
            scrollable_div = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".m6QErb.DxyBCb.kA9KIf.dS8AEf"))
            )
            last_height = driver.execute_script("return arguments[0].scrollHeight", scrollable_div)
            # get number of currently loaded comments
            n_comment_divs = len(driver.find_elements(By.CSS_SELECTOR, ".jftiEf.fontBodyMedium "))
            # limits to 500 reviews
            while n_comment_divs < total_reviews:
                # scroll to the bottom
                driver.execute_script(
                    "arguments[0].scrollTop = arguments[0].scrollHeight", scrollable_div
                )
                # wait for the page to load
                time.sleep(1.3)
                # recalculate number of loaded divs
                try:
                    # wait for the number of comments to change with a maximum wait time of 10 seconds
                    WebDriverWait(driver, 10).until(lambda x: len(x.find_elements(By.CSS_SELECTOR, ".jftiEf.fontBodyMedium ")) > n_comment_divs)
                    # recalculate number of loaded divs
                    n_comment_divs = len(driver.find_elements(By.CSS_SELECTOR, ".jftiEf.fontBodyMedium "))
                except TimeoutException:
                    # if after 10 seconds the number of comments hasn't changed, break
                    break
                # calculate new scroll height and compare with the last scroll height
                new_height = driver.execute_script("return arguments[0].scrollHeight", scrollable_div)
                if new_height == last_height:
                    break
                last_height = new_height
        except:
            print("Couldn't scroll...")

        # open the comments fully (see more)
        try:
            # gather all see more buttons and click them
            all_see_more_buttons = driver.find_elements(By.CSS_SELECTOR, ".w8nwRe.kyuRq")
            for button in all_see_more_buttons:
                button.click()
        except:
            print("Couldn't click all see more buttons.")

        # extract text
        try:
            comment_divs = driver.find_elements(By.CSS_SELECTOR, ".jftiEf.fontBodyMedium ")
            # there is necessarily a name, rating and a date but not comment
            for comment in comment_divs:
                # get name
                names.append(comment.find_element(By.CLASS_NAME, "d4r55 ").text)
                # get review but on the case of hotels, it might be different disposition
                try:
                    rating.append(comment.find_element(By.CLASS_NAME, "kvMYJc").get_attribute("aria-label"))
                except:
                    rating.append(comment.find_element(By.CLASS_NAME, "fzvQIb").text)
                # get dates, in case of hotels, might be different
                try:
                    dates.append(comment.find_element(By.CLASS_NAME, "rsqaWe").text)
                except:
                    date_element = comment.find_element(By.CLASS_NAME, "xRkPPb")
                    date_text = driver.execute_script('return arguments[0].firstChild.textContent;', date_element).strip()
                    dates.append(date_text)
                # verify that there is comment and add, else add empty string
                text = ""
                try:
                    myened_element = comment.find_element(By.CLASS_NAME, "MyEned")
                    text = myened_element.find_element(By.CLASS_NAME, "wiI7pd").text
                except NoSuchElementException:
                    text = ""
                comments.append(text)
        except:
            print("Something happened when trying to parse reviews...")

        finally:
            driver.quit()

        zipped = list(zip(names, rating, dates, comments))
        dictionary = {key: (v1, v2, v3) for key, v1, v2, v3 in zipped}

        return dictionary


    ### DATA CLEANING

    data_dict = scrape_reviews("https://www.google.com/maps/place/?q=place_id:" + my_place_id)

    df = pd.DataFrame.from_dict(data_dict, orient='index', columns=['grading', 'date', 'Text'], ).reset_index()
    df = df.dropna()

    # Remplacer les valeurs dans grading par des int
    df["grading"] = [int(x[0]) if x[0].isdigit() and "/" not in x else int(5*float(x.split('/')[0])/float(x.split('/')[1])) for x in df["grading"]]

    # Remplace date par un dictionnaire {months: number, years: number}
    def transform_date(date):
        weeks_or_days = ['week', 'day', 'semaine', 'jour']
        date_array = date.split()
        if any(word in date_array for word in weeks_or_days):
            return 0
        elif 'month' in date or 'mois' in date:
            if date_array[0].isdigit():
                return int(date_array[0])
            return 1
        elif date_array[0].isdigit():
            return int(date_array[0])*12
        return 12

    df["date"] = df["date"].apply(transform_date)

    df.index_name = 'ID'
    # Renommer la colonne d'index en "ID"
    df = df.rename_axis('ID').reset_index()
    df.set_index('ID', inplace=True)
    # Ajouter la colonne Language à la DataFrame

    # Pour que les détections soient consistent
    DetectorFactory.seed = 0
    def detect_language(text):
        try:
            return detect(text)
        except LangDetectException:
            return 'unknown'

    df['Language'] = df['Text'].apply(detect_language)
    df.drop(df.loc[df['Language'] == 'unknown'].index, inplace=True)

    def translate_text(df):
        for i, row in df.iterrows():
            if row['Language'] != 'en':
                text = row['Text']
                translated_text = GoogleTranslator(source='auto', target='en').translate(text)
                df.at[i, 'Text'] = translated_text
                df.at[i, 'Language'] = 'en'
        return df

    # Apply the function to the DataFrame
    df = translate_text(df)


    ### SENTIMENT ANALYSIS WITH ROBERTA

    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    nlp = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment", tokenizer=tokenizer)

    res = pd.DataFrame(columns=['label', 'score'])
    for index, row in df.iterrows():
        text = row['Text']
        langue = row['Language']

        # If the language is English and the token length doesn't exceed the limit
        if langue == 'en':
            tokens = tokenizer.encode(text, return_tensors='pt')

            if len(tokens[0]) <= 512:
                dico = pd.DataFrame(nlp(text))
                dico['ID'] = index
                res = pd.concat([res, dico], axis=0)

    # Remplacement des valeurs
    res['label'] = res['label'].replace('LABEL_0', 'NEGATIVE')
    res['label'] = res['label'].replace('LABEL_1', 'NEUTRAL')
    res['label'] = res['label'].replace('LABEL_2', 'POSITIVE')
    res["ID"] = res['ID'].astype(int)
    res = res.rename(columns={"label": "label (roberta)", "score": "score (roberta)"})
    res.set_index('ID', inplace=True)

    # Merge the two dataframes
    df = df.merge(res, left_index=True, right_index=True, how='left')



    ### DATA VISUALISATIONS

    figs_json = []

    ## Pourcentage de sentiments positifs, neutres, négatifs au fil du temps (3 courbes, un pour chaque)

    # Linear graph
    df_groupby_date_label = df.groupby(['date', 'label (roberta)']).size().unstack(fill_value=0)
    for col in ['NEGATIVE', 'NEUTRAL', 'POSITIVE']:
        if col not in df_groupby_date_label.columns:
            df_groupby_date_label[col] = 0
    df_groupby_date_label['POSITIVE'] = df_groupby_date_label['POSITIVE'].apply(lambda x:1.5*x)
    df_groupby_date_label['TOTAL'] = df_groupby_date_label[['NEGATIVE', 'NEUTRAL', 'POSITIVE']].sum(axis=1)
    df_groupby_date_label.drop(df_groupby_date_label[df_groupby_date_label.index > 12].index, inplace=True)
    df_groupby_date_label = df_groupby_date_label.sort_index(ascending=False)
    # Since graph_objects automatically sort the index in ascending, we put it in string, so it doesn't sort
    df_groupby_date_label.index = df_groupby_date_label.index.astype(str)
    fig1 = go.Figure()

    fig1.add_trace(go.Line(x=df_groupby_date_label.index,
                        y=df_groupby_date_label['POSITIVE']/df_groupby_date_label['TOTAL']*100,
                        line=dict(color='#00cc96'),
                        name='Positive',
                        text=['{:.2f}%'.format(v) for v in df_groupby_date_label['POSITIVE']/df_groupby_date_label['TOTAL']*100], hovertemplate='%{text}<extra></extra>'))

    fig1.add_trace(go.Line(x=df_groupby_date_label.index,
                        y=df_groupby_date_label['NEGATIVE']/df_groupby_date_label['TOTAL']*100,
                        name='Negative',
                        line=dict(color='#eb533a'),
                        text=['{:.2f}%'.format(v) for v in df_groupby_date_label['NEGATIVE']/df_groupby_date_label['TOTAL']*100], hovertemplate='%{text}<extra></extra>'))

    fig1.add_trace(go.Line(x=df_groupby_date_label.index,
                        y=df_groupby_date_label['NEUTRAL']/df_groupby_date_label['TOTAL']*100,
                        name='Neutral',
                        line=dict(color='#636efa'),
                        text=['{:.2f}%'.format(v) for v in df_groupby_date_label['NEUTRAL']/df_groupby_date_label['TOTAL']*100], hovertemplate='%{text}<extra></extra>'))

    fig1.update_layout(title='Sentiment Analysis over Time', xaxis_title='Month(s) ago', yaxis_title='Percentage %', yaxis_range=[0,100])
    # fig1.show()

    figs_json.append(pio.to_json(fig1))


    ## Graph Positif, neutre, négatif : diagramme circulaire (camembert)

    df_pie = df.groupby('label (roberta)').size()

    fig = go.Figure(data=[go.Pie(labels=df_pie.index, values=df_pie.values)])

    # fig.show()

    figs_json.append(pio.to_json(fig))


    ## Graph MOYENNE ETOILES

    df_groupby_date_score = df.groupby(['date', 'grading']).size().unstack(fill_value=0)

    df_groupby_date_score['Mean'] = (np.array(df_groupby_date_score.columns[:5]) * df_groupby_date_score.values).sum(axis=1) / df_groupby_date_score.values.sum(axis=1)

    df_groupby_date_score.drop(df_groupby_date_score[df_groupby_date_score.index > 12].index, inplace=True)
    df_groupby_date_score = df_groupby_date_score.sort_index(ascending=False)
    # Since graph_objects automatically sort the index in ascending, we put it in string, so it doesn't sort
    df_groupby_date_score.index = df_groupby_date_score.index.astype(str)

    fig = go.Figure()

    fig.add_trace(go.Line(x=df_groupby_date_score.index,
                        y=df_groupby_date_score['Mean'],
                        name='Rating',
                        text=['{:.2f}'.format(v) for v in df_groupby_date_score['Mean']], hovertemplate='%{text}<extra></extra>'))

    fig.update_layout(title='Star Rating over Time', xaxis_title='Month(s) ago', yaxis_title='Percentage', yaxis_range=[0,5])

    # fig.show()

    # WORD ANALYSIS


    checkpoint_map = available_checkpoints()
    aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(checkpoint='english',auto_device=True)

    def analyse_sent(text):
        atepc_result = aspect_extractor.extract_aspect(inference_source=text,  pred_sentiment=True)
        return atepc_result[0]['aspect'], atepc_result[0]['sentiment'], atepc_result[0]['confidence']

    def count_word(list):
        dict_res = {}
        for word in list.split():
            a = Word(word).lemmatize()
            if a not in dict_res:
                dict_res[a] = 1
            else:
                val = dict_res[a] + 1
                dict_res[a] = val
        sorted_dict = dict(sorted(dict_res.items(), key=lambda x: x[1], reverse=True))
        return sorted_dict

    def graph(pos, neg, name):
        values = [pos, neg]
        labels = ["percentage of positive reviews", "percentage of negative reviews"]
        colors = ['green', 'red']
        fig = go.Figure(data=[go.Pie(values=values, labels=labels, marker=dict(colors=colors))])

        fig.update_layout(title_text='Analysis for the word "'+name+'"')
        return fig.to_json()

    def absa_data(data):
        work_data = data[:100][:] #Environ 1min15 pour 25 avis
        # work_data = data
        aspect = []
        sentiment = []
        confidence = []

        for index, row in work_data.iterrows():
            text = []
            text.append(row['Text'])
            a, s, c = analyse_sent(text)
            aspect.append(a)
            sentiment.append(s)
            confidence.append(c)

        work_data["Aspect"] = aspect
        work_data["Sentiment"] = sentiment
        work_data["Confidence"] = confidence

        work_data.to_csv("export_csv_file.csv")
        data = pd.read_csv("export_csv_file.csv", sep=",")
        
        positive = ""
        neutral = ""
        negative = ""
        for idx, row in data.iterrows():
            aspect = eval(row.Aspect)
            sentiment = eval(row.Sentiment)
            for i in range(0, len(aspect)):
                if sentiment[i] == "Positive":
                    positive += " " + aspect[i]
                if sentiment[i] == "Neutral":
                    neutral += " " + aspect[i]
                else:
                    negative += " " + aspect[i]
        positive = positive.lower()
        # neutral = neutral.lower()
        negative = negative.lower()

        dict_positive = count_word(positive)
        # dict_neutral = count_word(neutral)
        dict_negative = count_word(negative)

        first_3_items = islice(dict_positive.items(), 3)
        dict_mix = {}
        for word, count in first_3_items:
            list_pos_neg = []
            list_pos_neg.append(count * 1.5)
            if word in dict_negative:
                list_pos_neg.append(dict_negative[word])
            else:
                list_pos_neg.append(0)
            dict_mix[word] = list_pos_neg
            sum = list_pos_neg[0] + list_pos_neg[1]
            list_pos_neg.append(list_pos_neg[0] / sum)
            list_pos_neg.append(list_pos_neg[1] / sum)

        list_graph = []
        for key in dict_mix:
            list_graph.append(graph(dict_mix[key][2],dict_mix[key][3], key))
        return list_graph

    figs_json.append(pio.to_json(fig))

    absa = absa_data(df)
    for figure in absa:
        figs_json.append(figure)


    return figs_json
