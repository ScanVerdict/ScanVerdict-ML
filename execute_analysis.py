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


def execute_analysis(my_place_id):
    ### DATA ACQUISITION

    def scrape_reviews(my_place_id):

        google_url = "https://www.google.com/maps/place/?q=place_id:"+my_place_id

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
            if total_reviews > 300:
                total_reviews = 300
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

    data_dict = scrape_reviews("https://www.google.com/maps/place/Googleplex/@37.4220656,-122.0862784,17z/data=!3m1!4b1!4m6!3m5!1s0x808fba02425dad8f:0x6c296c66619367e0!8m2!3d37.4220656!4d-122.0840897!16zL20vMDNiYnkx?entry=ttu")

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

    ## Mots importants
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    # Convertir la colonne "text" en une liste de tous les commentaires
    commentaires = df['Text'].tolist()
    # Concaténer tous les commentaires en une seule chaîne de caractères
    texte_complet = ' '.join(commentaires)
    # Diviser la chaîne de caractères en mots individuels
    mots = texte_complet.split()
    # Utiliser l'étiquetage morpho-syntaxique pour identifier les noms communs
    noms_communs = [mot for mot, pos in nltk.pos_tag(mots) if pos.startswith('NN')]
    # Compter les occurrences de chaque nom commun
    compteur_noms_communs = Counter(noms_communs)
    # Récupérer les trois noms communs les plus fréquents et leurs occurrences
    top_noms_communs = compteur_noms_communs.most_common(3)

    # Créer une liste de tous les commentaires positifs, négatifs et neutres
    commentaires_positifs = df[df['label (roberta)'] == 'POSITIVE']['Text']
    commentaires_negatifs = df[df['label (roberta)'] == 'NEGATIVE']['Text']
    commentaires_neutres = df[df['label (roberta)'] == 'NEUTRAL']['Text']

    for i in top_noms_communs:
        nom = i[0]
        # Compter le nombre de commentaires positifs, négatifs, et neutres pour le nom courant
        nb_positifs = commentaires_positifs.str.contains(nom, flags=re.IGNORECASE, regex=True).sum()
        nb_negatifs = commentaires_negatifs.str.contains(nom, flags=re.IGNORECASE, regex=True).sum()
        nb_neutres = commentaires_neutres.str.contains(nom, flags=re.IGNORECASE, regex=True).sum()

        # Créer une liste contenant le nombre de commentaires positifs, négatifs, et neutres pour le nom courant
        liste = [nb_positifs, nb_negatifs, nb_neutres]

        # Creer une pie chart pour le nom courant
        fig = go.Figure(data=[go.Pie(labels=['Positif', 'Négatif', 'Neutre'], values=liste)])
        fig.update_layout(title_text='Sentiment des commentaires contenant le mot "' + nom + '"')
        fig.show()
        figs_json.append(pio.to_json(fig))

    ## Pourcentage de sentiments positifs, neutres, négatifs au fil du temps (3 courbes, un pour chaque)

    # Subgraph1
    df_groupby_date_label = df.groupby(['date', 'label (roberta)']).size().unstack(fill_value=0)
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
    fig1.show()


    ## Subgraph2

    y_values = df_groupby_date_label['POSITIVE']/df_groupby_date_label['TOTAL']*100
    x_numeric = np.arange(len(df_groupby_date_label.index)).reshape(-1, 1)

    model = LinearRegression()
    model.fit(x_numeric, y_values)
    line = model.predict(x_numeric)

    hover_text = ['Y: {:.2f}%<br>Percent Change with previous month: {:.2f}%<br>Absolute Change with previous month: {:.2f}%'.format(y_values[i], (y_values[i]-y_values[i-1])/y_values[i-1]*100, y_values[i]-y_values[i-1]) if y_values[i]-y_values[i-1] <= 0 else 'Y: {:.2f}%<br>Percent Change with previous month: +{:.2f}%<br>Absolute Change with previous month: +{:.2f}%'.format(y_values[i], (y_values[i]-y_values[i-1])/y_values[i-1]*100, y_values[i]-y_values[i-1]) for i in range(1, len(y_values))]
    hover_text.insert(0, 'Y: '+"{:.2f}".format(y_values[0])+'%<br>Percent Change with previous month: 0%<br>Absolute Change with previous month: 0%')

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df_groupby_date_label.index,
                            y=y_values,
                            mode='markers',
                            marker=dict(color='#00cc96'),
                            name='Positive',
                            text=hover_text,
                            hovertemplate='%{text}<extra></extra>'))

    fig2.add_trace(go.Scatter(x=df_groupby_date_label.index,
                            y=line,
                            mode='lines',
                            line=dict(color='yellow'),
                            name='Regression Line',
                            text=['Percentage Change over the whole period: {:.2f}%'.format(line[-1]-line[0]) if (line[-1]-line[0]) <= 0 else 'Percentage Change over the whole period: +{:.2f}%'.format(line[-1]-line[0])  for i in range(len(y_values))],
                            hovertemplate='%{text}<extra></extra>'))

    fig2.update_layout(title='Positive Change over Time', xaxis_title='Month(s) ago', yaxis_title='Percentage', yaxis_range=[0,100])
    fig2.show()


    ## Subgraph3

    y_values = df_groupby_date_label['NEGATIVE']/df_groupby_date_label['TOTAL']*100
    x_numeric = np.arange(len(df_groupby_date_label.index)).reshape(-1, 1)

    model = LinearRegression()
    model.fit(x_numeric, y_values)
    line = model.predict(x_numeric)

    hover_text = ['Y: {:.2f}%<br>Percent Change with previous month: {:.2f}%<br>Absolute Change with previous month: {:.2f}%'.format(y_values[i], (y_values[i]-y_values[i-1])/y_values[i-1]*100, y_values[i]-y_values[i-1]) if y_values[i]-y_values[i-1] <= 0 else 'Y: {:.2f}%<br>Percent Change with previous month: +{:.2f}%<br>Absolute Change with previous month: +{:.2f}%'.format(y_values[i], (y_values[i]-y_values[i-1])/y_values[i-1]*100, y_values[i]-y_values[i-1]) for i in range(1, len(y_values))]
    hover_text.insert(0, 'Y: '+"{:.2f}".format(y_values[0])+'%<br>Percent Change with previous month: 0%<br>Absolute Change with previous month: 0%')

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=df_groupby_date_label.index,
                            y=y_values,
                            mode='markers',
                            marker=dict(color='#eb533a'),
                            name='Negative',
                            text=hover_text,
                            hovertemplate='%{text}<extra></extra>'))

    fig3.add_trace(go.Scatter(x=df_groupby_date_label.index,
                            y=line,
                            mode='lines',
                            line=dict(color='yellow'),
                            name='Regression Line',
                            text=['Percentage Change over the whole period: {:.2f}%'.format(line[-1]-line[0]) if (line[-1]-line[0]) <= 0 else 'Percentage Change over the whole period: +{:.2f}%'.format(line[-1]-line[0]) for i in range(len(y_values))],
                            hovertemplate='%{text}<extra></extra>'))

    fig3.update_layout(title='Negative Change over Time', xaxis_title='Month(s) ago', yaxis_title='Percentage', yaxis_range=[0,100])
    fig3.show()


    ## Subgraph4

    y_values = df_groupby_date_label['NEUTRAL']/df_groupby_date_label['TOTAL']*100
    x_numeric = np.arange(len(df_groupby_date_label.index)).reshape(-1, 1)

    model = LinearRegression()
    model.fit(x_numeric, y_values)
    line = model.predict(x_numeric)

    hover_text = ['Y: {:.2f}%<br>Percent Change with previous month: {:.2f}%<br>Absolute Change with previous month: {:.2f}%'.format(y_values[i], (y_values[i]-y_values[i-1])/y_values[i-1]*100, y_values[i]-y_values[i-1]) if y_values[i]-y_values[i-1] <= 0 else 'Y: {:.2f}%<br>Percent Change with previous month: +{:.2f}%<br>Absolute Change with previous month: +{:.2f}%'.format(y_values[i], (y_values[i]-y_values[i-1])/y_values[i-1]*100, y_values[i]-y_values[i-1]) for i in range(1, len(y_values))]
    hover_text.insert(0, 'Y: '+"{:.2f}".format(y_values[0])+'%<br>Percent Change with previous month: 0%<br>Absolute Change with previous month: 0%')

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=df_groupby_date_label.index,
                            y=y_values,
                            mode='markers',
                            marker=dict(color='#636efa'),
                            name='Neutral',
                            text=hover_text,
                            hovertemplate='%{text}<extra></extra>'))

    fig4.add_trace(go.Scatter(x=df_groupby_date_label.index,
                            y=line,
                            mode='lines',
                            line=dict(color='yellow'),
                            name='Regression Line',
                            text=['Percentage Change over the whole period: {:.2f}%'.format(line[-1]-line[0]) if (line[-1]-line[0]) <= 0 else 'Percentage Change over the whole period: +{:.2f}%'.format(line[-1]-line[0]) for i in range(len(y_values))],
                            hovertemplate='%{text}<extra></extra>'))

    fig4.update_layout(title='Neutral Change over Time', xaxis_title='Month(s) ago', yaxis_title='Percentage', yaxis_range=[0,100])
    fig4.show()


    ## MAIN GRAPH

    # Create subplots with 4 rows and 1 column
    fig = sp.make_subplots(rows=4, cols=1)

    # Add all figures
    for trace in fig1.data:
        fig.add_trace(trace, row=1, col=1)
    for trace in fig2.data:
        fig.add_trace(trace, row=2, col=1)
    for trace in fig3.data:
        fig.add_trace(trace, row=3, col=1)
    for trace in fig4.data:
        fig.add_trace(trace, row=4, col=1)

    fig.update_yaxes(range=[0, 100])
    fig.update_layout(title_text="Sentiment Analysis over Time", height=800)

    fig.add_annotation(text="Month(s) ago", xref="paper", yref="paper", x=0.5, y=-0.1, showarrow=False)
    fig.add_annotation(text="Percentage %", textangle=-90, xref="paper", yref="paper", x=-0.1, y=0.5, showarrow=False)

    # Show the combined subplots
    fig.show()

    figs_json.append(pio.to_json(fig))


    ## Graph Positif, neutre, négatif : diagramme circulaire (camembert)

    df_pie = df.groupby('label (roberta)').size()

    fig = go.Figure(data=[go.Pie(labels=df_pie.index, values=df_pie.values)])

    fig.show()

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

    fig.show()

    figs_json.append(pio.to_json(fig))

    print(figs_json)
    return figs_json