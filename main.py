# pip3 install bs4
# pip3 install requests
# pip3 install lxml
# pip3 install pandas
from bs4 import BeautifulSoup
import requests
from datetime import datetime as dt
from datetime import date
import pandas as pd
import matplotlib.pyplot as plt


def get_content(url):
    try:
        source = requests.get(url).text
        return BeautifulSoup(source, 'lxml')
    except requests.exceptions.RequestException as e:
        return False


def remove_null_values(data):
    data = data[~data.post_title.isnull()].copy()
    data = data[~data.post_date.isnull()].copy()
    data = data[~data.post_link.isnull()].copy()
    data = data[~data.post_author.isnull()].copy()
    return data


# get posts by targetTimeYearMonth and timeLimit which is
# representing upper boundary of dates
def get_posts(target_time_year_month, timeLimit):
    data = pd.DataFrame()
    url_content = (get_content("https://edition.cnn.com/article/sitemap-" + target_time_year_month + ".html"))
    if not url_content:
        print("URL is corrupted, please contact administrator")
        return
    articles_elements = url_content.find_all('div', class_='sitemap-entry')
    posts = articles_elements[1]
    posts = posts.find_all('li')
    id_generator = 0
    for post in posts:
        post_date = post.span.text
        if dt.strptime(post_date, "%Y-%m-%d").date() > timeLimit:
            continue
        post_id = "id-" + str(id_generator)
        id_generator += 1
        post_content = post.find('span', class_='sitemap-link')
        post_title = post_content.a.text
        post_link = post_content.a['href']
        # in order to get the author it needs to access the post itself
        article_page = get_content(post_link)
        try:
            post_author = article_page.find('span', class_='metadata__byline__author').text
        except Exception as ex:
            # print("Ignoring this data as there is no title")
            continue
        post_row = {"post_id": post_id, "post_title": post_title, "post_date": post_date, "post_link":post_link, "post_author": post_author}
        data = data.append(post_row, ignore_index=True)
    return remove_null_values(data)


# get_date_count_distr groups posts by date and counts them
# returns sorted date count combination
def get_date_count_distr(data):
    data_counted = data.value_counts(['post_date']).reset_index(name='count')
    data_counted["post_date"] = pd.to_datetime(data_counted["post_date"])
    data_counted = data_counted.sort_values(by="post_date")
    return data_counted


# calculate_common_words calculates top keywords by using common words calculation
def calculate_common_words(data):
    titles = data["post_title"]
    common_words_count = {}
    unnecessary_words = {"the", "to", "of", "in", "a", "on", "and", "for", "his", "over", "what",
                         "at", "as", "with", "from", "it", "but", "how", "is", "now", "her", "after",
                         "by", "an", "will", "are", "he"}
    # it is splitting the titles into words and finding duplicated words across all the titles
    for title in titles:
        title = title.lower()
        words = title.split()
        for word in words:
            count = 0
            for keyword in common_words_count:
                if word == keyword:
                    common_words_count[keyword] += 1
                    count += 1
            if count == 0 and not unnecessary_words.__contains__(word):
                common_words_count[word] = 0

    # sort common_words_count by the count in descending order
    common_words_count = {k: v for k, v in sorted(common_words_count.items(), reverse=True, key=lambda item: item[1])}
    data_items = common_words_count.items()
    common_words_count = pd.DataFrame(list(data_items))
    common_words_count.columns = ['Keyword', "Occurrences"]
    return common_words_count


# populate_topic_names populates new column called topic and matches it with the predefined topics
# posts have been annotated manually for quality purposes
def populate_topic_names(data, year):
    topics = pd.read_csv("data" + str(year) + "annotations.csv")
    for index_data, row in data.iterrows():
        post_link = row["post_link"]
        if len(topics[topics['post_link'].str.contains(post_link)]):
            row_index_topic_data = topics[topics['post_link'].str.contains(post_link)].index[0]
            data.at[index_data, "topic"] = topics["topic"][row_index_topic_data]
    return data


# EDA of the data by plotting top keywords and common words being used
def exploratory_data_analysis(data, year):
    data_date_count_distr = get_date_count_distr(data)

    data_common_words = calculate_common_words(data)
    common_words_count_not_zeros = data_common_words[(data_common_words != 0).all(1)]
    common_words_count_subset = common_words_count_not_zeros[0:10]
    common_words_count_subset.Occurrences.value_counts(normalize=True)

    fig_eda, axes_eda = plt.subplots(1, 2)
    data_date_count_distr.plot(kind='line', figsize=(5, 5), fontsize=10, ax=axes_eda[0], x='post_date', y='count')
    axes_eda[0].set_title("Number of news for December " + str(year), fontsize=12)
    common_words_count_subset.plot(kind='barh', figsize=(5, 5), fontsize=10, ax=axes_eda[1],
                                   x='Keyword', y='Occurrences')
    axes_eda[1].set_title('Top keywords usage for December ' + str(year), fontsize=12)
    plt.show()


# comparison_analysis does comparison analysis between 2 years by topic distribution
def comparison_analysis(data2019_annotated_ca, data2020_annotated_ca):
    fig, axes = plt.subplots(1, 2)
    data2019_annotated_ca['topic'].value_counts().plot(kind='pie', figsize=(5, 5), fontsize=10, ax=axes[0])
    axes[0].set_title('Topic distribution for 2019', fontsize=12)
    data2020_annotated_ca['topic'].value_counts().plot(kind='pie', figsize=(5, 5), fontsize=10, ax=axes[1])
    axes[1].set_title('Topic distribution for 2020', fontsize=12)
    plt.show()


# user_interaction is responsible for
# 1.Getting topic name from user -> Returning all the posts with their id and title, topic
# 2.Getting id of the post from the user  ->  Returning the post with the link to the post
def user_interaction(data_annotated_ui):
    data_title_id_topic = data_annotated_ui.copy(deep=True)
    data_id_link = data_annotated_ui.copy(deep=True)
    data_title_id_topic.drop(['post_author', 'post_date', 'post_link'], axis=1, inplace=True)
    data_id_link.drop(['post_author', 'post_date', 'post_title', 'topic'], axis=1, inplace=True)
    print("Please insert topic name by typing 'topic=CHOSEN_TOPIC'!")
    while True:
        try:
            input_entry = str(input())
            if input_entry[:6] == 'topic=':
                input_topic = input_entry[6:]
                if len(data_title_id_topic[data_title_id_topic['topic'].str.contains(input_topic)]):
                    row_index_topic = data_title_id_topic[
                        data_title_id_topic['topic'].str.contains(input_topic)].index
                    for index in row_index_topic:
                        print(data_title_id_topic.iloc[[index]])
                        print("\n\n Please choose the article by typing  'id=CHOSEN_ID'")
                else:
                    print("Please enter correct topic name! or type 'exit'  and press 'Enter' to exit the program")
            elif input_entry[:3] == "id=":
                input_id = input_entry[3:]
                if len(data_id_link[data_id_link['post_id'].str.contains(input_id)]):
                    row_index_topic = data_id_link[data_id_link['post_id'].str.contains(input_id)].index[0]
                    print(data_id_link.iloc[[row_index_topic]].post_link)
                else:
                    print("Please enter correct id! or type 'exit'  and press 'Enter' to exit the program")
            elif input_entry == "exit":
                print("Exiting the program. Thank you for using our services :)")
                break
            else:
                print("This command doesnt exist,"
                      " please try correct command by typing either topic=TOPIC or id=CHOSEN_ID "
                      "or type 'exit'  and press 'Enter' to exit the program")
        except ValueError:
            print("Please enter correct topic name! or type 'exit'  and press 'Enter' to exit the program")


if __name__ == '__main__':
    # this is needed to be able to see full url texts as they can be long for pandas dataframe
    pd.options.display.max_colwidth = 200
    # this is needed to show more columns in big pandas dataframe
    pd.set_option('display.max_columns', None)

    print("Collecting posts from CNN in 2019 December 1-15th...")
    data2019 = get_posts("2019-12", date(2019, 12, 15))
    print("Collection of posts from CNN for 2019 is completed")
    print("Collecting posts from CNN in 2020 December 1-15th...")
    data2020 = get_posts("2020-12", date(2020, 12, 15))
    print("Collection of posts from CNN for 2020 is completed")

    # populate predefined topics
    data2019_annotated = populate_topic_names(data2019, 2019)
    data2020_annotated = populate_topic_names(data2020, 2020)

    exploratory_data_analysis(data2019, 2019)
    comparison_analysis(data2019_annotated, data2020_annotated)
    user_interaction(data2020_annotated)


