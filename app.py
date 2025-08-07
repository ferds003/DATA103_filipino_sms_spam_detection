import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import pickle
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from wordcloud import WordCloud
from PIL import Image
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler


MODEL_PATHS = {
    "Random Forest": "models/rf_model_3.pkl",
    "Multinomial Naive Bayes": "models/mNB_model_3.pkl",
    "Support Vector Machine": "models/svm_model_3.pkl",
    "Complement Naive Bayes": "models/cNB_model_3.pkl",
}


@st.cache_resource
def load_model(model_name):
    model_path = MODEL_PATHS[model_name]
    with open(model_path, "rb") as f:
        return pickle.load(f)


sample_df = pd.read_csv("src/test.csv")
eda_df = pd.read_csv("src/eda_df.csv")
ml_df = pd.read_csv("src/cleaned_sms_text_labels_only_v2.csv")


## define feature importance per model and put it into a plotly chart when classifying
def explain_prediction(model, text, model_name, top_n=10):
    # Unpack classifier as saved models are in made pipeline saved in mlflow
    if hasattr(model, "named_steps"):
        clf = model.named_steps.get("classifier")
        vectorizer = model.named_steps.get(
            "vectorizer"
        )  # i was having errors since im calling a different vectorizer!
    else:
        clf = model
    # Compute feature contributions for the chosen models; make contributions in absolute values
    X_input = vectorizer.transform([text])
    feature_names = np.array(vectorizer.get_feature_names_out())
    if not X_input.nnz:
        return pd.DataFrame(
            columns=["Token", "Contribution"]
        )  ## Handles edge cases like empty input

    if model_name == "Multinomial Naive Bayes":
        class_log_prob = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
        contributions = X_input.toarray()[0] * class_log_prob
        contributions = np.abs(contributions)
    elif model_name == "Complement Naive Bayes":
        class_log_prob = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
        contributions = X_input.toarray()[0] * class_log_prob
        contributions = np.abs(contributions)
    elif model_name == "Random Forest":
        contributions = clf.feature_importances_ * X_input.toarray()[0]
        contributions = np.abs(contributions)
    elif model_name == "Support Vector Machine":
        contributions = clf.coef_.toarray()[0] * X_input.toarray()[0]
        contributions = np.abs(contributions)
    else:
        return st.error("Invalid model name")

    token_scores = list(zip(feature_names, contributions))
    token_scores = sorted(token_scores, key=lambda x: abs(x[1]), reverse=True)
    return pd.DataFrame(token_scores[:top_n], columns=["Token", "Contribution"])


## --FOR THE WORDCLOUD AT EdATAB--
def get_wordcloud_data(text):
    wc = WordCloud(
        width=1200,
        height=600,
        background_color="white",
        colormap="viridis",
        max_font_size=60,
        prefer_horizontal=0.8,
        collocations=False,
        min_word_length=3,
        max_words=75,
        relative_scaling=0.5,
        contour_color="black",
        contour_width=0.2,
    ).generate(text)
    elements = []
    for (word, freq), font_size, position, orientation, color in wc.layout_:
        elements.append((word, freq, position))  # position is already a tuple (x, y)
    return elements


## --FOR THE WORDCLOUD AT EdATAB--
def plot_wordcloud(elements, title):
    words, frequencies, positions = zip(*elements)
    x = [pos[0] for pos in positions]
    y = [-pos[1] for pos in positions]  # Flip Y to display properly
    sizes = [freq * 200 for freq in frequencies]

    fig = go.Figure(
        data=[
            go.Scatter(
                x=x,
                y=y,
                mode="text",
                text=words,
                textfont=dict(size=sizes),
                hoverinfo="text",
                textposition="middle center",
            )
        ]
    )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        showlegend=False,
        xaxis=dict(showgrid=False, visible=False),
        yaxis=dict(showgrid=False, visible=False),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


with open("src/combined_stopwords.pkl", "rb") as f:
    combined_stopwords = pickle.load(f)
vectorizer = TfidfVectorizer(stop_words=list(combined_stopwords), max_features=20)

bow_cv = CountVectorizer(stop_words=list(combined_stopwords), max_features=1000)
tfidf = TfidfVectorizer(stop_words=list(combined_stopwords), max_features=1000)

## ----- START OF APP ------
st.set_page_config(
    layout="wide", page_title="DATA103_filipino_spam_detection", page_icon="😎"
)

st.title("Spam SMS Detection in the Filipino Context")

## SIDEBAR
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    About the Team:

    Demo created by [Ferds Magallanes](https://ferds003.github.io), Hitika Motwani, Neil Penaflor, and Mark Abergos using Streamlit and Hugging Face Spaces.
    Purposes to demonstrate their NLP classification project for their minor data science class.

    Contributions:
    - Data Curation - Ferds, Hitika
    - EDA - Hitika, Neil
    - Features Selection and NLP_Training - Mark, Ferds
    - Eval and Demo - Ferds

    Acknowledgements:

    The team would like to thank Doc Jet Virtusio for the support and teachings he gave in our minor class :))

    """
)

## TABS PER PROJECT TASK
(
    DemoTAB,
    DataCurationTAB,
    EdATAB,
    FeatureSelectionTAB,
    TrainingPipelineTAB,
    ModelEvaluationTAB,
    ConTAB,
) = st.tabs(
    [
        "Demo",
        "Data Curation",
        "EDA",
        "Feature Selection",
        "Training Pipeline",
        "Model Evaluation",
        "Conclusion and Recommendations",
    ]
)

with DemoTAB:
    st.write("")
    st.markdown("""
    Hi there! Input your sample sms messages for us to classify if it is spam or not.
    Correspondingly, we will provide what text (tokens) signify in percentage is spam or not.
    """)

    ## Provide user with sample spam and ham messages
    with st.expander(
        "📋 Try a sample message! This is from our test.csv so this data is not trained on our model"
    ):
        st.markdown(
            "Select a sample SMS message from below: Label is 0 for ham 🍗 and 1 for spam 🥫"
        )
        label_map = {0: "Ham", 1: "Spam"}
        sample_index = st.selectbox(
            "Select a sample SMS message",
            sample_df.index,
            format_func=lambda x: f"SMS {x} - {label_map[sample_df.loc[x, 'label']]}: {sample_df.loc[x, 'text'][:50]}",
        )
        if st.button("Use this sample"):
            st.session_state["1"] = sample_df.loc[sample_index, "text"]

    ## Model selection
    selected_model_name = st.selectbox(
        "Select Classification Model", list(MODEL_PATHS.keys())
    )
    text = st.text_area("Enter SMS to classify here!", height=100, key="1")

    ## CASE WHEN BUTTON IS PRESSED
    if st.button("Classify"):
        if text:
            with st.spinner("Analyzing..."):
                clf = load_model(selected_model_name)
                prediction = clf.predict([text])[0]
                pred_proba = clf.predict_proba([text])[0]

                st.success(f"Prediction: {'Spam' if prediction == 1 else 'Not Spam'}")
                st.info(f"Probability of Spam: {pred_proba[1]:.2%}")
                st.info(f"Probability of Not Spam: {pred_proba[0]:.2%}")

                st.markdown("### Feature Importance")
                explain_df = explain_prediction(clf, text, selected_model_name)
                if (
                    explain_df is not None and not explain_df.empty
                ):  ## calling the  function
                    fig1 = px.bar(
                        explain_df,
                        x="Contribution",
                        y="Token",
                        orientation="h",
                        title="Top Contributing Tokens to Prediction",
                        labels={"Contribution": "Impact Score"},
                        color="Contribution",
                        color_continuous_scale="RdBu",
                        template="plotly_dark",
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                    print(
                        "Top tokens:", explain_df.head(8)
                    )  ## DEBUGGING LINE; Can be checked on streamlit terminal
                else:
                    st.warning("Unable to compute token contribution for this model.")
        else:
            st.warning("Please input text to classify.")
    st.markdown("---")
    st.markdown(""" 
    ## Changelogs:
    - Version 2 (August 2, 2025): Improvements across `precision` and `recall` metrics on training by random oversampling ham classes on `X_train` in training pipeline using `imbalanced-learn`package. Latest deployed models trained under these run params.
    - Version 1 (July 28, 2025): Initial demo of the project with 4 traditional ML classifiers using TFIDF vectorizer.
    """)

with DataCurationTAB:
    st.markdown(""" 
     Data cleaning and pre-processing is necessary as we are considering three datasets with different contexts. Below is a summary of the data treatment and insights done to make the versions of the dataset. <mark>We avoided the use of the UCL SMS repository for this project as this does not capture the filipino context.</mark>

- For [Dataset 1](https://www.kaggle.com/datasets/scottleechua/ph-spam-marketing-sms-w-timestamps):
  - drop any null values; drop any full redactions done in `text` column through regex. Drops 74% of the dataset as text sms data is salient to the project.
  - checked any redactions of the similar <> format within `text` feature. Concluded that any other text with <> format are coming from spam and ads category
  - Drops `date_read` column. Renamed `date_received` column
  - Made a label columns that considers the text as label `spam` if it is within the category `spam` and `ads` based on `category` column
  - applied `get_carrier` function to get sms local provider.
- For [Dataset 2](https://www.kaggle.com/datasets/bwandowando/philippine-spam-sms-messages):
  - drop any null values; all data will be considered under the label `spam` for its sms text messages data.
  - checked any redactions of the similar <> format within `text` feature. found `<REAL NAME>` redactions; replaced it with a blankspace
  - dropped `hashed_cellphone_number` and `carrier` column to apply own `get_carrier` function that considers also DITO sms provider.
  - renamed column `masked_celphone_number` to `sender` and `date` to `date_received` similar to dataset 1.
- For [Dataset 3](https://github.com/Yissuh/Filipino-Spam-SMS-Detection-Model/blob/main/data-set.csv):
  - drop any null values; dropped any `<<Content not supported.>>|<#>` tags and any other tags that are labeled under ham messages.
  - renamed column `message` to `text` in conformity with other datasets.
  - checked any redactions of the similar <> format within `text` feature. found `<CODE>`, `<DATE>`, `<Last 4 digits of acct num>`, `<REFERENCE NUMBER>`, `<TIME>`, `<space>` redactions to be ham messages; replaced it with a blankspace.

**Cleaning for Merged Dataset**
- Each datasets considered was checked for null values afterwards and checked if this is acceptable for processing; Decision was to just do an SQL query `NOT NULL` to filter values on target features for each respective tasks
- Used `concat` function to merged these three datasets knowing that column names are the same for all.
- Used `drop_duplicates` to remove 43 duplicate observations.
- Explicitly defined `date_received` as a datetime for EDA.

**NLP Processing on Merged Dataset**
- further processing done to get rid of unwanted characters; retrieves pure lowercased alphanumberic characters with no extract whitespace.
- gets rid of any `http-like` text which are urls; This was considered as ham messages often provide urls and lessen tokens considered in vectorization.
- noticed that both spam and ham have mixed english and tagalog text.   Used `langdetect` package to sort if english or tagalog then used packages `spacy` for english and `calamancy` for filipino/tagalog stopwords removal, punctuation removal, and tokenization towards BOW and TF-IDF vectorization.
- `lemmatization` was also considered for both languages.           
                """)

    st.text("Below is the version of the dataset used for EDA purposes")
    st.dataframe(eda_df, use_container_width=True)
    st.markdown("### Quick Stats for this dataset")
    st.write(eda_df.describe())

    # Optional let user download it
    st.download_button(
        label="📥 Download CSV",
        data=eda_df.to_csv(index=False).encode(
            "utf-8"
        ),  # Might be redundant since i have the csv already
        file_name="spam_eda_df.csv",
        mime="text/csv",
    )

    st.text("Below is the version of the dataset used for ML Training purposes")
    st.dataframe(ml_df, use_container_width=True)
    st.markdown("### Quick Stats for this dataset")
    st.write(ml_df.describe())

    # Optional let user download it
    st.download_button(
        label="📥 Download CSV",
        data=ml_df.to_csv(index=False).encode(
            "utf-8"
        ),  # Might be redundant since i have the csv already
        file_name="spam_ml_df.csv",
        mime="text/csv",
    )
    st.markdown("""
    **--Datasets Considered--**

    Below are the following datasets sourced along with a simple description of its features:

    > [ph-spam-marketing-sms-w-timestamps](https://www.kaggle.com/datasets/scottleechua/ph-spam-marketing-sms-w-timestamps) from Kaggle made by u/Scott_Lee_Chua:

    | column | count | description | dtype |
    |---|---|---|---|
    | data_received | 1622 | Datetime SMS was received in timezone UTC+8. | object |
    | sender | 1622 | A partially-masked phone number, unmasked alphanumeric Sender ID, or one of three special values: <br><br><br>-redacted_contact if sender is a person in my personal contact book; <br><br>-redacted_individual if sender is a person not in my contacts and the message is solicited (e.g., updates from delivery riders); or <br><br>-redacted_business if sender is a business/service and all their messages are solicited.<br> | object |
    | category | 1622 | Takes one of five possible values: spam, ads, gov, notifs, or OTP. See categories below.<br><br><br>spam — unsolicited messages from unknown senders.<br><br>ads — marketing messages from known businesses or services, such as GCash and Smart.<br><br>gov — public service announcements from Philippine government agencies, such as NTC and NDRRMC. <br><br>notifs — a catch-all category for legitimate and private messages, such as transaction confirmations, delivery updates, and a handful of personal conversations. <br><br>OTP — genuine one-time passwords <br> | object |
    | text | 1622 | Full text for spam, ads, and gov. | object |

    > [🇵🇭 Philippine Spam/ Scam SMS](https://www.kaggle.com/datasets/bwandowando/philippine-spam-sms-messages) from Kaggle made by u/bwandowando:

    | column | count | description | dtype |
    |---|---|---|---|
    | masked_celphone_number | 945 | cellphone number that is masked instead on the first five numbers and last three numbers.  | object |
    | hashed_celphone_number | 945 | part of the XML data given that provides its unique identifier. | object |
    | date | 945 | date when text was received. | object |
    | text | 945 | Full text for spam, ads, and gov. | object |
    | carrier | 945 | SMS registry fix that is associated with the first five numbers of the cellphone number. | object |


    >[Filipino-Spam-SMS-Detection-Model](https://github.com/Yissuh/Filipino-Spam-SMS-Detection-Model/blob/main/data-set.csv) from Github made by u/Yissuh and TUP-M students:

    | column | count | description | dtype |  |
    |---|---|---|---|---|
    | text | 873 | Full text for spam, ads, and gov. | object |  |
    | label | 873 | text that is labeled is either spam or ham. | object |  |

    <br>**--Final Version of the Dataset Considerd--**

    With this, the dataset considered have 2 versions being considered for this project, one for EDA and one for ML_training that is saved in an SQLite database for querying. For simplicity, csv_files are provided done after data cleaning and pre-processing that you may check further in succeeding sections. Below is a summary of the version dataset:

    > Version of the Dataset for EDA:

    | column | count | description | dtype |
    |---|---|---|---|
    | date_received | 1713 | date when text was received. | datetime |
    | sender | 1713 | partially-masked phone numbers. | object |
    | preprocessed_text | 1713 | cleaned data without redactions and only alphanumeric characters. | object |
    | carrier | 1713 | SMS registry fix from sender's number. | object |
    | label | 1713 | label if text is either spam or ham. | category |


    > Version of the Dataset for ML Training:

    | column | count | description | dtype |
    |---|---|---|---|
    | text | 3379 | cleaned data without redactions and only alphanumeric characters. | object |
    | label | 3379 | label if text is either spam or ham. | category |

    > ham and spam count for ML Training: We recognize that there is a class imbalance between the two labels. We treat this as an acceptable data for training.

    | label | text |
    |-------|------|
    | ham   | 418  |
    | spam  | 2964 |
                """)

with EdATAB:
    st.markdown("""
    The objective of the EDA for this project is to make an interactive visualization for the exploration of the made dataset in regards to the SMS registration act of 2022. 
    This will be displayed in the demo. The following are the questions answered through the EDA:
    
    - How many scam messages were received before/after SIM registration?
    - Which telecom prefixes are most targeted or send the most spam?
    - What words are most associated with scams?
    """)

    st.markdown("""
    Summary of the insights from the EDA are the following:
    - ham messages can be characterized to be typically shorter in text length while spam messages averages from 50 to 200 text length within the filipino-context.
    - Most frequent SMS carrier that does unsolicited messages is coming from `Smart`.
    - Most frequent messages associated with spam are the terms `bonus`, `BDO`, `com`, `free`, and `win` most likely persistent with unsolicited online gambling ads and solicitations to deposit on a unsafe link.
    - Most frequent messages associated with ham are the terms `access`, `android`, `app`, `code` most likely persistent with government announcements and warnings. Other context for this would be retrieving one-time-passwords (OTP) code. 
    - scam/spam messages are still prevalent despite the implementation of the sim registration act with known operations of this peaking during the middle of the week during the afternoon.

    Additionally, EDA done for ml_training dataset version under `cleaned_sms_text_labels_only_v2.csv` is provided.            
                """)

    ## -- EDA VISUALIZATIONS HERE --
    st.write("### Text Length Distribution")
    eda_df["length"] = eda_df["preprocessed_text"].apply(len)
    eda_df.head()
    fig2 = px.histogram(
        eda_df,
        x="length",
        nbins=50,  # optional, adjust bins
        color_discrete_sequence=["#636EFA"],  # optional style
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.write("### Prefix-wise spam distribution")
    fig3 = px.histogram(eda_df, x="carrier", color_discrete_sequence=["#636EFA"])
    st.plotly_chart(fig3, use_container_width=True)

    copy = eda_df.copy()
    copy["date_received"] = pd.to_datetime(copy["date_received"], format="mixed")
    copy["date_received"] = pd.to_datetime(copy["date_received"]).dt.normalize()
    copy["date_count"] = copy.groupby("date_received")["date_received"].transform(
        "size"
    )
    copy.sort_values(by="date_received", ascending=True, inplace=True)

    st.write("### Time-series plot (Messages per Month/Year")
    fig4 = px.line(
        copy,
        x="date_received",
        y="date_count",
        labels={"x": "Year", "y": "Number of Messages"},
        color_discrete_sequence=["#636EFA"],
    )
    st.plotly_chart(fig4, use_container_width=True)

    # Convert values in 'date_received' column to datetime datatype
    eda_df["date_received"] = pd.to_datetime(eda_df["date_received"], format="mixed")
    # Extract year from 'date_received' column
    eda_df["year"] = eda_df["date_received"].dt.year
    # Extract name of day from 'date_received' column
    eda_df["day_name"] = eda_df["date_received"].dt.day_name()
    # Extract hour of day from 'date_received' column
    eda_df["hour_of_day"] = eda_df["date_received"].dt.hour
    # Extract name of month from 'date_received' column
    eda_df["month_name"] = eda_df["date_received"].dt.month_name()
    # Check whether day in 'date_received' column falls on a weekend
    eda_df["is_weekend"] = eda_df["date_received"].dt.dayofweek.isin([5, 6]).astype(int)

    # Create a dataframe with the extracted features
    datetime_features_df = eda_df[
        ["date_received", "year", "day_name", "hour_of_day", "month_name", "is_weekend"]
    ]
    year_counts = eda_df["year"].value_counts().sort_index()

    st.write("### Distribution of Messages by Year")
    fig5 = px.bar(
        x=year_counts.index,
        y=year_counts.values,
        labels={"x": "Year", "y": "Number of Messages"},
        color=year_counts.values,
        color_continuous_scale=px.colors.sequential.Viridis,
    )
    st.plotly_chart(fig5, use_container_width=True)

    day_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    day_name_counts = eda_df["day_name"].value_counts().reindex(day_order).fillna(0)

    st.write("### Distribution of Messages by Day of the Week")
    fig6 = px.line(
        x=day_name_counts.index,
        y=day_name_counts.values,
        labels={"x": "Day of the Week", "y": "Number of Messages"},
        markers=True,
        line_shape="linear",
        color_discrete_sequence=["#636EFA"],
    )
    st.plotly_chart(fig6, use_container_width=True)

    hour_of_day_counts = eda_df["hour_of_day"].value_counts().sort_index()
    st.write("### Distribution of Messages by Hour of the Day")
    fig7 = px.bar(
        hour_of_day_counts,
        x=hour_of_day_counts.index,
        y=hour_of_day_counts.values,
        labels={"x": "Hour of the Day (0-23)", "y": "Number of Messages"},
        color=hour_of_day_counts.values,
        color_continuous_scale=px.colors.sequential.Viridis,
    )
    st.plotly_chart(fig7, use_container_width=True)

    month_order = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]

    month_name_counts = (
        eda_df["month_name"].value_counts().reindex(month_order).fillna(0)
    )

    st.write("### Distribution of Messages by Month")
    fig8 = px.bar(
        month_name_counts,
        x=month_name_counts.values,
        y=month_name_counts.index,
        orientation="h",
        labels={"x": "Number of Messages", "y": "Month"},
        color=month_name_counts.values,
        color_continuous_scale=px.colors.sequential.Viridis,
    )
    st.plotly_chart(fig8, use_container_width=True)

    is_weekend_counts = (
        eda_df["is_weekend"].map({0: "Weekday", 1: "Weekend"}).value_counts()
    )
    weekend_order = ["Weekday", "Weekend"]
    is_weekend_counts = is_weekend_counts.reindex(weekend_order).fillna(0)

    st.write("### Distribution of Messages: Weekday vs. Weekend")
    fig9 = px.pie(
        names=is_weekend_counts.index,
        values=is_weekend_counts.values,
        color_discrete_sequence=px.colors.sequential.Viridis,
    )
    st.plotly_chart(fig9, use_container_width=True)

    st.write("## EDA for second version of the dataset to be used for training")

    ## checking most common spam message values and its respective label
    st.markdown("Checking most common `spam texts` for this version of the dataset")
    spam_df = ml_df[ml_df["label"] == "spam"]
    st.dataframe(spam_df.value_counts(), use_container_width=True)

    st.markdown("Checking most common `ham texts` for this version of the dataset")
    ham_df = ml_df[ml_df["label"] == "ham"]
    st.dataframe(ham_df.value_counts(), use_container_width=True)

    st.markdown("### ☁️ WordCloud of Ham and Spam Messages")
    st.write(
        "NOTE: issue on displaying this properly on streamlit as text are intersecting but this is acceptable"
    )
    ham_text = " ".join(ml_df[ml_df["label"] == "ham"]["text"].astype(str).tolist())
    spam_text = " ".join(ml_df[ml_df["label"] == "spam"]["text"].astype(str).tolist())

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Ham 🍗")
        ham_elements = get_wordcloud_data(ham_text)
        plot_wordcloud(ham_elements, "Ham Messages WordCloud")

    with col2:
        st.subheader("Spam 🥫")
        spam_elements = get_wordcloud_data(spam_text)
        plot_wordcloud(spam_elements, "Spam Messages WordCloud")

with FeatureSelectionTAB:
    st.markdown("""
    We are only considering Bag-of-Words (BOW) using CountVectorizer and Term Frequency-Inverse Document Frequency (TF-IDF) for features purely on the text. 
    :blue-background[The team did not consider length of the words and other potential feature that can be inferred from the text.] 
    
    BOW considers the count of the words, converted as tokens at this point, within the sentence. TF-IDF considers the frequency of the token throughout the whole corpus of documents.
    Below is the implementation done to determine what feature is suited for the project. 
    Based on the heatmap for tfidf, we were able to determine the top terms that have the most reoccurence between documents therefroe it is appropriate that `TF-IDF` will be used as the feature for training pipeline.
    
    Insight will be to make suspections on these top terms for spam: `bonus`, `account`, `araw`;
    Ham messages that are frequent will be government announcements and OTPs therefore words like `access`, `code` and `app` are expected to associated with ham messages.
                """)

    st.markdown("### Top-N Terms Bar Chart for TF-IDF")
    tfidf_vect = TfidfVectorizer(stop_words=list(combined_stopwords), max_features=20)
    ml_df["text"] = ml_df["text"].fillna("")
    X_tfidf = tfidf_vect.fit_transform(ml_df["text"])
    counts = X_tfidf.toarray().sum(axis=0)
    terms = tfidf_vect.get_feature_names_out()

    df_top_terms = pd.DataFrame({"term": terms, "count": counts})

    fig10 = px.bar(
        df_top_terms.sort_values(by="count", ascending=False),
        x="term",
        y="count",
        title="Top Terms by Tfidf Frequency",
        text="count",
        color_discrete_sequence=["#636EFA"],
    )
    fig10.update_traces(textposition="outside")
    st.plotly_chart(fig10, use_container_width=True)

    ham_df1 = ham_df.copy()
    ham_df1["text"] = ham_df1["text"].fillna("")
    X = vectorizer.fit_transform(ham_df1["text"])
    tfidf_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

    st.write("### TF-IDF Heatmap: Top Terms vs Documents for Ham Messages")
    fig11 = px.imshow(
        tfidf_df.T,
        labels=dict(x="Document", y="Term", color="TF-IDF"),
        aspect="auto",
        color_continuous_scale="Viridis",
    )
    st.plotly_chart(fig11, use_container_width=True)

    st.write("### TF-IDF Heatmap: Top Terms vs Documents for Spam Messages")
    spam_df1 = spam_df.copy()
    spam_df1["text"] = spam_df1["text"].fillna("")

    X = vectorizer.fit_transform(spam_df1["text"])
    tfidf_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

    fig12 = px.imshow(
        tfidf_df.T,
        labels=dict(x="Document", y="Term", color="TF-IDF"),
        aspect="auto",
        color_continuous_scale="Viridis",
    )
    st.plotly_chart(fig12, use_container_width=True)

    st.write("### Scatter Plot for top terms between features for selection")
    bow_cv.fit(ml_df["text"].fillna(""))
    X_ham = bow_cv.transform(ham_df["text"].fillna("")).toarray().sum(axis=0)
    X_spam = bow_cv.transform(spam_df["text"].fillna("")).toarray().sum(axis=0)
    label = ["spam" if s > h else "ham" for h, s in zip(X_ham, X_spam)]

    X_counts = bow_cv.fit_transform(ml_df["text"]).toarray().sum(axis=0)
    X_tfidf = tfidf.fit_transform(ml_df["text"]).toarray().sum(axis=0)

    df_terms = pd.DataFrame(
        {
            "term": bow_cv.get_feature_names_out(),
            "count": X_counts,
            "tfidf": X_tfidf,
            "label": label,
        }
    )

    fig13 = px.scatter(
        df_terms,
        x="count",
        y="tfidf",
        text="term",
        title="TF-IDF vs Frequency",
        hover_data=["label"],
    )
    fig13.update_layout(
        plot_bgcolor="#111111",  # dark gray background
        paper_bgcolor="#111111",
        font_color="white",  # makes text visible
        title_font=dict(size=20, color="white"),
        xaxis=dict(color="white", gridcolor="gray"),
        yaxis=dict(color="white", gridcolor="gray"),
    )
    fig13.update_traces(
        marker=dict(size=8, color="deepskyblue", line=dict(width=1, color="white")),
        textposition="top center",
        textfont=dict(color="white", size=12),
    )
    st.plotly_chart(fig13, use_container_width=True)

with TrainingPipelineTAB:
    st.markdown("### Training Pipeline")
    st.markdown("""
    The project will consider a train-val-test split for a cross-validation (cv) training with hyperparameter tuning considered per fold-run.

    The group will consider :blue-background[our (4) traditional and explainable classifiers] that are known to be used for spam detection. These are to be the two variants of `Naive-Bayes (NB)`, multinomial and complement (noted to handle class imbalances well), `Support Vector Machine`, and `RandomForest`.

    The project utilized `MLflow` to track training and artificats (evaluation metrics) per general run when the model is called; We have put this all under a function.""")

    split_code = """
    
    ##define train-test vals
    X = ml_df['text']
    y = ml_df['label']
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    ##splitting
    X_temp, X_test, y_temp, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)

    ## will set split for validation to be also 0.2
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.3, random_state=42)
    """
    st.markdown("## Define Train-Val-Test Split")
    st.code(split_code, language="python")

    helper_code = """
    
    # I want to save also a confusion matrix in a image format to be displayed and checked
    def plot_and_log_cf_matrix(y_true, y_pred, labels, model_type, run_id):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title(f'Confusion Matrix from run: {model_type}_{run_id}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')

        # Saving the temp file here
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            plt.savefig(temp_file.name)
        img_path = f'confusion_matrix_{model_type}_{run_id}.png'
        plt.savefig(img_path)
        plt.close()
        mlflow.log_artifact(img_path)
        os.remove(img_path) ## partial cache, will not be saved in drive

    # I want to save the cross-val performance in mlflow too
    def plot_cv_perf(cv_results,run_id):
        mean_scores = cv_results['mean_test_score']
        std_scores = cv_results['std_test_score']
        plt.figure()
        plt.errorbar(range(len(mean_scores)), mean_scores, yerr=std_scores, fmt='-o')
        plt.title('Cross-Validation Performance per Parameter Set')
        plt.xlabel('Parameter Set Index')
        plt.ylabel('CV Accuracy')
        img_path = f'cv_performance_{run_id}.png'
        plt.savefig(img_path)
        plt.close()
        mlflow.log_artifact(img_path)
        os.remove(img_path) ## partial cache, will not be saved in drive

    # log the eval metrics in mlflow
    def evaluate_and_log_metrics(y_true, y_pred, model_type, val_name, run_id):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        mlflow.log_metric(f'{model_type}_{run_id}_accuracy', acc)
        mlflow.log_metric(f'{model_type}_{run_id}_precision', prec)
        mlflow.log_metric(f'{model_type}_{run_id}_recall', rec)
        mlflow.log_metric(f'{model_type}_{run_id}_f1_score', f1)

        report = classification_report(y_true, y_pred)
        report_path = f'classification_report_{model_type}_{val_name}_{run_id}.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        mlflow.log_artifact(report_path)
        os.remove(report_path)

        return acc, prec, rec, f1
    """

    st.markdown("### Helper Functions for MLflow")
    st.code(helper_code, language="python")

    train_code = """
    
    ## init a function for mlflow tracking
    def train(run_name='default_run', preprocessor='tfidf',model_type='svm', cv_folds=5):
        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
        vectorizer = TfidfVectorizer(ngram_range=(1,2)) if preprocessor == 'tfidf' else CountVectorizer()

        ## IMPORTANT CHANGE THIS TO IMPROVE ACCURACY
        #hyper-parameter tuning through gridsearch CV initialize here; call also model
        if model_type == 'svm':
            model = SVC(probability=True)
            param_grid = {
                'classifier__C': [0.1, 1, 10],
                'classifier__kernel': ['linear', 'rbf']
            }

        elif model_type == 'random_forest':
        model = RandomForestClassifier(class_weight='balanced')
        param_grid = {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 10, 20],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 5, 10]
        }

        elif model_type == 'complementNB':
        model = ComplementNB()
        param_grid = {
            'classifier__alpha': [0.3, 0.5, 1.0, 1.5],
        }

        elif model_type == 'multinomialNB':
        model = MultinomialNB()
        param_grid = {
            'classifier__alpha': [0.3, 0.5, 1.0, 1.5],
        }
        else:
        raise ValueError(f"Invalid model type: {model_type}")

        pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', model)
        ])

        # calling the stratifieed k-fold cv and grid search
        strat_cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=strat_cv,
            scoring='accuracy',
            n_jobs=-1
        )

        ## THIS IS THE TRAINING PART IN FUNCTION
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        ## LOGGING PARAMS IN MLFLOW
        mlflow.log_param("preprocessor", preprocessor)
        mlflow.log_param("model_type", model_type)
        mlflow.log_params(grid_search.best_params_)

        mlflow.log_metric("best_cv_accuracy", grid_search.best_score_)

        plot_cv_perf(grid_search.cv_results_, run_id) ## variables needed for function: cv_results,run_id

        # ! Validation set evaluation !
        val_preds = best_model.predict(X_val)
        evaluate_and_log_metrics(y_val, val_preds, model_type, 'val', run_id) ## variables needed for function: y_true, y_pred, model_type, val_name, run_id
        plot_and_log_cf_matrix(y_val, val_preds, label_encoder.classes_, model_type, run_id) ## variables needed for function: y_true, y_pred, labels, model_type, run_id
        val_acc = accuracy_score(y_val, val_preds)
        val_f1 = f1_score(y_val, val_preds, average='weighted', zero_division=0)


        # ! Test set evaluation !
        test_preds=best_model.predict(X_test)
        evaluate_and_log_metrics(y_test, test_preds, model_type, 'test', run_id) ## variables needed for function: y_true, y_pred, model_type, val_name, run_id
        plot_and_log_cf_matrix(y_test, test_preds, label_encoder.classes_, model_type, run_id)
        test_acc = accuracy_score(y_test, test_preds)
        test_f1 = f1_score(y_test, test_preds, average='weighted', zero_division=0)

        #save best model
        mlflow.sklearn.log_model(best_model, f"best_model for run {run_name}")
        print(f"\n\n--- RUN TEST FOR {run_name} using {preprocessor} and {model_type}; {run_id} ---")
        print(f"Best CV Accuracy: {grid_search.best_score_:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}, F1: {val_f1:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}")
        print(f"Best Parameters: {grid_search.best_params_}\n\n")
    """
    st.markdown("### Training Pipeline Made with MLflow configurations")
    st.code(train_code, language="python")

    sample_run = """
    
    --- RUN TEST FOR multinomialNB_v2 using tfidf and multinomialNB; 747c6ac17b154a9284aad6f69b62e38a ---
    Best CV Accuracy: 0.9268
    Validation Accuracy: 0.9324, F1: 0.9214
    Test Accuracy: 0.9339, F1: 0.9235
    Best Parameters: {'classifier__alpha': 0.3}


    🏃 View run multinomialNB_v2 at: http://localhost:2000/#/experiments/616295389774529782/runs/747c6ac17b154a9284aad6f69b62e38a
    🧪 View experiment at: http://localhost:2000/#/experiments/616295389774529782
    2025/07/28 14:30:00 WARNING mlflow.models.model: `artifact_path` is deprecated. Please use `name` instead.
    2025/07/28 14:30:04 WARNING mlflow.models.model: Model logged without a signature and input example. Please set `input_example` parameter when logging the model to auto infer the model signature.


    --- RUN TEST FOR complementNB_v2 using tfidf and complementNB; 71b18fe5ba9f4dd38d90c3033c298963 ---
    Best CV Accuracy: 0.9541
    Validation Accuracy: 0.9577, F1: 0.9564
    Test Accuracy: 0.9645, F1: 0.9641
    Best Parameters: {'classifier__alpha': 0.3}


    🏃 View run complementNB_v2 at: http://localhost:2000/#/experiments/616295389774529782/runs/71b18fe5ba9f4dd38d90c3033c298963
    🧪 View experiment at: http://localhost:2000/#/experiments/616295389774529782
    2025/07/28 14:33:22 WARNING mlflow.models.model: `artifact_path` is deprecated. Please use `name` instead.
    2025/07/28 14:33:26 WARNING mlflow.models.model: Model logged without a signature and input example. Please set `input_example` parameter when logging the model to auto infer the model signature.


    --- RUN TEST FOR random_forest_v2 using tfidf and random_forest; e2ddf375fb104e8881b25d0cdbd5c2ac ---
    Best CV Accuracy: 0.9522
    Validation Accuracy: 0.9563, F1: 0.9566
    Test Accuracy: 0.9675, F1: 0.9678
    Best Parameters: {'classifier__max_depth': None, 'classifier__min_samples_leaf': 1, 'classifier__min_samples_split': 2, 'classifier__n_estimators': 50}


    🏃 View run random_forest_v2 at: http://localhost:2000/#/experiments/616295389774529782/runs/e2ddf375fb104e8881b25d0cdbd5c2ac
    🧪 View experiment at: http://localhost:2000/#/experiments/616295389774529782
    2025/07/28 14:34:20 WARNING mlflow.models.model: `artifact_path` is deprecated. Please use `name` instead.
    2025/07/28 14:34:25 WARNING mlflow.models.model: Model logged without a signature and input example. Please set `input_example` parameter when logging the model to auto infer the model signature.


    --- RUN TEST FOR svm_v2 using tfidf and svm; 13449e470cb24e6c8839ca883fcf43c5 ---
    Best CV Accuracy: 0.9474
    Validation Accuracy: 0.9606, F1: 0.9579
    Test Accuracy: 0.9665, F1: 0.9646
    Best Parameters: {'classifier__C': 10, 'classifier__kernel': 'linear'}


    🏃 View run svm_v2 at: http://localhost:2000/#/experiments/616295389774529782/runs/13449e470cb24e6c8839ca883fcf43c5
    🧪 View experiment at: http://localhost:2000/#/experiments/616295389774529782
    """

    st.markdown("### Example output of run of cell; Check also sample of mflow ui")
    st.code(sample_run, language="python")
    img1 = Image.open("img/mlflow_ui.png")
    st.image(img1, caption="MLFlow UI", use_container_width=True)


with ModelEvaluationTAB:
    st.markdown("## Summary of best model configuration and model metrics")
    st.markdown("""
    The model training above already provides how the model metrics are extracted; All evaluation metrics and visualization are saved as artificats under `mlflow`. 
    
    With this, presented below is the training summary done under initial run parameters for `preprocessor=tfidf` and `cv_folds=5` for all models considered of the study.

    Models considered are the following: `complement_NB (cNB)`, `multinomial_NB (mNB)`, `random_forest (rf)`, `support_vector_machine (svm)`
                
    REVISION (August 4, 2025): Upon advise, we have done a oversampling of the ham classes. Improvements on `precision` and `recall` is observed in 3rd Run.""")

    st.markdown("""
                
    ### Conclusion: 
    - The best model that performed well are `rf` and `svm` classifiers with :blue-background[val_accuracy of 0.95 for both models and test_accuracy of 0.97 for both models too!]. 
    - Assumption for the high accuracy is that these models covered many hyperparameters during training and therefore it was able to find the optimal hyperparameters for the accuracy.
    - Another key metric being checked is the precision and recall for the `ham` class. Due to the dataset having a class imbalance, its performance on recall is vital to determine how it deals with true positives and its classification for the minority class which is the ham messages.
    -This is bad practice but given the scarcity and inability to do sampling techniques properly to combat the imbalance (like SMOTE), we are accepting precision and recall that are similar of value to each other.
    - Based on these insights :blue-background[both rf and svm models] are the best models for this run as they were able to achieve _acceptable values_ for both validation and test evaluations.
                """)

    st.markdown("---")
    st.markdown("## 1st Run (July 28, 2025)")

    st.markdown(
        """This run used the default `ngram_range` for the TF-IDF vectorizer, which is `(1, 1)`. This means that only single-word tokens `(unigrams)` were considered during feature extraction."""
    )

    st.markdown(""" 
               ### 1st_Run Validation Accuracy

    | models | val_accuracy | precision_0 | recall_0 | f1-score | support | hyper_params | tfidf_ngram_range |
    |---|---|---|---|---|---|---|---|
    | cNB | 0.94 | 0.79 | 0.73 | 0.76 | {0:90,1:620} | 'classifier__alpha': [0.3, 0.5, 1.0, 1.5] | (1,1) |
    | mNB | 0.93 | 0.98 | 0.44 | 0.66 | {0:90,1:620} | 'classifier__alpha': [0.3, 0.5, 1.0, 1.5] | (1,1) |
    | rf | 0.95 | 0.85 | 0.79 | 0.82 | {0:90,1:620} | 'classifier__n_estimators':[50, 100, 200]<br>'classifier__max_depth':[None, 10, 20]<br>'classifier__min_samples_split':[1,2,5,10] | (1,1) |
    | svm | 0.95 | 0.89 | 0.72 | 0.80 | {0:90,1:620} | 'classifier__C': [0.1, 1, 10]<br>'classifier__kernel': ['linear','rbf'] | (1,1) |               
                               """)
    st.markdown("""
                ### 1st_Run Test Accuracy

    | models | test_accuracy | precision_0 | recall_0 | f1-score_0 | support | hyper_params | tfidf_ngram_range |
    |---|---|---|---|---|---|---|---|
    | cNB | 0.96 | 0.84 | 0.80 | 0.82 | {0:125,1:889} | 'classifier__alpha': [0.3, 0.5, 1.0, 1.5] | (1,1) |
    | mNB | 0.93 | 0.98 | 0.48 | 0.65 | {0:125,1:889} | 'classifier__alpha': [0.3, 0.5, 1.0, 1.5] | (1,1) |
    | rf | 0.97 | 0.92 | 0.85 | 0.88 | {0:125,1:889} | 'classifier__n_estimators':[50, 100, 200]<br>'classifier__max_depth':[None, 10, 20]<br>'classifier__min_samples_split':[1,2,5,10] | (1,1) |
    | svm | 0.97 | 0.90 | 0.81 | 0.85 | {0:125,1:889} | 'classifier__C': [0.1, 1, 10]<br>'classifier__kernel': ['linear','rbf'] | (1,1) |
                """)

    st.markdown("#### SVM Performance Metrics")
    img2 = Image.open(
        "img/1_run/svm/confusion_matrix_svm_cd28b67a16254e80b055bf50bab61d3d.png"
    )
    img3 = Image.open(
        "img/1_run/svm/cv_performance_svm_cd28b67a16254e80b055bf50bab61d3d.png"
    )
    col1, col2 = st.columns(2)
    with col1:
        st.image(img2, caption="Confusion Matrix", use_container_width=True)
    with col2:
        st.image(img3, caption="CV Performance", use_container_width=True)

    st.markdown("#### RF Performance Metrics")
    img4 = Image.open(
        "img/1_run/rf/confusion_matrix_random_forest_5d8f4c16e8aa4120b3eede32f17bbf78.png"
    )
    img5 = Image.open(
        "img/1_run/rf/cv_performance_5d8f4c16e8aa4120b3eede32f17bbf78.png"
    )
    col1, col2 = st.columns(2)
    with col1:
        st.image(img4, caption="Confusion Matrix", use_container_width=True)
    with col2:
        st.image(img5, caption="CV Performance", use_container_width=True)

    st.markdown("#### Multinomial Naive Bayes Performance Metrics")
    img6 = Image.open(
        "img/1_run/mNB/confusion_matrix_multinomialNB_981ba728e07048f48cf7ab2f9c9cc559.png"
    )
    img7 = Image.open(
        "img/1_run/mNB/cv_performance_981ba728e07048f48cf7ab2f9c9cc559.png"
    )
    col1, col2 = st.columns(2)
    with col1:
        st.image(img6, caption="Confusion Matrix", use_container_width=True)
    with col2:
        st.image(img7, caption="CV Performance", use_container_width=True)

    st.markdown("#### Complement Naive Bayes Performance Metrics")
    img8 = Image.open(
        "img/1_run/cNB/confusion_matrix_complementNB_af86bad539074f32bc26a9410173d98a.png"
    )
    img9 = Image.open(
        "img/1_run/cNB/cv_performance_af86bad539074f32bc26a9410173d98a.png"
    )
    col1, col2 = st.columns(2)
    with col1:
        st.image(img8, caption="Confusion Matrix", use_container_width=True)
    with col2:
        st.image(img9, caption="CV Performance", use_container_width=True)

    st.markdown("---")
    st.write("## 2nd Run (July 28, 2025)")
    st.markdown("""
                For this second run, the  `ngram_range` for the TF-IDF vectorizer was changed, which is now `(1, 2)`. 
                This means that unigrams from previous run and two-word tokens `(bigrams)` were considered during feature extraction.
                """)

    st.markdown("""
                ### 2nd_Run Validation Accuracy

    | models | val_accuracy | precision_0 | recall_0 | f1-score_0 | support | hyper_params | tfidf_ngram_range |
    |---|---|---|---|---|---|---|---|
    | cNB | 0.96 | 0.88 | 0.77 | 0.82 | {0:90,1:620} | 'classifier__alpha': [0.3, 0.5, 1.0, 1.5] | (1,2) |
    | mNB | 0.93 | 1.00 | 0.47 | 0.64 | {0:90,1:620} | 'classifier__alpha': [0.3, 0.5, 1.0, 1.5] | (1,2) |
    | rf | 0.95 | 0.85 | 0.79 | 0.82 | {0:90,1:620} | 'classifier__n_estimators':[50, 100, 200]<br>'classifier__max_depth':[None, 10, 20]<br>'classifier__min_samples_split':[1,2,5,10] | (1,2) |
    | svm | 0.95 | 0.89 | 0.72 | 0.80 | {0:90,1:620} | 'classifier__C': [0.1, 1, 10]<br>'classifier__kernel': ['linear','rbf'] | (1,2) |
                """)
    st.markdown("""
                ### 2nd_Run Test Accuracy

    | models | val_accuracy | precision_0 | recall_0 | f1-score_0 | support | hyper_params | tfidf_ngram_range |
    |---|---|---|---|---|---|---|---|
    | cNB | 0.96 | 0.87 | 0.83 | 0.85 | {0:125,1:889} | 'classifier__alpha': [0.3, 0.5, 1.0, 1.5] | (1,2) |
    | mNB | 0.96 | 0.87 | 0.83 | 0.85 | {0:125,1:889} | 'classifier__alpha': [0.3, 0.5, 1.0, 1.5] | (1,2) |
    | rf | 0.97 | 0.90 | 0.87 | 0.89 | {0:125,1:889} | 'classifier__n_estimators':[50, 100, 200]<br>'classifier__max_depth':[None, 10, 20]<br>'classifier__min_samples_split':[1,2,5,10] | (1,2) |
    | svm | 0.97 | 0.90 | 0.81 | 0.85 | {0:125,1:889} | 'classifier__C': [0.1, 1, 10]<br>'classifier__kernel': ['linear','rbf'] | (1,2) |                
                """)
    st.markdown("#### SVM Performance Metrics")
    img10 = Image.open(
        "img/2_run/svm/confusion_matrix_svm_23523aa756d14d508efc9d7460aa2496.png"
    )
    img11 = Image.open(
        "img/2_run/svm/cv_performance_23523aa756d14d508efc9d7460aa2496.png"
    )
    col1, col2 = st.columns(2)
    with col1:
        st.image(img10, caption="Confusion Matrix", use_container_width=True)
    with col2:
        st.image(img11, caption="CV Performance", use_container_width=True)

    st.markdown("#### RF Performance Metrics")
    img13 = Image.open(
        "img/2_run/rf/confusion_matrix_random_forest_83f0e0d7df324e37908e3c80a008e51b.png"
    )
    img14 = Image.open(
        "img/2_run/rf/cv_performance_83f0e0d7df324e37908e3c80a008e51b.png"
    )
    col1, col2 = st.columns(2)
    with col1:
        st.image(img13, caption="Confusion Matrix", use_container_width=True)
    with col2:
        st.image(img14, caption="CV Performance", use_container_width=True)

    st.markdown("#### Multinomial Naive Bayes Performance Metrics")
    img15 = Image.open(
        "img/2_run/mNB/confusion_matrix_multinomialNB_0bd6e47a7ec541b590db14519fded6aa.png"
    )
    img16 = Image.open(
        "img/2_run/mNB/cv_performance_0bd6e47a7ec541b590db14519fded6aa.png"
    )
    col1, col2 = st.columns(2)
    with col1:
        st.image(img15, caption="Confusion Matrix", use_container_width=True)
    with col2:
        st.image(img16, caption="CV Performance", use_container_width=True)

    st.markdown("#### Complement Naive Bayes Performance Metrics")
    img17 = Image.open(
        "img/2_run/cNB/confusion_matrix_complementNB_f99dedcca3404ec483a07fe2773c9194.png"
    )
    img18 = Image.open(
        "img/2_run/cNB/cv_performance_f99dedcca3404ec483a07fe2773c9194.png"
    )
    col1, col2 = st.columns(2)
    with col1:
        st.image(img17, caption="Confusion Matrix", use_container_width=True)
    with col2:
        st.image(img18, caption="CV Performance", use_container_width=True)

    st.markdown("## 3rd Run (August 4, 2025)")
    st.markdown("""
                For this third run, we address the class imbalance by oversampling of the ham class using the `imbalance-learn` package with the aim that ham class is twice that of spam during training in pipeline. 
                Improvement on `precision` and `recall` is observed overall. The performance metrics are shown below with `random forest` and `complement NB` considered the best models.
                """)

    st.markdown("""### 3rd_Run Validation Accuracy""")
    st.markdown("""
    | models | val_accuracy | precision_0 | recall_0 | f1-score_0 | support | hyper_params | tfidf_range |
    |---|---|---|---|---|---|---|---|
    | cNB | 0.96 | 0.86 | 0.84 | 0.85 | {0:95,1:615} | 'classifier__alpha': [0.3, 0.5, 1.0, 1.5] | (1,2) + X_train = {2(ham):spam} |
    | mNB | 0.95 | 0.75 | 0.91 | 0.83 | {0:95,1:615} | 'classifier__alpha': [0.3, 0.5, 1.0, 1.5] | (1,2) + X_train = {2(ham):spam} |
    | rf | 0.97 | 0.79 | 0.86 | 0.82 | {0:95,1:615} | 'classifier__n_estimators':[50, 100, 200]<br>'classifier__max_depth':[None, 10, 20]<br>'classifier__min_samples_split':[1,2,5,10] | (1,2) + X_train = {2(ham):spam} |
    | svm | 0.96 | 0.96 | 0.72 | 0.82 | {0:95,1:615} | 'classifier__C': [0.1, 1, 10]<br>'classifier__kernel': ['linear','rbf'] | (1,2) + X_train = {2(ham):spam} |            
                
                """)

    st.markdown("""
                
    ### 3rd_Run Test Accuracy

    | models | test_accuracy | precision_0 | recall_0 | f1-score_0 | support | hyper_params | tfidf_range |
    |---|---|---|---|---|---|---|---|
    | cNB | 0.89 | 0.89 | 0.86 | 0.87 | {0:125,1:889} | 'classifier__alpha': [0.3, 0.5, 1.0, 1.5] | (1,2) + X_train = {2(ham):spam} |
    | mNB | 0.95 | 0.76 | 0.91 | 0.83 | {0:125,1:889} | 'classifier__alpha': [0.3, 0.5, 1.0, 1.5] | (1,2) + X_train = {2(ham):spam} |
    | rf | 0.96 | 0.80 | 0.92 | 0.86 | {0:125,1:889} | 'classifier__n_estimators':[50, 100, 200]<br>'classifier__max_depth':[None, 10, 20]<br>'classifier__min_samples_split':[1,2,5,10] | (1,2) + X_train = {2(ham):spam} |
    | svm | 0.96 | 0.96 | 0.71 | 0.82 | {0:125,1:889} | 'classifier__C': [0.1, 1, 10]<br>'classifier__kernel': ['linear','rbf'] | (1,2) + X_train = {2(ham):spam} |                
                """)
    st.markdown("#### SVM Performance Metrics")
    img19 = Image.open(
        "img/3_run/svm/confusion_matrix_svm_05edf44674ee4ffeb25b1284d0a08e83.png"
    )
    img20 = Image.open(
        "img/3_run/svm/cv_performance_05edf44674ee4ffeb25b1284d0a08e83.png"
    )
    col1, col2 = st.columns(2)
    with col1:
        st.image(img19, caption="Confusion Matrix", use_container_width=True)
    with col2:
        st.image(img20, caption="CV Performance", use_container_width=True)

    st.markdown("#### RF Performance Metrics")
    img21 = Image.open(
        "img/3_run/rf/confusion_matrix_random_forest_498f3ef34e954cdeb074bce4766180af.png"
    )
    img22 = Image.open(
        "img/3_run/rf/cv_performance_498f3ef34e954cdeb074bce4766180af.png"
    )
    col1, col2 = st.columns(2)
    with col1:
        st.image(img21, caption="Confusion Matrix", use_container_width=True)
    with col2:
        st.image(img22, caption="CV Performance", use_container_width=True)

    st.markdown("#### Multinomial Naive Bayes Performance Metrics")
    img23 = Image.open(
        "img/3_run/mNB/confusion_matrix_multinomialNB_5c755dd20b2c44aa92dff382a9a9073f.png"
    )
    img24 = Image.open(
        "img/3_run/mNB/cv_performance_5c755dd20b2c44aa92dff382a9a9073f.png"
    )
    col1, col2 = st.columns(2)
    with col1:
        st.image(img23, caption="Confusion Matrix", use_container_width=True)
    with col2:
        st.image(img24, caption="CV Performance", use_container_width=True)

    st.markdown("#### Complement Naive Bayes Performance Metrics")
    img25 = Image.open(
        "img/3_run/cNB/confusion_matrix_complementNB_9497e0a34f76466b8c6ab91aa4ec433e.png"
    )
    img26 = Image.open(
        "img/3_run/cNB/cv_performance_9497e0a34f76466b8c6ab91aa4ec433e.png"
    )
    col1, col2 = st.columns(2)
    with col1:
        st.image(img25, caption="Confusion Matrix", use_container_width=True)
    with col2:
        st.image(img26, caption="CV Performance", use_container_width=True)

with ConTAB:
    st.markdown("""
                ## Conclusion

    The project was able to achieve spam classifiers specific to the filipino-context using three datasets that can be a direct tool for assessement for the SIM registration act of 2022.

    This current project was able to do the following unique implementations that stood among the other public projects / references we have checked to localize spam classification:
    - additional EDA insights for the state of the filipino-context messages like plotly graphs and tfidf heatmaps to determine which words are most common in spam and ham messages
    - train-val-test cross-validation training with hyperparameter tuning directly using the `mlflow` package
    - Considers traditional machine learning classifiers of the two NB variants, SVM, and RF.
    - demo app available in `HuggingFace Space` for further collaboration and feedback to target stakeholders and SIM users. """)

    st.markdown("---")
    st.markdown("""
        ## Recommendations

    Despite the contributions made above, the project can be further improved on these following aspects:
    - There is a class imbalance by 3x spam than ham class due to sources. Additional data sources to make spam a minority class will significantly improve evaluation metrics particularly on the precision for positive classifications as positive and recall that undermines the true negatives
    - The use of advanced NLP techniques that consider the whole context of the sentence like BERT embeddings;
    - Tuning the hyperparameters further to greatly scope the potential improvement on evaluation metrics
    - The use of deep learning models and XAI techniques to improve accuracy and transparency or even fine-tune models like DOST-ASTI's roBERTa sentiment analysis that can be used for classification problem
    - conversion of XML data directly from extraction into a data visualization with the use of trained classifiers to map out spam/ham in a timeseries plot. (was hoping to do that! will do it talaga when I have time! -Ferds)
                    """)

    st.markdown("""
                ### References: Kindly check them below!
                
    To support future researchers and practitioners exploring spam classification in the Filipino context, we have curated the following references. These resources were instrumental in shaping our approach and may serve as valuable starting points for continued efforts to counter spammers and scammers in the country!      
    
    [Filipino-Spam-SMS-Detection-Model](https://github.com/Yissuh/Filipino-Spam-SMS-Detection-Model/tree/main)
    - A text mining project implemented for Data Analytics course as a third year student in Technological University of the Philippines - Manila. The project focuses on developing a localized spam SMS detection in the Philippines.

    [SIM Spam Slam](https://ktrin-u.github.io/CS132-Arbitrary/) 
    - A simple website done by UP Students also did a scam classification on chosen datasets; consider what we are doing as a replication study of what they made with the dataset updated (kudos for the maintainers); We werent able to do the clustering and other methods they have done.

    [Philippine-Scam-Analysis-ML](https://github.com/HiroshiJoe/Philippine-Scam-Analysis-ML/blob/main/PH_Scam_Analysis_and_Machine_Learning.ipynb)
    - This repository includes analysis and models for distinguishing "Scam" from "Non-scam" messages. 
    The data, originally extracted from images via OCR for the ScamGuard project, is used here to apply 
    sentiment analysis and machine learning techniques to improve scam detection.

    [A Machine Learning Approach for Efficient Spam Detection in Short Messaging System (SMS)](https://ieeexplore.ieee.org/document/10322491)

    - A conference paper done by PUP Batangas students. Based on their abstract, Short Message Service (SMS) 
    is widely used due to its affordability and convenience, but spam messages pose serious risks such as fraud, 
    identity theft, and financial loss. To address this, machine learning models were applied to classify and filter 
    spam from legitimate messages. The dataset consisted of 60% ham and 40% spam, combining internet-sourced and 
    self-collected samples. Among the models tested, Bernoulli Naive Bayes achieved the best results with 96.63% accuracy after optimization.         
                """)
