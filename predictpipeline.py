import apache_beam as beam
import logging
import joblib
import numpy as np
import pandas as pd
import argparse
import re
from google.cloud import storage
from apache_beam.options.pipeline_options import StandardOptions, GoogleCloudOptions, SetupOptions, PipelineOptions
from past.builtins import unicode

# Build and run the pipeline
def run(argv=None):
    pipeline_options = PipelineOptions(flags=argv)
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--output_topic',
      required=True,
      help=(
          'Output PubSub topic of the form '
          '"projects/<PROJECT>/topics/<TOPIC>".'))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
      '--input_topic',
      help=(
          'Input PubSub topic of the form '
          '"projects/<PROJECT>/topics/<TOPIC>".'))
    group.add_argument(
      '--input_subscription',
      help=(
          'Input PubSub subscription of the form '
          '"projects/<PROJECT>/subscriptions/<SUBSCRIPTION>."'))
    known_args, pipeline_args = parser.parse_known_args(argv)
    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(StandardOptions).streaming = True

    google_cloud_options = pipeline_options.view_as(GoogleCloudOptions)
    google_cloud_options.project = 'quickstart-1569373998669'
    google_cloud_options.job_name = 'alexa-prediction'
    google_cloud_options.staging_location = 'gs://p2_bucket/staging/'
    google_cloud_options.temp_location = 'gs://p2_bucket/temp/'
    pipeline_options.view_as(StandardOptions).runner = 'DataflowRunner'
    pipeline_options.view_as(SetupOptions).save_main_session = True
    pipeline_options.view_as(SetupOptions).setup_file = "./setup.py"
    logging.info("Pipeline arguments: {}".format(pipeline_options))

    p = beam.Pipeline(options=pipeline_options)
    (p
    | "Read data from PubSub" >> beam.io.ReadFromPubSub(subscription=known_args.input_subscription).with_output_types(bytes)
    | 'decode' >> beam.Map(lambda x: x.decode('utf-8','ignore'))
    | "predicting" >> beam.ParDo(PredictSklearn(project='egenproject', bucket_name='p2_bucket', model_path='amazon_alexa.csv', destination_name='amazon_alexa.csv')).with_output_types(str)
    #| 'encode' >> beam.Map(lambda x: x.encode('utf-8','ignore')).with_output_types(bytes)
    | "Write to PubSub" >> beam.io.WriteStringsToPubSub(known_args.output_topic))

    result = p.run()
    result.wait_until_finish()

# Function to download model from bucket
def download_blob(bucket_name=None, source_blob_name=None, project=None, destination_file_name=None):
    storage_client = storage.Client(project)
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

class PredictSklearn(beam.DoFn):

    def __init__(self, project=None, bucket_name=None, model_path=None, destination_name=None):
        self._model = None
        self._project = project
        self._bucket_name = bucket_name
        self._model_path = model_path
        self._destination_name = destination_name

    def download_blob(self, bucket_name=None, source_blob_name=None, project=None, destination_file_name=None):
        from google.cloud import storage
        storage_client = storage.Client(project)
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)

    def setup(self):
        """Download sklearn model from GCS"""
        logging.info(
            "Sklearn model initialisation {}".format(self._model_path))
        self.download_blob(bucket_name=self._bucket_name, source_blob_name=self._model_path,
                      project=self._project, destination_file_name=self._destination_name)

    def process(self, element):
        import re
        import nltk
        nltk.download('stopwords')
        from nltk.corpus import stopwords
        data = pd.read_csv(r"amazon_alexa.csv", encoding = "ISO-8859-1")
        reviews = data["verified_reviews"].tolist()
        labels = data["feedback"].values
        processed_reviews = []
        def preprocess(text):
            text = re.sub(r'\W', ' ', str(text))
            text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
            text = re.sub(r'\^[a-zA-Z]\s+', ' ', text)
            text = re.sub(r'\s+', ' ', text, flags=re.I)
            text = re.sub(r'^b\s+', '', text)
            return text

        for text in reviews:
            processed_reviews.append(preprocess(text))

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(processed_reviews, labels, test_size=0.2, random_state=0)

        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(max_features=2000, min_df=5, max_df=0.75, stop_words=stopwords.words('english'))
        X_train1 = vectorizer.fit_transform(X_train).toarray()
        X_test1 = vectorizer.transform(X_test).toarray()

        from sklearn.ensemble import RandomForestClassifier
        rfc = RandomForestClassifier(n_estimators=200, random_state=42)
        rfc.fit(X_train1, y_train)
        processedelement=list(preprocess(element.encode('ISO-8859-1','ignore')))
        processedelement=vectorizer.transform(processedelement)
        y_pred = rfc.predict(processedelement)
        return str(y_pred)

# log the output
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run()