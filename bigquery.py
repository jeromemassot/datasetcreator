from google.cloud import bigquery
from google.cloud import storage
import uuid
import re

def clean_caption(text:str) -> str:
    rule1 = r'[Ff][Ii][Gg](?:[Uu][Rr][Ee])?\s*\d*(?:[,-:.])?\s*\d*(?:[,-:.])?\d*\s*(?:[,-:.])?\d*\s*'
    rule2 = r'[Tt][Aa][Bb](?:[Ll][Ee])?\s*\d*(?:[,-:.])?\s*\d*(?:[,-:.])?\d*\s*(?:[,-:.])?\d*\s*'
    text = re.sub(rule1, '', text)
    text = re.sub(rule2, '', text)
    text = text.replace('"', '')
    return text


def create_url(text:str) -> str:
    text = text.replace(
        "/content/drive/MyDrive/26- Colab Notebooks/VISION/Documents Segmentation/datasets/books/petrophysics/figures/",
        "gs://petrophysics_figures/"
    )
    return text


# first the index needed to be modified to be used as
# unique identifier in BigQuery table
initial_uri = "gs://petrophysics_figures/figs_captions.csv"
modified_uri = "gs://petrophysics_figures/figs_captions_bq.csv"

storage_client = storage.Client()
bucket = storage_client.bucket('petrophysics_figures')
blob = bucket.blob('figs_captions.csv')
bq_blob = bucket.blob('figs_captions_bq.csv')

with blob.open("r") as f:
    modified_str = ''
    for i, line in enumerate(f.readlines()):
        if i==0:
            modified_str += 'id|url|category|coords|caption|tags|origin|document|page_index|status|backup\n'
        else:
            components = line.split('|')
            components[9] = create_url(components[9])
            figure_id = str(uuid.uuid5(uuid.NAMESPACE_X500 , components[9]))
            figure_url = components[9].replace('\n', '')
            figure_category = 'None'

            figure_caption = components[8]
            figure_caption = clean_caption(figure_caption)

            figure_tags = 'None'
            figure_status = 'Not Validated'
            figure_backup = 'None'

            origin = 'petrophysics_figures'
            document = components[1]
            page_index = components[10]

            coords = '|'.join(components[4:8])

            bq_schema_data = [
                figure_id, figure_url, figure_category, coords, figure_caption, 
                figure_tags, origin, document, page_index, figure_status, figure_backup
            ]

            new_line = "|".join(bq_schema_data)
            modified_str += new_line + '\n'

with bq_blob.open("w") as f:
    f.write(modified_str)

# construct a BigQuery client object.
bq_client = bigquery.Client()

# set table_id to the ID of the table to create.
table_id = "petroglyphs-nlp.geosciences_ai_datasets.geosciences-captioned-figures"

job_config = bigquery.LoadJobConfig(
    schema=[
        bigquery.SchemaField("id", "STRING", "REQUIRED"),
        bigquery.SchemaField("url", "STRING", "REQUIRED"),
        bigquery.SchemaField("category", "STRING", "NULLABLE"),
        bigquery.SchemaField("coords", "STRING", "NULLABLE"),
        bigquery.SchemaField("caption", "STRING", "NULLABLE"),
        bigquery.SchemaField("tags", "STRING", "NULLABLE"),
        bigquery.SchemaField("origin", "STRING", "REQUIRED"),
        bigquery.SchemaField("document", "STRING", "REQUIRED"),
        bigquery.SchemaField("page_index", "STRING", "REQUIRED"),
        bigquery.SchemaField("status", "STRING", "NULLABLE"), 
        bigquery.SchemaField("backup_url", "STRING", "NULLABLE")
    ],
    skip_leading_rows=1,
    field_delimiter="|",
    source_format=bigquery.SourceFormat.CSV,
    write_disposition=bigquery.WriteDisposition.WRITE_APPEND
)

# create the bigquery job
load_job = bq_client.load_table_from_uri(
    modified_uri, table_id, job_config=job_config
)

# waits for the job to complete
load_job.result()

# make an API request
destination_table = bq_client.get_table(table_id)
print("Loaded {} rows.".format(destination_table.num_rows))