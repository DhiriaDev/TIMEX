import timex as tx

# data,stato,ricoverati_con_sintomi,terapia_intensiva,totale_ospedalizzati,isolamento_domiciliare,totale_positivi,variazione_totale_positivi,nuovi_positivi,dimessi_guariti,deceduti,casi_da_sospetto_diagnostico,casi_da_screening,totale_casi,tamponi,casi_testati,note

# PARAMETERS
param_file_nameJSON = 'configuration_test.json'

# data ingestion
print('-> INGESTION')
df, param_config = tx.data_ingestion(param_file_nameJSON)   # ingestion of data

print('-> DESCRIPTION')
tx.data_description(df, param_config, verbose='yes')

print('-> SELECTION')
df = tx.data_selection(df, param_config)

print('-> ADD DIFF COLUMN')
df = tx.add_diff_column(df, df.columns[1], 'incremento_column')

print('-> DESCRIPTION')
tx.data_description(df, param_config, verbose='yes')

print('-> DESCRIPTION')
tx.data_frame_visualization(df, param_config, visualization_type="hist", verbose="yes")
tx.data_frame_visualization(df, param_config, visualization_type="line", mode="stacked", verbose="yes")