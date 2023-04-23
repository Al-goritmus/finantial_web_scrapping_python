
"""# Get files in folder"""
import os
import pandas as pd
import numpy as np
import uuid
import matplotlib.pyplot     as plt
import matplotlib.patches    as mpatches
import seaborn               as sns
import sklearn.metrics       as Metrics
import json
import sweetviz as sv


path_dir = '/mnt/f/PythonFinanceProjects/'


def conditions_for_reading(filename):
    valid_extensions = ['.xlsx', '.csv', '.txt', '.sql']
    return any(filename.lower().endswith(ext) for ext in valid_extensions)


files_name = [os.path.join(file) for file in os.listdir(path_dir) if conditions_for_reading(file)]

# Test
for file_name in files_name:
    with open(file_name, encoding='latin-1') as f:
        print(f.read())



"""# Functions


"""

def normalize_str_categorical(df_serie,func_type='upper'):
  if func_type == 'upper':
    return df_serie.str.upper().str.strip()
  elif func_type == 'lower':
    return df_serie.str.lower().str.strip()
def remove_accents_cols(df_cols):
    return df_cols.str.replace('ñ','ni').str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')

def remove_special_chars(df_cols):
    return df_cols.str.replace(r'[$@&/.:-]',' ', regex=True)
def regular_camel_case(snake_str):
    components = snake_str.split('_')
    return components[0] + ''.join(x.title() for x in components[1:])
def regular_snake_case(df_cols):
    cols = df_cols.str.replace('ñ','ni')
    cols = cols.str.lower().str.replace('/',' ').str.replace('.',' ').str.strip()
    cols = cols.str.replace(r'\s+',' ',regex=True)
    cols = cols.str.replace(' ','_')
    return cols

def create_markdown_cell(src_list):

  cell = {
      "cell_type": "markdown",
      "outputs": [],
      "execution_count":'null',
      "source": src_list,
      "metadata": {
        "id": f"{uuid.uuid4()}"
      }
  }

  return cell

def create_code_cells(src_list):
  
  cell = {"cell_type": "code",
      "execution_count": 'null',
      "metadata": {
        "id": f"{uuid.uuid4()}"
      },
      "outputs": [],
      "source":src_list}

  return cell

def create_notebook_text(cells):

  json_cells = json.dumps(cells)

  str_notebook = f'''{{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {{
    "colab": {{
      "provenance": []
    }},
    "kernelspec": {{
      "name": "python3",
      "display_name": "Python 3"
    }},
    "language_info": {{
      "name": "python"
    }}
  }},
  "cells": {json_cells}
}}
'''

  return str_notebook

def add_import_cells():

  return [
      {
          "cell_type": "markdown",
          "source": [
              "# Import libraries"
          ],
          "metadata": {
            "id": f"{uuid.uuid4()}"
          }
      },
      {
          "cell_type": "code",
          "execution_count": 'null',
          "metadata": {
            "id": f"{uuid.uuid4()}"
          },
          "outputs": [],
          "source": [
            "!pip install sweetviz\n",
          ]
      },
      {
          "cell_type": "code",
          "outputs": [],
          "execution_count": 'null',
          "metadata": {
            "id": f"{uuid.uuid4()}"
          },
          "outputs": [],
          "source": [
            "import pandas as pd\n",
            "import psycopg2\n",
            "import numpy as np\n",
            "import matplotlib.pyplot     as plt\n",
            "import matplotlib.patches    as mpatches\n",
            "import seaborn               as sns\n",
            "import sweetviz as sv\n",
            "import sklearn.metrics       as Metrics\n",
            "from google.colab import drive\n",
            "from pandas_profiling import ProfileReport\n",
            "%matplotlib inline"
          ]
      }
   ]

def add_db_config(cells):

  return cells + [
      {
          "cell_type": "markdown",
          "source": [
              "# Set DB Conn"
          ],
          "metadata": {
            "id": f"{uuid.uuid4()}"
          }
      },
      {
          "cell_type": "code",
          "execution_count": 'null',
          "metadata": {
            "id": f"{uuid.uuid4()}"
          },
          "outputs": [],
          "source": [
            "DB_params = {\n",
            f"\t'host':'{DB_params['host']}',\n",
            f"\t'port':{DB_params['port']},\n",
            f"\t'database':'{DB_params['database']}',\n",
            f"\t'user':'{DB_params['user']}',\n",
            f"\t'password':'{DB_params['password']}',\n",
            "}"
          ]
      }
   ]

def add_read_from_db(cells, query):
  return cells + [
      {
          "cell_type": "markdown",
          "source": [
              "# Read data from DB"
          ],
          "metadata": {
            "id": f"{uuid.uuid4()}"
          }
      },
      {
          "cell_type": "code",
          "execution_count": 'null',
          "metadata": {
            "id": f"{uuid.uuid4()}"
          },
          "outputs": [],
          "source": [
            "psy_conn = psycopg2.connect(**DB_params)"
          ]
      },
      {
          "cell_type": "code",
          "execution_count": 'null',
          "metadata": {
            "id": f"{uuid.uuid4()}"
          },
          "outputs": [],
          "source": [
            f"query = '{query}'"
          ]
      },
      {
          "cell_type": "code",
          "execution_count": 'null',
          "metadata": {
            "id": f"{uuid.uuid4()}"
          },
          "outputs": [],
          "source": [
            "df = pd.read_sql_query(query, psy_conn)"
          ]
      },
      {
          "cell_type": "code",
          "execution_count": 'null',
          "metadata": {
            "id": f"{uuid.uuid4()}"
          },
          "outputs": [],
          "source": [
            f"df"
          ]
      }
   ]

def add_readfile_cells(cells, path_dirname,filename,sheet):
  
  isExcel = filename.endswith('xlsx')

  return cells + [
      {
          "cell_type": "markdown",
          "source": [
              "# Read file"
          ],
          "metadata": {
            "id": f"{uuid.uuid4()}"
          }
      },
      {
          "cell_type": "code",
          "execution_count": 'null',
          "metadata": {
            "id": f"{uuid.uuid4()}"
          },
          "outputs": [],
          "source": [
            "drive.mount('/content/drive')\n"
          ]
      },
      {
          "cell_type": "code",
          "execution_count": 'null',
          "metadata": {
            "id": f"{uuid.uuid4()}"
          },
          "outputs": [],
          "source": [
            f"path_dir = '{path_dirname}'\n",
            f"filename = '{filename}'\n",
            f"sheet = '{sheet}'\n" if isExcel else ''
          ]
      },
      {
          "cell_type": "code",
          "execution_count": 'null',
          "metadata": {
            "id": f"{uuid.uuid4()}"
          },
          "outputs": [],
          "source": [
            "df = pd.read_excel(f'{path_dir}{filename}', sheet_name=sheet, header=0)" if isExcel else "df = pd.read_csv(f'{path_dir}{filename}')"
          ]
      },
      {
          "cell_type": "code",
          "execution_count": 'null',
          "metadata": {
            "id": f"{uuid.uuid4()}"
          },
          "outputs": [],
          "source": [
            f"df"
          ]
      }
   ]

def add_normalize_cols_cells(cells):

  return cells + [
      {
          "cell_type": "markdown",
          "source": [
              "# Normalize_cols"
          ],
          "metadata": {
            "id": f"{uuid.uuid4()}"
          }
      },
      {
          "cell_type": "code",
          "execution_count": 'null',
          "metadata": {
            "id": f"{uuid.uuid4()}"
          },
          "outputs": [],
          "source": [
          "def remove_accents_cols(df_cols):\n"
          "    return df_cols.str.replace('ñ','ni').str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')\n",
          "def remove_special_chars(df_cols):\n"
          "    return df_cols.str.replace(r'[$@&/.:-]',' ', regex=True)\n",
          "def regular_camel_case(snake_str):\n"
          "    components = snake_str.split('_')\n"
          "    return components[0] + ''.join(x.title() for x in components[1:])\n",
          "def regular_snake_case(df_cols):\n"
          "    cols = df_cols.str.replace('ñ','ni')\n"
          "    cols = cols.str.lower().str.replace('/',' ').str.replace('.',' ').str.strip()\n"
          "    cols = cols.str.replace(r'\s+',' ',regex=True)\n"
          "    cols = cols.str.replace(' ','_')\n"
          "    return cols"
          ]
      },
      {
          "cell_type": "code",
          "execution_count": 'null',
          "metadata": {
            "id": f"{uuid.uuid4()}"
          },
          "outputs": [],
          "source": [
              "df.columns = remove_accents_cols(df.columns)\n",
              "df.columns = remove_special_chars(df.columns)\n",
              "df.columns = regular_snake_case(df.columns)"
          ]
      },
      {
          "cell_type": "code",
          "execution_count": 'null',
          "metadata": {
            "id": f"{uuid.uuid4()}"
          },
          "outputs": [],
          "source": [
            f"df"
          ]
      }]

def add_general_stats_cells(cells):
  return cells + [
      {
          "cell_type": "markdown",
          "source": [
              "# General stats cells"
          ],
          "metadata": {
            "id": f"{uuid.uuid4()}"
          }
      },
      {
          "cell_type": "code",
          "execution_count": 'null',
          "metadata": {
            "id": f"{uuid.uuid4()}"
          },
          "outputs": [],
          "source": ["df.describe()",
          ]
      },
      {
          "cell_type": "code",
          "execution_count": 'null',
          "metadata": {
            "id": f"{uuid.uuid4()}"
          },
          "outputs": [],
          "source": ["df.dtypes\n"
          ]
      },
      {
          "cell_type": "code",
          "execution_count": 'null',
          "metadata": {
            "id": f"{uuid.uuid4()}"
          },
          "outputs": [],
          "source": [
            "df.info()"
          ]
      },
      {
          "cell_type": "code",
          "execution_count": 'null',
          "metadata": {
            "id": f"{uuid.uuid4()}"
          },
          "outputs": [],
          "source": [
            "print('recuento de columnas por tipo: ', df.dtypes.value_counts())\n",
            "print('sumatoria de valores nulos en el dataframe: ', df.isna().sum())",
          ]
      }
      ]

def add_profiling_cells(cells):
  return cells + [
      {
          "cell_type": "markdown",
          "source": [
              "# Profiling_library_install_cells"
          ],
          "metadata": {
            "id": f"{uuid.uuid4()}"
          }
      },
      {
          "cell_type": "code",
          "execution_count": 'null',
          "metadata": {
            "id": f"{uuid.uuid4()}"
          },
          "outputs": [],
          "source": ['profile = ProfileReport(df)\n'
              'df.profile_report()\n',
            'profile.to_file("reporte_html_para_clientes.html")'
          ]
      }]

def add_cell(cells,new_cells):
  return cells + new_cells

def add_sweetviz_cells(cells_p, path_dir,f,sheet):
  return cells_p + [
      {
          "cell_type": "markdown",
          "source": [
              "# Sweetviz report"
          ],
          "metadata": {
            "id": f"{uuid.uuid4()}"
          }
      },
      {
          "cell_type": "code",
          "execution_count": 'null',
          "metadata": {
            "id": f"{uuid.uuid4()}"
          },
          "outputs": [],
          "source": ['sweet_report = sv.analyze(df)\n',
                     f"sweet_report.show_html(f'{{path_dir}}sw_report_{f}_{sheet}.html')"
          ]
      }]

  pass

def create_notebook_file(filename,str_content):
  with open(f'{path_dir}EDA_{filename}.ipynb', 'w') as writefile:
    writefile.write(str_content)

def object_cells(serie):
  n_rows = serie.size
  col = serie.name
  # print('my col',serie.name)
  
  isna_val = serie.isna().sum()
  non_null_values = n_rows - isna_val

  null_nums = pd.to_numeric(serie, errors='coerce').isna().sum()
  converted_num = n_rows - null_nums

  null_dates = pd.to_datetime(serie, errors='coerce').isna().sum()
  converted_date = n_rows - null_dates

  contains_str = serie.str.contains('^[a-zA-Z]').any()

  cells_c = []

  # Validador de tipos de columna
  if (converted_num == 0 and converted_date == 0) or (contains_str == True):
    
    cells_c = add_cell(cells_c,[create_code_cells([
        f"df['{col}'] = df['{col}'].astype(str)\n",
        f"df['{col}'] = df['{col}'].str.replace('.0','')\n",
        f"df['{col}'].unique()"])
      ])

    cells_c = add_cell(cells_c,[create_code_cells([f"df['{col}'].value_counts()"])])
    cells_c = add_cell(cells_c,[create_code_cells([f"plt.figure(figsize=(10,5))\n",f"df['{col}'].value_counts()[:15].plot(kind='pie')"])])
    cells_c = add_cell(cells_c,[create_code_cells([f"plt.figure(figsize=(30,5))\n",f"df['{col}'].value_counts()[:15].plot(kind='barh')"])])

  # TODO: Rewrite next code                              
  elif converted_date > converted_num :

    cells_c = add_cell(cells_c,[create_code_cells([f"df['{col}'] = pd.to_datetime(df[f'{col}'], errors='coerce')"])])
    cells_c = add_cell(cells_c,[create_code_cells([f"print('la columna {col} es: ',df[f'{col}'].dtype)"])])

  elif converted_num >= converted_date:

    cells_c = add_cell(cells_c,[create_code_cells([f"df['{col}'] = pd.to_numeric(df[f'{col}'], errors='coerce')"])])
    cells_c = add_cell(cells_c,[create_code_cells([f"print('la media es: ',df['{col}'].mean())"])])
    cells_c = add_cell(cells_c,[create_code_cells([f"print('la moda es: ',df['{col}'].mode())"])])
    cells_c = add_cell(cells_c,[create_code_cells([f"print('la mediana es: ',df['{col}'].median())"])])
    cells_c = add_cell(cells_c,[create_code_cells([f"print('el valor mínimo de {col} es: : ',df['{col}'].min())"])])
    cells_c = add_cell(cells_c,[create_code_cells([f"print('el valor máximo de {col} es: : ',df['{col}'].max())"])])
    cells_c = add_cell(cells_c,[create_code_cells([f"print('el rango de {col} es: : ',df['{col}'].max() - df['{col}'].min())"])])
    cells_c = add_cell(cells_c,[create_code_cells([f"print('la desviación éstandar de {col} es: : ',df['{col}'].std())"])])
    cells_c = add_cell(cells_c,[create_code_cells([f"sns.histplot(data = df,x = '{col}')\n", f"plt.axvline(x=df.{col}.mean(),color='red',linestyle='dashed',linewidth=2)"])])



  return cells_c

def int_cells(serie):
  col = serie.name
  n_rows = serie.size
  unique_counts = df[col].unique().sum()

  cells_c = []

  if unique_counts < 0.2 * n_rows:

    cells_c = add_cell(cells_c,[create_code_cells([f"df['{col}'] = df['{col}'].astype(str)"])])
    cells_c = add_cell(cells_c,[create_code_cells([f"df['{col}'] = df['{col}'].str.replace('.0','')"])])
    cells_c = add_cell(cells_c,[create_code_cells([f"df['{col}'].unique()"])])
    cells_c = add_cell(cells_c,[create_code_cells([f"df['{col}'].value_counts()"])])
    cells_c = add_cell(cells_c,[create_code_cells([f"plt.figure(figsize=(10,5))\n",
                                    f"df['{col}'].value_counts()[:15].plot(kind='pie')"])])
    cells_c = add_cell(cells_c,[create_code_cells([f"plt.figure(figsize=(30,15))\n",
                                    f"df['{col}'].value_counts()[:15].plot(kind='barh')"])])
  else:

    cells_c = add_cell(cells_c,[create_code_cells([f"df['{col}'] = pd.to_numeric(df[f'{col}'], errors='coerce')"])])
    cells_c = add_cell(cells_c,[create_code_cells([f"print('la media es: ',df['{col}'].mean())"])])
    cells_c = add_cell(cells_c,[create_code_cells([f"print('la moda es: ',df['{col}'].mode())"])])
    cells_c = add_cell(cells_c,[create_code_cells([f"print('la mediana es: ',df['{col}'].median())"])])
    cells_c = add_cell(cells_c,[create_code_cells([f"print('el valor mínimo de {col} es: : ',df['{col}'].min())"])])
    cells_c = add_cell(cells_c,[create_code_cells([f"print('el valor máximo de {col} es: : ',df['{col}'].max())"])])
    cells_c = add_cell(cells_c,[create_code_cells([f"print('el rango de {col} es: : ',df['{col}'].max() - df['{col}'].min())"])])
    cells_c = add_cell(cells_c,[create_code_cells([f"print('la desviación éstandar de {col} es: : ',df['{col}'].std())"])])
    cells_c = add_cell(cells_c,[create_code_cells([f"sns.histplot(data = df,x = '{col}')\n",f"plt.axvline(x=df['{col}'].mean(),color='red',linestyle='dashed',linewidth=2)"])])

  return cells_c

def float_cells(serie):
  col = serie.name
  cells_c = []

  cells_c = add_cell(cells_c,[create_code_cells([f"df['{col}'] = pd.to_numeric(df[f'{col}'], errors='coerce')"])])
  cells_c = add_cell(cells_c,[create_code_cells([f"print('la media es: ',df['{col}'].mean())"])])
  cells_c = add_cell(cells_c,[create_code_cells([f"print('la moda es: ',df['{col}'].mode())"])])
  cells_c = add_cell(cells_c,[create_code_cells([f"print('la mediana es: ',df['{col}'].median())"])])
  cells_c = add_cell(cells_c,[create_code_cells([f"print('el valor mínimo de {col} es: : ',df['{col}'].min())"])])
  cells_c = add_cell(cells_c,[create_code_cells([f"print('el valor máximo de {col} es: : ',df['{col}'].max())"])])
  cells_c = add_cell(cells_c,[create_code_cells([f"print('el rango de {col} es: : ',df['{col}'].max() - df['{col}'].min())"])])
  cells_c = add_cell(cells_c,[create_code_cells([f"print('la desviación éstandar de {col} es: : ',df['{col}'].std())"])])
  cells_c = add_cell(cells_c,[create_code_cells([f"sns.histplot(data = df,x = '{col}')\n",f"plt.axvline(x=df['{col}'].mean(),color='red',linestyle='dashed',linewidth=2)"])])
  return cells_c

def datetime_cells(serie):
  col = serie.name
  cells_c = []
  
  cells_c = add_cell(cells_c,[create_code_cells([f"df['{col}'] = pd.to_datetime(df[f'{col}'], errors='coerce')"])])
  cells_c = add_cell(cells_c,[create_code_cells([f"print('la columna {col} es: ',df[f'{col}'].dtype)"])])
  return cells_c

def bool_cells(serie):
  col = serie.name
  cells_c = []

  cells_c = add_cell(cells_c,[create_code_cells([f"df['{col}'].unique()"])])
  cells_c = add_cell(cells_c,[create_code_cells([f"df['{col}'].value_counts()"])])
  cells_c = add_cell(cells_c,[create_code_cells([f"plt.figure(figsize=(10,5))\n",
                                    f"df['{col}'].value_counts()[:15].plot(kind='pie')"])])
  cells_c = add_cell(cells_c,[create_code_cells([f"plt.figure(figsize=(30,15))\n",
                                  f"df['{col}'].value_counts()[:15].plot(kind='barh')"])])
  return cells_c

actions_for_dtype = {
    'object': object_cells,
    'int64': int_cells,
    'float64': float_cells,
    'datetime64[ns]': datetime_cells,
    'bool': bool_cells
}

def add_columns_analysis_cells(cells_p,df_p):
  # add lines to cell analysis
  # new_cells = add code
  # cells.append(create_markdown_cell([f"## {j}"]))
  # cells.append(create_code_cells([f"df['{j}'].dtype"]))
  new_cells = add_cell(cells_p,[create_markdown_cell([f"# General Analyst by columns"])])
  for col in df_p.columns:
    col_type = df_p[col].dtype
    new_cells = add_cell(new_cells,[
        create_markdown_cell([f"## {col}\n",
                              f"type: {df_p[col].dtype}"]),
        # create_code_cells([f"df_p['{col}'].dtype"]),
                                    ])
    # new_cells = new_cells + actions_for_dtype[col_type] 
    # print(col,col_type)
    new_cells = add_cell(new_cells,actions_for_dtype[f'{col_type}'](df_p[col])) 

  return new_cells

# add_cell(cells,actions_for_dtype[f"{df['nombre1'].dtype}"](df['nombre1'],'nombre1'))

"""# Create reports
Ahora pongo algun texto aqui
"""

text_report_file = ''
text_report_sheet = ''
counter_structured_files = 0
for f in files_name:
  if '.xlsx' in f:
    try:
      xls = pd.ExcelFile(f'{path_dir}{f}')
      counter_structured_files = counter_structured_files + 1
      sheets = xls.sheet_names
      cols_total = 0
      vol_total = 0

      for sheet in sheets:
        print(f'reading {f} in sheet: {sheet}')
        
        # Read df from file by sheets
        df = pd.read_excel(f'{path_dir}{f}', sheet_name=sheet, header=0)
        #new_header = df.iloc[0] #grab the first row for the header 
        #df = df[1:] #take the data less the header row 
        #df.columns = new_header #set the header row as the df header

        # Format columns
        df.columns = remove_accents_cols(df.columns)
        df.columns = remove_special_chars(df.columns)
        df.columns = regular_snake_case(df.columns)

        cols = len(df.columns)
        cols_total = cols_total + cols
        vol = df.shape[0]*df.shape[1]
        vol_total = vol_total + vol
        
        text_report_sheet = text_report_sheet+f'{f}\t{sheet}\t{cols}\t{df.shape}\t{vol}\n'

        # Create cells for generated Notebook
        cells = add_import_cells()
        cells = add_readfile_cells(cells, path_dir,f,sheet)
        cells = add_normalize_cols_cells(cells)
        cells = add_general_stats_cells(cells)
        cells = add_columns_analysis_cells(cells,df)
        # cells = add_profiling_cells(cells)
        cells = add_sweetviz_cells(cells, path_dir,f,sheet)

        # Create Notebook .ipynb file
        format_file_string = create_notebook_text(cells)
        create_notebook_file(f"{f}_{sheet}",format_file_string)

        # sweet_report = sv.analyze(df)
        # sweet_report.show_html(f'{path_dir}{f}_sweetviz_report.html')


        text_report_file = text_report_file+f'{f}\t{len(sheets)}\t{sheets}\t{cols_total}\t{vol_total}\n'
    except Exception as e:
      print('Error reading',f)
      print(f'Error: {e}')
  elif '.csv' in f:
    try:
      df = pd.read_csv(f'{path_dir}{f}', sep=',')
      counter_structured_files = counter_structured_files + 1
      cols = len(df.columns)
      cols_total = cols_total + cols
      vol = df.shape[0]*df.shape[1]
      vol_total = vol_total + vol

      sheet = ''

      # Format columns
      df.columns = remove_accents_cols(df.columns)
      df.columns = remove_special_chars(df.columns)
      df.columns = regular_snake_case(df.columns)


      text_report_sheet = text_report_sheet+f'{f}\t1\t{cols}\t{df.shape}\t{vol}\n'
      text_report_file = text_report_file+f'{f}\t1\t\t{cols_total}\t{vol_total}\n'

      # Create cells for generated Notebook
      cells = add_import_cells()
      cells = add_readfile_cells(cells, path_dir,f,sheet)
      cells = add_normalize_cols_cells(cells)
      cells = add_general_stats_cells(cells)
      cells = add_columns_analysis_cells(cells,df)
      # cells = add_profiling_cells(cells)
      cells = add_sweetviz_cells(cells, path_dir,f,sheet)

      print(f,df.columns)
      print(cells)
      # Create Notebook .ipynb file
      format_file_string = create_notebook_text(cells)
      create_notebook_file(f"{f}_{sheet}",format_file_string)
    except:
      print('error reading csv',f)

  elif '.txt' in f:
    pass
    try:
      query = open(f'{path_dir}{f}').read()
      #df = pd.read_sql_query(query, psy_conn)

      cols_total = 0
      vol_total = 0
      
      # print(df.columns)
      # Format columns
      df.columns = remove_accents_cols(df.columns)
      df.columns = remove_special_chars(df.columns)
      df.columns = regular_snake_case(df.columns)

      cols = len(df.columns)
      cols_total = cols_total + cols
      vol = df.shape[0]*df.shape[1]
      vol_total = vol_total + vol
      
      # text_report_sheet = text_report_sheet+f'{f}\t{sheet}\t{cols}\t{df.shape}\t{vol}\n'

      # Create cells for generated Notebook
      cells = add_import_cells()
      cells = add_db_config(cells)
      cells = add_read_from_db(cells, query)
      cells = add_normalize_cols_cells(cells)
      cells = add_general_stats_cells(cells)
      cells = add_columns_analysis_cells(cells,df)
      # cells = add_profiling_cells(cells)
      # cells = add_sweetviz_cells(cells, path_dir,f,sheet)

      # Create Notebook .ipynb file
      format_file_string = create_notebook_text(cells)
      create_notebook_file(f"{f}",format_file_string)

      sweet_report = sv.analyze(df)
      sweet_report.show_html(f'{path_dir}{f}_sweetviz_report.html')


      # text_report_file = text_report_file+f'{f}\t{len(sheets)}\t{sheets}\t{cols_total}\t{vol_total}\n'

    except Exception as e:
      print('error reading csv',f)
      print(f'error: {e}')
    # print(f'{df.shape}\t{sheet}')
  # print(f,len(sheets),cols_total,sheets)
# text_report_file = text_report_file+f'{len(files_name)} files xlsx or csv'

#with open(f'{path_dir}text_report_file.txt', 'w') as writefile:
#    writefile.write(text_report_file)

with open(f'{path_dir}text_report_sheet.txt', 'w') as writefile:
    writefile.write(text_report_sheet)