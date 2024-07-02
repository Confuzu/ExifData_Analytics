# ExifData_Analytics

ExifData Analytics is a toolbox designed for the analytical evaluation of EXIF metadata from AI-generated images. This project provides the ability to gain insights into image generation parameters and trends. It's useful for understanding the usage patterns of different models and settings used in AI image generation.

## Key Features

- Metadata Extraction and Normalization
- Database Management for metadata entries
- Parameter Analysis (Sampler, CFG scale, Size, Model, VAE, Denoising strength)
- Keyword Analysis with Keywords provided by the user, a txt file with Keywords or the default Keyword list
- TF-IDF Analysis for important terms in metadata
- Visualization of parameter frequencies and TF-IDF scores
  

## Project Structure

The project consists of three main scripts:

1. `DB_filler.py`: 
   - Responsible for executing ExifTool and managing user input for data extraction.
   - Transfers data to the database operations script. 

2. `parameter_statistic.py`:
   - Performs statistical evaluation of the data from the database.
   - Analyzes various parameters and generates visualizations and text reports.

3. `parameter_statistic_DB.py`:
   - Handles database operations, including inserting and updating metadata entries.
   - Provides functions for data retrieval and database management.


## How to Use

```
install Python3
```
```
install ExifTool by Phil Harvey  https://exiftool.org/install.html  and have it in your PATH and accessible from the command line
```
```
pip install -r requirements.txt
```
## Configuration 

You can change the settings in the `config.ini` ore use the default values:
- change the Logging directory
  
- choose another Database name
  
- Increase or Decrease Batch Size for Processing. <br /> 
  Number of images to process in each batch.<br /> 
  Adjust based on memory availability and processing speed requirements.<br />
  
- Increase or Decrease Number of worker threads.<br /> 
  Adjust based on available CPU cores and desired parallelism. <br /> 

- Example with:
  ```
  batch_size = 100
  max_workers = 24
  ```
  On a 12 Core 24 Thread CPU + SSD + 32GB  25k Images in 01:38 min data extracted and written to the Database

  
 ### Image Metadata Extraction Script
   - In the command prompt or terminal, run:
     ```
     python DB_filler.py
     ```
   - Follow the prompts to enter the directory containing your images and specify if subfolders should be included.

 ### Metadata Analysis Script
   - After the metadata extraction is complete, run:
     ```
     python parameter_statistic.py
     ```
   - This will generate statistics and plots based on your image metadata.
   - The results, statistics, plots and Logs will be saved in the project directory, Plots in the `output_plots` Folder and Logs in`LOG_files`.

 ### Database Management Script
   - Managing the SQLite database operations and it provides functions for database maintenance, optimization, and retrieval of statistics.

     ```
     python parameter_statistic_DB.py
     ```
     **Key Functions**:
  
   - `database_stats`: Retrieves general statistics about the database contents.
   - `optimize_database`: Optimizes the database for better performance.
   - `metadata_sample`: Get sample of image metadata 
   - `check_database_integrity`: Verifies the integrity of the database.
   - `backup_database`: Creates a backup of the database in the specified directory.
   - `database_size`: Returns the current size of the database file.
   - `last_modified`: Retrieves the last modification date of the database.
   - `clear_database`: Removes all records from the database.

### Data Examples

  - **keyword_analysis.txt**
    ```
    Keyword: Searched Keyword
    Count: 0
    Top Models: Model Name
    Top Samplers: Sampler Name

    Keyword: Second Searched Keyword
    Count: 0
    Top Models: Model Name
    Top Samplers: Sampler Name
    
    ...
  
  - **prompt_word_counts.txt**
    ```
    Analysis for prompt words:
    prompt1: 154
    prompt2: 146
    prompt3: 134

    ...    
    
    ```
  - **negative_prompt_word_counts.txt**
    ```
    Analysis for negative prompt words:
    negative prompt1: 154
    negative prompt2: 146
    negative prompt3: 134
    negative prompt4: 130
    
    ...
   
    ```
    - **parameter_counts.txt**
        Always the full DB  
    ```
    Analysis for Sampler:
    Sampler
    Sampler Name  65
    Sampler Name  55
    Sampler Name  29
    ...


    Analysis for CFG scale:
    CFG scale
    6     35
    4     29
    8     29
    ...

    Analysis for Size:
    Size
    576x768    149
    768x768      5
    768x576      3
    ...

    Analysis for Model:
    Model
    Model Name  30
    Model Name  28
    Model Name  22
    ...
    

    Analysis for VAE:
    VAE
    VAE Name  124
    VAE Name   30
    VAE Name   14
    ...

    Analysis for Denoising strength:
    Denoising strength
    0.35  154
    0.47  129
    0.72   97
    ...
    ```
    
     - **tfidf_analysis.txt**
    ```
    TF-IDF Analysis of Prompts:
    prompt1: 47.638000568343095
    prompt2: 31.389964942854668
    prompt3: 24.169336638845234
        
    ...
    ```
### Plot Example

![CFG scale_counts](https://github.com/Confuzu/ExifData_Analytics/assets/133601702/0fd8bad7-ce1a-4398-9f4c-9579988ed9aa)

    

### Acknowledgements
  Thanks to Phil Harvey for his awesome exif data tool https://exiftool.org

    
