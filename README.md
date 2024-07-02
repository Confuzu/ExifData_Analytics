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

### Image Metadata Extraction Script
   - In the command prompt or terminal, run:
     ```sh
     python DB_filler.py
     ```
   - Follow the prompts to enter the directory containing your images and specify if subfolders should be included.

 ### Metadata Analysis Script
   - After the metadata extraction is complete, run:
     ```sh
     python parameter_statistic.py
     ```
   - This will generate statistics and plots based on your image metadata.
   - The results, statistics, plots and Logs will be saved in the project directory, Plots in the `output_plots` Folder and Logs in`LOG_files`.

 ### Database Management Script
   - Managing the SQLite database operations and it provides functions for database maintenance, optimization, and retrieval of statistics.

     ```sh
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


