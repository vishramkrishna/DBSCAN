import sys
import dbscan

import dbscan
import data_processing
import file_io




def script_basic_DBSCAN():
    """
    Script to run the DBSCAN.
    
    Args:
        None
    Returns:
        None
    """
    print('Starting anomaly detection...')

    # 1. Import testing data
    print('Importing data...')

   
    testing_data = file_io.import_csv_data('Data/dataset/')

    # 2. Get the list of features
    print('Creating list of features...')
    features = file_io.read_features_list('Data/Features/features.txt')

    # 3. Process the data
    print('Processing the data...')
    encoded_testing_data = data_processing.apply_encoder(testing_data[0],features)


    # 4. Run the DBSCAN 
    print('Running the DBSCAN...') 
    predictions  = dbscan.run_DBCSAN(encoded_testing_data,  features)
    print('Finished anomaly detection.')
    
    print('Calculating the number of outliers...')
    data_processing.calculate_the_number_of_outliers(predictions)
    
    outliers = data_processing.create_data_frame_from_predictions(predictions, testing_data[0])
    file_io.export_data_frame_to_csv(outliers)


    return



def main():
      
    """
    argument = sys.argv[1]
    """
    argument = "0"
    if argument == "0":
       script_basic_DBSCAN()   

    return

if __name__ == '__main__':
    main()
