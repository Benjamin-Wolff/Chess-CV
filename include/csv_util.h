/*
  Bruce A. Maxwell

  Utility functions for reading and writing CSV files with a specific format

  Each line of the csv file is a filename in the first column, followed by numeric data for the remaining columns
  Each line of the csv file has to have the same number of columns
 */

#ifndef CVS_UTIL_H
#define CVS_UTIL_H

#include <vector>
#include <map>

/*
  Given a filename, and feature label name, and the image features, by
  default the function will append a line of data to the CSV format
  file.  If reset_file is true, then it will open the file in 'write'
  mode and clear the existing contents.

  The label name is written to the first position in the row of
  data. The values in image_data are all written to the file as
  floats.

  The function returns a non-zero value in case of an error.
 */
int append_image_data_csv( const char *filename, char label_name, char pieceColor, std::vector<float> &image_data, int reset_file );

/*
  Given a file with the format of a string as the first column and
  floating point numbers as the remaining columns, this function
  returns the labels as a std::vector of character arrays, and the
  remaining data as a 2D std::vector<float>.

  labels will contain all of the image segment label names.
  data will contain the features calculated from each image.

  If echo_file is true, it prints out the contents of the file as read
  into memory.

  The function returns a non-zero value if something goes wrong.
 */
int read_image_data_csv( const char *filename, std::vector<std::string> &labels, std::vector<std::vector<float>> &data, int echo_file );

/**
 * Function to read image data into features, but reads into a hashMap-style structure
 * @param filename   the name of the csv file to parse
 * @param dataMap    an unordered map where the key is the image file name and the value is the feature vector
 * 
 * @returns 0 if the function is a success, non-zero value otherwise
*/
int read_image_data_csv2( char *filename, std::unordered_map<std::string, std::vector<float>> &dataMap);

#endif
