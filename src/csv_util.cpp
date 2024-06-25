/*
  Author: Benjamin Wolff
  Date: February 18, 2024
  
  Based on Dr. Bruce Maxwell's csv_util file to store and extract data from csv files
*/

#include <cstdio>
#include <cstring>
#include <vector>
#include "opencv2/opencv.hpp"
#include <map>

/*
  reads a string from a CSV file. the 0-terminated string is returned in the char array os.

  The function returns false if it is successfully read. It returns true if it reaches the end of the line or the file.
 */
int getstring( FILE *fp, char os[] ) {
  int p = 0;
  int eol = 0;
  
  for(;;) {
    char ch = fgetc( fp );
    if( ch == ',' ) {
      break;
    }
    else if( ch == '\n' || ch == EOF ) {
      eol = 1;
      break;
    }
    // printf("%c", ch ); // uncomment for debugging
    os[p] = ch;
    p++;
  }
  // printf("\n"); // uncomment for debugging
  os[p] = '\0';

  return(eol); // return true if eol
}

int getint(FILE *fp, int *v) {
  char s[256];
  int p = 0;
  int eol = 0;

  for(;;) {
    char ch = fgetc( fp );
    if( ch == ',') {
      break;
    }
    else if(ch == '\n' || ch == EOF) {
      eol = 1;
      break;
    }
      
    s[p] = ch;
    p++;
  }
  s[p] = '\0'; // terminator
  *v = atoi(s);

  return(eol); // return true if eol
}

/*
  Utility function for reading one float value from a CSV file

  The value is stored in the v parameter

  The function returns true if it reaches the end of a line or the file
 */
int getfloat(FILE *fp, float *v) {
  char s[256];
  int p = 0;
  int eol = 0;

  for(;;) {
    char ch = fgetc( fp );
    if( ch == ',') {
      break;
    }
    else if(ch == '\n' || ch == EOF) {
      eol = 1;
      break;
    }
      
    s[p] = ch;
    p++;
  }
  s[p] = '\0'; // terminator
  *v = atof(s);

  return(eol); // return true if eol
}

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
int append_image_data_csv( const char *filename, char label_name, char pieceColor, std::vector<float> &image_data, int reset_file ) {
  char buffer[256];
  char mode[8];
  FILE *fp;

  strcpy(mode, "a");

  if( reset_file ) {
    strcpy( mode, "w" );
  }
  
  fp = fopen( filename, mode );
  if(!fp) {
    printf("Unable to open output file %s\n", filename );
    exit(-1);
  }


  // write the filename and the feature vector to the CSV file
  buffer[0] = label_name;
  buffer[1] = ',';
  buffer[2] = pieceColor;
  buffer[3] = '\0';
  std::fwrite(buffer, sizeof(char), strlen(buffer), fp );
  for(int i=0;i<image_data.size();i++) {
    char tmp[256];
    sprintf(tmp, ",%.4f", image_data[i] );
    std::fwrite(tmp, sizeof(char), strlen(tmp), fp );
  }
      
  std::fwrite("\n", sizeof(char), 1, fp); // EOL

  fclose(fp);
  
  return(0);
}

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
int read_image_data_csv( const char *filename, std::vector<std::string> &labels, 
                         std::vector<std::vector<float>> &data, int echo_file ) {
  FILE *fp;
  float fval;
  char label[256];
  char color[256];

  fp = fopen(filename, "r");
  if( !fp ) {
    printf("Unable to open feature file\n");
    return(-1);
  }

  //printf("Reading %s\n", filename);
  for(;;) {
    std::vector<float> dvec;
    
    
    // read the filename
    if( getstring( fp, label ) ) {
      break;
    }

    if( getstring( fp, color ) ) {
      break;
    }
    // printf("Evaluting %s\n", filename);

    // read the whole feature file into memory
    for(;;) {
      // get next feature
      float eol = getfloat( fp, &fval );
      dvec.push_back( fval );
      if( eol ) break;
    }
    // printf("read %lu features\n", dvec.size() );

    data.push_back(dvec);

    char *fname = new char[strlen(label)+strlen(color)+1];
    strcpy(fname, color);
    strcat(fname, label);
    labels.push_back( fname );
  }
  fclose(fp);
  //printf("Finished reading CSV file\n");

  if(echo_file) {
    for(int i=0;i<data.size();i++) {
      for(int j=0;j<data[i].size();j++) {
	      printf("%.4f  ", data[i][j] );
      }
      printf("\n");
    }
    printf("\n");
  }

  return(0);
}


/* Version of the above function that allows for reading the data into a hashmap, 
      where the file is the key and the feature vector is the value
*/
int read_image_data_csv2( char* filename, std::unordered_map<std::string, std::vector<float>> &dataMap) {
  FILE *fp;
  float fval;
  char label[256];
  char color[256];

  fp = fopen(filename, "r");
  if( !fp ) {
    printf("Unable to open feature file\n");
    return(-1);
  }

  printf("Reading %s\n", filename);
  for(;;) {
    std::vector<float> dvec;
    
    
    // read the filename
    if( getstring( fp, label ) ) {
      break;
    }

    if( getstring( fp, color ) ) {
      break;
    }
    // printf("Evaluting %s\n", filename);
    printf("Reading value: %s %s\b", label, color);

    // read the whole feature file into memory
    for(;;) {
      // get next feature
      float eol = getfloat( fp, &fval );
      dvec.push_back( fval );
      if( eol ) break;
    }
    // printf("read %lu features\n", dvec.size() );

    //data.push_back(dvec);

    char *fname = new char[strlen(label)+strlen(color)+1];
    strcpy(fname, color);
    strcat(fname, label);

    //filenames.push_back( fname );
    dataMap[fname] = dvec;
  }
  fclose(fp);
  printf("Finished reading CSV file\n");

  return(0);
}



