/*
  Author: Benjamin Wolff
  Date: April 16, 2024
  
  Headers for the operations related to detecting pieces and squares for the chess board image.
*/

#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>

#include "processingOps.hpp"

const char CSV_LIGHT_FILE_PATH[] = "light_features.csv";
const char CSV_DARK_FILE_PATH[] = "dark_features.csv";

const char PIECE_CLASSIFIER_FILE_PATH[] = "chess_piece_classifier_vgg16.onnx";

const std::string PIECE_VALUES[12] = {"bb", "bk", "bn", "bp", "bq", "br", "wb", "wk", "wn", "wp", "wq", "wr"};


/**
 * Check to see if the chessboard square specified is empty or not.
 * @param image         cv::Mat representing the image of the chessboard
 * @param currentRect   cv::Rect representing the area of the square in the image
 * @param isDarkSquare  bool representing if the square is a dark square (true) or a light square (false)
 * 
 * @returns true if the space is identified to be empty, false otherwise
*/
bool isEmptySpace(cv::Mat &image, cv::Rect &currentRect, bool isDarkSquare);

/**
 * Uses the neural network to get the predicted piece label for the location.
 * @param image         cv::Mat representing the image of the chessboard
 * @param currentRect   cv::Rect representing the square of interest
 * 
 * @returns a std::string containing the piece prediction
*/
std::string getNNPieceLabel(cv::Mat &image, cv::Rect &currentRect);

/**
 * Find the predicted piece labels for each square on the board.
 *   "ee" for empty, and "b" or "w" for black and white followed by the letter for the piece.
 * @param dst           cv::Mat represeting the image
 * @param rectangles    vector of cv::Rect's representing each square on the chess board
 * @param squareLabels  the resulting vector of strings containing the labels for each square
 * @param showLabels    boolean representing if we want to show the labels on dst
 * 
 * @returns 0 if the function returns successfully
*/
int getPieceLabels(cv::Mat &dst, std::vector<cv::Rect> rectangles, std::vector<std::string> &squareLabels, bool showLabels=false);


/**
 * Find the predicted piece labels for each square on the board using the neural network.
 *   "ee" for empty, and "b" or "w" for black and white followed by the letter for the piece.
 * @param dst           cv::Mat represeting the image
 * @param rectangles    vector of cv::Rect's representing each square on the chess board
 * @param squareLabels  the resulting vector of strings containing the labels for each square
 * @param showLabels    boolean representing if we want to show the labels on dst
 * 
 * @returns 0 if the function returns successfully
*/
int getPieceLabelsNN(cv::Mat &dst, std::vector<cv::Rect> rectangles, std::vector<std::string> &squareLabels, bool showLabels=false);

/**
 * Computes the 2D histogram for an image based on the image's r and g values
 * @param image      the cv::Mat image to find the histogram for
 * @param numBins    the number of bins for each side of the histogram
 * 
 * @returns a cv::Mat for the 2D histogram, where the rows are normalized r values and the columns are normalized g values
*/
cv::Mat getHistogramFeature(cv::Mat &image, int numBins);

/**
 * Convert the given Mat of the histogram to a vector of floats.
 * @param histogram a cv::Mat for the histogram
 * @param result    a vector of floats for the result
*/
void convertMatToVec(cv::Mat &histogram, std::vector<float> &result);

/**
 * Computes the histogram intersection difference between the given histograms
 * @param h1    vector of floats for the first histogram
 * @param h2    vector of floats for the second histogram
 * 
 * @returns a float representing the difference
*/
float computeHistogramIntersectionDifference(std::vector<float> h1, std::vector<float> h2);

/**
 * Computes the histogram differences between the image of the square and other square images and returns the best label
 * @param image         a cv::Mat storing the relevant image
 * @param currentRect   a cv::Rect for the rectangle of the square of interest on the board
 * @param labels        a vector of strings representing the labels of the existing data
 * @param featureData   a vector of vectors of histogram features from the existing data
 * @param nBins         an int that states how many bins the histograms will be split into
 * 
 * @returns a string representing the best label
*/
std::string computeHistogramDiffs(cv::Mat &image, cv::Rect currentRect, std::vector<std::string> &labels, std::vector<std::vector<float>> featureData, int nBins=16);


/**
 * Adds a label and the relevant features to the relevant features.csv file.
 * @param src           cv::Mat representing the source image
 * @param rectangle     cv::Rect representing the box for the square of interest
 * @param label         char representing the label of the square ('e', 'p', 'n', 'b', 'r', 'q', 'k')
 * @param pieceColor    char representing the piece color ('b', 'w', 'e')
 * @param isDarkSquare  bool for if the square is dark (true) or light (false)
*/
void addLabelFeatures(cv::Mat src, cv::Rect rectangle, char label, char pieceColor, bool isDarkSquare);

/**
 * Allow the user to label images based on the show squares
 * @param src   cv::Mat representing the source image
 * 
 * @returns 0 if the function returns successfully
*/
int labelImages(cv::Mat &src);
