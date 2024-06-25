/*
  Author: Benjamin Wolff
  Date: April 18, 2024
  
  Headers for the operations related to analyzing the chess position and working with chess-related APIs
*/

#pragma once
#include <iostream>
#include <unordered_map>
#include <string>
#include <cpr/cpr.h>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

/**
 * Makes an API call to StockFish chess engine based on the fen, and displays the evaluation and best move if obtained.
 * @param dst       cv::Mat representing the image
 * @param fen       string of the 'fen' representation of the board's pieces
 * @param squares   vector of cv::Rect's representing the squares so the best move can be displayed.
 * 
 * @returns 0 if the function returns successfully
*/
int getChessAnalysis(cv::Mat image, std::string fen, std::vector<cv::Rect> squares);

/**
 * Converts the labels of the chessboard to the chess "fen" format, a format that an API related to chess can read
 * @param squareLabels  vector of strings representing the labels for each square on the board
 * 
 * @returns a string in the "fen" format
*/
std::string getFenFromLabels(std::vector<std::string> squareLabels);