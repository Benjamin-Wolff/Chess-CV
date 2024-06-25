/* Author: Benjamin Wolff
   Date: April 12, 2024

   Main file for my CS5330 Final Project.
   Given an image (or video) of a chess board with chess pieces on it, use computer vision techniques to
        determine where the chess board's squares are, which pieces are where, 
        and provides some insights based on the chess position.
*/

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
//#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <unordered_set>

#include "csv_util.h"
#include "processingOps.hpp"
#include "pieceDetectionOps.hpp"
#include "chessAnalysis.hpp"



/**
 * Function to handle the chess board computer vision workflow,
 *  including board square location, thresholding, and piece detection
 * @param src               a cv::Mat storing the source frame data
 * @param dst               a cv::Mat where the resulting frame is expected to be stored
 * @param currentDisplay    a char storing the value of the current display, indicating what to show
*/
void handleBoardFlow(cv::Mat &src, cv::Mat &dst, char currentDisplay) {

    src.copyTo(dst);
    cv::Size newSize(428, 524);

    // show results of hough transform
    if (currentDisplay == 'h') {
        cv::Mat resized;
        // cv::Size newSize(src.size().width, src.size().height);
        std::vector<cv::Vec4i> lines;
        // calculates lines from hough transform
        calcHoughLines(src, resized, newSize, lines, true);

        resized.copyTo(dst);
        displayLines(dst, lines);

    }
    // shows intersections between hough lines
    else if (currentDisplay == 'i') {
        cv::Mat resized;
        std::vector<cv::Vec4i> lines;

        // calculates lines from hough transform
        calcHoughLines(src, resized, newSize, lines);
        resized.copyTo(dst);

        // Find intersections between lines
        std::vector<cv::Point2f> intersections;
        getIntersections(dst, lines, newSize, intersections);

        // get back our normal size points
        src.copyTo(dst);
        std::vector<cv::Point2f> originalPoints = scalePointsToOriginal(dst, intersections, src.size(), newSize, true);
    }
    // show squares formed by intersections
    else if (currentDisplay == 's') {
        cv::Mat resized;
        std::vector<cv::Vec4i> lines;

        // calculates lines from hough transform
        calcHoughLines(src, resized, newSize, lines);
        resized.copyTo(dst);

        // Find intersections between lines
        std::vector<cv::Point2f> intersections;
        getIntersections(dst, lines, newSize, intersections);

        src.copyTo(dst);
        std::vector<cv::Point2f> originalPoints = scalePointsToOriginal(dst, intersections, src.size(), newSize);

        // find rectangles based on the intersections
        std::vector<cv::Rect> rectangles;
        setRectangles(dst, originalPoints, rectangles, true);

    }
    // show piece labelings
    else if (currentDisplay == 'p') {
        cv::Mat resized;
        std::vector<cv::Vec4i> lines;

        // calculates lines from hough transform
        calcHoughLines(src, resized, newSize, lines);
        resized.copyTo(dst);

        // Find intersections between lines
        std::vector<cv::Point2f> intersections;
        getIntersections(dst, lines, newSize, intersections);

        src.copyTo(dst);
        std::vector<cv::Point2f> originalPoints = scalePointsToOriginal(dst, intersections, src.size(), newSize);

        // find rectangles based on the intersections
        std::vector<cv::Rect> rectangles;
        setRectangles(dst, originalPoints, rectangles);

        // get all the labels for the pieces
        std::vector<std::string> squareLabels;
        getPieceLabels(dst, rectangles, squareLabels, true);

    }
    // get the analysis
    else if (currentDisplay == 'x') {
        cv::Mat resized;
        std::vector<cv::Vec4i> lines;

        // calculates lines from hough transform
        calcHoughLines(src, resized, newSize, lines);
        resized.copyTo(dst);

        // Find intersections between lines
        std::vector<cv::Point2f> intersections;
        getIntersections(dst, lines, newSize, intersections);
        
        src.copyTo(dst);
        std::vector<cv::Point2f> originalPoints = scalePointsToOriginal(dst, intersections, src.size(), newSize);

        // find rectangles based on the intersections
        std::vector<cv::Rect> rectangles;
        setRectangles(dst, originalPoints, rectangles);

        // get all the labels for the pieces
        std::vector<std::string> squareLabels;
        getPieceLabels(dst, rectangles, squareLabels);

        std::string fen = getFenFromLabels(squareLabels);
        getChessAnalysis(dst, fen, rectangles);

    }
    // show all current steps
    else if (currentDisplay == 'a') {
        cv::Mat resized;
        std::vector<cv::Vec4i> lines;

        // calculates lines from hough transform
        calcHoughLines(src, resized, newSize, lines);
        resized.copyTo(dst);

        // Find intersections between lines
        std::vector<cv::Point2f> intersections;
        getIntersections(dst, lines, newSize, intersections);
        
        src.copyTo(dst);
        std::vector<cv::Point2f> originalPoints = scalePointsToOriginal(dst, intersections, src.size(), newSize);

        // find rectangles based on the intersections
        std::vector<cv::Rect> rectangles;
        setRectangles(dst, originalPoints, rectangles);

        // get all the labels for the pieces
        std::vector<std::string> squareLabels;
        getPieceLabels(dst, rectangles, squareLabels);

        std::string fen = getFenFromLabels(squareLabels);
        getChessAnalysis(dst, fen, rectangles);

    }

    return;
}

/**
 * Function to handle image display and allow for the chess board workflow
 * @param imgPath           a string representing the path to the image of the chess board
 * @param possibleButtons   a set of characters representing the buttons that correspond to actions
 * 
 * @returns 0 if the function was successful
*/
int handleImgDisplay(std::string imgPath, std::unordered_set<char> possibleButtons) {
    cv::Mat src = imread(imgPath, cv::IMREAD_COLOR);
    cv::Mat dst;

    // checks if image is empty, returns 1 if so
    if (src.empty() ) {
        std::cout << "Could not read the following image: " << imgPath << ". Please try again!" << std::endl;
        return 1;
    }

    cv::imshow("Original Image", src);
    int key = cv::waitKey(0);

    while (key != 'q') {
        // create a new window based on the requested image transformation
        if (possibleButtons.find(key) != possibleButtons.end() && key != 'n') {
            handleBoardFlow(src, dst, key);
            cv::imshow(std::string(1, char(key)), dst);
        }

        key = cv::waitKey(0);
    }
    return 0;
}


int handleLabelDisplay(std::string imgPath) {

    cv::Mat src = imread(imgPath, cv::IMREAD_COLOR);

    // checks if image is empty, returns 1 if so
    if (src.empty() ) {
        std::cout << "Could not read the following image: " << imgPath << ". Please try again!" << std::endl;
        return 1;
    }

    cv::imshow("Original Image", src);
    int key = cv::waitKey(0);

    printf("Shown is the initial image. Label each square as 'p' for pawn, 'b' for bishop, 'n' for knight, 'r' for rook, 'q' for queen, 'k' for king. Press any other key to skip\n");
    printf("After that, enter 'b' for black or 'w' for white\n");

    labelImages(src);

    return 0;
}

int handleSavingDisplay() {
    std::string imgPath = "images/IMG_1248.jpg";
    cv::Mat src, dst, resized, temp;
    cv::Size newSize(428, 524);

    src = cv::imread(imgPath, cv::IMREAD_COLOR);

    // checks if image is empty, returns 1 if so
    if (src.empty() ) {
        std::cout << "Could not read the following image: " << imgPath << ". Please try again!" << std::endl;
        return 1;
    }

    std::vector<cv::Vec4i> lines;

    // calculates lines from hough transform
    calcHoughLines(src, resized, newSize, lines);
    resized.copyTo(dst);

    // Find intersections between lines
    std::vector<cv::Point2f> intersections;
    getIntersections(dst, lines, newSize, intersections);

    src.copyTo(dst);
    std::vector<cv::Point2f> originalPoints = scalePointsToOriginal(dst, intersections, src.size(), newSize);

    // find rectangles based on the intersections
    std::vector<cv::Rect> rectangles;
    setRectangles(dst, originalPoints, rectangles, true);
    
    int current = 0;
    for (cv::Rect currentRect : rectangles) {
        cv::Mat temp = src(currentRect);
        cv::imwrite("im5_" + std::to_string(current) + ".jpg", temp);
        current++;
    }
   
    return 0;
}

/**
 * Main function to take in an image of a chessboard with pieces on it and evaluate the position
 */
int main(int argc, char *argv[])
{
    std::string displayType;
    int ret;
    std::string imgPath;

    if (argc == 1) {
        std::cout << "Must include an image path" << std::endl;
    }
    else if (argc == 2) {
        displayType = "img";
        imgPath = argv[1];
    }
    else if (argc == 3) {
        displayType = argv[1];
        imgPath = argv[2];
    }
    else {
        std::cout << "Usage: segmentation [img or label or *NONE*] [imgPath]" << std::endl;
        return -1;
    }

    // options are normal, hough transform, intersections, squares, pieces, analysis, all (demo)
    std::unordered_set<char> possibleButtons = {'n', 'h', 'i', 's', 'p', 'x', 'a'};

    // enters the related path for handling live video or an image;
    // if (displayType == "vid") {
    //     ret = handleVidDisplay(possibleButtons);
    // }
    if (displayType == "img") {
        ret = handleImgDisplay(imgPath, possibleButtons);
    }
    else if (displayType == "label") {
        ret = handleLabelDisplay(imgPath);
    }
    else if (displayType == "save") {
        ret = handleSavingDisplay();
    }
    else {
        std::cout << "Invalid display type: " <<  displayType << " - [img or vid]" << std::endl;
        ret = -1;
    }

    return ret;
}

