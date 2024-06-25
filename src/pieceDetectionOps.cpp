/* Author: Benjamin Wolff
   Date: April 17, 2024

   Implementation of the code for detecting pieces on the chess board squares.
*/

#include <iostream>
#include <numeric>
#include <unordered_set>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include "pieceDetectionOps.hpp"
#include "csv_util.h"


/**
 * Check to see if the chessboard square specified is empty or not.
 * @param image         cv::Mat representing the image of the chessboard
 * @param currentRect   cv::Rect representing the area of the square in the image
 * @param isDarkSquare  bool representing if the square is a dark square (true) or a light square (false)
 * 
 * @returns true if the space is identified to be empty, false otherwise
*/
bool isEmptySpace(cv::Mat &image, cv::Rect &currentRect, bool isDarkSquare) {
    float startX = currentRect.width * 0.2;
    float endX = currentRect.width * 0.8;
    float startY = currentRect.height * 0.2;
    float endY = currentRect.height * 0.8;

    cv::Mat square = image(currentRect);
    cv::Mat temp;

    cv::cvtColor(square, temp, cv::COLOR_BGR2GRAY);
    //cv::GaussianBlur(temp, temp, cv::Size(5, 5), 0);

    // Applies Canny
    cv::Canny(temp, temp, 10, 250);
    cv::Mat regionToCheck = temp(cv::Rect(cv::Point2f(startX, startY), cv::Point2f(endX, endY)));

    double cannySum = cv::sum(regionToCheck)[0];
    // printf("Canny sum: %f    isDark: %d\n", cannySum, isDarkSquare);
    // printf("Result: %d\n", (isDarkSquare ? (cannySum < 7000) : (cannySum < 60000)));


    return cannySum < 7000;

}

/**
 * Check to see if the chessboard square specified is empty or not.
 * @param image         cv::Mat representing the image of the chessboard
 * @param currentRect   cv::Rect representing the area of the square in the image
 * @param isDarkSquare  bool representing if the square is a dark square (true) or a light square (false)
 * 
 * @returns true if the space is identified to be empty, false otherwise
*/
bool isEmptySpace2(cv::Mat &image, cv::Rect &currentRect, bool isDarkSquare) {
    float startX = currentRect.width * 0.3;
    float endX = currentRect.width * 0.7;
    float startY = currentRect.height * 0.3;
    float endY = currentRect.height * 0.6;

    cv::Scalar blackPiecesLightMean(94.463211, 121.884900, 129.456013);
    cv::Scalar blackPiecesDarkMean(36.129934, 50.572634, 35.326354);
    cv::Scalar emptyLightSpacesMean(132.099640, 169.842131, 180.652940);
    cv::Scalar emptyDarkSpacesMean(45.951506, 63.028035, 40.079179);
    cv::Scalar whitePiecesLightMean(107.143386, 148.214789, 161.085827);
    cv::Scalar whitePiecesDarkMean(53.660771, 79.108487, 70.062115);

    cv::Mat square = image(currentRect);

    // Compute the average color of the ROI
    cv::Scalar squareMean = cv::mean(square);
    bool isCloserToEmpty;
    float emptyDifference;
    if (isDarkSquare) {
        emptyDifference = SSD(squareMean, emptyDarkSpacesMean);
        printf("Dark\n");
        printf("Empty, black, white: %f   %f   %f\n", emptyDifference, SSD(squareMean, blackPiecesDarkMean), SSD(squareMean, whitePiecesDarkMean));
        isCloserToEmpty = emptyDifference < SSD(squareMean, blackPiecesDarkMean) && emptyDifference < SSD(squareMean, whitePiecesDarkMean);
    }
    else {
        emptyDifference = SSD(squareMean, emptyLightSpacesMean);
        printf("Light\n");
        printf("Empty, black, white: %f   %f   %f\n", emptyDifference, SSD(squareMean, blackPiecesLightMean), SSD(squareMean, whitePiecesLightMean));
        isCloserToEmpty = emptyDifference < SSD(squareMean, blackPiecesLightMean) && emptyDifference < SSD(squareMean, whitePiecesLightMean);
    }

    // Determine if the square is closer to an empty space or a space with a piece
    return isCloserToEmpty;

}



/**
 * Uses the neural network to get the predicted piece label for the location.
 * @param image         cv::Mat representing the image of the chessboard
 * @param currentRect   cv::Rect representing the square of interest
 * 
 * @returns a std::string containing the piece prediction
*/
std::string getNNPieceLabel(cv::Mat &image, cv::Rect &currentRect) {
    cv::Size dnnImageSize(224, 224);
    cv::Mat square = image(currentRect);
    cv::Mat input = cv::dnn::blobFromImage(square, 1.0, cv::Size(224, 224), cv::Scalar(0, 0, 0), true, false);

    cv::dnn::Net net = cv::dnn::readNetFromONNX(PIECE_CLASSIFIER_FILE_PATH);
    net.setInput(input);

    // Forward pass to get output
    cv::Mat output = net.forward();

    // Find the index of the top prediction
    cv::Point classIdPoint;
    double confidence;
    cv::minMaxLoc(output.reshape(1, 1), nullptr, &confidence, nullptr, &classIdPoint);
    int classId = classIdPoint.x;

    printf("class ID: %d\n", classId);
    // Get the label corresponding to the predicted class
    std::string label = PIECE_VALUES[classId];
    printf("label name: %s\n", label.c_str());
    // cv::imshow("square", square);
    // cv::waitKey(0);
    // cv::destroyWindow("square");

    return label;
}


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
int getPieceLabels(cv::Mat &dst, std::vector<cv::Rect> rectangles, std::vector<std::string> &squareLabels, bool showLabels) {
    std::vector<std::string> lightLabels, darkLabels;
    std::vector<std::vector<float>> lightData, darkData;
    // read the relevant data
    read_image_data_csv(CSV_LIGHT_FILE_PATH, lightLabels, lightData, 0);
    read_image_data_csv(CSV_DARK_FILE_PATH, darkLabels, darkData, 0);
    std::string currentLabel;
    cv::Mat temp;

    dst.copyTo(temp);

    int current = 0;
    bool isDarkSquare = false;
    for (cv::Rect currentRect : rectangles) {
        // see if we can easily determine if space is empty  
        // printf("current: %d\n", current);
        if (isEmptySpace(temp, currentRect, isDarkSquare)) {
            // printf("Empty\n");
            if (showLabels) {
                cv::putText(dst, //target image
                    "ee",
                    cv::Point(currentRect.x + (0.25 * currentRect.width), currentRect.y + (0.6 * currentRect.height)),
                    cv::FONT_HERSHEY_DUPLEX,
                    3.0,
                    CV_RGB(0, 255, 0), //font color
                    4);
            }
            squareLabels.push_back("ee");
        }
        // otherwise, use histogram intersection to compare
        else {// if (false) {
            currentLabel = isDarkSquare ? computeHistogramDiffs(dst, currentRect, darkLabels, darkData) : computeHistogramDiffs(dst, currentRect, lightLabels, lightData);
            squareLabels.push_back(currentLabel);
                if (showLabels) {
                    cv::putText(dst, //target image
                                currentLabel,
                                cv::Point(currentRect.x + (0.25 * currentRect.width), currentRect.y + (0.6 * currentRect.height)),
                                cv::FONT_HERSHEY_DUPLEX,
                                3.0,
                                CV_RGB(0, 255, 0), //font color
                                4);
                }
        }
        // switch from dark to light unless starting at new row
        if (current % 8 != 7) {
            isDarkSquare = !isDarkSquare;
        }
        current++;
    }

    return 0;
}

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
int getPieceLabelsNN(cv::Mat &dst, std::vector<cv::Rect> rectangles, std::vector<std::string> &squareLabels, bool showLabels) {
    std::string currentLabel;
    cv::Mat temp;

    dst.copyTo(temp);

    bool isDarkSquare = false;
    for (cv::Rect currentRect : rectangles) {
        // see if we can easily determine if space is empty  
        if (isEmptySpace(temp, currentRect, isDarkSquare)) {
            if (showLabels) {
                cv::putText(dst, //target image
                    "ee",
                    cv::Point(currentRect.x + (0.25 * currentRect.width), currentRect.y + (0.6 * currentRect.height)),
                    cv::FONT_HERSHEY_DUPLEX,
                    3.0,
                    CV_RGB(0, 255, 0), //font color
                    4);
            }
            squareLabels.push_back("ee");
        }
        // otherwise, use histogram intersection to compare
        else { //if (false) {
            currentLabel = getNNPieceLabel(dst, currentRect);
            squareLabels.push_back(currentLabel);
                if (showLabels) {
                    cv::putText(dst, //target image
                                currentLabel,
                                cv::Point(currentRect.x + (0.25 * currentRect.width), currentRect.y + (0.6 * currentRect.height)),
                                cv::FONT_HERSHEY_DUPLEX,
                                3.0,
                                CV_RGB(0, 255, 0), //font color
                                4);
                }
            }

            isDarkSquare = !isDarkSquare;
    }

    return 0;
}


/**
 * Takes a histogram and normalizes it by dividing each value by the number of pixels in the image
 * @param histogram     a cv::Mat used to store the unnormalized histogram
 * @param hTotal        a float used to store the total number of pixels in the image
*/
void normalizeHistogram(cv::Mat &histogram, float hTotal) {
    float *hRow;
    for (int i = 0; i < histogram.rows; i++) {
        hRow = histogram.ptr<float>(i);
        for (int j = 0; j < histogram.cols; j++) {
            hRow[j] = hRow[j] / hTotal;
        }
    }
}

/**
 * Convert the given Mat of the histogram to a vector of floats.
 * @param histogram a cv::Mat for the histogram
 * @param result    a vector of floats for the result
*/
void convertMatToVec(cv::Mat &histogram, std::vector<float> &result) {
    if (histogram.isContinuous()) {
        result.assign((float*)histogram.data, (float*)histogram.data + histogram.total()*histogram.channels());
    } 
    else {
        printf("Something wrong with conversion\n");
    }

    // printf("Sum of histogram vector: %f\n", std::accumulate(result.begin(), result.end(), 0.0));

}

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
std::string computeHistogramDiffs(cv::Mat &image, cv::Rect currentRect, std::vector<std::string> &labels, std::vector<std::vector<float>> featureData, int nBins) {
    cv::Mat square = image(currentRect);

    cv::Mat featuresMat = getHistogramFeature(square, nBins);
    std::vector<float> features;
    convertMatToVec(featuresMat, features);
    float bestDiff = FLT_MAX;
    float currentDiff;
    std::string bestLabel = "";

    //std::cout << "Calculating scores for histogram... " << std::endl;
    for (int i = 0; i < labels.size(); i++) {
        currentDiff = computeHistogramIntersectionDifference(features, featureData[i]);

        if (currentDiff < bestDiff) {
            bestDiff = currentDiff;
            bestLabel = labels[i];
            //printf("Current Diff and label: %s  %f    %d\n", bestLabel.c_str(), bestDiff, i);

        }
    }

    // cv::imshow("sq",square);
    // cv::waitKey(0);
    // cv::destroyWindow("sq");

    return bestLabel;
}


/**
 * Computes the histogram intersection difference between the given histograms
 * @param h1    vector of floats for the first histogram
 * @param h2    vector of floats for the second histogram
 * 
 * @returns a float representing the difference
*/
float computeHistogramIntersectionDifference(std::vector<float> h1, std::vector<float> h2) {
    float intersection = 0.0;

    if (h1.size() != h2.size()) {
        printf("Size issues...\n");
    }

    for (int i = 0; i < h1.size(); i++) {
        if (std::min(h1[i], h2[i]) != 0.0) {
        }
        intersection += std::min(h1[i], h2[i]);
    }

    // printf("intersection is: %f\n", intersection);
    return 1.0 - intersection;
}


/**
 * Convert the vector of floats to a Mat of the histogram.
 * @param result    a vector of floats for the histogram
 * @param histogram a cv::Mat for the histogram
*/
void convertVecToMat(std::vector<float> &result, cv::Mat &histogram, int numBins=16) {
    histogram = cv::Mat(numBins, numBins, CV_32FC1, result.data());
}

/**
 * Computes the 2D histogram for an image based on the image's r and g values
 * @param image      the cv::Mat image to find the histogram for
 * @param numBins    the number of bins for each side of the histogram
 * 
 * @returns a cv::Mat for the 2D histogram, where the rows are normalized r values and the columns are normalized g values
*/
cv::Mat getHistogramFeature(cv::Mat &image, int numBins=16) {
    cv::Mat histogram = cv::Mat::zeros(numBins, numBins, CV_32FC1);
    cv::Vec3b *row;
    uchar blue, red, green;
    float r, g;
    int rIndex, gIndex;
    float total;
    
    for (int i = 0; i < image.rows; i++) {
        row = image.ptr<cv::Vec3b>(i);
 
        for (int j = 0; j < image.cols; j++) {
            blue = row[j][0];
            green = row[j][1];
            red = row[j][2];

            total = static_cast<float>(blue + green + red);
            if (total == 0) {
                r = 0;
                g = 0;
            }
            else {
                r = red / total;
                g = green / total;
            }

            rIndex = static_cast<int>(r * (numBins-1) + .5);
            gIndex = static_cast<int>(g * (numBins-1) + .5);
            histogram.at<float>(rIndex, gIndex) = histogram.at<float>(rIndex, gIndex) + 1;

        }
    }

    float hTotal = image.rows * image.cols;
    normalizeHistogram(histogram, hTotal);
    // printf("Sum of histogram feature: %f\n", cv::sum(histogram)[0]);

    return histogram;
}



/**
 * Computes the histogram intersection (difference) between 2 histograms using the histogram difference
 * @param h1         a cv::Mat representing the histogram for the first image
 * @param h2         a cv::Mat representing the histogram for the second image
 * @param numBins    the number of bins for each side of each histogram
 * 
 * @returns a float for the histogram intersection (difference) between the two histograms
*/
float computeHistogramIntersectionDifference(cv::Mat h1, cv::Mat h2, int numBins) {
    float intersection = 0.0;
    float *row1, *row2;

    for (int i = 0; i < numBins; i++) {
        row1 = h1.ptr<float>(i); 
        row2 = h2.ptr<float>(i); 
        for (int j = 0; j < numBins; j++) {
            intersection += std::min(row1[j], row2[j]);
        }
    }

    return 1.0 - intersection;
}


/**
 * Adds a label and the relevant features to the relevant features.csv file.
 * @param src           cv::Mat representing the source image
 * @param rectangle     cv::Rect representing the box for the square of interest
 * @param label         char representing the label of the square ('e', 'p', 'n', 'b', 'r', 'q', 'k')
 * @param pieceColor    char representing the piece color ('b', 'w', 'e')
 * @param isDarkSquare  bool for if the square is dark (true) or light (false)
*/
void addLabelFeatures(cv::Mat src, cv::Rect rectangle, char label, char pieceColor, bool isDarkSquare) {
    cv::Scalar boardColors[] = { cv::Scalar(125, 150, 160), cv::Scalar(30, 35, 15) };
    cv::Scalar pieceColors[] = { cv::Scalar(20, 25, 25), cv::Scalar(110, 175, 215) };
    // cv::Scalar boardShadowColors[] = { cv::Scalar(95, 120, 130), cv::Scalar(18, 25, 10) };
    cv::Mat square = src(rectangle);

    int nBins = 16;
    cv::Mat histogram = getHistogramFeature(square, nBins);
    std::vector<float> histVec;
    convertMatToVec(histogram, histVec);

    append_image_data_csv((isDarkSquare ? CSV_DARK_FILE_PATH : CSV_LIGHT_FILE_PATH), label, pieceColor, histVec, 0);


    return;
}

/**
 * Allow the user to label images based on the show squares
 * @param src   cv::Mat representing the source image
 * 
 * @returns 0 if the function returns successfully
*/
int labelImages(cv::Mat &src) {
    cv::Mat resized, temp, temp2;
    cv::Size newSize(428, 524);
    std::vector<cv::Vec4i> lines;
    // possible labels are empty, pawn, bishop, knight (n), rook, queen, king  
    std::unordered_set<char> possibleLabels = {'e', 'p', 'b', 'n', 'r', 'q', 'k'};

    // calculates lines from hough transform
    calcHoughLines(src, resized, newSize, lines);
    resized.copyTo(temp);

    // Find intersections between lines
    std::vector<cv::Point2f> intersections;
    getIntersections(temp, lines, newSize, intersections);

    src.copyTo(temp);
    std::vector<cv::Point2f> originalPoints = scalePointsToOriginal(temp, intersections, src.size(), newSize);

    // find rectangles based on the intersections
    std::vector<cv::Rect> rectangles;
    setRectangles(temp, originalPoints, rectangles);

    src.copyTo(temp2);

    int current = 0;
    bool isDarkSquare = false;
    for (cv::Rect currentRect : rectangles) {
        cv::rectangle(temp2, currentRect, cv::Scalar(0, 255, 0), 5);
        
        cv::imshow("Rectangle " + std::to_string(current), temp2);
        int key = cv::waitKey(0);
        src.copyTo(temp2);
        if (possibleLabels.find(key) != possibleLabels.end()) {
            printf("Identified as: %c\n", static_cast<char>(key));
            int color = cv::waitKey(0);

            // add the label and its features to the relevant features.csv file
            addLabelFeatures(temp2, currentRect, static_cast<char>(key), static_cast<char>(color), isDarkSquare);
        }

        cv::destroyWindow("Rectangle " + std::to_string(current));
        
        if (current % 8 != 7) {
            isDarkSquare = !isDarkSquare;
        }
        current++;
    }

    return 0;
}
