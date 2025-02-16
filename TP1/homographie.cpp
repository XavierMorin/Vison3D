#include "opencv2/highgui.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
//   https://docs.opencv.org/4.5.4/d3/d63/classcv_1_1Mat.html

using namespace std;

#include <iostream>
#include <vector>
#include <string>

cv::Mat creeCanevasEtAjusteLesCoords(cv::Mat& image, std::vector<int>& orect, std::vector<int>& coords) {

    // Cree le nouveau canevas
    int xmin = orect[0];
    int ymin = orect[1];
    int xmax = orect[2];
    int ymax = orect[3];

    int largeur = xmax - xmin;
    int hauteur = ymax - ymin;

    cv::Mat canevas = cv::Mat::zeros(cv::Size(largeur, hauteur), image.type());

    // Trouve la position de image A dans le canevas
    int x_offset = std::max(0, -xmin);
    int y_offset = std::max(0, -ymin);

    // Ajuste les coordonees de l'image A par rapport au nouveau canevas
    for(int i =0; i < coords.size(); i++)
    {
        if(i %2 ==0)
            coords[i] += x_offset;
        else
            coords[i] += y_offset;
    }

    int roi_largeur = std::min(image.cols, largeur - x_offset);
    int roi_hauteur = std::min(image.rows, hauteur - y_offset);

    if (roi_largeur <= 0 || roi_hauteur <= 0) {
        // L'image n'est pas dans le canevas
        return canevas;
    }

    // Parties de l'image A qui rentre dans le nouveau canevas
    cv::Rect roi_image(std::max(0, xmin), std::max(0, ymin), roi_largeur, roi_hauteur);

    // Emplacement de l'image A dans le nouveau canevas
    cv::Rect roi_canevas(x_offset, y_offset, roi_largeur, roi_hauteur);

    // Copie l'image A dans le nouveau canevas
    image(roi_image).copyTo(canevas(roi_canevas));

    return canevas;
}

cv::Mat creerMasqueAPartirDesCoords(const cv::Size& tailleImage, const std::vector<int>& coords) {
    std::vector<cv::Point> points;
    for (size_t i = 0; i < coords.size(); i += 2) {
        points.emplace_back(coords[i], coords[i + 1]);
    }

    cv::Mat masque = cv::Mat::zeros(tailleImage, CV_8UC1);
    std::vector<std::vector<cv::Point>> remplissageTout{points};
    cv::fillPoly(masque, remplissageTout, cv::Scalar(255));

    return masque;
}

cv::Mat trouverHomographieBversA(std::vector<int>& coords_a, std::vector<int>& coords_b)
{

    std::vector<cv::Point2f> points_a, points_b;
    for (size_t i = 0; i < coords_a.size(); i += 2) {
        points_a.emplace_back(coords_a[i], coords_a[i + 1]);
        points_b.emplace_back(coords_b[i], coords_b[i + 1]);
    }

    cv::Mat homographie_ba = cv::findHomography(points_b, points_a, cv::RANSAC);

    return homographie_ba;
}


void trouverHomographieBversAetSuperposerSurA(
    std::vector<int>& coords_a,
    std::vector<int>& coords_b,
    std::string& chemin_image_a,
    std::string& chemin_image_b,
    std::string& chemin_image_sortie,
    std::vector<int> orect ={})
{

    cv::Mat image_a = cv::imread(chemin_image_a);
    cv::Mat image_b = cv::imread(chemin_image_b);

    if (image_a.empty() || image_b.empty()) {
        std::cerr << "Impossible de charger une ou les deux images !" << std::endl;
        return;
    }

    cv::Mat resultat;
    if(!orect.empty())
        resultat = creeCanevasEtAjusteLesCoords(image_a, orect, coords_a);
    else
        resultat = image_a.clone();

    cv::Mat image_b_deformee;
    cv::Mat homographie_ba = trouverHomographieBversA(coords_a, coords_b);
    cv::warpPerspective(image_b, image_b_deformee, homographie_ba, resultat.size());

    cv::Mat masque = creerMasqueAPartirDesCoords(resultat.size(), coords_a);
    image_b_deformee.copyTo(resultat, masque);

    // Sauvegarde le résultat si le chemin de sortie est fourni
    if (!chemin_image_sortie.empty()) {
        cv::imwrite(chemin_image_sortie, resultat);
        std::cout << "Image de sortie sauvegardée à " << chemin_image_sortie << std::endl;
    }
    else {
        cv::imshow("Image B déformée sur Image A", resultat);
        cv::waitKey(0);
    }
}

// ./homographie -a 270 490 315 488 315 620 270 640 -b 0 0 1000 0 1000 1400 0 1400 -ia "../images/stephen-wilkes_times-square-nyc-day-to-night-1.jpg" -ib "../images/bon-cop-bad-cop.jpg"
int main(int argc, char *argv[])
{
    std::string chemin_image_a, chemin_image_b, chemin_image_sortie;
    std::vector<int> coords_a, coords_b, orect;

    // Parse les arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-ia") {
            if (i + 1 < argc) {
                chemin_image_a = argv[++i];
            }
        } else if (arg == "-ib") {
            if (i + 1 < argc) {
                chemin_image_b = argv[++i];
            }
        } else if (arg == "-io") {
            if (i + 1 < argc) {
                chemin_image_sortie = argv[++i];
            }
        } else if (arg == "-a") {
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                coords_a.push_back(std::stoi(argv[++i]));
            }
        } else if (arg == "-b") {
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                coords_b.push_back(std::stoi(argv[++i]));
            }
        } else if (arg == "-orect"){
            while (i + 1 < argc) {
                orect.push_back(std::stoi(argv[++i]));
            }
        }
    }

    if (coords_a.empty() || coords_a.size() != 8) {
        std::cout << "Veuillez entrer exactement 4 coordonnées pour a. Sous la forme -a x1 y1 x2 y2 x3 y3 x4 y4";
        return 0;
    }
    if (coords_b.empty() || coords_b.size() != 8) {
        std::cout << "Veuillez entrer exactement 4 coordonnées pour b. Sous la forme -b x1 y1 x2 y2 x3 y3 x4 y4";
        return 0;
    }

    if (!orect.empty() && orect.size() != 4) {
        std::cout << "Veuillez entrer exactement 4 coordonnées pour orect. Sous la forme -orec xmin ymin xmax ymax";
        return 0;
    }

    if (chemin_image_a.empty() || chemin_image_b.empty())
    {
        cv::Mat homographie_ba = trouverHomographieBversA(coords_a,coords_b);
        std::cout << homographie_ba;
        return 0;
    }

    trouverHomographieBversAetSuperposerSurA(coords_a, coords_b, chemin_image_a, chemin_image_b, chemin_image_sortie, orect);
    return 0;
}


