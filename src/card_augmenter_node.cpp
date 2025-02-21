#include "get_package_path.cpp"
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include <random>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace fs = std::filesystem;

class CardAugmenter  {
    public:
        CardAugmenter(const std::string& cards_dir, const std::string& output_dir);
        void processAllCards();
        inline const std::vector<cv::Mat>& getCardImages() const;
        inline const std::vector<std::string>& getBackgroundImagesPaths() const;
        inline std::string getOutputDir() const;
        inline void setCardsDir(const std::string& cards_dir);
        inline void setOutputDir(const std::string& output_dir);
    private:
        inline cv::Mat loadCardImage(const std::string& image_path);
        cv::Mat augmentCard(const cv::Mat& card, const cv::Mat background);
        inline cv::Mat loadRandomBackground();
        inline void saveGeneratedImage(const cv::Mat& image, int index);
        std::vector<cv::Mat> card_images_;
        std::vector<std::string> background_images_paths_;
        std::string output_dir_;
        mutable std::mt19937 rng_;
    };

CardAugmenter::CardAugmenter(const std::string& cards_dir, const std::string& output_dir)
    : rng_(std::random_device{}()), output_dir_(output_dir) {
    std::cout << "CardAugmenterNode konstruktor kezdete" << std::endl;
    setCardsDir(cards_dir);
    std::cout << "Kártyák betöltve a mappából: " << cards_dir.c_str() << std::endl;
    std::string backgrounds_dir =   get_package_path() +"/img/background_samples";  
    for (const auto& entry : fs::directory_iterator(backgrounds_dir)) {
        background_images_paths_.emplace_back(entry.path().string());
        if (background_images_paths_.size() >= 45) break;
    }
    if (background_images_paths_.empty() || card_images_.empty()) {
        std::cerr << "Hiba: Üres kártya- vagy háttérkép-mappa." << std::endl;
        exit(1);
    }
    std::cout << "CardAugmenter konstruktor vége" << std::endl;
}

inline const std::vector<cv::Mat>& CardAugmenter::getCardImages() const { return card_images_; }
inline const std::vector<std::string>& CardAugmenter::getBackgroundImagesPaths() const { return background_images_paths_; }
inline std::string CardAugmenter::getOutputDir() const { return output_dir_; }

inline void CardAugmenter::setCardsDir(const std::string& cards_dir) {
    card_images_.clear();
    std::cout << "Kártyák betöltése mappából: " << cards_dir.c_str() << std::endl;
    for (const auto& entry : fs::directory_iterator(cards_dir)) {
        auto card_image = loadCardImage(entry.path().string());
        if (!card_image.empty()) {
            // 3csatornás kép konvertálása 4csatornásra
            if (card_image.channels() == 3) {
                cv::cvtColor(card_image, card_image, cv::COLOR_BGR2BGRA);
                std::cout << "Kártyakép konvertálva RGBA formátumba: " << entry.path().c_str() << std::endl;
            }
            card_images_.emplace_back(card_image);
            std::cout << "Kép betöltve: " << entry.path().c_str() << std::endl;
        } else {
            std::cerr << "Hiba a kártyakép betöltésében: " << entry.path().c_str() << std::endl;
        }
    }
    std::cout << "Kártyák betöltése befejezve, összesen " << card_images_.size() << " kép található." << std::endl;
}

inline void CardAugmenter::setOutputDir(const std::string& output_dir) {
    if (!output_dir.empty()) {
        output_dir_ = output_dir;
    } else {
        std::cerr << "Hiba: Az output mappa elérési útja nem lehet üres." << std::endl;
    }
}
inline cv::Mat CardAugmenter::loadCardImage(const std::string& image_path) {
    return cv::imread(image_path, cv::IMREAD_UNCHANGED);
}
inline cv::Mat CardAugmenter::loadRandomBackground() {
    std::uniform_int_distribution<int> dist(0, background_images_paths_.size() - 1);
    std::string background_path = background_images_paths_[dist(rng_)];
    std::cout << "Háttérkép betöltése innen: " << background_path.c_str() << std::endl;
    return cv::imread(background_path);
}
//egy kártyás verzió
/*cv::Mat CardAugmenterNode::augmentCard(const cv::Mat& card, const cv::Mat& background) {
    if (card.empty() || background.empty()) {
        RCLCPP_ERROR(this->get_logger(), "A kártya vagy a háttér üres kép!");
        return cv::Mat();
    }
    cv::Mat augmented_background = background.clone();
    // kártya méretének skálázása a háttérhez képest
    double scale_factor = std::min(
        static_cast<double>(background.cols) * 0.2 / card.cols,  
        static_cast<double>(background.rows) * 0.2 / card.rows
    );

    cv::Mat resized_card;
    cv::resize(card, resized_card, cv::Size(), scale_factor, scale_factor);
    // Elforgatas + uj keret merete
    int max_angle = 30;
    double angle = std::uniform_real_distribution<double>(-max_angle, max_angle)(rng_);
    double radians = angle * CV_PI / 180.0;
    int new_width = std::abs(resized_card.cols * std::cos(radians)) + std::abs(resized_card.rows * std::sin(radians));
    int new_height = std::abs(resized_card.cols * std::sin(radians)) + std::abs(resized_card.rows * std::cos(radians));
    // uj keret létrehozása és elforgatás
    cv::Mat larger_card(new_height, new_width, resized_card.type(), cv::Scalar(0, 0, 0, 0));
    cv::Mat rotation_matrix = cv::getRotationMatrix2D(cv::Point2f(resized_card.cols / 2.0, resized_card.rows / 2.0), angle, 1.0);
    rotation_matrix.at<double>(0, 2) += (new_width - resized_card.cols) / 2.0;
    rotation_matrix.at<double>(1, 2) += (new_height - resized_card.rows) / 2.0;
    cv::warpAffine(resized_card, larger_card, rotation_matrix, larger_card.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0, 0));
    if (larger_card.empty()) {
        RCLCPP_ERROR(this->get_logger(), "Az elforgatott kártya üres!");
        return augmented_background;
    }
    // hatterre helyezes
    cv::Point position(
        std::uniform_int_distribution<int>(0, background.cols - larger_card.cols)(rng_),
        std::uniform_int_distribution<int>(0, background.rows - larger_card.rows)(rng_)
    );

    cv::Mat roi = augmented_background(cv::Rect(position.x, position.y, larger_card.cols, larger_card.rows));

    for (int y = 0; y < larger_card.rows; ++y) {
        for (int x = 0; x < larger_card.cols; ++x) {
            cv::Vec4b pixel = larger_card.at<cv::Vec4b>(y, x);
            if (pixel[3] > 0) {
                roi.at<cv::Vec3b>(y, x) = cv::Vec3b(pixel[0], pixel[1], pixel[2]);
            }
        }
    }
    return augmented_background;
}*/
cv::Mat CardAugmenter::augmentCard(const cv::Mat& card, cv::Mat background) {
    if (card.empty() || background.empty()) {
        std::cerr << "Hiba: A kártya vagy a háttér üres kép!" << std::endl;
        return background;
    }

    //kártya átméretezése
    double scale_factor = std::min(
        static_cast<double>(background.cols) * 0.2 / card.cols,
        static_cast<double>(background.rows) * 0.2 / card.rows
    );

    cv::Mat resized_card;
    cv::resize(card, resized_card, cv::Size(), scale_factor, scale_factor);

    //forgatas, uj meret
    int max_angle = 30;
    double angle = std::uniform_real_distribution<double>(-max_angle, max_angle)(rng_);
    double radians = angle * CV_PI / 180.0;
    int new_width = std::abs(resized_card.cols * std::cos(radians)) + std::abs(resized_card.rows * std::sin(radians));
    int new_height = std::abs(resized_card.cols * std::sin(radians)) + std::abs(resized_card.rows * std::cos(radians));

    cv::Mat larger_card(new_height, new_width, resized_card.type(), cv::Scalar(0, 0, 0, 0));
    cv::Mat rotation_matrix = cv::getRotationMatrix2D(
        cv::Point2f(resized_card.cols / 2.0, resized_card.rows / 2.0), angle, 1.0);
    rotation_matrix.at<double>(0, 2) += (new_width - resized_card.cols) / 2.0;
    rotation_matrix.at<double>(1, 2) += (new_height - resized_card.rows) / 2.0;
    cv::warpAffine(resized_card, larger_card, rotation_matrix, larger_card.size(),
                   cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0, 0));

    if (larger_card.empty()) {
        std::cout << "Az elforgatott kártya üres!" << std::endl;
        return background;
    }

    //random lerakas
    std::uniform_int_distribution<int> x_dist(0, background.cols - larger_card.cols);
    std::uniform_int_distribution<int> y_dist(0, background.rows - larger_card.rows);
    cv::Point position(x_dist(rng_), y_dist(rng_));
    for (int y = 0; y < larger_card.rows; ++y) {
        for (int x = 0; x < larger_card.cols; ++x) {
            cv::Vec4b pixel = larger_card.at<cv::Vec4b>(y, x);
            if (pixel[3] > 0) { // Csak a nem átlátszó pixelek
                background.at<cv::Vec3b>(position.y + y, position.x + x) =
                    cv::Vec3b(pixel[0], pixel[1], pixel[2]);
            }
        }
    }

    return background;
}
inline void CardAugmenter::saveGeneratedImage(const cv::Mat& image, int index) {
    if (image.empty()) {
        std::cerr << "Hiba: Az elmenteni kívánt kép üres, nem lehet menteni." << std::endl;
        return;
    }
    std::string output_path = output_dir_ + "/generated_image_new_" + std::to_string(index) + ".jpg";
    cv::imwrite(output_path, image);
    std::cout << "Kép elmentve: " << output_path.c_str() << std::endl;
}
//egy kártyás verzió
/*void CardAugmenterNode::processAllCards() {
    RCLCPP_INFO(this->get_logger(), "processAllCards elkezdődött.");
    for (size_t i = 0; i < card_images_.size(); ++i) {
        RCLCPP_INFO(this->get_logger(), "Háttér betöltése...");
        cv::Mat background = loadRandomBackground();
        if (background.empty()) {
            RCLCPP_ERROR(this->get_logger(), "A háttérkép betöltése sikertelen.");
            continue;
        }
        RCLCPP_INFO(this->get_logger(), "Kártya augmentálása...");
        cv::Mat augmented_image = augmentCard(card_images_[i], background);
        RCLCPP_INFO(this->get_logger(), "Generált kép mentése...");
        saveGeneratedImage(augmented_image, i);
    }
    RCLCPP_INFO(this->get_logger(), "processAllCards befejeződött.");
}*/
void CardAugmenter::processAllCards() {
    std::cout << "processAllCards elkezdődött." << std::endl;
    for (size_t i = 0; i < card_images_.size(); ++i) {
        std::cout << "Háttér betöltése..." << std::endl;
        cv::Mat background = loadRandomBackground();
        if (background.empty()) {
            std::cerr << "A háttérkép betöltése sikertelen." << std::endl;
            continue;
        }
        //random masik kartya
        std::uniform_int_distribution<int> dist(0, card_images_.size() - 1);
        const cv::Mat& second_card = card_images_[dist(rng_)];
        if (second_card.empty()) {
            std::clog << "A második kártya üres, nem generálható kép." << std::endl;
            continue;
        }

        std::cout << "Kártyák augmentálása..." << std::endl;
        cv::Mat augmented_image = background.clone();
        //augm
        augmented_image = augmentCard(card_images_[i], augmented_image);
        augmented_image = augmentCard(second_card, augmented_image);

        std::cout << "Generált kép mentése..." << std::endl;
        saveGeneratedImage(augmented_image, i);
    }
    std::cout << "processAllCards befejeződött." << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Használat: " << argv[0] << " <kártyák mappa elérési útja> <output mappa elérési útja>\n";
        return 1;
    }
    std::string cards_dir = argv[1];
    std::string output_dir = argv[2];
    if (!fs::exists(cards_dir) || !fs::is_directory(cards_dir)) {
        std::cerr << "Hiba: A megadott kártyák mappa nem létezik vagy nem mappa: " << cards_dir << "\n";
        return 1;
    }
    if (!fs::exists(output_dir)) {
        std::cout << "Az output mappa nem létezik, létrehozás: " << output_dir << "\n";
        fs::create_directories(output_dir);  
    } else if (!fs::is_directory(output_dir)) {
        std::cerr << "Hiba: A megadott output útvonal nem mappa: " << output_dir << "\n";
        return 1;
    }
    
  
    return 0;
}
