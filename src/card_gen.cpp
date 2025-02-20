#include <unordered_map>
#include <opencv2/opencv.hpp>
#include <opencv2/freetype.hpp>
#include <filesystem>
#include <unistd.h>
#include <iostream>
#include <vector>
#include <string>

class CardGenerator{

private:
    std::vector<std::string> names;
    std::vector<std::string> nCodes;
    std::vector<std::string> ids;
    std::vector<std::string> images;
    std::vector<std::string> barcodes;
    std::vector<std::string> cardIds;
    std::vector<std::string> cardTemplates;
public:
    CardGenerator(){
        std::cout<<"Node inicializálása"<<std::endl;
        //todo - ok for now
        //std::string pkg_path = package_path();
        std::string pkg_path = "";
        std::cout<<"Kérem adja meg a forrás képek mappáit tartalmazó mappa elérési útját: ";
        std::cin >> pkg_path;
        names = {"John Doe", "Jane Doe", "John Smith"};
        nCodes = {"A1B2C3", "D4E5F6", "G7H8I9"};
        ids = {"56981237", "66982238", "45981239"};
        cardIds = {"15268647", "25268648", "85268649"};
        //todo - updates:seems aight
        images = {pkg_path+"/portraits/portrait1.png", pkg_path+"/portraits/portrait2.png", pkg_path+"/portraits/portrait3.png"};
        barcodes = {pkg_path+"/barcodes/rsz_barcode1.png", pkg_path+"/barcodes/rsz_barcode2.png", pkg_path+"/barcodes/rsz_barcode3.png"};
        cardTemplates = {pkg_path+"/cardtemplates/student_c.png", pkg_path+"/cardtemplates/employee_c.png"};
    }

    void generate_card(const std::string& output_dir) {
        std::cout<<"Kartyak generalasa..." << std::endl;

        // Ellenőrizze, hogy a kimeneti könyvtár létezik-e, ha nem, hozza létre
        if (!std::filesystem::exists(output_dir)) {
            std::filesystem::create_directories(output_dir);
        }

    int total_combinations = cardTemplates.size() * names.size() * nCodes.size() * ids.size() * images.size() * barcodes.size();
    int counter = 0;

    for (int i = 0; i < total_combinations; ++i) {
        // Számolja ki az indexeket a különböző vektorokból
        int template_idx = (i / (names.size() * nCodes.size() * ids.size() * images.size() * barcodes.size())) % cardTemplates.size();
        int name_idx = (i / (nCodes.size() * ids.size() * images.size() * barcodes.size())) % names.size();
        int nCode_idx = (i / (ids.size() * images.size() * barcodes.size())) % nCodes.size();
        int id_idx = (i / (images.size() * barcodes.size())) % ids.size();
        int image_idx = (i / barcodes.size()) % images.size();
        int barcode_idx = i % barcodes.size();

        // Töltse be a kártya sablonját, képet és vonalkódot
        cv::Mat card(359, 553, CV_8UC3, cv::Scalar(255, 255, 255));
        auto ft2 = cv::freetype::createFreeType2();
        //todo
        std::cout<<"Kérem adja meg a betűtípus (.ttf fájl) elérési útját: ";
        std::string font_path = "";
        std::cin >> font_path;
        ft2->loadFontData(font_path, 0);

        cv::Mat card_template = cv::imread(cardTemplates[template_idx]);
        cv::Mat image = cv::imread(images[image_idx]);
        cv::Mat barcode = cv::imread(barcodes[barcode_idx]);

        // Helyezze el a képeket és a szöveget a kártyán
        if (cardTemplates[template_idx].find("student") != std::string::npos) {
            // Hallgatói kártya elrendezése
            cv::Rect ct(cv::Point(0, 0), card_template.size());
            cv::Rect bc(cv::Point(41, 29), barcode.size());
            cv::Rect pt(cv::Point(382, 122), image.size());
            card_template.copyTo(card(ct));
            barcode.copyTo(card(bc));
            image.copyTo(card(pt));

            ft2->putText(card, names[name_idx], cv::Point(61, 297), 15, cv::Scalar(0, 0, 0), 1, cv::LINE_AA, false);
            ft2->putText(card, "H" + cardIds[id_idx], cv::Point(158, 230), 15, cv::Scalar(0, 0, 0), 1, cv::LINE_AA, false);
            ft2->putText(card, nCodes[nCode_idx], cv::Point(157, 260), 15, cv::Scalar(0, 0, 0), 1, cv::LINE_AA, false);
        } else {
            // Alkalmazotti kártya elrendezése
            cv::Rect ct(cv::Point(0, 0), card_template.size());
            cv::Rect bc(cv::Point(49, 22), barcode.size());
            cv::Rect pt(cv::Point(386, 123), image.size());
            card_template.copyTo(card(ct));
            barcode.copyTo(card(bc));
            image.copyTo(card(pt));

            ft2->putText(card, names[name_idx], cv::Point(65, 297), 15, cv::Scalar(0, 0, 0), 1, cv::LINE_AA, false);
            ft2->putText(card, "A" + cardIds[id_idx], cv::Point(165, 230), 15, cv::Scalar(0, 0, 0), 1, cv::LINE_AA, false);
            ft2->putText(card, ids[id_idx], cv::Point(165, 260), 15, cv::Scalar(0, 0, 0), 1, cv::LINE_AA, false);
        }

        // Kártáya mentése
        std::string card_filename = output_dir + "/card" + std::to_string(counter++) + ".png";
        cv::imwrite(card_filename, card);
        std::cout<<"Kartya mentve a kovetkezo helyre: " << card_filename.c_str() << std::endl;
    }
}
};



int main(int argc, char **argv) {
    CardGenerator card_gen;

    //todo
    //card_gen->generate_card(package_path() + "/generated_cards");
    std::string pkg_path = "";
    std::cout<<"Kérem adja meg a kimeneti mappa elérési útját: ";
    std::cin >> pkg_path;
    card_gen.generate_card(pkg_path + "/generated_cards");

    return 0;
}