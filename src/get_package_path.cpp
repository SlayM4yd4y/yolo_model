#include <iostream>
#include <cstdlib>
#include <string>
#include <array>

std::string get_package_path() {
    std::array<char, 128> buffer;
    std::string result;
    FILE* pipe = popen("find $HOME -type d -name 'yolo_model' 2>/dev/null | head -n 1", "r");
    if (!pipe) return "NULL";
    while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
        result += buffer.data();
    }
    pclose(pipe);

    if (!result.empty()) {
        result.pop_back();  
        return result;
    }
    return "NULL";
}

int main() {
    std::cout << get_package_path() << std::endl;
    return 0;
}
