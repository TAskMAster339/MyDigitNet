#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <fstream>
#include <vector>
#include <cmath>
#include <iostream>
#include <cassert>
#include <string>
#include <chrono>
#include <filesystem>

#include "dataset.h"
#include "netWork.h"

// Константы пути и номер примеров
const std::string TRAIN_IMG_PATH = RESOURCE_PATH "data/train-images-idx3-ubyte/train-images.idx3-ubyte";
const std::string TRAIN_LBL_PATH = RESOURCE_PATH "data/train-labels-idx1-ubyte/train-labels.idx1-ubyte";
const std::string TEST_IMG_PATH = RESOURCE_PATH "data/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte";
const std::string TEST_LBL_PATH = RESOURCE_PATH "data/t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte";
const int EXAMPLES = 60000;

// Нейросетть
NetWork NW;

// Константы размера окна
const GLuint WIDTH = 400, HEIGHT = 400;
const GLuint DOWNSCALED_WIDTH = 28, DOWNSCALED_HEIGHT = 28;

struct Curve {
    std::vector<float> vertices;
};

std::vector<Curve> curves;
Curve currentCurve;
GLuint VBO, VAO, shaderProgram;
bool isDrawing = false;


const char* vertexShaderSource = R"(
    #version 330 core
    layout(location = 0) in vec2 aPos;

    void main() {
        gl_Position = vec4(aPos, 0.0, 1.0);
    }
)";

const char* fragmentShaderSource = R"(
    #version 330 core
    out vec4 FragColor;

    void main() {
        FragColor = vec4(1.0, 1.0, 1.0, 1.0); // Белый цвет
    }
)";

struct data_info{
    double* pixels;
    int digit;
};

data_info* PrepareData(const std::vector<std::vector<double>>& images, const std::vector<double>& labels, const data_NetWork& data_NW, int examples){
    data_info* data = new data_info[examples];
    for (int i = 0; i < examples; ++i){
        data[i].pixels = new double[data_NW.size[0]];
    }
    for (int i = 0; i < examples; ++i){
        data[i].digit = labels[i];
        for (int j = 0; j < data_NW.size[0]; ++j){
            data[i].pixels[j] = images[i][j];
        }
    }
    return data;
}
data_NetWork ReadDataNetWork(std::string path){
    data_NetWork data{};
    std::ifstream fin;
    fin.open(path);
    if (!fin.is_open()){
        std::cout << "Error reading the file " << path << std::endl;
        return data;
    }
    else
        std::cout << path << " loading...\n";
    std::string tmp;
    int L;
        fin >> tmp;
        if (tmp == "NetWork"){
            fin >> L;            
            data.L = L;
            data.size = new int[L];
            for (int i = 0; i < L; ++i){
                fin >> data.size[i];
            }
    }
    std::cout << path << " loading completed\n";
    fin.close();
    return data;
}
void show_predict(NetWork& NW, const char* path){
    std::ifstream fin;
    fin.open(path);
    if (!fin.is_open()){
        std::cout << "Error reading the file " << path << std::endl;
        return;
    }
    double* image = new double[784];
    for (int i = 0; i < 784; i++){
        fin >> image[i];
    }
    fin.close();
    std::vector<double> result;
    result = NW.MakePredict(image);
    double sum = 0.0;
    double max = -100.0;
    int max_index = -1;
    for (int i = 0; i < result.size(); i++){
        if (result[i] < 0){
            result[i] = 0;
        }
        if (result[i] >= max){
            max = result[i];
            max_index = i;
        }
        sum += result[i];
    }
    delete[] image;
    std::cout << "--------- Prediction ---------" << std::endl;
    for (int i = 0; i < result.size(); i++){
        std::cout << '\t' << i << " -> " << round((result[i] / sum) * 100) << '%' << std::endl;
    }
    std::cout << "   Neuron network thinks this is a digit - " << max_index << std::endl;
    std::cout << "--------- Prediction ---------" << std::endl;
}
void save_rescaled_image() {
    std::vector<unsigned char> pixels(WIDTH * HEIGHT * 3);
    glReadPixels(0, 0, WIDTH, HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());

    std::vector<float> downscaled(DOWNSCALED_WIDTH * DOWNSCALED_HEIGHT);

    int blockWidth = WIDTH / DOWNSCALED_WIDTH;
    int blockHeight = HEIGHT / DOWNSCALED_HEIGHT;

    for (int y = 0; y < DOWNSCALED_HEIGHT; ++y) {
        for (int x = 0; x < DOWNSCALED_WIDTH; ++x) {
            float brightnessSum = 0.0f;

            for (int dy = 0; dy < blockHeight; ++dy) {
                for (int dx = 0; dx < blockWidth; ++dx) {
                    int pixelX = x * blockWidth + dx;
                    int pixelY = (HEIGHT - 1) - (y * blockHeight + dy); // Инвертируем координату Y
                    int index = (pixelY * WIDTH + pixelX) * 3;

                    float r = pixels[index] / 255.0f;
                    float g = pixels[index + 1] / 255.0f;
                    float b = pixels[index + 2] / 255.0f;

                    // Вычисление яркости как средней интенсивности
                    float brightness = 0.2126f * r + 0.7152f * g + 0.0722f * b;
                    brightnessSum += brightness;
                }
            }

            float averageBrightness = brightnessSum / (blockWidth * blockHeight);
            downscaled[y * DOWNSCALED_WIDTH + x] = averageBrightness;
        }
    }

    std::ofstream outFile("output.txt");
    if (outFile.is_open()) {
        for (int y = 0; y < DOWNSCALED_HEIGHT; ++y) {
            for (int x = 0; x < DOWNSCALED_WIDTH; ++x) {
                outFile << downscaled[y * DOWNSCALED_WIDTH + x] << " ";
            }
            outFile << "\n";
        }
        outFile.close();
    } else {
        std::cerr << "Failed to save output.txt" << std::endl;
    }
}
// Функция для компиляции шейдеров
GLuint compileShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    int success;
    char infoLog[512];
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        std::cerr << "ERROR::SHADER::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    return shader;
}
void key_callback(GLFWwindow* winodw, int key, int scancode, int action, int mods){
    if (key == GLFW_KEY_C && action == GLFW_PRESS){
        curves.clear();
        std::cout << "Screen cleared" << std::endl;
    }
}
void add_point_to_curve(Curve& curve, float x, float y) {
    if (!curve.vertices.empty()) {
        float lastX = curve.vertices[curve.vertices.size() - 2];
        float lastY = curve.vertices[curve.vertices.size() - 1];

        // Расстояние между последней точкой и новой точкой
        float distance = std::sqrt((x - lastX) * (x - lastX) + (y - lastY) * (y - lastY));

        if (distance > 0.01f) { // Пороговое значение для добавления дополнительных точек
            int numPoints = static_cast<int>(distance / 0.01f);
            for (int i = 1; i <= numPoints; ++i) {
                float t = static_cast<float>(i) / numPoints;
                float intermediateX = lastX + t * (x - lastX);
                float intermediateY = lastY + t * (y - lastY);
                curve.vertices.push_back(intermediateX);
                curve.vertices.push_back(intermediateY);
            }
        }
    }
    curve.vertices.push_back(x);
    curve.vertices.push_back(y);
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            isDrawing = true;
            currentCurve.vertices.clear();

            double xpos, ypos;
            glfwGetCursorPos(window, &xpos, &ypos);

            float x = (2.0f * xpos) / WIDTH - 1.0f;
            float y = 1.0f - (2.0f * ypos) / HEIGHT;

            add_point_to_curve(currentCurve, x, y);

        } else if (action == GLFW_RELEASE) {
            isDrawing = false;
            curves.push_back(currentCurve);
        }
    }
    if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS){
        save_rescaled_image();
        show_predict(NW, "output.txt");
    }
}

void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
    if (isDrawing) {
        float x = (2.0f * xpos) / WIDTH - 1.0f;
        float y = 1.0f - (2.0f * ypos) / HEIGHT;

        add_point_to_curve(currentCurve, x, y);
    }
}

void display() {
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(shaderProgram);
    glBindVertexArray(VAO);

    for (const auto& curve : curves) {
        if (!curve.vertices.empty()) {
            glBindBuffer(GL_ARRAY_BUFFER, VBO);
            glBufferData(GL_ARRAY_BUFFER, curve.vertices.size() * sizeof(float), curve.vertices.data(), GL_DYNAMIC_DRAW);
            glDrawArrays(GL_LINE_STRIP, 0, curve.vertices.size() / 2);
        }
    }

    if (isDrawing && !currentCurve.vertices.empty()) {
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, currentCurve.vertices.size() * sizeof(float), currentCurve.vertices.data(), GL_DYNAMIC_DRAW);
        glDrawArrays(GL_LINE_STRIP, 0, currentCurve.vertices.size() / 2);
    }

    glBindVertexArray(0);
}

// Главная функция
int main() {

    std::cout << "\tHello and welcome to the MyDigitNet, a simple neuron network trained to guess hand-written digits.\n";
    std::cout << "\t\tThis network needs to load config from: " << RESOURCE_PATH "Config.txt" << ".\n";
    std::cout << "\t\t\tFirst of all you need to chose activation function.\n\n";
    // Цикл настройки нейросети

    data_NetWork NW_config = ReadDataNetWork( RESOURCE_PATH "Config.txt");
    double right_ans = 0, right, predict, maxra = 0;
    int epoch = 0;
    bool study, repeat = true;
    std::chrono::duration<double> time;
   
    NW.Init(NW_config);
    NW.PrintConfig();
    std::cout << "\t\tNow we are ready to use your neuron network.\n";
    while(repeat){
        std::cout << "This neuron network has been already trained,\nAnd you can use it without study process. It will load data from Weights.txt or you can start study process.\n" << std::endl;
        std::cout << "Write 1 if you want to train your neuron network.\nWrite 0 if you want to use trained neuron network.\n";
        std::cin >> study;
        std::cout << '\n';
        if (study) {
            // mnist training data
            std::vector<std::vector<double>> images = readImages(TRAIN_IMG_PATH);
            std::vector<double> labels = readLabels(TRAIN_LBL_PATH);  

            int examples = EXAMPLES;
            data_info* data = PrepareData(images, labels, NW_config, examples);
            auto begin = std::chrono::steady_clock::now();
            while (right_ans / examples * 100 < 100){
                right_ans = 0;
                auto t1 = std::chrono::steady_clock::now();
                for (int i = 0; i < examples; ++i){
                    NW.SetInput(data[i].pixels);
                    right = data[i].digit;
                    predict = NW.ForwardFeed();
                    if (predict != right){
                        NW.BackPropagation(right);
                        NW.WeightsUpdater(0.15 * exp(-epoch / 20.0));
                    }
                    else
                        right_ans++;
                }
                auto t2 = std::chrono::steady_clock::now();
                time = t2 - t1;
                if (right_ans > maxra)
                    maxra = right_ans;
                std::cout << "right answers " << right_ans / examples * 100 << '\t' << "maxra: " << maxra / examples * 100 << " epoch: " << epoch << std::endl;
                epoch++;
                if (epoch == 10)
                    break;
                auto end = std::chrono::steady_clock::now();
                time = end - begin;
                std::cout << "TIME: " << time.count() / 60.0 << " min" << std::endl;
                
                NW.SaveWeights();
            }
        }
        else{
            NW.ReadWeights();
            std::cout << '\n';
        }
        std::cout << "Woud you like to see accuracy of your neuron network?\n\nWrite 1 if you want to see NN accuracy.\nWrite 0 if you want to skip this step.\n";
        bool test_flag;
        std::cin >> test_flag;
        std::cout << '\n';
        if (test_flag){
            // mnist test data
            std::vector<std::vector<double>> testImages = readImages(TEST_IMG_PATH);
            std::vector<double> testLabels = readLabels(TEST_LBL_PATH); 

            int ex_tests = 10000;
            data_info* data_test;
            data_test = PrepareData(testImages, testLabels, NW_config, ex_tests);
            right_ans = 0;
            for (int i = 0; i < ex_tests; ++i){
                NW.SetInput(data_test[i].pixels);
                predict = NW.ForwardFeed();
                right = data_test[i].digit;
                if (right == predict)
                    right_ans++;
            }
            std::cout << "Right answers: " << right_ans / ex_tests * 100 << std::endl;
        }
        std::cout << "Do you want to repeat these steps?\n\nWrite 1 if you do.\nWrite 0 if you don't.\n";
        std::cin >> repeat;
        std::cout << '\n';
    }
    // Инициализация GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }
    std::cout << "Now you can see a floating window, where you can draw your digits using left mouse button.\nIf you want to clean the scren, you can press 'c'.\n";
    std::cout << "When your digit is ready to guess, you need to press right mouse button to send your picture to your neuron network.\nThen you can see its prediction.\n";
    // Настройки OpenGL
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    // Создание окна
    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "Write your digit here", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    // Установка контекста
    glfwMakeContextCurrent(window);

    gladLoadGL();
    glLineWidth(25.0);
    // Компиляция и линковка шейдеров
    GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vertexShaderSource);
    GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSource);
    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    // Проверка линковки шейдерной программы
    int success;
    char infoLog[512];
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, nullptr, infoLog);
        std::cerr << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    // Генерация VBO и VAO
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW); // Изначально пустой VBO

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    // Установка коллбэков
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);
    glfwSetKeyCallback(window, key_callback);

    // Основной цикл
    while (!glfwWindowShouldClose(window)) {
        display(); // Отображение сцены

        glfwSwapBuffers(window);
        glfwPollEvents(); // Обработка событий
    }

    // Очистка ресурсов
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteProgram(shaderProgram);

    // Завершение работы
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
