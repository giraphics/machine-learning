// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <onnxruntime_cxx_api.h>
#include <array>
#include <vector>
#include <algorithm>
#include <iostream>
#include <cstring>
#include <cmath>

// Drawing area size and scale
constexpr int SIZE = 50;
int DRAW_W = SIZE, DRAW_H = SIZE, SCALE = 8;
int WIN_W = DRAW_W * SCALE + 200, WIN_H = DRAW_H * SCALE;

int brush_width = 1; // Brush width in pixels (1-20)

// PBO/texture
GLuint pbo = 0, tex = 0, vao = 0, vbo = 0, ebo = 0;

// Drawing buffer (RGBA8)
std::vector<uint32_t> drawbuf;

// Mouse state
bool painting = false;

// MNIST model wrapper (same as before, but with cross-platform path)
template <typename T>
static void softmax(T& input) {
    float rowmax = *std::max_element(input.begin(), input.end());
    std::vector<float> y(input.size());
    float sum = 0.0f;
    for (size_t i = 0; i != input.size(); ++i) {
        sum += y[i] = std::exp(input[i] - rowmax);
    }
    for (size_t i = 0; i != input.size(); ++i) {
        input[i] = y[i] / sum;
    }
}

struct MNIST {
    MNIST() {
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        input_tensor_ = Ort::Value::CreateTensor<float>(memory_info, input_image_.data(), input_image_.size(),
            input_shape_.data(), input_shape_.size());
        output_tensor_ = Ort::Value::CreateTensor<float>(memory_info, results_.data(), results_.size(),
            output_shape_.data(), output_shape_.size());
    }
    std::ptrdiff_t Run() {
        const char* input_names[] = { "Input3" };
        const char* output_names[] = { "Plus214_Output_0" };
        Ort::RunOptions run_options;
        session_.Run(run_options, input_names, &input_tensor_, 1, output_names, &output_tensor_, 1);
        softmax(results_);
        result_ = std::distance(results_.begin(), std::max_element(results_.begin(), results_.end()));
        return result_;
    }

    static constexpr int MNIST_SIZE = 28;
    static constexpr const int width_ = MNIST_SIZE;
    static constexpr const int height_ = MNIST_SIZE;
    std::array<float, width_ * height_> input_image_{};
    std::array<float, 10> results_{};
    int64_t result_{ 0 };
private:
    Ort::Env env;
    Ort::Session session_{ env, L"mnist.onnx", Ort::SessionOptions{nullptr} };
    Ort::Value input_tensor_{ nullptr };
    std::array<int64_t, 4> input_shape_{ 1, 1, width_, height_ };
    Ort::Value output_tensor_{ nullptr };
    std::array<int64_t, 2> output_shape_{ 1, 10 };
};

std::unique_ptr<MNIST> mnist;

// Downsample drawbuf (RGBA8) to MNIST input (float, 0=white, 1=black)

void ConvertDrawbufToMnist() {
    auto MNIST_SIZE = MNIST::MNIST_SIZE;
    for (int y = 0; y < MNIST_SIZE; ++y) {
        for (int x = 0; x < MNIST_SIZE; ++x) {
            int x0 = x * DRAW_W / MNIST_SIZE;
            int x1 = (x + 1) * DRAW_W / MNIST_SIZE;
            int y0 = y * DRAW_H / MNIST_SIZE;
            int y1 = (y + 1) * DRAW_H / MNIST_SIZE;

            float sum = 0;
            int count = 0;

            for (int yy = y0; yy < y1; ++yy) {
                for (int xx = x0; xx < x1; ++xx) {
                    uint32_t pixel = drawbuf[yy * DRAW_W + xx];
                    uint8_t r = (pixel >> 16) & 0xFF;
                    uint8_t g = (pixel >> 8) & 0xFF;
                    uint8_t b = pixel & 0xFF;
                    float v = (r + g + b) < 384 ? 1.0f : 0.0f;
                    sum += v;
                    count++;
                }
            }

            mnist->input_image_[y * MNIST_SIZE + x] = (count > 0) ? (sum / count) : 0.0f;
        }
    }
}

// GLFW mouse callbacks
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);
    int x = xpos / SCALE, y = ypos / SCALE;
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) painting = true;
        else if (action == GLFW_RELEASE) {
            painting = false;
            ConvertDrawbufToMnist();
            mnist->Run();
            // Print result to console
            std::cout << "Predicted digit: " << mnist->result_ << std::endl;
            for (int i = 0; i < 10; ++i)
                std::cout << i << ": " << mnist->results_[i] << std::endl;
        }
    }
    if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS) {
        std::fill(drawbuf.begin(), drawbuf.end(), 0xFFFFFFFF);
    }
}

void cursor_pos_callback(GLFWwindow* window, double xpos, double ypos) {
    if (painting) {
        int x = xpos / SCALE, y = ypos / SCALE;
        int half = brush_width / 2;
        for (int dy = -half; dy <= half; ++dy) {
            for (int dx = -half; dx <= half; ++dx) {
                int xx = x + dx, yy = y + dy;
                if (xx >= 0 && xx < DRAW_W && yy >= 0 && yy < DRAW_H)
                    drawbuf[yy * DRAW_W + xx] = 0xFF000000; // Draw black
            }
        }
    }
}

// Upload drawbuf to PBO/texture
void update_pbo_texture() {
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    void* ptr = glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY);
    memcpy(ptr, drawbuf.data(), drawbuf.size() * sizeof(uint32_t));
    glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, DRAW_W, DRAW_H, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

// Simple shader sources
const char* vs_src = R"(
#version 330 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aTex;
out vec2 TexCoord;
void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    TexCoord = aTex;
}
)";
const char* fs_src = R"(
#version 330 core
in vec2 TexCoord;
out vec4 FragColor;
uniform sampler2D uTex0;
void main() {
    FragColor = texture(uTex0, TexCoord);
}
)";

GLuint compile_shader(GLenum type, const char* src) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char info[512];
        glGetShaderInfoLog(shader, 512, nullptr, info);
        std::cerr << "Shader compile error: " << info << std::endl;
    }
    return shader;
}

GLuint create_program() {
    GLuint vs = compile_shader(GL_VERTEX_SHADER, vs_src);
    GLuint fs = compile_shader(GL_FRAGMENT_SHADER, fs_src);
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);
    return prog;
}

// Setup quad VAO/VBO/EBO
void setup_quad() {
    float vertices[] = {
        // pos      // tex (flipped vertically)
         1, -1,     1, 1,
         1,  1,     1, 0,
        -1,  1,     0, 0,
        -1, -1,     0, 1
    };
    unsigned short indices[] = { 0, 1, 2, 2, 3, 0 };
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &ebo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glBindVertexArray(0);
}

// Draw the MNIST output as a bar graph
void draw_bar_graph(const MNIST& mnist) {
    float x0 = (float)(DRAW_W * SCALE + 20) / WIN_W * 2 - 1;
    float barw = 0.1f;
    float maxv = *std::max_element(mnist.results_.begin(), mnist.results_.end());
    for (int i = 0; i < 10; ++i) {
        float v = mnist.results_[i] / maxv;
        float y0 = 1.0f - i * 0.18f;
        float y1 = y0 - 0.12f;
        float x1 = x0 + v * 0.7f;
        float color[3] = { (i == mnist.result_) ? 0.5f : 0.2f, (i == mnist.result_) ? 1.0f : 0.2f, 0.2f };
        glColor3fv(color);
        glBegin(GL_QUADS);
        glVertex2f(x0, y0);
        glVertex2f(x1, y0);
        glVertex2f(x1, y1);
        glVertex2f(x0, y1);
        glEnd();
    }
}

int main(int argc, char** argv) {
    // Parse command line for --size N and --brush N
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--size" && i + 1 < argc) {
            int sz = std::stoi(argv[i + 1]);
            if (sz > 0) {
                DRAW_W = DRAW_H = sz;
            }
            ++i;
        } else if (std::string(argv[i]) == "--brush" && i + 1 < argc) {
            int bw = std::stoi(argv[i + 1]);
            if (bw < 1) bw = 1;
            if (bw > 20) bw = 20;
            brush_width = bw;
            ++i;
        }
    }
    WIN_W = DRAW_W * SCALE + 200;
    WIN_H = DRAW_H * SCALE;
    drawbuf.resize(DRAW_W * DRAW_H, 0xFFFFFFFF);

    if (!glfwInit()) return -1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* window = glfwCreateWindow(WIN_W, WIN_H, "GLFW Handwriting Recognizer", nullptr, nullptr);
    if (!window) { glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);
    glewExperimental = GL_TRUE;
    glewInit();

    // Setup PBO and texture
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, DRAW_W * DRAW_H * sizeof(uint32_t), nullptr, GL_STREAM_DRAW);
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, DRAW_W, DRAW_H, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    setup_quad();
    GLuint prog = create_program();

    // Setup ONNX model
    try { mnist = std::make_unique<MNIST>(); }
    catch (const Ort::Exception& e) { std::cerr << e.what() << std::endl; return -1; }

    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_pos_callback);

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        glClearColor(1, 1, 1, 1);
        glClear(GL_COLOR_BUFFER_BIT);

        // Update PBO/texture
        update_pbo_texture();

        // Draw drawing area (scaled up)
        glViewport(0, 0, DRAW_W * SCALE, DRAW_H * SCALE);
        glUseProgram(prog);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, tex);
        glUniform1i(glGetUniformLocation(prog, "uTex0"), 0);
        glBindVertexArray(vao);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, 0);
        glBindVertexArray(0);
        glUseProgram(0);

        // Draw bar graph
        glViewport(0, 0, WIN_W, WIN_H);
        draw_bar_graph(*mnist);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
    glDeleteBuffers(1, &ebo);
    glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &tex);
    glDeleteProgram(prog);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
