cmake_minimum_required(VERSION 3.10)
project(MyDigitNet)

# Установка стандарта C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Указать путь к директории с заголовочными файлами
include_directories(include)

# Добавляем файлы исходного кода
add_executable(${PROJECT_NAME}
src/main.cpp
src/glad.c
src/activateFunction.cpp
src/dataset.cpp
src/Matrix.cpp
src/netWork.cpp
)
# Устанавливаем рабочию директорию
set_property(TARGET ${PROJECT_NAME} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
# Добавляем путь к исходникам кода(Нужно для корректности относительных путей)
add_definitions(-DRESOURCE_PATH="${CMAKE_SOURCE_DIR}/src/")
# Определяем платформу и подключаем нужные библиотеки
if(WIN32) # Windows
    message(STATUS "Configuring for Windows")
    
    
    #glad
    set(GLAD_SOURCES ${CMAKE_SOURCE_DIR}/src/glad.c)
    add_library(glad ${GLAD_SOURCES})

    #GLFW
    add_subdirectory(lib/glfw)
    include_directories(lib/glfw/include)
    target_link_libraries(${PROJECT_NAME} glad glfw ${GLFW_LIBRARIES} )


elseif(APPLE) # macOS
    message(STATUS "Configuring for macOS")
    
    # Установка пути к заголовкам и библиотекам GLFW через Homebrew
    find_package(PkgConfig REQUIRED)
    pkg_search_module(GLFW REQUIRED glfw3)
    
    # Линкуем библиотеки OpenGL и GLFW
    target_link_libraries(${PROJECT_NAME} ${GLFW_LIBRARIES})
    target_link_libraries(${PROJECT_NAME} "-framework OpenGL")
    
    # Путь к заголовочным файлам GLFW
    target_include_directories(${PROJECT_NAME} PRIVATE ${GLFW_INCLUDE_DIRS})
    target_include_directories(${PROJECT_NAME} PRIVATE lib/glad/include)

elseif(UNIX) # Linux
    message(STATUS "Configuring for Linux")
    
    # Установка пакетов с помощью PkgConfig
    find_package(PkgConfig REQUIRED)
    pkg_search_module(GLFW REQUIRED glfw3)
    find_package(OpenGL REQUIRED)
    
    # Линкуем библиотеки OpenGL и GLFW
    target_link_libraries(${PROJECT_NAME} ${GLFW_LIBRARIES})
    target_link_libraries(${PROJECT_NAME} ${OPENGL_gl_LIBRARY})
    
    # Путь к заголовочным файлам GLFW и GLAD
    target_include_directories(${PROJECT_NAME} PRIVATE ${GLFW_INCLUDE_DIRS})
    target_include_directories(${PROJECT_NAME} PRIVATE ${OPENGL_INCLUDE_DIR})
    target_include_directories(${PROJECT_NAME} PRIVATE lib/glad/include)

else()
    message(FATAL_ERROR "Unsupported platform")
endif()