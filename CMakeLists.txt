cmake_minimum_required(VERSION 3.16)
project(safety_service CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# LinQ TPU: AUTO — включить, если найдены libtpu и заголовки; ON — обязательно; OFF — только OpenCV DNN
set(WITH_LINQ "AUTO" CACHE STRING "Сборка с бэкендом LinQ TPU: AUTO | ON | OFF")

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio dnn)
find_package(Threads REQUIRED)

add_executable(safety_service
  src/main.cpp
  src/model.cpp
  src/backend_opencv.cpp
  src/backend_linq.cpp
  src/backend_remote.cpp
  src/tracker.cpp
  src/rules.cpp
  src/events.cpp
  src/camera.cpp
  src/web.cpp
)
target_include_directories(safety_service PRIVATE src third_party ${OpenCV_INCLUDE_DIRS})
target_link_libraries(safety_service PRIVATE ${OpenCV_LIBS} Threads::Threads)

# https:// для удалённого бэкенда (прокси Jupyter по https) — если есть libssl-dev
find_package(OpenSSL QUIET)
if(OPENSSL_FOUND)
  message(STATUS "OpenSSL ${OPENSSL_VERSION}: https для удалённого бэкенда")
  target_compile_definitions(safety_service PRIVATE CPPHTTPLIB_OPENSSL_SUPPORT=1)
  target_link_libraries(safety_service PRIVATE OpenSSL::SSL OpenSSL::Crypto)
else()
  message(STATUS "OpenSSL не найден: удалённый бэкенд только по http:// (sudo apt install libssl-dev)")
endif()

if(NOT WITH_LINQ STREQUAL "OFF")
  find_path(LIBTPU_INCLUDE_DIR libtpu/device.hpp)
  find_library(LIBTPU_LIBRARY tpu)
  find_library(UXS_LIBRARY uxs)
  if(LIBTPU_INCLUDE_DIR AND LIBTPU_LIBRARY)
    message(STATUS "LinQ TPU: ${LIBTPU_LIBRARY}")
    target_compile_definitions(safety_service PRIVATE HAVE_LINQ=1)
    target_include_directories(safety_service PRIVATE ${LIBTPU_INCLUDE_DIR})
    target_link_libraries(safety_service PRIVATE ${LIBTPU_LIBRARY})
    if(UXS_LIBRARY)
      target_link_libraries(safety_service PRIVATE ${UXS_LIBRARY})
    else()
      target_link_libraries(safety_service PRIVATE uxs)
    endif()
  elseif(WITH_LINQ STREQUAL "ON")
    message(FATAL_ERROR "WITH_LINQ=ON, но libtpu не найдена (установите libtpu и libtpu-dev)")
  else()
    message(STATUS "LinQ TPU: libtpu не найдена — сборка только с OpenCV DNN")
  endif()
endif()
