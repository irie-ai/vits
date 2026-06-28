# 必要なCMakeの最低バージョン
cmake_minimum_required(VERSION 3.8)

# プロジェクト名
project(VITS)

# 使用するC++のバージョン
set(CMAKE_CXX_STANDARD 17)

# C++17を必須にする
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# LibTorchがあるフォルダを指定
# libtorchフォルダを展開した場所を指定する
# 例：C:/libtorch
set(CMAKE_PREFIX_PATH "C:/libtorch")

# LibTorchを検索する
# 成功するとTorch_INCLUDE_DIRSやTORCH_LIBRARIESなどが使える
find_package(Torch REQUIRED)

enable_testing()

add_library(vits_core
    "src/commons.cpp")

target_include_directories(vits_core PUBLIC
    include)

# LibTorchライブラリをリンクする
# これを書かないとtorch::Tensorなどが使えない
target_link_libraries(vits_core PUBLIC "${TORCH_LIBRARIES}")

# 実行ファイルを作成する
#
# VITS     ← 実行ファイル名
# main.cpp     ← コンパイル対象
add_executable(VITS
    "main.cpp")

target_link_libraries(VITS PRIVATE vits_core)

add_executable(test_commons
    "tests/test_commons.cpp")

target_link_libraries(test_commons PRIVATE vits_core)

add_test(NAME test_commons COMMAND test_commons)

# Visual Studio(MSVC)用の設定
if(MSVC)

    # libtorch/lib 以下にあるDLLを取得
    file(GLOB TORCH_DLLS
        "${TORCH_INSTALL_PREFIX}/lib/*.dll"
    )

    # ビルド完了後
    add_custom_command(TARGET VITS

        # ビルド後に実行
        POST_BUILD

        # DLLをexeと同じフォルダへコピーする
        COMMAND ${CMAKE_COMMAND}
            -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:VITS>
    )

    add_custom_command(TARGET test_commons
        POST_BUILD
        COMMAND ${CMAKE_COMMAND}
            -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:test_commons>
    )

endif()
