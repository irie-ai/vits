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
    "src/commons.cpp"
    "src/modules.cpp"
    "src/attentions.cpp"
    "src/models.cpp"
    "src/losses.cpp"
    "src/mel_processing.cpp"
    "src/text_processing.cpp"
    "src/data_utils.cpp"
    "src/utils.cpp"
    "src/monotonic_align.cpp"
    "src/inference.cpp")

target_include_directories(vits_core PUBLIC
    include)

if(MSVC)
    target_compile_options(vits_core PUBLIC /utf-8)
endif()

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

add_executable(hello_vits
    "examples/hello_vits.cpp")

target_link_libraries(hello_vits PRIVATE vits_core)

add_test(NAME example_hello_vits COMMAND hello_vits)

add_executable(load_model
    "examples/load_model.cpp")

target_link_libraries(load_model PRIVATE vits_core)

add_executable(vits_inference
    "examples/inference.cpp")

target_link_libraries(vits_inference PRIVATE vits_core)

add_executable(vits_benchmark
    "examples/benchmark.cpp")

target_link_libraries(vits_benchmark PRIVATE vits_core)

add_executable(ljs_train_smoke
    "examples/ljs_train_smoke.cpp")

target_link_libraries(ljs_train_smoke PRIVATE vits_core)

add_executable(test_commons
    "tests/test_commons.cpp")

target_link_libraries(test_commons PRIVATE vits_core)

add_test(NAME test_commons COMMAND test_commons)

add_executable(test_modules
    "tests/test_modules.cpp")

target_link_libraries(test_modules PRIVATE vits_core)

add_test(NAME test_modules COMMAND test_modules)

add_executable(test_attentions
    "tests/test_attentions.cpp")

target_link_libraries(test_attentions PRIVATE vits_core)

add_test(NAME test_attentions COMMAND test_attentions)

add_executable(test_models
    "tests/test_models.cpp")

target_link_libraries(test_models PRIVATE vits_core)

add_test(NAME test_models COMMAND test_models)

add_executable(test_losses
    "tests/test_losses.cpp")

target_link_libraries(test_losses PRIVATE vits_core)

add_test(NAME test_losses COMMAND test_losses)

add_executable(test_mel_processing
    "tests/test_mel_processing.cpp")

target_link_libraries(test_mel_processing PRIVATE vits_core)

add_test(NAME test_mel_processing COMMAND test_mel_processing)

add_executable(test_text_processing
    "tests/test_text_processing.cpp")

target_link_libraries(test_text_processing PRIVATE vits_core)

add_test(NAME test_text_processing COMMAND test_text_processing)

add_executable(test_data_utils
    "tests/test_data_utils.cpp")

target_link_libraries(test_data_utils PRIVATE vits_core)

add_test(NAME test_data_utils COMMAND test_data_utils)

add_executable(test_utils
    "tests/test_utils.cpp")

target_link_libraries(test_utils PRIVATE vits_core)

add_test(NAME test_utils COMMAND test_utils)

add_executable(test_monotonic_align
    "tests/test_monotonic_align.cpp")

target_link_libraries(test_monotonic_align PRIVATE vits_core)

add_test(NAME test_monotonic_align COMMAND test_monotonic_align)

add_executable(test_inference
    "tests/test_inference.cpp")

target_link_libraries(test_inference PRIVATE vits_core)

add_test(NAME test_inference COMMAND test_inference)

add_executable(test_training
    "tests/test_training.cpp")

target_link_libraries(test_training PRIVATE vits_core)

add_test(NAME test_training COMMAND test_training)

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

    add_custom_command(TARGET hello_vits
        POST_BUILD
        COMMAND ${CMAKE_COMMAND}
            -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:hello_vits>
    )

    add_custom_command(TARGET load_model
        POST_BUILD
        COMMAND ${CMAKE_COMMAND}
            -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:load_model>
    )

    add_custom_command(TARGET vits_inference
        POST_BUILD
        COMMAND ${CMAKE_COMMAND}
            -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:vits_inference>
    )

    add_custom_command(TARGET vits_benchmark
        POST_BUILD
        COMMAND ${CMAKE_COMMAND}
            -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:vits_benchmark>
    )

    add_custom_command(TARGET ljs_train_smoke
        POST_BUILD
        COMMAND ${CMAKE_COMMAND}
            -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:ljs_train_smoke>
    )

    add_custom_command(TARGET test_modules
        POST_BUILD
        COMMAND ${CMAKE_COMMAND}
            -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:test_modules>
    )

    add_custom_command(TARGET test_attentions
        POST_BUILD
        COMMAND ${CMAKE_COMMAND}
            -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:test_attentions>
    )

    add_custom_command(TARGET test_models
        POST_BUILD
        COMMAND ${CMAKE_COMMAND}
            -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:test_models>
    )

    add_custom_command(TARGET test_losses
        POST_BUILD
        COMMAND ${CMAKE_COMMAND}
            -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:test_losses>
    )

    add_custom_command(TARGET test_mel_processing
        POST_BUILD
        COMMAND ${CMAKE_COMMAND}
            -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:test_mel_processing>
    )

    add_custom_command(TARGET test_text_processing
        POST_BUILD
        COMMAND ${CMAKE_COMMAND}
            -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:test_text_processing>
    )

    add_custom_command(TARGET test_data_utils
        POST_BUILD
        COMMAND ${CMAKE_COMMAND}
            -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:test_data_utils>
    )

    add_custom_command(TARGET test_utils
        POST_BUILD
        COMMAND ${CMAKE_COMMAND}
            -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:test_utils>
    )

    add_custom_command(TARGET test_monotonic_align
        POST_BUILD
        COMMAND ${CMAKE_COMMAND}
            -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:test_monotonic_align>
    )

    add_custom_command(TARGET test_inference
        POST_BUILD
        COMMAND ${CMAKE_COMMAND}
            -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:test_inference>
    )

    add_custom_command(TARGET test_training
        POST_BUILD
        COMMAND ${CMAKE_COMMAND}
            -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:test_training>
    )

endif()
