# Download/Copy model in build process may cause out of memory
# so get an RWKV model before running docker build in local
# $ wget -O rwkv-model.pth https://huggingface.co/BlinkDL/rwkv-4-pile-169m/resolve/main/RWKV-4-Pile-169M-20220807-8023.pth?download=true

# Convert model format
# $ python python/convert_pytorch_to_ggml.py rwkv-model.pth rwkv-model.bin FP16
# $ python python/quantize.py rwkv-model.bin rwkv-model-Q5_1.bin Q5_1
# $ rm rwkv-model.pth rwkv-model.bin

FROM python:3.9.18-slim
WORKDIR /home

# slim has no CMAKE_CXX_COMPILER: need "gcc clang clang-tools"
# https://stackoverflow.com/a/69866667
RUN apt-get update && \
    apt-get install -y gcc clang clang-tools cmake && \
    pip install torch numpy tokenizers Flask flask-cors

COPY . .

# Build rwkv library
RUN cmake .
# **Anaconda & M1 users**: please verify that `CMAKE_SYSTEM_PROCESSOR: arm64` after running `cmake .` — if it detects `x86_64`, edit the `CMakeLists.txt` file under the `# Compile flags` to add `set(CMAKE_SYSTEM_PROCESSOR "arm64")`.
RUN cmake --build . --config Release

# start Flask chat server
EXPOSE 5050
CMD ["flask", "--app", "python/flask_chat_api", "run", "-h", "0.0.0.0", "-p", "5050"]

# Run up container by following command
# $ docker run -it --rm -p 5050:5050 -v $(pwd)/rwkv-model-Q5_1.bin:/home/rwkv-model-Q5_1.bin rwkv.cpp