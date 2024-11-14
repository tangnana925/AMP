# 使用一个基础镜像，这里我们使用 Python 3.6

FROM nvcr.io/nvidia/pytorch:21.02-py3
# FROM nvcr.io/nvidia/pytorch:20.07-py3

# 设置pip主要源和备用源(切换为国内源，如不是在国内请忽略
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/ && \
pip config set global.extra-index-url https://pypi.org/simple/


# RUN apt-get update && apt-get install -y software-properties-common
# RUN add-apt-repository ppa:deadsnakes/ppa
# RUN apt-get update && apt-get install -y python${PYTHON_VERSION}

# # 使Python 3.7成为默认版本，这样子会导致在基础镜像中基于3.6安装的软件无法使用
# RUN update-alternatives --install $(which python) python3 $(which python3.6) 1
# RUN update-alternatives --install $(which python) python3 $(which python3.7) 2
# RUN update-alternatives --set python3 $(which python3.7)

# # 同样处理pip
# RUN update-alternatives --install /usr/bin/pip3 pip3 $(which pip3.6) 1
# RUN update-alternatives --install /usr/bin/pip3 pip3 /usr/bin/pip3.7 2
# RUN update-alternatives --set pip3 /usr/bin/pip3.7

# # 创建软链接
# RUN ln -sf /usr/bin/python3.7 /opt/conda/bin/python

# 清理APT缓存
# RUN apt-get clean && rm -rf /var/lib/apt/lists/*


# 更新 pip
RUN python3 -m pip install --upgrade pip


# 设置工作目录
WORKDIR /root

# 以可编辑模式安装你的应用
RUN python3 -m pip install ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests
# RUN python3 -m pip install tqdm spur torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
RUN python3 -m pip install tqdm torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

ENV HOME=/root

# 运行你的应用
CMD ["python3","-u","-m","your_applications"]
