# Select the base image
#FROM nvcr.io/nvidia/pytorch:21.12-py3
#FROM python:3.9.7-slim
#FROM nvidia/cuda:11.4.0-runtime-ubuntu20.04
FROM nvidia/cuda:11.4.2-devel-ubuntu20.04
#FROM nvidia/cuda:11.6.2-runtime-ubuntu20.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
# installations for python
RUN apt-get update && apt-get install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget libbz2-dev liblzma-dev
# installations for blender and other
RUN apt-get update && apt-get install -y git subversion cmake libx11-dev libxxf86vm-dev libxcursor-dev libxi-dev libxrandr-dev libxinerama-dev libglew-dev libwayland-dev wayland-protocols libegl-dev libxkbcommon-dev libdbus-1-dev linux-libc-dev sudo
# install prolog dependencies
RUN apt-get update && apt-get install -y gringo swi-prolog
RUN apt-get upgrade -y
#RUN cp /usr/lib/python3.9/lib-dynload/_bz2.cpython-38-x86_64-linux-gnu.so  /usr/local/lib/python3.9/
# install python and required packages
RUN mkdir /home/python
WORKDIR /home/python
RUN wget https://www.python.org/ftp/python/3.9.7/Python-3.9.7.tgz
RUN tar xzf Python-3.9.7.tgz
WORKDIR /home/python/Python-3.9.7
RUN ./configure --enable-optimizations
RUN make install

COPY ./modules/requirements.txt /home/python/requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
#RUN pip3 install git+https://github.com/yuce/pyswip@master#egg=pyswip
RUN pip3 install -r /home/python/requirements.txt
RUN ln -s /usr/local/bin/python3.9 /usr/bin/python & ln -s /usr/local/bin/pip3.9 /usr/bin/pip

# get blender
RUN mkdir /home/blender-git
WORKDIR /home/blender-git
RUN git clone -b blender-v3.0-release --single-branch https://git.blender.org/blender.git

RUN mkdir /home/blender-git/lib
WORKDIR /home/blender-git/lib
RUN svn checkout https://svn.blender.org/svnroot/bf-blender/tags/blender-3.0-release/lib/linux_centos7_x86_64

WORKDIR /home/blender-git/blender
RUN make update

# install dependencies
#RUN /home/blender-git/blender/build_files/build_environment/install_deps.sh

# remove old libpng
#RUN mv /usr/lib/x86_64-linux-gnu/libpng.a /usr/lib/x86_64-linux-gnu/libpng_no_pic.a

# compile libpng.a with -fPIC flag
#RUN mkdir /home/libpng
#WORKDIR /home/libpng
#RUN wget https://download.sourceforge.net/libpng/libpng-1.6.37.tar.gz
#RUN tar xvfz libpng-1.6.37.tar.gz
#WORKDIR /home/libpng/libpng-1.6.37
#RUN ./configure --enable-shared --prefix=/home/libp --with-pic=yes
#RUN sudo make
#RUN mv /home/libpng/libpng.a /usr/lib/x86_64-linux-gnu/libpng.a

# use precompiled blender lib
#RUN wget https://github.com/Argmaster/PyR3/releases/download/bpy-binaries/libpng-with-pic-linux.tar.gz
#RUN tar xvfz libpng-with-pic-linux.tar.gz
#RUN mv /home/libpng/lib/libpng.a /usr/lib/x86_64-linux-gnu/libpng.a

WORKDIR /home/blender-git/blender
RUN rm ./CMakeLists.txt
COPY ./modules/CMakeLists.txt ./CMakeLists.txt
RUN make
RUN cp -a /home/blender-git/build_linux/bin/. /usr/local/lib/python3.9/site-packages/

#SHELL ["/bin/bash", "-c"]
#RUN echo 'alias python='python3'' >> ~/.bashrc
#RUN . ~/.bashrc


# install detectron
#RUN mkdir /home/detectron
#WORKDIR /home/detectron
#RUN git clone https://github.com/facebookresearch/detectron
#WORKDIR /home/detectron/detectron
#RUN make
#RUN export PYTHONPATH=$PYTHONPATH:/home/detectron/detectron/detectron
#RUN . ~/.bashrc

EXPOSE 8282
# create workdir
RUN mkdir /home/workdir
WORKDIR /home/workdir

# copy and extract precompiled blender module
#RUN tar xvfz /home/Master_thesis/modules/blender_module.tar.gz -C /usr/local/lib/python3.9/site-packages


ENV PYTHONPATH "${PYTHONPATH}:./"
ENV NVIDIA_VISIBLE_DEVICES all