#! /bin/bash
G='\033[1;32m'
N='\033[0m'
echo -e "${G}Building started${N}"
bazel build --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --config=opt //tensorflow/tools/pip_package:build_pip_package
echo -e "${G}Building finished, generating whl file ....${N}"
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
echo -e "${G}Copying new whl to safe place${N}"
cp -r --backup=t /tmp/tensorflow_pkg/* ../tensorflow_whl_build/automated_builds
echo -e "${G}Deploy new whl${N}"
sudo -H pip3 install --upgrade ../tensorflow_whl_build/automated_builds/tensorflow-1.8.0-cp36-cp36m-linux_x86_64.whl
echo -e "${G}New whl deployed"
