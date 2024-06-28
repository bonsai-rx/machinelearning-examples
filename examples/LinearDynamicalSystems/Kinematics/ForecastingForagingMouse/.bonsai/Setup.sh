#! /bin/bash

CURRENT_DIR="$(pwd)"
SETUP_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
cd $SETUP_SCRIPT_DIR

if [ ! -f "./Bonsai.exe" ]; then
    CONFIG="./Bonsai.config"
    if [ -f "$CONFIG" ]; then
        VERSION=$(xmllint --xpath '//PackageConfiguration/Packages/Package[@id="Bonsai"]/@version' "$CONFIG" | sed -e 's/^[^"]*"//' -e 's/"$//')
        RELEASE="https://github.com/bonsai-rx/bonsai/releases/download/$VERSION/Bonsai.zip"
    else
        RELEASE="https://github.com/bonsai-rx/bonsai/releases/latest/download/Bonsai.zip"
    fi
    wget $RELEASE -O "temp.zip"
    mv -f "NuGet.config" "temp.config"
    unzip -d "." -o "temp.zip"
    mv -f "temp.config" "NuGet.config"
    rm -rf "temp.zip"
    rm -rf "Bonsai32.exe"
fi

source ./activate
source ./run --no-editor
cd $CURRENT_DIR