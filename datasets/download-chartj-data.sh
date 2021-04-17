#!/bin/bash
fileid="1OvsqMY7QqDwBUsZS1wLOhHyZWP-ugDSp"
filename="chartj.zip"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
unzip chartj.zip
rm chartj.zip
rm cookie