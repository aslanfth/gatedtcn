#!/bin/bash
fileid="1WnCI2m9DRZq51_LnAyIT8EsyJnXzgcaO"
filename="speech.zip"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
unzip speech.zip
rm speech.zip
rm cookie