#!/bin/sh
echo "Downloading MNLI data..."
wget https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip
unzip multinli_1.0.zip
rm multinli_1.0.zip
echo "Done!"
