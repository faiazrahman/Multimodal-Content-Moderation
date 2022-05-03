#!/bin/sh
echo "Downloading SAMSum data..."
wget --user-agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10.8; rv:21.0) Gecko/20100101 Firefox/21.0" https://arxiv.org/src/1911.12237v2/anc/corpus.7z
7za e corpus.7z
rm corpus.7z
echo "Done!"
