Run the `download_data.sh` shell script to download the data from its URL and unzip it.
When it asks "Would you like to replace the existing file: ...", type in N (for no)

```
bash download_data.sh
```

Notes
- The data is stored as a 7zip file (with extension `.7z`), so we use the `7za`
  Linux command to uncompress it (with the `e` flag, for "extract")
- We have to pass a user agent to `wget`, otherwise it fails with 403 Forbidden
  error (likely because the site thinks that a bot, etc. is trying to access
  the data)

## Dataset
The SAMSum dataset contains about 16k messenger-like conversations with summaries. Conversations were created and written down by linguists fluent in English. Linguists were asked to create conversations similar to those they write on a daily basis, reflecting the proportion of topics of their real-life messenger convesations. The style and register are diversified - conversations could be informal, semi-formal or formal, they may contain slang words, emoticons and typos. Then, the conversations were annotated with summaries. It was assumed that summaries should be a concise brief of what people talked about in the conversation in third person.
The SAMSum dataset was prepared by Samsung R&D Institute Poland and is distributed for research purposes (non-commercial licence: CC BY-NC-ND 4.0).

## Paper
The dataset and experiments performed using it were described in paper: "SAMSum Corpus: A Human-annotated Dialogue Dataset for Abstractive Summarization". Please cite our paper if you use this dataset:

```
@inproceedings{gliwa-etal-2019-samsum,
    title = "{SAMS}um Corpus: A Human-annotated Dialogue Dataset for Abstractive Summarization",
    author = "Gliwa, Bogdan  and
      Mochol, Iwona  and
      Biesek, Maciej  and
      Wawer, Aleksander",
    booktitle = "Proceedings of the 2nd Workshop on New Frontiers in Summarization",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D19-5409",
    doi = "10.18653/v1/D19-5409",
    pages = "70--79"
}
```
