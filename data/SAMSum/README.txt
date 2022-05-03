Run the `download_data.sh` shell script to download the data from its URL and unzip it.
When it asks "Would you like to replace the existing file: ...", type in N (for no).

```
bash download_data.sh
```

Notes
- The data is stored as a 7zip file (with extension `.7z`), so we use the `7za`
  Linux command to uncompress it (with the `e` flag, for "extract")
- We have to pass a user agent to `wget`, otherwise it fails with 403 Forbidden
  error (likely because the site thinks that a bot, etc. is trying to access
  the data)
- If your Linux distribution does not have the 7za command, you can install it;
  if that fails, you can directly download the data from the link
  https://arxiv.org/src/1911.12237v2/anc/corpus.7z
  and move the `train.json`, `test.json`, and `val.json` files manually into
  this directory

Data Format
- The data is stored in separate `train.json`, `test.json`, and `val.json` files
- Each file is a JSON containing a list of objects, as follows
```json
[
  {
    "id": "13862856",
    "summary": "Hannah needs Betty's number but Amanda doesn't have it. She needs to contact Larry.",
    "dialogue": "Hannah: Hey, do you have Betty's number?\nAmanda: Lemme check\nHannah: <file_gif>\nAmanda: Sorry, can't find it.\nAmanda: Ask Larry\nAmanda: He called her last time we were at the park together\nHannah: I don't know him well\nHannah: <file_gif>\nAmanda: Don't be shy, he's very nice\nHannah: If you say so..\nHannah: I'd rather you texted him\nAmanda: Just text him 🙂\nHannah: Urgh.. Alright\nHannah: Bye\nAmanda: Bye bye"
  },
  {
    "id": "13729565",
    "summary": "Eric and Rob are going to watch a stand-up on youtube.",
    "dialogue": "Eric: MACHINE!\r\nRob: That's so gr8!\r\nEric: I know! And shows how Americans see Russian ;)\r\nRob: And it's really funny!\r\nEric: I know! I especially like the train part!\r\nRob: Hahaha! No one talks to the machine like that!\r\nEric: Is this his only stand-up?\r\nRob: Idk. I'll check.\r\nEric: Sure.\r\nRob: Turns out no! There are some of his stand-ups on youtube.\r\nEric: Gr8! I'll watch them now!\r\nRob: Me too!\r\nEric: MACHINE!\r\nRob: MACHINE!\r\nEric: TTYL?\r\nRob: Sure :)"
  },
  {
    "id": "13680171",
    "summary": "Lenny can't decide which trousers to buy. Bob advised Lenny on that topic. Lenny goes with Bob's advice to pick the trousers that are of best quality.",
    "dialogue": "Lenny: Babe, can you help me with something?\r\nBob: Sure, what's up?\r\nLenny: Which one should I pick?\r\nBob: Send me photos\r\nLenny:  <file_photo>\r\nLenny:  <file_photo>\r\nLenny:  <file_photo>\r\nBob: I like the first ones best\r\nLenny: But I already have purple trousers. Does it make sense to have two pairs?\r\nBob: I have four black pairs :D :D\r\nLenny: yeah, but shouldn't I pick a different color?\r\nBob: what matters is what you'll give you the most outfit options\r\nLenny: So I guess I'll buy the first or the third pair then\r\nBob: Pick the best quality then\r\nLenny: ur right, thx\r\nBob: no prob :)"
  },
]
```

# Original README (from SAMSum)

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
