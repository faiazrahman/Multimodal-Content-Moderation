MMHS150K Dataset
Paper: "Exploring Hate Speech Detection in Multimodal Publications"
Raul Gomez, Jaume Gibert, Lluis Gomez, Dimosthenis Karatzas
in WACV 2020
- More info in : https://gombru.github.io/2019/10/09/MMHS/

# Dataset Contents
	MMHS150K_GT.json
		Python dict with an entry per tweet, where key is the tweet ID and fields are:
			tweet_url
			labels: array with 3 numeric labels [0-5] indicating the label by each one of the three AMT annotators
					0 - NotHate, 1 - Racist, 2 - Sexist, 3 - Homophobe, 4 - Religion, 5 - OtherHate
			img_url
			tweet_text
			labels_str: array with the 3 labels strings

    splits/train_ids.txt
	splits/val_ids.txt
	splits/test_ids.txt
		Contain the tweet IDs used in the 3 splits

    img_resized/
		Images resized such that their shortest size is 500 pixels
        These are .jpg files with filename the same as the tweet ID

	img_txt/
		Text extracted from the images using OCR (optical character recognition)
        These are .json files with filename the same as the tweet ID
        e.g. Contains only a single line
        ```
        {"img_text": "STRONG BORDERS, NO CRIME"}
        ```

	hatespeech_keywords.txt
		Contains the keywords that were used to gather the tweets

# Data Preprocessing and Dataloader Overview
Thus, our data preprocessing will
- Load the Python dict mapping tweet_id to tweet_url, labels, img_url,
  tweet_text, labels
- Select the majority label (if none, do not include this example)
- Get the tweet id from the dict key
- Get the text from the dict value field 'tweet_text'
- Get the image from img_resized/{tweet id}
- Get the image's OCR text from img_txt/{tweet id}
- Create a pd.DataFrame with ['tweet_id', 'text', 'image_ocr_text'] and save it
  to a .pkl

Then, our torch.utils.data.Dataset will
- Load the pd.DataFrame from the serialized .pkl file
- Return the tensors for the text, image, and image_ocr_text
