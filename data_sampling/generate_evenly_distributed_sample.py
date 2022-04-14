import argparse
import pandas as pd

TRAIN_DATA_DEFAULT = "../data/Fakeddit/multimodal_train.tsv"
TEST_DATA_DEFAULT = "../data/Fakeddit/multimodal_test_public.tsv"
TRAIN_SAMPLE_SIZE = 10000
TEST_SAMPLE_SIZE = 1000
DATA_DELIMITER = "\t"

def generate_and_save_sample(data_path: str, sample_size: int) -> str:
    """ Returns filepath to saved sample """
    print(f"Running for {data_path}...")
    print(f"  Sample size: {sample_size}")
    target_num_positive = sample_size // 2
    target_num_negative = sample_size - target_num_positive
    target_amounts_for_each_class = {
        "0": target_num_positive,
        "1": target_num_negative // 5,
        "2": target_num_negative // 5,
        "3": target_num_negative // 5,
        "4": target_num_negative // 5,
        "5": target_num_negative // 5,
    }
    curr_amounts_for_each_class = {
        "0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0,
    }

    output_filename = data_path.replace(".tsv", "") + "_" + str(sample_size) + "_sampled.tsv"
    print(f"Creating sample and saving to to {output_filename}")

    curr_num_samples = 0
    on_header_line = True
    with open(output_filename, "w") as output_file:
        with open(data_path, "r") as data_file:
            for line in data_file:
                # Always copy over the first line (the header)
                if on_header_line:
                    output_file.write(line)
                    on_header_line = False
                    continue

                # The 6-way label is the last column in the raw data
                label = line.split(DATA_DELIMITER)[-1].strip()
                if (label in target_amounts_for_each_class.keys()
                    and curr_amounts_for_each_class[label] < target_amounts_for_each_class[label]):
                    output_file.write(line)
                    curr_amounts_for_each_class[label] += 1
                    curr_num_samples += 1

                if curr_num_samples >= sample_size:
                    break

    # Verify even distribution of 6-way labels
    sample_df = pd.read_csv(output_filename, sep="\t", header=0)
    df = sample_df.groupby("2_way_label")["id"].nunique()
    print(df)
    df = sample_df.groupby("3_way_label")["id"].nunique()
    print(df)
    df = sample_df.groupby("6_way_label")["id"].nunique()
    print(df)

    return output_filename

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", type=str, default=TRAIN_DATA_DEFAULT)
    parser.add_argument("--test_data_path", type=str, default=TEST_DATA_DEFAULT)
    parser.add_argument("--train_sample_size", type=int, default=TRAIN_SAMPLE_SIZE)
    parser.add_argument("--test_sample_size", type=int, default=TEST_SAMPLE_SIZE)
    args = parser.parse_args()

    generate_and_save_sample(args.train_data_path, args.train_sample_size)
    generate_and_save_sample(args.test_data_path, args.test_sample_size)
