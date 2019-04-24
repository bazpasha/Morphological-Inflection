import argparse
import data_processing
import os
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True, help="Input XML OpenCorpora file")
    parser.add_argument("-o", "--output-directory", type=str, required=True, help="Output directory")
    parser.add_argument("-t", "--train-part", type=float, default=0.8, help="Train part")
    parser.add_argument("-v", "--val-part", type=float, default=0.1, help="Validate part")
    parser.add_argument("-p", "--parts-of-speech", type=str, nargs="+", help="Parts of speech")
    args = parser.parse_args()

    data = data_processing.parse_and_filter(args.input, args.parts_of_speech)
    mapper = data_processing.Mapper(data)

    mapper.to_json(os.path.join(args.output_directory, "mapper.json"))
    train, val, test = data_processing.train_val_test_split(d, train_part=args.train_part, val_part=args.val_part)

    with open(os.path.join(args.output_directory, "train.json"), "w") as _out:
        json.dump(train, _out)

    with open(os.path.join(args.output_directory, "validate.json"), "w") as _out:
        json.dump(val, _out)

    with open(os.path.join(args.output_directory, "test.json"), "w") as _out:
        json.dump(test, _out)