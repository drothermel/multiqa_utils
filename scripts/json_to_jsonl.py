import json
import jsonlines
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Json to JsonLines")
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    filename = args.input
    outname = args.input + "l"
    
    jsondata = json.load(open(filename))
    writer = jsonlines.Writer(open(outname, 'w+'))
    writer.write_all(jsondata)
    writer.close()
    print("Wrote:", outname)



