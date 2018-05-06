import re
import logging
import argparse

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)
logger = logging.getLogger('NameCleaner')

class NameCleaner(object):

    SWAPS = {
        "\"": "'",
        "`": "'",
        "à": "a",
        "á": "a",
        "è": "e",
        "é": "e",
        "ë": "e",
        "í": "i",
        "ñ": "n",
        "ó": "o",
        "ö": "o",
        "ú": "u",
        "ü": "u",
        "‘": "'",
        "’": "'",
        "!": "",
        "-": " ",
        ",": ""
    }

    DROPS = re.compile(r'[\(\)/]')

    def __init__(self, lines):
        self.lines = lines


    def strip(self):
        self.lines = [x.strip() for x in self.lines]

    def lower(self):
        self.lines = [x.lower() for x in self.lines]

    def drop_some(self):
        processed = []

        re.compile(r'\d\$,')
        for line in self.lines:
            if not NameCleaner.DROPS.findall(line):
                processed.append(line)
            else:
                logger.info("Dropping line " + line)

        self.lines = processed



    def do_swaps(self):
        items = NameCleaner.SWAPS.items()
        for i, line in enumerate(self.lines):

            orig = line

            for bad, good in items:
                line = line.replace(bad, good)

            if line != orig:
                logging.info("Replaced chars in '" + orig + "' to give '" + line + "'")
            self.lines[i] = line

    def results(self):
        return self.lines



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Clean a list of names')
    parser.add_argument('--file', help='The file to clean', required=True)

    args = parser.parse_args()

    with open(args.file) as f:
        content = f.readlines()

    logger.info("Loaded " + str(len(content)) + " lines")

    nc = NameCleaner(content)
    nc.strip()
    nc.lower()
    nc.drop_some()
    nc.do_swaps()

    with open(args.file + ".cleaned.txt", 'w') as f:
        for result in nc.results():
            f.write("%s\n" % result)