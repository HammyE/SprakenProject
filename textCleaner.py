with open("raw_data/excitedtag.txt") as file:
    text = file.read()
    lines = text.split("\n")
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith('"'):
            line = line[1:]
            passing = True
            while passing:
                index += 1
                line += ' ' + lines[index]
                if line.endswith('"'):
                    line = line[:-1]
                    passing = False
        print(line.replace("#excited", '').replace("#Excited", '').strip())
        index += 1
