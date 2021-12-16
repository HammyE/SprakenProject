name = "sports"
with open(f"raw_data/{name}_raw.txt") as file:
    text = file.read()
    lines = text.split("\n")
    index = 0
    with open(f"parsed_data/{name}.txt", 'w') as out:
        line = lines[index]
        low_line = line.lower()
        out.write(line.replace(f"#{name}", '').strip())
        print(line.replace(f"#{name}", '').strip())
        index += 1
        while index < len(lines):
            line = lines[index]
            # if line.startswith('"'):
            #     line = line[1:]
            #     passing = True
            #     while passing:
            #         index += 1
            #         line += ' ' + lines[index]
            #         if line.endswith('"'):
            #             line = line[:-1]
            #             passing = False
            line = line.lower()
            out.write('\n' + line.replace(f"#{name}", '').strip())
            print(line.replace(f"#{name}", '').strip())
            index += 1
