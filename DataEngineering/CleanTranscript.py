
def cleanFile(filename:str):
    new_script = []
    with open(filename, 'r') as transcript:
        script = transcript.readlines()
        script.pop(0)
        new_script = [line[line.rindex('\t')+1:-1] for line in script]
        new_script = ' '.join(new_script)


    return new_script





