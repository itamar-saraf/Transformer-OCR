inputfiles = ['dataset/annotation_test.txt', 'dataset/annotation_val.txt', 'dataset/annotation_train.txt']
outputfiles = ['dataset/test.txt', 'dataset/val.txt', 'dataset/train.txt']

for inputfile, outputfile in zip(inputfiles, outputfiles):

    with open(inputfile, 'r') as input:
        with open(outputfile, 'a') as ouptput:
            for line in input:
                seperated = line.split()
                seperated[1] = '\t' + seperated[0].split('_')[1] + '\n'
                prefix = seperated[0].split('.')
                seperated[0] = './dataset' + prefix[1] + '.jpg'
                path_label = ''.join(seperated)
                ouptput.write(path_label)
