import numpy as np
txtfile = '../1Aero/Train_Set_1/FS_P01_dev_001.txt'
opfile = 'processedinput'
with open(txtfile, 'r') as f:
    op = np.array([], dtype = 'int64')
    contents = f.read()
    xontents = contents.rstrip().split('\n')
    for i in range(len(xontents)):
        line1 = xontents[i].strip().split('\t')
        ran = int(100*(float(line1[1])))-int(100*(float(line1[0])))
        #print(ran)
        if line1[2] == 'NS':
            for inter in range(ran):
                op = np.append(op, 0)
        elif line1[2] == 'S':
            for inter in range(ran):
                op = np.append(op, 1)
    outputs = np.split(op, 90)
    outputs = np.array(list(outputs))
    print(outputs.shape)
    for i in range(90):
        print(outputs[i])
    np.save(opfile, op)